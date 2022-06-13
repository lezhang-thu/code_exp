import numpy as np
import torch
import torch.nn.functional as F

from captioning.utils.rewards import get_scores
import captioning.ctree.cytree as tree


class MBTrainer(torch.nn.Module):
    def __init__(self, trainer, opt):
        super(MBTrainer, self).__init__()
        self.trainer = trainer
        self.opt = opt

    def forward(self, fc_feats, att_feats, att_masks, sample_probs, gen_result,
                targ_vs):
        fetch_values = list()
        _, log_probs = self.trainer(fc_feats,
                                    att_feats,
                                    att_masks,
                                    gen_result=gen_result,
                                    opt={
                                        'sample_method':
                                        self.opt.train_sample_method,
                                        'beam_size': self.opt.train_beam_size,
                                        'sample_n': 1
                                    },
                                    fetch_values=fetch_values,
                                    mode='sample')
        values = torch.cat(fetch_values, 1)
        mask = gen_result > 0
        mask = torch.cat(
            [mask.new_full((mask.shape[0], 1), True), mask[:, :-1]], 1)
        targ_vs = targ_vs.unsqueeze(1).expand_as(values)

        x = (0.5 * (values - targ_vs)**2 * mask[:, :values.shape[1]]).mean()
        y = (mask.unsqueeze(-1) * sample_probs * log_probs).mean()
        loss = x
        return loss


class Runner:
    def __init__(self, predictor, nenvs, noptepochs, envsperbatch, opt,
                 loader):
        self.predictor = predictor
        self.nenvs = nenvs
        self.noptepochs = noptepochs
        self.envsperbatch = envsperbatch
        self.opt = opt
        self.loader = loader

        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.discount = 1.0
        self.value_delta_max = 0.01
        self.num_simulations = 10
        self.num_epochs = 10

        def f(epoch):
            if epoch < 0.5 * self.num_epochs:
                return 1.0
            elif epoch < 0.75 * self.num_epochs:
                return 0.5
            return 0.25

        self.temperature = f

    @torch.no_grad()
    def run(self, iteration, epoch):
        mb_fc_feats = []
        mb_att_feats = []
        mb_att_masks = []

        mb_sample_probs = []
        mb_gen_result = []
        mb_values = []

        # collect data - start
        for _ in range(self.nenvs // self.opt.batch_size):
            data = self.loader.get_batch('train')
            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1

            for x, y in zip(
                [data['fc_feats'], data['att_feats'], data['att_masks']],
                [mb_fc_feats, mb_att_feats, mb_att_masks]):
                if x is not None:
                    y.append(x.numpy())

            x = [data['fc_feats'], data['att_feats'], data['att_masks']]
            x = [None if _ is None else _.cuda() for _ in x]
            fc_feats, att_feats, att_masks = x

            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self.predictor._prepare_feature(
                fc_feats, att_feats, att_masks)

            # minimax value storage
            min_max_stats_lst = tree.MinMaxStatsList(self.opt.batch_size)
            min_max_stats_lst.set_delta(self.value_delta_max)

            roots = tree.Roots(self.opt.batch_size, self.opt.vocab_size + 1)
            noises = [
                np.random.dirichlet([self.root_dirichlet_alpha] *
                                    (self.opt.vocab_size + 1)).astype(
                                        np.float32).tolist()
                for _ in range(self.opt.batch_size)
            ]
            prefixes = p_att_feats.new_zeros((self.opt.batch_size, 0),
                                             dtype=torch.long)

            policies, _ = self.predictor._prefix_next(p_fc_feats, p_att_feats,
                                                      pp_att_feats,
                                                      p_att_masks, prefixes)
            roots.prepare(self.root_exploration_fraction, noises,
                          policies.tolist())

            seq_probs = []
            unfinished = p_att_feats.new_ones((self.opt.batch_size, ),
                                              dtype=torch.bool)

            # env - start
            for idx_outer in range(self.opt.max_length):
                # simulations - start
                for idx_inner in range(self.num_simulations):
                    # prepare a result wrapper to transport results between Python and C++ parts
                    results = tree.ResultsWrapper(self.opt.batch_size)
                    tree.batch_traverse(roots, self.pb_c_base, self.pb_c_init,
                                        self.discount, min_max_stats_lst,
                                        results)

                    # type of `v`: `list` of `list` (Python)
                    v = roots.get_trajectories()
                    lens = np.asarray([len(item) for item in v])
                    mask = lens[:, None] > np.arange(lens.max())
                    x = np.zeros(mask.shape, dtype=int)
                    x[mask] = np.concatenate(v)

                    eos_env = np.zeros((self.opt.batch_size, ), dtype=bool)
                    eos_env[lens == 0] = True
                    eos_env[lens > 0] = x[lens > 0, lens[lens > 0] - 1] == 0
                    eos_env = torch.from_numpy(eos_env).cuda()

                    t = torch.cat([prefixes, torch.from_numpy(x).cuda()], -1)
                    policies, values = self.predictor._prefix_next(
                        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, t)
                    policies[eos_env, 0] = -1.0
                    if eos_env.sum() > 0:
                        x = get_scores(
                            np.asarray(data['gts'])[eos_env.cpu().numpy()],
                            t[eos_env], self.opt)
                        values[eos_env] = torch.from_numpy(x).float().cuda()
                    tree.batch_back_propagate(self.discount, values.tolist(),
                                              policies.tolist(),
                                              min_max_stats_lst, results)
                # simulations - end
                x = roots.get_distributions()
                z = [1 for _ in range(self.opt.vocab_size + 1)]
                for idx, y in enumerate(x):
                    # if root is terminal, [] is returned
                    if len(y) == 0:
                        x[idx] = z
                x = np.asarray(x)**(1 / self.temperature(epoch))
                x = x / x.sum(-1, keepdims=True)
                seq_probs.append(x)

                it = torch.distributions.Categorical(
                    probs=torch.from_numpy(x).cuda()).sample()
                it[~unfinished] = 0

                for idx, y in enumerate(unfinished):
                    if y:
                        roots.update_with_move(idx, it[idx])
                unfinished &= it != 0

                prefixes = torch.cat([prefixes, it.unsqueeze_(-1)], -1)
                if unfinished.sum() == 0:
                    break
            # env - end
            roots.release_forest()

            seq_probs = np.stack(seq_probs, 1)
            x = np.zeros((self.opt.batch_size, self.opt.max_length,
                          self.opt.vocab_size + 1),
                         dtype=np.float32)
            x[:, :seq_probs.shape[1], :] = seq_probs
            seq_probs = x

            x = torch.zeros((self.opt.batch_size, self.opt.max_length),
                            dtype=np.long,
                            device=prefixes.device)
            x[:, :prefixes.shape[1]] = prefixes
            prefixes = x

            rewards_sample = np.asarray(get_scores(data['gts'], prefixes,
                                                   self.opt),
                                        dtype=np.float32)
            prefixes = prefixes.cpu().numpy()

            trajectory = [seq_probs, prefixes, rewards_sample]
            for x, y in zip(trajectory,
                            [mb_sample_probs, mb_gen_result, mb_values]):
                y.append(x)
        # collect data - end

        max_att_num = np.max([_.shape[1] for _ in mb_att_feats])
        for k, x in enumerate(mb_att_feats):
            after = max_att_num - x.shape[1]
            mb_att_feats[k] = np.pad(x, ((0, 0), (0, after), (0, 0)),
                                     mode="constant")
            mb_att_masks[k] = np.pad(mb_att_masks[k], ((0, 0), (0, after)),
                                     mode="constant")

        mb_fc_feats, mb_att_feats, mb_att_masks, \
        mb_sample_probs, mb_gen_result, mb_values = [None if len(_) == 0 else np.concatenate(_) for _ in [
            mb_fc_feats, mb_att_feats, mb_att_masks,
            mb_sample_probs, mb_gen_result, mb_values
        ]]

        return (iteration, epoch, mb_fc_feats, mb_att_feats, mb_att_masks,
                mb_sample_probs, mb_gen_result, mb_values)


class Trainer(object):
    def __init__(self, optimizer, predictor, trainer, nenvs, noptepochs,
                 envsperbatch, opt, loader):
        self.optimizer = optimizer

        self.nenvs = nenvs
        self.noptepochs = noptepochs
        self.envsperbatch = envsperbatch

        self.mbtrainer = MBTrainer(trainer, opt)
        self.runner = Runner(predictor, nenvs, noptepochs, envsperbatch, opt,
                             loader)

    def train(self, iteration, epoch):
        iteration, epoch, \
        mb_fc_feats, mb_att_feats, mb_att_masks, \
        mb_sample_probs, mb_gen_result, mb_values = self.runner.run(iteration, epoch)

        envinds = np.arange(self.nenvs)
        for _ in range(self.noptepochs):
            np.random.shuffle(envinds)
            for start in range(0, self.nenvs, self.envsperbatch):
                end = start + self.envsperbatch
                mbenvinds = envinds[start:end]
                fc_feats, att_feats, att_masks, sample_probs, gen_result, targ_vs = \
                    [None if x is None else torch.from_numpy(x).cuda()
                     for x in [None if _ is None else _[mbenvinds] for _ in [
                        mb_fc_feats, mb_att_feats, mb_att_masks,
                        mb_sample_probs, mb_gen_result, mb_values
                    ]]]
                self.optimizer.zero_grad()
                loss = self.mbtrainer(fc_feats, att_feats, att_masks,
                                      sample_probs, gen_result, targ_vs)
                loss.backward()
                self.optimizer.step()
        return iteration, epoch
