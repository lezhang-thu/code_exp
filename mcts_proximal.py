import numpy as np
import torch

from captioning.utils.rewards import get_scores
import captioning.ctree.cytree as tree


class MBTrainer(torch.nn.Module):
    def __init__(self, trainer, opt, clip_range):
        super(MBTrainer, self).__init__()
        self.trainer = trainer
        self.opt = opt

    def forward(self, fc_feats, att_feats, att_masks, sample_probs, gen_result,
                targ_vs):
        ent_coef = 0.1
        trajectory = [sample_logprobs, gen_result, advs]
        for k, _ in enumerate(trajectory):
            trajectory[k] = _.reshape(-1, _.shape[-1])
        sample_probs, gen_result, values = trajectory
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
        values = torch.stack(fetch_values, 1)
        mask = gen_result > 0
        mask = torch.cat(
            [mask.new_full((mask.shape[0], 1), True), mask[:, :-1]], 1)

        x = mask.unsqueeze(-1)
        loss = (sample_probs[x] * log_prob[x]).mean()
        loss += F.huber_loss(values[mask], targ_vs.unsqueeze(-1))
        return loss


class Runner(object):
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
        self.num_simulations = 50
        self.num_epochs = int(1e4)

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

            # debug - start
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self.predictor._prepare_feature(
                fc_feats, att_feats, att_masks)
            # minimax value storage
            min_max_stats_lst = tree.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self.value_delta_max)

            roots = tree.Roots(self.opt.batch_size, self.opt.vocab_size + 1)
            noises = [
                np.random.dirichlet([self.root_dirichlet_alpha] *
                                    (self.opt.vocab_size + 1)).astype(
                                        np.float32).tolist()
                for _ in range(self.opt.batch_size)
            ]
            prefixes = p_att_feats.new_zeros((batch_size, 0), dtype=torch.long)
            seq_probs = []
            unfinished = p_att_feats.new_zeros((batch_size, ),
                                               dtype=torch.bool)
            policies, _ = self.predictor(p_fc_feats,
                                         p_att_feats,
                                         pp_att_feats,
                                         p_att_masks,
                                         prefixes,
                                         mode='prefix_next')
            roots.prepare(self.root_exploration_fraction, noises, policies)

            for _ in range(self.opt.seq_len + 1):
                for _ in range(self.num_simulations):
                    # prepare a result wrapper to transport results between python and c++ parts
                    results = tree.ResultsWrapper(self.opt.batch_size)
                    tree.batch_traverse(roots, self.pb_c_base, self.pb_c_init,
                                        self.discount, results)
                    # type of `x`: `list` of `list` (Python)
                    x = results.get_trajectories()
                    lens = np.asarray([len(item) for item in x])
                    mask = lens[:, None] > np.asarray(lens.max())
                    out = np.zeros(mask.shape, dtype=np.int64)
                    out[mask] = np.concatenate(x)
                    x = out
                    eos_env = np.full((self.opt.batch_size, ), False)
                    eos_env[out[np.arange(self.opt.batch_size),
                                lens - 1] == 0] = True

                    policies, values = self.predictor(
                        p_fc_feats,
                        p_att_feats,
                        pp_att_feats,
                        p_att_masks,
                        torch.cat(
                            [prefixes, torch.from_numpy(x).cuda()], -1),
                        mode='prefix_next')
                    policies[eos_env, 0] = -1.0
                    values[eos_env] = get_scores(
                        np.asarray(data['gts'])[eos_env], x[eos_env], self.opt)
                    tree.batch_back_propagate(self.discount, values.tolist(),
                                              policies.tolist(),
                                              min_max_stats_lst, results)
                    seq_probs.append(policies.unsqueeze(1))
                it = torch.distributions.Categorical(probs=torch.from_numpy(
                    np.asarray(roots.get_distributions()))**(
                        1 / self.temperature(epoch))).sample()
                it[~unfinished] = 0
                for idx, x in enumerate(unfinished):
                    if x:
                        roots.update_with_move(idx, it[idx])
                unfinished &= it != 0
                prefixes = torch.cat([prefixes, it], -1)
                if unfinished.sum() == 0:
                    break
            roots.release_forest()
            rewards_sample = get_scores(data['gts'], prefixes, self.opt)
            seq_probs = torch.cat(seq_probs, 1).cpu().numpy()
            prefixes = prefixes.cpu().numpy()
            # debug - end

            trajectory = [seq_probs, prefixes, rewards_sample]
            for x, y in zip(trajectory,
                            [mb_sample_logprobs, mb_gen_result, mb_advs]):
                y.append(x)
        max_att_num = np.max([_.shape[1] for _ in mb_att_feats])
        for k, x in enumerate(mb_att_feats):
            after = max_att_num - x.shape[1]
            mb_att_feats[k] = np.pad(x, ((0, 0), (0, after), (0, 0)),
                                     mode="constant")
            mb_att_masks[k] = np.pad(mb_att_masks[k], ((0, 0), (0, after)),
                                     mode="constant")

        mb_fc_feats, mb_att_feats, mb_att_masks, \
        mb_sample_logprobs, mb_gen_result, mb_values = [None if len(_) == 0 else np.vstack(_) for _ in [
            mb_fc_feats, mb_att_feats, mb_att_masks,
            mb_sample_probs, mb_gen_result, mb_values
        ]]
        return iteration, epoch, mb_fc_feats, mb_att_feats, mb_att_masks, \
               mb_sample_probs, mb_gen_result, mb_values


class Trainer(object):
    def __init__(self,
                 optimizer,
                 predictor,
                 trainer,
                 nenvs,
                 noptepochs,
                 envsperbatch,
                 opt,
                 loader,
                 clip_range=0.1):
        self.optimizer = optimizer

        self.nenvs = nenvs
        self.noptepochs = noptepochs
        self.envsperbatch = envsperbatch

        self.mbtrainer = MBTrainer(trainer, opt, clip_range)
        self.runner = Runner(predictor, nenvs, noptepochs, envsperbatch, opt,
                             loader)

    def train(self, iteration, epoch):
        self.mbtrainer.trainer.eval()
        self.runner.predictor.eval()

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
                loss = self.mbtrainer(fc_feats, att_feats, att_masks,
                                      sample_probs, gen_result, targ_vs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return iteration, epoch, None
