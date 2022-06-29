import random
from collections import deque

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

    def forward(self, fc_feats, att_feats, att_masks, sample_probs, targ_vs,
                gen_result):
        log_probs, values = self.trainer(
            fc_feats, att_feats,
            torch.cat(
                [gen_result.new_zeros((self.opt.batch_size, 1)), gen_result],
                1)[..., :-1], att_masks)
        mask = gen_result > 0
        mask = torch.cat(
            [mask.new_full((mask.shape[0], 1), True), mask[:, :-1]], 1)
        x = (F.huber_loss(values, targ_vs, reduction='none') *
             mask).sum() / mask.sum()
        #y = (mask * (sample_probs * log_probs).sum(-1)).mean()
        #return -y
        return x


class Runner:
    def __init__(self, trainer, opt, loader):
        self.trainer = trainer
        self.opt = opt
        self.loader = loader

        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.alpha = [self.root_dirichlet_alpha] * (self.opt.vocab_size + 1)
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.discount = 1.0
        self.value_delta_max = 0.01
        self.num_simulations = 50
        self.num_epochs = 10

        def f(epoch):
            if epoch < 0.5 * self.num_epochs:
                return 1.0
            elif epoch < 0.75 * self.num_epochs:
                return 0.5
            return 0.25

        self.temperature = f

        self.buffer_size = int(5e2)
        self.replay_buffer = deque(maxlen=self.buffer_size)

    @torch.no_grad()
    def evaluate_greedy(self, data):
        self.trainer.eval()
        gts, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self.prepare_input(
            data)
        greedy_x, _ = self.trainer(fc_feats,
                                   att_feats,
                                   att_masks,
                                   mode='sample')
        rewards_greedy = np.asarray(get_scores(gts, greedy_x, self.opt),
                                    dtype=np.float32)
        return rewards_greedy.mean()

    @torch.no_grad()
    def evaluate_mcts(self, data):
        self.trainer.eval()
        mcts_x = self.mcts(data, rollout=False)[-1]
        gts = np.asarray(data['gts'], dtype=object)
        rewards_sample = np.asarray(get_scores(gts, mcts_x, self.opt),
                                    dtype=np.float32)
        return rewards_sample.mean()

    def policies_values(self, eos_env, p_fc_feats, p_att_feats, pp_att_feats,
                        gts, idx_outer):
        policies = p_att_feats.new_zeros(
            (self.opt.batch_size, self.opt.vocab_size + 1))
        policies[eos_env, 0] = -1.0
        values = policies.new_zeros((self.opt.batch_size, ))

        rollout = None
        if (~eos_env).sum() > 0:
            x = p_fc_feats
            if x is not None:
                x = x[~eos_env]

            x_policy, rollout = self.trainer._prefix_rollout(
                x, p_att_feats[~eos_env], pp_att_feats[~eos_env],
                p_att_masks[~eos_env],
                t[~eos_env][:, :idx_outer +
                            np.max(lens[(~eos_env).cpu().numpy()])])
            policies[~eos_env] = x_policy

        for y, targ in zip((eos_env, ~eos_env), (t[eos_env], rollout)):
            if y.sum() > 0:
                x = get_scores(gts[y.cpu().numpy()], targ, self.opt)
                values[y] = torch.from_numpy(x).float().cuda()
        return policies, values

    def simulate(self, roots, min_max_stats_lst, prefixes, p_fc_feats,
                 p_att_feats, pp_att_feats, gts, idx_outer, rollout):
        for idx_inner in range(self.num_simulations):
            # prepare a result wrapper to transport results between Python and C++ parts
            results = tree.ResultsWrapper(self.opt.batch_size)
            tree.batch_traverse(roots, self.pb_c_base, self.pb_c_init,
                                self.discount, min_max_stats_lst, results)

            # type of `v`: `list` of `list` (Python)
            v = roots.get_trajectories()
            lens = np.asarray([len(item) for item in v])
            mask = lens[:, None] > np.arange(lens.max())
            x = np.zeros(mask.shape, dtype=int)
            x[mask] = np.concatenate(v)

            eos_env = np.zeros((self.opt.batch_size, ), dtype=bool)
            eos_env[lens == 0] = True
            eos_env[lens > 0] = x[lens > 0, lens[lens > 0] - 1] == 0
            eos_env[lens == self.opt.max_length - idx_outer] = True
            eos_env = torch.from_numpy(eos_env).cuda()

            t = torch.cat([prefixes, torch.from_numpy(x).cuda()], -1)
            if rollout:
                policies, values = self.policies_values(
                    eos_env, p_fc_feats, p_att_feats, pp_att_feats, gts,
                    idx_outer)
            else:
                policies, values = self.trainer._prefix_next(
                    p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, t)
                policies[eos_env, 0] = -1.0
            tree.batch_back_propagate(self.discount, values.tolist(),
                                      policies.tolist(), min_max_stats_lst,
                                      results)

    def init_tree(self, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks):
        # minimax value storage
        min_max_stats_lst = tree.MinMaxStatsList(self.opt.batch_size)
        min_max_stats_lst.set_delta(self.value_delta_max)

        roots = tree.Roots(self.opt.batch_size, self.opt.vocab_size + 1,
                           self.opt.max_length)
        noises = [
            np.random.default_rng().dirichlet(self.alpha).astype(
                np.float32).tolist() for _ in range(self.opt.batch_size)
        ]
        prefixes = p_att_feats.new_zeros((self.opt.batch_size, 0),
                                         dtype=torch.long)

        policies, _ = self.trainer._prefix_next(p_fc_feats, p_att_feats,
                                                pp_att_feats, p_att_masks,
                                                prefixes)
        roots.prepare(self.root_exploration_fraction, noises,
                      policies.tolist())
        return roots, min_max_stats_lst, prefixes

    def prepare_input(self, data):
        gts = np.asarray(data['gts'], dtype=object)

        x = [data['fc_feats'], data['att_feats'], data['att_masks']]
        x = [None if _ is None else _.cuda() for _ in x]
        fc_feats, att_feats, att_masks = x

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self.trainer._prepare_feature(
            fc_feats, att_feats, att_masks)
        return gts, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks

    @torhc.no_grad()
    def mcts(self, data, rollout=True):
        gts, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self.prepare_input(
            data)
        roots, min_max_stats_lst, prefixes = self.init_tree(
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks)
        seq_probs = []
        seq_values = []
        unfinished = p_att_feats.new_ones((self.opt.batch_size, ),
                                          dtype=torch.bool)
        # env - start
        for idx_outer in range(self.opt.max_length):
            self.simulate(roots, min_max_stats_lst, prefixes, p_fc_feats,
                          p_att_feats, pp_att_feats, gts, idx_outer, rollout)
            x = roots.get_distributions()
            z = [1 for _ in range(self.opt.vocab_size + 1)]
            for idx, y in enumerate(x):
                # if root is terminal, [] is returned
                if len(y) == 0:
                    x[idx] = z
            x = np.asarray(x)**(1 / self.temperature(epoch))
            x = x / x.sum(-1, keepdims=True)
            seq_probs.append(x)
            seq_values.append(np.asarray(roots.get_values(), dtype=np.float32))
            if rollout:
                it = torch.distributions.Categorical(
                    probs=torch.from_numpy(x).cuda()).sample()
            else:
                it = torch.argmax(torch.from_numpy(x).cuda(), -1)
            it[~unfinished] = 0

            for idx, y in enumerate(unfinished):
                if y:
                    roots.update_with_move(idx, it[idx], min_max_stats_lst,
                                           self.discount)
            unfinished &= it != 0

            prefixes = torch.cat([prefixes, it.unsqueeze_(-1)], -1)
            if unfinished.sum() == 0:
                break
        # env - end
        roots.release_forest()
        return ret_numpy(seq_probs, seq_values, prefixes)

    def ret_numpy(self, seq_probs, seq_values, prefixes):
        seq_probs = np.stack(seq_probs, 1)
        x = np.zeros((self.opt.batch_size, self.opt.max_length,
                      self.opt.vocab_size + 1),
                     dtype=np.float32)
        x[:, :seq_probs.shape[1], :] = seq_probs
        seq_probs = x

        seq_values = np.stack(seq_values, 1)
        x = np.zeros((self.opt.batch_size, self.opt.max_length),
                     dtype=np.float32)
        x[:, :seq_values.shape[1]] = seq_values
        seq_values = x

        x = torch.zeros((self.opt.batch_size, self.opt.max_length),
                        dtype=np.long,
                        device=prefixes.device)
        x[:, :prefixes.shape[1]] = prefixes
        prefixes = x
        prefixes = prefixes.cpu().numpy()
        return seq_probs, seq_values, prefixes

    @torch.no_grad()
    def run(self, epoch):
        # collect data - start
        data = self.loader.get_batch('train')
        if data['bounds']['wrapped']:
            epoch += 1

        seq_probs, seq_values, prefixes = self.mcts(data)
        self.replay_buffer.extend(
            zip(
                [None] * self.opt.batch_size if data['fc_feats'] is None else
                list(data['fc_feats'].numpy()),
                list(data['att_feats'].numpy()),
                list(data['att_masks'].numpy()),
                list(seq_probs),
                list(seq_values),
                list(prefixes),
            ))
        # collect data - end
        return epoch

    def replay_sample(self):
        x = random.sample(self.replay_buffer, self.opt.batch_size)
        z = []
        for k in range(6):
            z.append([_[k] for _ in x])
        mb_fc_feats, mb_att_feats, mb_att_masks, mb_sample_probs, mb_values, mb_gen_result = z

        max_att_num = np.max([_.shape[0] for _ in mb_att_feats])
        for k, x in enumerate(mb_att_feats):
            after = max_att_num - x.shape[0]
            mb_att_feats[k] = np.pad(x, ((0, after), (0, 0)), mode="constant")
            mb_att_masks[k] = np.pad(mb_att_masks[k], ((0, after)),
                                     mode="constant")

        return [
            None if x[0] is None else np.stack(x) for x in [
                mb_fc_feats, mb_att_feats, mb_att_masks, mb_sample_probs,
                mb_values, mb_gen_result
            ]
        ]


class Trainer(object):
    def __init__(self, optimizer, trainer, opt, loader):
        self.optimizer = optimizer
        self.mbtrainer = MBTrainer(trainer, opt)
        self.runner = Runner(trainer, opt, loader)

    def train(self, iteration, epoch):
        k = 1
        self.runner.trainer.eval()
        for _ in range(k):
            epoch = self.runner.run(epoch)
        self.mbtrainer.trainer.train()
        for _ in range(4):
            fc_feats, att_feats, att_masks, sample_probs, targ_vs, gen_result = [
                None if x is None else torch.from_numpy(x).cuda()
                for x in self.runner.replay_sample()
            ]
            self.optimizer.zero_grad()
            loss = self.mbtrainer(fc_feats, att_feats, att_masks, sample_probs,
                                  targ_vs, gen_result)
            loss.backward()
            self.optimizer.step()
        return iteration + k, epoch
