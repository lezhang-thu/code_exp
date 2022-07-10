import os
import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from captioning.utils.rewards import get_scores
import captioning.ctree.cytree as tree


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        z = []
        for k in range(5):
            z.append([self._storage[j][k] for j in idxes])
        mb_att_feats, mb_att_masks, mb_greedy, mb_src, mb_labels = z

        max_att_num = np.max([_.shape[0] for _ in mb_att_feats])
        for k, x in enumerate(mb_att_feats):
            after = max_att_num - x.shape[0]
            mb_att_feats[k] = np.pad(x, ((0, after), (0, 0)), mode="constant")
            mb_att_masks[k] = np.pad(
                mb_att_masks[k], ((0, after)), mode="constant"
            )
        return [
            np.stack(x)
            for x in [mb_att_feats, mb_greedy, mb_src, mb_att_masks, mb_labels]
        ]

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        """
        idxes = [
            random.randint(0,
                           len(self._storage) - 1) for _ in range(batch_size)
        ]
        return self._encode_sample(idxes)


class MBTrainer(torch.nn.Module):
    def __init__(self, trainer, opt):
        super(MBTrainer, self).__init__()
        self.trainer = trainer
        self.opt = opt

    def forward(
        self, fc_feats, att_feats, seq_greedy, src_x, att_masks, better_labels
    ):
        logits = self.trainer.is_better(
            fc_feats, att_feats, seq_greedy, src_x, att_masks
        )
        #print(logits)
        #exit(0)

        # debug
        #if random.uniform(0, 1) < 5e-4:
        #    x = logits.argmax(-1)
        #    print(x)
        #    print(better_labels)
        #    print((x == better_labels).sum().item() / self.opt.batch_size)
        return F.huber_loss(logits, better_labels)


class Runner:
    def __init__(self, trainer, opt, loader):
        self.trainer = trainer
        for p in self.trainer.model.encoder.parameters():
            p.requires_grad = False
        self.freeze = copy.deepcopy(trainer)
        # debug
        m_path = os.path.join(opt.start_from, 'model_' + opt.id + '.pth')
        missing_keys, unexpected_keys = self.trainer.load_state_dict(
            torch.load(m_path), strict=False
        )
        print(
            "missing_keys: {}, unexpected_keys: {}".format(
                missing_keys, unexpected_keys
            )
        )
        print(
            '#' * 6, "Model reload from {} success ...".format(m_path), '#' * 6
        )

        self.opt = opt
        self.loader = loader

        self.root_dirichlet_alpha = 0.3
        #self.root_exploration_fraction = 0.25
        self.root_exploration_fraction = 0.0
        self.alpha = [self.root_dirichlet_alpha] * (self.opt.vocab_size + 1)
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.discount = 1.0
        self.value_delta_max = 0.01
        self.num_simulations = 20
        self.num_epochs = 10

        def f(epoch):
            if epoch < 0.5 * self.num_epochs:
                return 1.0
            elif epoch < 0.75 * self.num_epochs:
                return 0.5
            return 0.25

        self.temperature = f

        # unsolved
        self.buffer_size = int(1e6)
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # debug
        self.list_shared = None
        self.greedy_res = None

        self.memory = None
        self.att_masks = None

        self.run_transitions = None
        self.print_thres = 5e-4

    @torch.no_grad()
    def evaluate_greedy(self, data, beam_size=1):
        self.freeze.eval()
        x = [data['fc_feats'], data['att_feats'], data['att_masks']]
        x = [None if _ is None else _.cuda() for _ in x]
        fc_feats, att_feats, att_masks = x
        gts = np.asarray(data['gts'], dtype=object)
        greedy_x, _ = self.freeze(
            fc_feats,
            att_feats,
            att_masks,
            opt={
                'beam_size': beam_size,
                'sample_n': 1,
                'group_size': 1
            },
            mode='sample'
        )
        #print(greedy_x[23, ...])
        #rewards_greedy = np.asarray(get_scores(gts, greedy_x, self.opt),
        #                            dtype=np.float32)
        #return rewards_greedy.mean()
        return greedy_x

    @torch.no_grad()
    def evaluate_mcts(self, data):
        self.trainer.eval()
        self.freeze.eval()
        mcts_x = self.mcts(data, 0, do_rollout=False)[-1]

        gts = np.asarray(data['gts'], dtype=object)
        #rewards_sample = np.asarray(get_scores(gts, torch.from_numpy(mcts_x),
        #                                       self.opt),
        #                            dtype=np.float32)
        #return rewards_sample.mean()
        return torch.from_numpy(mcts_x)

    def policies_values(
        self, eos_env, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, gts,
        idx_outer, t, lens, greedy_score, do_rollout
    ):
        policies = p_att_feats.new_zeros(
            (self.opt.batch_size, self.opt.vocab_size + 1)
        )
        policies[eos_env, 0] = -1.0

        rollout = None
        if (~eos_env).sum() > 0:
            x = p_fc_feats
            if x is not None:
                x = x[~eos_env]

            #x_policy, rollout = self.trainer._prefix_rollout(
            x_policy, rollout = self.freeze._prefix_rollout(
                x, p_att_feats[~eos_env], pp_att_feats[~eos_env],
                p_att_masks[~eos_env],
                t[~eos_env][:, :idx_outer +
                            np.max(lens[(~eos_env).cpu().numpy()])], do_rollout
            )
            policies[~eos_env] = x_policy

        q = t.new_zeros((self.opt.batch_size, self.opt.max_length))
        q[eos_env, :t.shape[1]] = t[eos_env]
        if (~eos_env).sum() > 0:
            q[~eos_env, :rollout.shape[1]] = rollout
        #equal_seq = (q == torch.from_numpy(
        #    self.greedy_res
        #).cuda()).sum(-1) == self.opt.max_length

        if do_rollout:
            values = torch.from_numpy(get_scores(gts, q,
                                                 self.opt)).float().cuda()
            values -= greedy_score
            values[torch.abs(values) < 1e-3] = 0.0
            u = values.cpu().numpy()
            #x_gts = torch.sign(values).to(torch.float32)
            #u = x_gts.to(torch.long).cpu().numpy() + 1

            #if random.uniform(0, 1) < 0.5:
            if True:
                x = self.trainer.get_sign(
                    self.memory, self.att_masks,
                    torch.from_numpy(self.greedy_res).cuda(), q
                ).to(torch.float32)
            else:
                x = x_gts

            remove = lens == 0
            #remove = equal_seq.cpu().numpy()
            for k, join in enumerate(~remove):
                if join:
                    self.replay_buffer.add(
                        (
                            self.list_shared['att_feats'][k],
                            self.list_shared['att_masks'][k],
                            self.greedy_res[k], q.cpu().numpy()[k], u[k]
                        )
                    )
            # debug
            self.run_transitions += (~remove).sum().item()

            #if random.uniform(0, 1) < self.print_thres:
            #    print('training')
            #    print('x')
            #    print(x + 1)
            #    print('x_gts')
            #    print(x_gts + 1)
            #    print((x == x_gts).sum() / self.opt.batch_size)
        else:
            # debug
            #values = torch.from_numpy(get_scores(gts, q,
            #                                     self.opt)).float().cuda()
            #values -= greedy_score
            #values[torch.abs(values) < 1e-3] = 0.0
            # debug
            #return policies, values
            
            #x_gts = torch.sign(values).to(torch.float32)
            
            x = self.trainer.get_sign(
                self.memory, self.att_masks,
                torch.from_numpy(self.greedy_res).cuda(), q
            ).to(torch.float32)

            #if random.uniform(0, 1) < self.print_thres:
            #    print('x')
            #    print(x + 1)
            #    print('x_gts')
            #    print(x_gts + 1)
            #    print((x == x_gts).sum() / self.opt.batch_size)

            # debug
            #return policies, x_gts

        return policies, x
        #return policies, values

    #def rollout_eval(
    #    self, eos_env, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, gts,
    #    idx_outer, t, lens
    #):
    #    policies = p_att_feats.new_zeros(
    #        (self.opt.batch_size, self.opt.vocab_size + 1)
    #    )
    #    policies[eos_env, 0] = -1.0

    #    rollout = None
    #    if (~eos_env).sum() > 0:
    #        x = p_fc_feats
    #        if x is not None:
    #            x = x[~eos_env]

    #        x_policy, rollout = self.freeze._prefix_rollout(
    #            x, p_att_feats[~eos_env], pp_att_feats[~eos_env],
    #            p_att_masks[~eos_env],
    #            t[~eos_env][:, :idx_outer +
    #                        np.max(lens[(~eos_env).cpu().numpy()])]
    #        )
    #        policies[~eos_env] = x_policy

    #    gen_results = t.new_zeros((self.opt.batch_size, self.opt.max_length))
    #    gen_results[eos_env, :t.shape[-1]] = t[eos_env]
    #    if (~eos_env).sum() > 0:
    #        gen_results[~eos_env] = rollout

    #    _, values = self.trainer._prefix_next(
    #        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, gen_results
    #    )
    #    return policies, values

    def simulate(
        self, roots, min_max_stats_lst, prefixes, p_fc_feats, p_att_feats,
        pp_att_feats, p_att_masks, gts, idx_outer, do_rollout, greedy_score
    ):
        for idx_inner in range(self.num_simulations):
            # prepare a result wrapper to transport results between Python and C++ parts
            results = tree.ResultsWrapper(self.opt.batch_size)
            tree.batch_traverse(
                roots, self.pb_c_base, self.pb_c_init, self.discount,
                min_max_stats_lst, results
            )

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
            #if rollout:
            if True:
                policies, values = self.policies_values(
                    eos_env, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks,
                    gts, idx_outer, t, lens, greedy_score, do_rollout
                )
            else:
                pass
                #policies, values = self.rollout_eval(
                #    eos_env, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks,
                #    gts, idx_outer, t, lens
                #)
                #policies, values = self.trainer._prefix_next(
                #    p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, t)
                #policies[eos_env, 0] = -1.0
            tree.batch_back_propagate(
                self.discount, values.tolist(), policies.tolist(),
                min_max_stats_lst, results
            )

    def init_tree(self, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks):
        # minimax value storage
        min_max_stats_lst = tree.MinMaxStatsList(self.opt.batch_size)
        min_max_stats_lst.set_delta(self.value_delta_max)

        roots = tree.Roots(
            self.opt.batch_size, self.opt.vocab_size + 1, self.opt.max_length
        )
        noises = [
            np.random.default_rng().dirichlet(self.alpha).astype(
                np.float32
            ).tolist() for _ in range(self.opt.batch_size)
        ]
        prefixes = p_att_feats.new_zeros(
            (self.opt.batch_size, 0), dtype=torch.long
        )

        policies, _ = self.freeze._prefix_next(
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, prefixes
        )
        roots.prepare(self.root_exploration_fraction, noises, policies.tolist())
        return roots, min_max_stats_lst, prefixes

    def prepare_input(self, data):
        gts = np.asarray(data['gts'], dtype=object)

        x = [data['fc_feats'], data['att_feats'], data['att_masks']]
        x = [None if _ is None else _.cuda() for _ in x]
        fc_feats, att_feats, att_masks = x

        greedy_res, _ = self.freeze(
            fc_feats,
            att_feats,
            att_masks,
            mode='sample',
            opt={'sample_method': 'greedy'}
        )
        greedy_score = torch.from_numpy(get_scores(gts, greedy_res,
                                                   self.opt)).cuda()
        # debug
        x = np.zeros((self.opt.batch_size, self.opt.max_length), dtype=np.long)
        x[:, :greedy_res.shape[1]] = greedy_res.cpu().numpy()
        self.greedy_res = x

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self.freeze._prepare_feature(
            fc_feats, att_feats, att_masks
        )
        self.att_masks = p_att_masks
        self.memory = pp_att_feats
        return gts, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, greedy_score

    @torch.no_grad()
    def mcts(self, data, epoch, do_rollout=True):
        gts, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, greedy_score = self.prepare_input(
            data
        )
        roots, min_max_stats_lst, prefixes = self.init_tree(
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks
        )
        seq_probs = []
        unfinished = p_att_feats.new_ones(
            (self.opt.batch_size, ), dtype=torch.bool
        )

        # env - start
        for idx_outer in range(self.opt.max_length):
            self.simulate(
                roots, min_max_stats_lst, prefixes, p_fc_feats, p_att_feats,
                pp_att_feats, p_att_masks, gts, idx_outer, do_rollout,
                greedy_score
            )
            x = roots.get_distributions()
            z = [1 for _ in range(self.opt.vocab_size + 1)]
            for idx, y in enumerate(x):
                # if root is terminal, [] is returned
                if len(y) == 0:
                    x[idx] = z
            x = np.asarray(x)**(1 / self.temperature(epoch))
            x = x / x.sum(-1, keepdims=True)
            seq_probs.append(x)
            #if do_rollout:
            if False:
                it = torch.distributions.Categorical(
                    probs=torch.from_numpy(x).cuda()
                ).sample()
            else:
                it = torch.argmax(torch.from_numpy(x).cuda(), -1)
            it[~unfinished] = 0

            for idx, y in enumerate(unfinished):
                if y:
                    roots.update_with_move(
                        idx, it[idx], min_max_stats_lst, self.discount
                    )
            unfinished &= it != 0

            prefixes = torch.cat([prefixes, it.unsqueeze_(-1)], -1)
            if unfinished.sum() == 0:
                break
        # env - end
        roots.release_forest()
        seq_values = np.asarray(
            get_scores(gts, prefixes, self.opt), dtype=np.float32
        )
        return self.ret_numpy(seq_probs, seq_values, prefixes)

    def ret_numpy(self, seq_probs, seq_values, prefixes):
        seq_probs = np.stack(seq_probs, 1)
        x = np.zeros(
            (self.opt.batch_size, self.opt.max_length, self.opt.vocab_size + 1),
            dtype=np.float32
        )
        x[:, :seq_probs.shape[1], :] = seq_probs
        seq_probs = x

        x = torch.zeros(
            (self.opt.batch_size, self.opt.max_length),
            dtype=np.long,
            device=prefixes.device
        )
        x[:, :prefixes.shape[1]] = prefixes
        prefixes = x
        prefixes = prefixes.cpu().numpy()
        return seq_probs, seq_values, prefixes

    #def random_sample(self, data):
    #    gts = np.asarray(data['gts'], dtype=object)

    #    x = [data['fc_feats'], data['att_feats'], data['att_masks']]
    #    x = [None if _ is None else _.cuda() for _ in x]
    #    fc_feats, att_feats, att_masks = x
    #    prefixes, _ = self.freeze(
    #        fc_feats,
    #        att_feats,
    #        att_masks,
    #        opt={
    #            'sample_method': 'sample',
    #            'beam_size': 1,
    #            'sample_n': 1
    #        },
    #        mode='sample'
    #    )
    #    seq_probs = np.zeros(
    #        (self.opt.batch_size, self.opt.max_length, self.opt.vocab_size + 1),
    #        dtype=np.float32
    #    )

    #    seq_values = np.asarray(
    #        get_scores(gts, prefixes, self.opt), dtype=np.float32
    #    )

    #    x = torch.zeros(
    #        (self.opt.batch_size, self.opt.max_length),
    #        dtype=np.long,
    #        device=prefixes.device
    #    )
    #    x[:, :prefixes.shape[1]] = prefixes
    #    prefixes = x.cpu().numpy()
    #    return seq_probs, seq_values, prefixes

    @torch.no_grad()
    def run(self, epoch):
        # debug
        #self.loader.reset_iterator('val')
        #data = self.loader.get_batch('val')

        # collect data - start
        data = self.loader.get_batch('train')

        # debug
        self.list_shared = dict()
        for key in ('att_feats', 'att_masks'):
            self.list_shared[key] = list(data[key].numpy())

        if data['bounds']['wrapped']:
            epoch += 1

        seq_probs, seq_values, prefixes = self.mcts(data, epoch)
        #seq_probs, seq_values, prefixes = self.random_sample(data)
        #self.replay_buffer.extend(
        #    zip(
        #        [None] * self.opt.batch_size
        #        if data['fc_feats'] is None else list(data['fc_feats'].numpy()),
        #        list(data['att_feats'].numpy()),
        #        list(data['att_masks'].numpy()),
        #        list(seq_probs),
        #        list(seq_values),
        #        list(prefixes),
        #    )
        #)
        # collect data - end
        return epoch


class Trainer(object):
    def __init__(self, optimizer, trainer, opt, loader):
        self.optimizer = optimizer
        self.mbtrainer = MBTrainer(trainer, opt)
        self.runner = Runner(trainer, opt, loader)
        self.stat = None

    def train(self, iteration, epoch):
        #k = 20
        k = 1
        self.runner.trainer.eval()
        self.runner.run_transitions = 0
        #self.runner.loader.reset_iterator('val')
        for _ in range(k):
            #print(_)
            epoch = self.runner.run(epoch)
        self.mbtrainer.trainer.train()
        loss = None
        m = self.runner.run_transitions // self.runner.opt.batch_size
        print('m: {}'.format(m))
        for _ in range(m * 4):
            att_feats, seq_greedy, src_x, att_masks, better_labels = [
                torch.from_numpy(x).cuda() for x in
                self.runner.replay_buffer.sample(self.mbtrainer.opt.batch_size)
            ]
            self.optimizer.zero_grad()
            loss = self.mbtrainer(
                None, att_feats, seq_greedy, src_x, att_masks, better_labels
            )
            loss.backward()
            self.stat = loss.item(
            ) if self.stat is None else self.stat * 0.9 + loss.item() * 0.1
            self.optimizer.step()
        print(self.stat)
        return iteration + k, epoch
