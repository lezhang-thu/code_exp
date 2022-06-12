import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
import copy

import time
import os
from six.moves import cPickle
import traceback
from collections import defaultdict

import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
import skimage.io
import captioning.utils.eval_utils as eval_utils
import captioning.utils.misc as utils
from captioning.utils.rewards import init_scorer
from captioning.modules.loss_wrapper import LossWrapper
from captioning.modules.proximal import Trainer
import logging
import sys


def setup_logging(root_dir):
    logger_name = "log"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    fh = logging.FileHandler(os.path.join(root_dir, logger_name + '.txt'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def eval_func(opt, model, loader, split, beam_size):
    # eval model
    opt.verbose_loss = 0
    eval_kwargs = {
        'split': split,
        'dataset': opt.input_json,
        'beam_size': beam_size
    }
    eval_kwargs.update(vars(opt))
    eval_kwargs['beam_size'] = beam_size

    val_loss, predictions, lang_stats = eval_utils.eval_split(
        model, None, loader, eval_kwargs)

    # Save model if is improving on validation result
    if opt.language_eval == 1:
        current_score = lang_stats['CIDEr']
    else:
        current_score = -val_loss
    return current_score


def train(opt):
    seed = 23
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ################################
    # Build dataloader
    ################################
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    ##########################
    # Initialize infos
    ##########################
    infos = {
        'iter': 0,
        'epoch': 0,
        'loader_state_dict': None,
        'vocab': loader.get_vocab(),
    }
    # Load old infos (if there is) and check if models are compatible
    if opt.start_from is not None and os.path.isfile(
            os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl')):
        with open(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl'),
                  'rb') as f:
            #print("arrive here")
            #exit(0)
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            #need_be_same = [
            #    "caption_model", "rnn_type", "rnn_size", "num_layers"
            #]
            need_be_same = ["rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert getattr(saved_model_opt, checkme) == getattr(
                    opt, checkme
                ), "Command line argument and saved model disagree on '%s' " % checkme
            print(
                '#' * 6, "Infos reload from {} success ...".format(
                    os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl')),
                '#' * 6)

    infos['opt'] = opt
    #########################
    # Build logger
    #########################
    # naive dict logger
    histories = defaultdict(dict)
    if opt.start_from is not None and os.path.isfile(
            os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
        with open(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl'),
                  'rb') as f:
            histories.update(utils.pickle_load(f))

    ##########################
    # Build model
    ##########################
    #opt.vocab = loader.get_vocab()

    opt.vocab = infos["vocab"]
    loader.dataset.ix_to_word = infos['vocab']
    model = models.setup(opt).cuda()

    #debug
    vocab = infos["vocab"]
    #print("type(vocab): {}".format(type(vocab)))
    #x_counter = 0
    #with open("vocab-different-2.txt", 'w') as f:
    #    for key, value in sorted(vocab.items(), key=lambda x: int(x[0])):
    #        f.write("{}\n".format(value))
    #print("win")
    #exit(0)

    #del opt.vocab

    # Load pretrained weights:
    if opt.start_from is not None and os.path.isfile(
            os.path.join(opt.start_from, 'model_' + opt.id + '.pth')):
        m_path = os.path.join(opt.start_from, 'model_' + opt.id + '.pth')
        missing_keys, unexpected_keys = model.load_state_dict(
            torch.load(m_path), strict=False)
        print("missing_keys: {}, unexpected_keys: {}".format(
            missing_keys, unexpected_keys))
        print('#' * 6, "Model reload from {} success ...".format(m_path),
              '#' * 6)

    predictor = model
    predictor.vocab = getattr(model, 'vocab', None)  # nasty
    c = copy.deepcopy
    trainer = c(predictor)

    ##########################
    #  Build optimizer
    ##########################
    optimizer = torch.optim.AdamW(
        trainer.parameters(),
        lr=1e-4,  # TODO
        weight_decay=1e-4)

    # Load the optimizer
    if opt.start_from is not None and os.path.isfile(
            os.path.join(opt.start_from, "optimizer.pth")):
        optimizer.load_state_dict(
            torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    #########################
    # Get ready to start
    #########################
    #iteration = infos['iter']
    #epoch = infos['epoch']
    iteration = 0
    epoch = 0
    # For back compatibility
    if 'iterators' in infos:
        infos['loader_state_dict'] = {
            split: {
                'index_list': infos['split_ix'][split],
                'iter_counter': infos['iterators'][split]
            }
            for split in ['train', 'val', 'test']
        }
    loader.load_state_dict(infos['loader_state_dict'])
    #if opt.load_best_score == 1:
    #    best_val_score = infos.get('best_val_score', None)

    best_val_score = eval_func(opt,
                               predictor,
                               loader,
                               split='val',
                               beam_size=1)
    print("best_val_score/beam_size=1: {}".format(best_val_score))
    #exit(0)

    init_scorer(opt.cached_tokens)

    # Start training
    nenvs = 4 * opt.batch_size
    assert nenvs % opt.batch_size == 0
    x = nenvs // opt.batch_size
    opt.save_checkpoint_every = 16
    assert opt.save_checkpoint_every % x == 0

    logger = setup_logging("log_{}".format(opt.id))
    logger.info("opt.batch_size: {}".format(opt.batch_size))
    trainer.train()
    predictor.eval()
    train_model = Trainer(optimizer,
                          predictor,
                          trainer,
                          nenvs=nenvs,
                          noptepochs=1,
                          envsperbatch=opt.batch_size,
                          opt=opt,
                          loader=loader)
    reload_period = 0
    try:
        while True:
            # Stop if reaching max epochs
            #if epoch >= opt.max_epochs and opt.max_epochs != -1:
            #    break
            iteration, epoch = train_model.train(iteration, epoch)
            print("iteration: {}".format(iteration))
            reload_period += 4
            # critical - start
            if reload_period == 16:
                predictor.load_state_dict(trainer.state_dict())
                reload_period = 0
            # critical - end

            # update infos
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['loader_state_dict'] = loader.state_dict()

            # make evaluation on validation set, and save model
            if iteration % opt.save_checkpoint_every == 0:
                current_score = eval_func(opt,
                                          predictor,
                                          loader,
                                          split='val',
                                          beam_size=1)
                test_score = eval_func(opt,
                                       predictor,
                                       loader,
                                       split='test',
                                       beam_size=5)
                logger.info('######## Iter (TEST) ' + str(iteration) +
                            ' ########')
                logger.info("test_score: {}".format(test_score))
                logger.info("val_current @iteration {}: {}".format(
                    iteration, current_score))
                logger.info("val_predictor: {}".format(best_val_score))

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    infos['best_val_score'] = best_val_score
                    utils.save_checkpoint(opt,
                                          predictor,
                                          infos,
                                          optimizer,
                                          append='best')

    except (RuntimeError, KeyboardInterrupt):
        pass


opt = opts.parse_opt()
train(opt)
