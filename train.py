from __future__ import division
from __future__ import print_function

import time
import argparse
import torch
import torch.nn as nn
from optims import Optim
import util
from util import utils
import lr_scheduler as L
from models import *
from collections import OrderedDict
from tqdm import tqdm
import sys
import os

# parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# sys.path.insert(0, parent_dir)
from util.nlp_utils import *


# config
def parse_args():
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-beam_search', default=True, action='store_true',
                        help="beam_search")
    parser.add_argument('-config', default='config.yaml', type=str,
                        help="config file")
    parser.add_argument('-model', default='seq2seq', type=str,
                        choices=['seq2seq', 'graph2seq', 'bow2seq', 'h_attention'])
    # parser.add_argument('-gpus', default=[1], type=int,
    #                     help="Use CUDA on the listed devices.")
    parser.add_argument('-gpus', default=[], type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-restore',
                        type=str, default=None,
                        help="restore checkpoint")
    parser.add_argument('-seed', type=int, default=1234,
                        help="Random seed")
    parser.add_argument('-notrain', default=False, action='store_true',
                        help="train or not")
    parser.add_argument('-log', default='', type=str,
                        help="log directory")

    # graph 相关的
    parser.add_argument('-verbose', default=False, action='store_true',
                        help="verbose")
    parser.add_argument('-adj', type=str, default="numsent",
                        help='adjacent matrix')
    parser.add_argument('-use_copy', default=False, action="store_true",
                        help='whether to use copy mechanism')
    parser.add_argument('-use_bert', default=False, action="store_true",
                        help='whether to use bert in the encoder')
    parser.add_argument('-use_content', default=False, action="store_true",
                        help='whether to use title in the seq2seq')
    parser.add_argument('-word_level_model', default='bert', choices=['bert', 'memory', 'word'],
                        help='whether to use bert or memory network or nothing in the word level of encoder')
    parser.add_argument('-graph_model', default='none', choices=['GCN', 'GNN', 'none'],
                        help='whether to use gcn in the encoder')
    parser.add_argument('-debug', default=False, action="store_true",
                        help='whether to use debug mode')

    opt = parser.parse_args()
    # 用config.data来得到config中的data选项
    config = util.utils.read_config(opt.config)
    return opt, config


# set opt and config as global variables
args, config = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)


# Training settings

def set_up_logging():
    # log为记录文件
    # config.log是记录的文件夹, 最后一定是/
    # opt.log是此次运行时记录的文件夹的名字
    if not os.path.exists(config.log):
        os.mkdir(config.log)
    if args.log == '':
        log_path = config.log + utils.format_time(time.localtime()) + '/'
    else:
        log_path = config.log + args.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logging = utils.logging(log_path + 'log.txt')  # 往这个文件里写记录
    logging_csv = utils.logging_csv(log_path + 'record.csv')  # 往这个文件里写记录
    logging("__________logging the args_________")
    for k, v in config.items():
        logging("%s:\t%s" % (str(k), str(v)))
    logging("\n")
    return logging, logging_csv, log_path


logging, logging_csv, log_path = set_up_logging()
use_cuda = torch.cuda.is_available()


def train(model, vocab, dataloader, scheduler, optim, updates):
    scores = []
    max_bleu = 0.
    for epoch in range(1, config.epoch + 1):
        total_acc = 0.
        total_loss = 0.
        start_time = time.time()

        if config.schedule:
            scheduler.step()
            print("Decaying learning rate to %g" % scheduler.get_lr()[0])

        model.train()

        train_data = dataloader.train_batches
        length_batch=len(train_data)
        print("Number of batch:   ",length_batch)

        for batch in tqdm(train_data, disable=not args.verbose):
            model.zero_grad()
            outputs = model(batch, use_cuda)
            target = batch.tgt
            if use_cuda:
                target = target.cuda()
            loss, acc = model.compute_loss(outputs.transpose(0, 1), target.transpose(0, 1)[1:])
            loss.backward()
            total_loss += loss.data.item()
            # report_correct += num_correct
            # report_total += num_total
            # report_tot_vocab += total_count
            # report_vocab += vocab_count
            total_acc += acc
            optim.step()
            updates += 1  # 进行了一次更新
            logging("training batches_______ time: %6.3f, epoch: %3d, updates: %8d, batch loss: %6.3f, train acc: %.3f"
                    % (time.time() - start_time, epoch, updates, loss.data.item(), acc))



        # 一个epoch之后判断是否得到最好的结果
                # logging中记录的是每次更新时的epoch，time，updates，correct等基本信息.
                # 还有score分数的信息
        logging("Finished an epoch ______time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.3f, train acc: %.3f"
                % (time.time() - start_time, epoch, updates, total_loss / length_batch,
                   total_acc / length_batch))
        print('evaluating after %d epochs...' % epoch)
        # TODO: fix eval and print bleu, ppl
        score = eval(model, vocab, dataloader, epoch, updates)
        print('Epoch: ',epoch, ',   Bleu sore: '+str(score), )
        scores.append(score)
        if score >= max_bleu:
            save_model(log_path + str(epoch) + '_' + str(updates) + '_' + str(score) + '_updates_checkpoint.pt', model, optim, updates)
            max_bleu = score
        save_model(log_path + 'save_'+str(epoch)+'_'+str(updates) +'_'+ str(score) + '_updates_checkpoint.pt', model, optim, updates)

    return max_bleu


def eval(model, vocab, dataloader, epoch, updates, do_test=False):
    model.eval()
    multi_ref, reference, candidate, source, tags, alignments = [], [], [], [], [], []
    if do_test:
        data_batches = dataloader.test_batches
    else:
        data_batches = dataloader.dev_batches
    i = 0
    for batch in tqdm(data_batches, disable=not args.verbose):
        if len(args.gpus) > 1 or not args.beam_search:
            samples, alignment = model.sample(batch, use_cuda)
        else:
            samples, alignment = model.beam_sample(batch, use_cuda, beam_size=config.beam_size)
        '''
        if i == 0:
            print(batch.examples[27].ori_title)
            print(alignment.shape)
            print([d for d in alignment.tolist()[27]])
            return
        '''
        # print("sample:")
        # print(samples)
        # print('\n')
        candidate += [vocab.id2sent(s) for s in samples]
        source += [example for example in batch.examples]
        # reference += [example.ori_target for example in batch.examples]
        multi_ref += [example.ori_target for example in batch.examples]
    utils.write_result_to_file(source, candidate, log_path)
    #text_result, bleu = utils.eval_bleu(reference, candidate, log_path)
    multi_text_result, bleu = utils.eval_multi_bleu(multi_ref, candidate, log_path)
    logging_csv([epoch, updates, multi_text_result])
    print(multi_text_result, flush=True)
    #print(text_result, flush=True)
    return bleu


def save_model(path, model, optim, updates):
    '''保存的模型是一个字典的形式, 有model, config, optim, updates.'''

    # 如果使用并行的话使用的是model.module.state_dict()
    model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates}
    torch.save(checkpoints, path)


def main():
    # 设定种子
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # checkpoint
    if args.restore:  # 存储已有模型的路径
        print('loading checkpoint...\n')
        checkpoints = torch.load(os.path.join(log_path, args.restore))

    # Load data
    print('loading data...\n')
    contentfile = os.path.join(config.data, "new_data.json")
    vocab = Vocab(config.vocab, contentfile, config.vocab_size)
    #print(vocab._word2id)
    start_time = time.time()

    dataloader = DataLoader(config, config.data, config.batch_size, vocab,args.debug)
    print("DATA loaded!")
    print('loading time cost: %.3f' % (time.time() - start_time))
    # print(dataloader.train_batches[0].tgt)
    # print(dataloader.train_batches[0].tgt_len)
    # print(dataloader.train_batches[0].tgt_mask)

    # model
    print('building model...\n')
    torch.backends.cudnn.benchmark = True
    # configure the model
    # Model and optimizer
    model = seq2seq(config, vocab, use_cuda, use_content=args.use_content)

    if args.restore:
        model.load_state_dict(checkpoints['model'])
    if use_cuda:
        model.cuda()
        # lm_model.cuda()
    if len(args.gpus) > 1:  # 并行
        model = nn.DataParallel(model, device_ids=args.gpus, dim=1)
    logging(repr(model) + "\n\n")  # 记录这个文件的框架

    # total number of parameters
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    logging('total number of parameters: %d\n\n' % param_count)

    # updates是已经进行了几个epoch, 防止中间出现程序中断的情况.
    if args.restore:
        updates = checkpoints['updates']
        ori_updates = updates
    else:
        updates = 0

    # optimizer
    if args.restore:
        optim = checkpoints['optim']
    else:
        optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                      lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)

    # if opt.pretrain:
    # pretrain_lm(lm_model, vocab)
    optim.set_parameters(model.parameters())
    if config.schedule:
        scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)
    else:
        scheduler = None

    if not args.notrain:
        max_bleu = train(model, vocab, dataloader, scheduler, optim, updates)
        logging("Best bleu score: %.2f\n" % (max_bleu))
    else:
        assert args.restore is not None
        eval(model, vocab, dataloader, 0, updates, do_test=False)


if __name__ == '__main__':
    main()
