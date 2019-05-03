#coding:utf-8
import random
import torch.backends.cudnn as cudnn
import torch.optim
from   torch.utils import data
import numpy as np
from   image_dataset import DataSet
import torchvision.transforms as transforms
import argparse
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.parallel
import os
import time
import shutil
import utils
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import aren
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description=' source example code for AREN-CVPR19')
parser.add_argument('--dataset', default='CUB200', metavar='NAME',
                    help = 'dataset name')
parser.add_argument('--imagedir', default='../data/CUB_200_2011', metavar='NAME',
                    help = 'image dir for loading images')
parser.add_argument('--txtdir', default='CUB', metavar='NAME',
                    help = 'dirs with txtfiles included')
parser.add_argument('--data_root', default= './data',type=str, metavar='DIRECTORY',
                    help='path to data directory')
parser.add_argument('--model_root', default='./models',metavar='DIRECTORY',
                    help = 'dataset to model directory')
parser.add_argument('--result_root', default='./results',metavar='DIRECTORY',
                    help = 'dataset to result directory')
parser.add_argument('--folds', default= 5,  type=int, metavar='NUM',
                    help='folders')
parser.add_argument('--cpp_map', default= 20,  type=int,metavar='NUM',
                    help='cpp_map is the number of compressed feature maps')
parser.add_argument('--epochs', default= 10,  type=int,metavar='NUM',
                    help='epochs is the total epochs')
parser.add_argument('--total_search', default= 1,  type=int,metavar='NUM',
                    help='parameter searching numbers')
parser.add_argument('--start_epoch', default= 0, type=int,metavar='NUM',
                    help='#start_epoch')
parser.add_argument('--final_epoch', default= 100,  type=int,metavar='NUM',
                    help='last number of epoch')
parser.add_argument('--seed', default= 272, type=int,metavar='NUM',
                    help='seeds')
parser.add_argument('--step', default= 30,  type=int,metavar='NUM',
                    help='for SGD the default lr reducing step')
parser.add_argument('--dropout', default= 0.4,  type=float,metavar='NUM',
                    help='dropout')
parser.add_argument('--pretrain', default= 1,  type=int, choices=[0,1],metavar='FLAG',
                    help='0 used imagenet model, 1 pretrained model')
parser.add_argument('--pre_model', default= './models/CUB-PS-P_Repro_72.7.pth.tar', type=str, metavar='FILE',
                    help='the path for pretrained model')
parser.add_argument('--fix_feature', default= 0,  choices=[0,1], type=int, metavar='FLAG',
                    help='fix the lr or not')
parser.add_argument('--lr_strategy', default= 'step_lr', metavar='METHOD',
                    choices=['step_lr','sgdr_lr'], type=str, help='lr type')
parser.add_argument('--lr', default= '0.001',  type=float, metavar='RANGE',
                    help='lr learning rate')
parser.add_argument('--momentum', default= 0.9,  type=float,metavar='NUM',
                    help='momentum')
parser.add_argument('--weight_decay', default= 0.0005,  type=float,metavar='NUM',
                    help='weight decay')
parser.add_argument('--threshold', default= 0.8,  type=float,metavar='NUM',
                    help='threshold')
parser.add_argument('--cycle_len', default= 10,  type=int,metavar='NUM',
                    help='cycle_len')
parser.add_argument('--cycle_mul', default= 2,  type=int,metavar='NUM',
                    help='cycle_mul')
parser.add_argument('--parts', default= 10,  type=int,metavar='NUM',
                    help='parts')
parser.add_argument('--ls_coef_part', default= 1, type=int, metavar='NUM',
                    help='part coef for loss')
parser.add_argument('--ls_coef_bi', default= 0, type=int,metavar='NUM',
                    help='bilinear coef for loss')
parser.add_argument('--coef_part', default= 1,  type=int,metavar='NUM',
                    help='part coef for prediction')
parser.add_argument('--coef_bi', default= 0,  type=int,metavar='NUM',
                    help='bilinear coef for prediction')
parser.add_argument('--print_freq', default= 10, type=int,metavar='NUM',
                    help='print frequence')
parser.add_argument('--determine', default= 1, choices=[0,1], type=int,metavar='FLAG',
                    help='for reproduce the results')
parser.add_argument('--output', default= 'CUB-PS-P', type=str, metavar='DIRECTORY',
                    help='name of output')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ClassAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, n_labels):
        self.reset(n_labels)

    def reset(self,n_labels):
        self.n_labels = n_labels
        self.acc = torch.zeros(n_labels)
        self.cnt = torch.Tensor([1e-8]*n_labels)
        self.pred_prob = []

    def update(self, val, cnt, pred_prob):
        self.acc += val
        self.cnt += cnt
        self.avg = 100*self.acc.dot(1.0/self.cnt).item()/self.n_labels
        self.pred_prob += pred_prob
        #print ('pred',len(self.pred_prob))

def accuracy(output_vec, target, n_labels):
    """Computes the precision@k for the specified values of k"""
    output = output_vec

    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    class_accuracy = torch.zeros(n_labels)
    class_cnt = torch.zeros(n_labels)
    prec = 0.0
    pred_prob = []
    for i in range(batch_size):
        t = target[i]
        pred_prob.append(output[i][t])

        if pred[i] == t:
            prec += 1
            class_accuracy[t] += 1
        class_cnt[t] += 1
    return prec*100.0 / batch_size, class_accuracy, class_cnt, pred_prob

def config_process(config):
    if config.dataset == 'CUB200':
        config.image_dir = os.path.join(config.imagedir, 'images/')
        config.class_file = os.path.join(config.data_root,config.txtdir,  'classes.txt')
        config.image_label = os.path.join(config.data_root, config.txtdir, 'image_label_PS.txt')
        config.attributes_file = os.path.join(config.data_root, config.txtdir, 'class_attributes.txt')
        config.train_classes = os.path.join(config.data_root, config.txtdir, 'trainvalclasses.txt')
        config.test_classes = os.path.join(config.data_root, config.txtdir, 'testclasses.txt')
    
    if not os.path.exists(config.result_root):
        os.makedirs(config.result_root)
    if not os.path.exists(config.model_root):
        os.makedirs(config.model_root)
    #namespace ==> dictionary
    return vars(config)

def grab_data(config, examples, labels, is_train = True):

    params = {'batch_size': 64,
              'num_workers': 4,
              'pin_memory':True,
              'sampler': None}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tr_transforms, ts_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.08, 1), (0.5, 4.0 / 3)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]), transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if is_train:
        params['shuffle'] = True
        params['sampler'] = None
        data_set = data.DataLoader(DataSet(config, examples, labels, tr_transforms, is_train),**params)
    else:
        params['shuffle'] = False
        data_set = data.DataLoader(DataSet(config, examples, labels,ts_transforms,is_train),**params)
    return data_set

def fix_seeds(config):
    seed = config['seed']
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False # ensure the deterministic
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.deterministic = True
    return config

def load_model(config,model,optimizer,fname):
    checkpoint = torch.load(fname)
    config['start_epoch'] = checkpoint['epoch']
    config['best_meas'] = checkpoint['best_meas']
    config['best_epoch'] = checkpoint['best_epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {}), best_prec {}"
          .format(fname, checkpoint['epoch'], checkpoint['best_meas']))

def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = config['lr']*(0.1 ** (epoch // config['step']))
    print('current step learning rate {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(config, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './models/{}_model_best.pth.tar'.format(config['output']))

def save_model(config, model,optimizer,epoch,best_meas,best_epoch,is_best,fname):
    save_checkpoint(config, {
        'epoch': epoch + 1,
        'arch': 'resnet101',
        'state_dict': model.state_dict(),
        'best_meas': best_meas,
        'best_epoch': best_epoch,
        'optimizer' : optimizer.state_dict(),
    }, is_best,fname)

def sgdr(period, batch_idx):
    '''returns normalised anytime sgdr schedule given period and batch_idx
        best performing settings reported in paper are T_0 = 10, T_mult=2
        so always use T_mult=2 ICLR-17 SGDR learning rate.'''
    batch_idx = float(batch_idx)
    restart_period = period
    while batch_idx / restart_period > 1.:
        batch_idx = batch_idx - restart_period
        restart_period = restart_period * 2.
    r = math.pi * (batch_idx / restart_period)
    return 0.5 * (1.0 + math.cos(r))

def set_optimizer_lr(lr,optimizer):
    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(config, model, optimizer, criterion, train_loader, epoch, lr_period, start_batch_idx):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    class_avg = ClassAverageMeter(config['n_train_lbl'])

    # switch to train mode
    model.train()
    end = time.time()

    for i, (inputs, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs = inputs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if config['lr_strategy'] == 'sgdr_lr':
            set_optimizer_lr(config['lr'] * sgdr(lr_period, i + start_batch_idx))

        output = model(inputs)
        m = inputs.size(0)

        ls_lmbd3 = config['ls_coef_bi']
        ls_lmbd2 = config['ls_coef_part']
        lmbd3    = config['coef_bi']
        lmbd2    = config['coef_part']

        loss = (ls_lmbd2 * criterion(output[0], target) + ls_lmbd3 * criterion(output[1], target))/(ls_lmbd2+ls_lmbd3)
        avg_output = (lmbd2 * output[0] + lmbd3 * output[1])/(lmbd2+lmbd3)

        # measure accuracy and record loss
        prec1,class_acc,class_cnt,pred_prob = accuracy(avg_output, target, config['n_classes'])
        losses.update(loss, m)
        top1.update(prec1, m)

        class_avg.update(class_acc,class_cnt,pred_prob)

        # gradient and SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # time measure
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}] '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f} (avg: {loss.avg:.4f}) '
                  'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f}) '
                  'Class avg {lbl_avg.avg:.3f} '.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, lbl_avg=class_avg, top1=top1))

    return class_avg.avg, top1.avg, losses.avg

def test(config, model, criterion, val_loader):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    class_avg = ClassAverageMeter(config['n_test_lbl'])

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(input)

            ls_lmbd3 = config['ls_coef_bi']
            ls_lmbd2 = config['ls_coef_part']
            lmbd3 = config['coef_bi']
            lmbd2 = config['coef_part']

            loss = (ls_lmbd2 * criterion(output[0], target) \
                    + ls_lmbd3 * criterion(output[1], target)) / (ls_lmbd2 + ls_lmbd3)
            avg_output = (lmbd2 * output[0] + lmbd3 * output[1]) / (lmbd2 + lmbd3)

            prec1,class_acc,class_cnt,prec_prob = accuracy(avg_output, target, config['n_test_lbl'])

            m = input.size(0)
            losses.update(loss, m)
            top1.update(prec1, m)

            class_avg.update(class_acc,class_cnt,prec_prob)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config['print_freq'] == 0:
                print('Test: [{0}/{1}] '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Loss {loss.val:.4f} (avg: {loss.avg:.4f}) '
                      'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f}) '
                      'Class avg {class_avg.avg:.3f} '.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    class_avg=class_avg, top1=top1))

    return class_avg.avg, top1.avg, class_avg.pred_prob

def train_test(config, model, optimizer, criterion, train_loader, valid_loader):
    if config['pretrain'] == 0:
        best_epoch = -1
        best_meas = -1
        best_pred_prob = None

    if config['lr_strategy'] == 'sgdr_lr':
        lr_period = config['cycle_len'] * len(train_loader)

    for epoch in range(config['start_epoch'], config['epochs']):

        start_batch_idx = len(train_loader) * epoch

        train_acc, train_top1, train_loss = train(config, model, optimizer, criterion, train_loader, epoch, lr_period, start_batch_idx)
        test_acc, test_top1, pred_prob  = test(config, model, criterion, valid_loader)

        is_best = test_acc > best_meas
        if is_best:
            best_epoch = epoch
            best_meas = test_acc
            best_pred_prob = pred_prob
            save_model(config, model,optimizer, epoch, best_meas, best_epoch, is_best,
                            './models/{}{}'.format(config['output'], '_checkpoint.pth.tar'))
        config['iter'] = epoch
        config['train_acc'] = train_acc
        config['train_top1'] = train_top1
        config['test_acc'] = test_acc
        config['test_top1'] = test_top1
        config['train_loss'] = train_loss.item()
        config['best_epoch'] = best_epoch
        config['best_meas'] = best_meas

        print('best valid {:.3f} best epoches {} pred meas {:.3f} {}'
              .format(best_meas, best_epoch, test_acc, utils.dict_str(config, False)))

        if config['lr_strategy'] == 'step_lr':
            adjust_learning_rate(optimizer, epoch, config)
    return best_meas, best_epoch, best_pred_prob

def main():
    config = fix_seeds(config_process(parser.parse_args()))
    examples, labels, class_map = utils.image_load(config['class_file'], config['image_label'])
    # train, test, label, train_attr, test_attr
    datasets = utils.split_byclass(config, examples, labels, np.loadtxt(config['attributes_file']), class_map)
    print('load the train: {} the test: {}'.format(len(datasets[0][0]), len(datasets[0][1])))
    best_cfg = config
    best_cfg = fix_seeds(best_cfg)
    best_cfg['epochs'] = config['final_epoch']
    best_cfg['n_classes']   = datasets[0][3].size(0)
    best_cfg['n_train_lbl'] = datasets[0][3].size(0)
    best_cfg['n_test_lbl']  = datasets[0][4].size(0)

    # data grab
    train_set = grab_data(best_cfg, datasets[0][0], datasets[0][2], True )
    test_set  = grab_data(best_cfg, datasets[0][1], datasets[0][2], False)

    # create model
    model = models.__dict__['resnet101'](pretrained=True)
    model = aren.AREN(best_cfg, model, Parameter(F.normalize(datasets[0][3])), Parameter(F.normalize(datasets[0][4])))
    model = nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=best_cfg['lr'],
                                     momentum=best_cfg['momentum'],
                                     weight_decay=best_cfg['weight_decay'])
    # load the fine-tuned checkpoint
    if best_cfg['pretrain'] == 1:
        load_model(best_cfg,model,optimizer,best_cfg['pre_model'])

    # best_meas, best_epoch, best_pred_prob = train_test(best_cfg, model, optimizer, criterion, train_set, test_set)
    best_meas, best_epoch, best_pred_prob = test(best_cfg, model, criterion, test_set)
    print('Reproducing CUB:PS ACA = {:.3f}%'.format(best_meas))

if __name__ == '__main__':
    '''for reproducing purpose on CUB:PS ZSL results! '''
    main()
