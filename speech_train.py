'''
Modified from https://github.com/jack-willturner/DeepCompression-PyTorch

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] Tao Lin, Sebastian U.stich, Luis Barba, Daniil Dmitriev, Martin Jaggi
    Dynamic Pruning with Feedback. ICLR2020
'''
import time
import random
import pathlib
from os.path import isfile
import copy

import numpy as np
import cv2
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader

from torch.autograd import Variable

import models
import config
import pruning
from utils import *
from data import DataLoader
import os
# for ignore ImageNet PIL EXIF UserWarning and ignore transparent images
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "(Palette )?images with Transparency", UserWarning)


import os
import sys
import time

# from model_dataloader import *

def readlines(datapath):
    # Read all the lines in a text file and return as a list
    with open(datapath, 'r') as f:
        lines = f.read().splitlines()
    return lines



class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=2

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss


# criterion_kl = KLLoss().cuda()

def hyperparam():
    args = config.config()
    return args

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def main(args):
    global arch_name
    if args.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num

    
    # Dataset loading
    print('==> Load data..')
    start_time = time.time()
    train_loader, val_loader, in_num, out_num  = DataLoader(args.batch_size, args.dataset,
                                          args.workers, args.datapath, 32,
                                          args.cuda)
    
    # if args.dataset == 'audio_12':
    # train_filename   = readlines("../TC_resnet/splits/{}.txt".format("train"))
    # valid_filename   = readlines("../TC_resnet/splits/{}.txt".format("valid"))
    # # transform_train = transforms.Compose([transforms.ToTensor()])
    # train_dataset    = SpeechCommandDataset_12("../TC_resnet/dataset", train_filename, True)
    # valid_dataset    = SpeechCommandDataset_12("../TC_resnet/dataset", valid_filename, False)
    # # train_loader = DataLoader(train_dataset,128,shuffle = True , drop_last = True)
    # # val_loader = DataLoader(valid_dataset,128,shuffle = True , drop_last = True)
    # train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, drop_last = True )
    # val_loader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle = True, drop_last = True )
    
    
    elapsed_time = time.time() - start_time
    print('===> Data loading time: {:,}m {:.2f}s'.format(
        int(elapsed_time//60), elapsed_time%60))
    print('===> Data loaded..') 
    
    # set model name
    arch_name = set_arch_name(args)
    print('\n=> creating model \'{}\''.format(arch_name))

    if not args.prune:  # base  
        model, image_size = models.__dict__[args.arch](data=args.dataset, num_layers=args.layers,
                                                        width_mult=args.width_mult,
                                                        depth_mult=args.depth_mult,
                                                        model_mult=args.model_mult)

    elif args.prune:    # for pruning
        pruner = pruning.__dict__[args.pruner]
        model, image_size = pruning.models.__dict__[args.arch](data=args.dataset, num_layers=args.layers,
                                                   width_mult=args.width_mult,
                                                   depth_mult=args.depth_mult,
                                                   model_mult=args.model_mult,
                                                   mnn=pruner.mnn,
                                                   n_bit = args.n_bits,
                                                               bins = 40,
                                                    in_num = in_num,
                                                    out_num = out_num
                                                              )
    
    assert model is not None, 'Unavailable model parameters!! exit...\n'

    # set criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    
    weight_bias = list()
    scale_weight = list()
    scale_activation = list()
    for name, param in model.named_parameters():
        if 'interval' in name:
            scale_weight.append(param)
            # print(param)
        elif 'scale' in name:
            # print(param)
            scale_activation.append(param)
        else:
            weight_bias.append(param)
        
    if not args.acti_quan:
        param_groups1 = [{'params': weight_bias, 'lr': args.lr}]
        param_groups2 = [{'params': scale_weight, 'lr': args.lr*0.0001}]
    else:
        param_groups1 = [{'params': weight_bias, 'lr': args.lr}]
        param_groups2 = [{'params': scale_weight, 'lr': args.lr*0.0001},
                        {'params': scale_activation, 'lr': args.lr*0.0001}]
    
    
    
    # param_groups1 = [{'params': weight_bias, 'lr': args.lr}]
    # #param_groups2 = [{'params': scale_weight, 'lr': args.lr * args.interval_factor}]
    # param_groups2 = [{'params': scale_weight, 'lr': args.lr * 0.0001}]
    
    optimizer = optim.SGD(param_groups1+param_groups2, lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay,
                            nesterov=args.nesterov)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr,
    #                         momentum=args.momentum, weight_decay=args.weight_decay,
    #                         nesterov=args.nesterov)

    scheduler = set_scheduler(optimizer, args)
    # print(args.gpuids)
    # set multi-gpu
    if args.cuda:
        
        model = model.cuda()
        criterion = criterion.cuda()
        model = nn.DataParallel(model, device_ids=args.gpuids,
                                output_device=args.gpuids[0])
        cudnn.benchmark = True
        
        # for distillation


    # load a pre-trained model 
    if args.load is not None:
        ckpt_file = pathlib.Path('checkpoint') / arch_name / args.dataset / args.load
        assert isfile(ckpt_file), '==> no checkpoint found \"{}\"'.format(args.load)

        print('==> Loading Checkpoint \'{}\''.format(args.load))
        # check pruning or quantization or transfer
        strict = False if args.prune else True
        # load a checkpoint 
        checkpoint = load_model(model, ckpt_file, main_gpu=args.gpuids[0], use_cuda=args.cuda, strict=strict)
        print('==> Loaded Checkpoint \'{}\''.format(args.load)) 

    writer = SummaryWriter()
    name = "(p:"+ str(args.prune_freq) + "_d:" + str(args.dpf_epoch) + ")"
    nmp = str(args.prune_freq)
    # for training
    if args.run_type == 'train':
        # init parameters
        start_epoch = 0
        global iterations
        iterations = 0
        best_acc1 = 0.0
        train_time = 0.0
        validate_time = 0.0

        f = open(args.txt_name_train, 'w')
        ff = open(args.txt_name_test, 'w')

        DS = 0 
        for epoch in range(start_epoch, args.epochs):

            print('\n==> {}/{} training'.format(
                    arch_name, args.dataset))
            print('==> Epoch: {}, lr = {}'.format(
                epoch, optimizer.param_groups[0]["lr"]))

            # train for one epoch
            print('===> [ Training ]')
            if epoch +1 == 10 :
                iterations = 0
            start_time = time.time()
            acc1_train = train(args, train_loader,
                epoch=epoch, model=model,
                criterion=criterion, optimizer=optimizer,DS=DS)


            elapsed_time = time.time() - start_time
            train_time += elapsed_time
            print('====> {:.2f} seconds to train this epoch\n'.format(
                elapsed_time))

            # evaluate on validation set
            print('===> [ Validation ]')
            start_time = time.time()
            acc1_valid = validate(args, val_loader,
                epoch=epoch, model=model, criterion=criterion)
            elapsed_time = time.time() - start_time
            validate_time += elapsed_time
            print('====> {:.2f} seconds to validate this epoch'.format(
                elapsed_time))

            tt1 = validate_t(args, val_loader,
                epoch=epoch, model=model, criterion=criterion)

            # learning rate schduling
            scheduler.step()

            acc1_train = round(acc1_train, 4)
            # acc5_train = round(acc5_train.item(), 4)
            acc1_valid = round(acc1_valid, 4)
            # acc5_valid = round(acc5_valid.item(), 4)

            # writer.add_scalars('Accuracy/train_All', {name:acc1_train}, epoch)
            # writer.add_scalars('Accuracy/train_P:'+nmp , {name:acc1_train}, epoch)

            # writer.add_scalar('Accuracy/train/top5', acc5_train, epoch)
            # writer.add_scalar('Accuracy/valid_{}_{}'.format(args.prune_freq,args.dpf_epoch)', acc1_valid, epoch)
            # writer.add_scalar('Accuracy/valid/top5', acc5_valid, epoch)

            f.write(str(acc1_train)+'\n')
            ff.write(str(acc1_valid)+'\n')
            
            # writer.add_scalars('Accuracy/test_All', {name:acc1_valid}, epoch)
            # writer.add_scalars('Accuracy/test_P:'+ nmp , {name:acc1_valid}, epoch)

            # remember best Acc@1 and save checkpoint and summary csv file
            state = model.state_dict()
            summary = [epoch, acc1_train, acc1_valid]

            is_best = acc1_valid > best_acc1
            best_acc1 = max(acc1_valid, best_acc1)
            if is_best:
                save_model(arch_name, args.dataset, state, args.save)
            save_summary(arch_name, args.dataset, args.save.split('.pth')[0], summary)

            # for pruning
            if args.prune:
                num_total, num_zero, sparsity = pruning.cal_sparsity(model)
                print('\n====> sparsity: {:.2f}% || num_zero/num_total: {}/{}'.format(sparsity, num_zero, num_total))
                if not DS :
                    if (sparsity+1) >= ((args.prune_rate)*100  ) :
                        DS = 1
                print(DS,",,,")

            # end of one epoch
            print()
        
        # calculate the total training time 
        avg_train_time = train_time / (args.epochs - start_epoch)
        avg_valid_time = validate_time / (args.epochs - start_epoch)
        total_train_time = train_time + validate_time
        print('====> average training time each epoch: {:,}m {:.2f}s'.format(
            int(avg_train_time//60), avg_train_time%60))
        print('====> average validation time each epoch: {:,}m {:.2f}s'.format(
            int(avg_valid_time//60), avg_valid_time%60))
        print('====> training time: {}h {}m {:.2f}s'.format(
            int(train_time//3600), int((train_time%3600)//60), train_time%60))
        print('====> validation time: {}h {}m {:.2f}s'.format(
            int(validate_time//3600), int((validate_time%3600)//60), validate_time%60))
        print('====> total training time: {}h {}m {:.2f}s'.format(
            int(total_train_time//3600), int((total_train_time%3600)//60), total_train_time%60))
        
        f.close()
        ff.close()
        writer.close()

        return best_acc1
    
    elif args.run_type == 'evaluate':   # for evaluation
        # for evaluation on validation set
        print('\n===> [ Evaluation ]')
        
        # main evaluation
        start_time = time.time()
        acc1 = validate(args, val_loader, None, model, criterion)
        elapsed_time = time.time() - start_time
        print('====> {:.2f} seconds to evaluate this model\n'.format(
            elapsed_time))
        
        acc1 = round(acc1.item(), 4)
        # acc5 = round(acc5.item(), 4)

        # writer.add_scalar('Accuracy/test_All', {'{}_{}'.format(args.prune_freq,args.dpf_epoch):acc1}, epoch)
        # writer.add_scalar('Accuracy/test_P:{}'.format(args.prune_freq), {'{}_{}'.format(args.prune_freq,args.dpf_epoch):acc1}, epoch)

        # writer.add_scalars('Accuracy/test_All', {name:acc1}, epoch)
        # writer.add_scalars('Accuracy/test_P:'+ nmp , {name:acc1}, epoch)
        # writer.close()
        # writer.add_scalar('Accuracy/test/top5', acc5, epoch)

        # save the result

        ckpt_name = '{}-{}-{}'.format(arch_name, args.dataset, args.load[:-4])
        save_eval([ckpt_name, acc1, acc5])

        if args.prune:
            _,_,sparsity = pruning.cal_sparsity(model)
            print('Sparsity : {}'.format(sparsity))
        return acc1
    else:
        assert False, 'Unkown --run-type! It should be \{train, evaluate\}.'

def train(args, train_loader, epoch, model, criterion, optimizer,DS, **kwargs):
    r"""Train model each epoch
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    correct = 0
    progress = ProgressMeter(len(train_loader), batch_time, data_time,
                             losses, top1, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    # with torch.autograd.set_detect_anomaly(True):
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input, target = input.to("cuda"), target.to("cuda")

        if args.cuda:
            target = target.cuda(non_blocking=True)
        
        
        # for pruning
        if args.prune:
            if args.method == 'ours_freeze':
                if (globals()['iterations'] + 1) % args.prune_freq == 0 and not DS:                 # if not DS:
                    if (epoch + 1) <= args.target_epoch:
                        target_sparsity = args.prune_rate - args.prune_rate * (
                                    1 - (globals()['iterations']) / (args.target_epoch * len(train_loader))) ** 3
                        if args.prune_type == 'structured':
                            filter_mask = pruning.get_filter_mask(model, target_sparsity, args)
                            pruning.filter_prune(model, filter_mask)
                        elif args.prune_type == 'unstructured':
                            threshold = pruning.get_weight_threshold(model, target_sparsity, args)
                            pruning.weight_prune(model, threshold, args)

                    else:
                        target_sparsity = args.prune_rate
                        if args.prune_type == 'structured':
                            filter_mask = pruning.get_filter_mask(model, target_sparsity, args)
                            pruning.filter_prune(model, filter_mask)
                        elif args.prune_type == 'unstructured':
                            threshold = pruning.get_weight_threshold(model, target_sparsity, args)
                            pruning.weight_prune(model, threshold, args)
                # ours
                if (epoch + 1) <= 10 : 
                    output = model(input, 3, args.n_bits, args.acti_n_bits, 0)
                # DPF
                elif DS == 0 : 
                    output = model(input, 0, args.n_bits, args.acti_n_bits, args.acti_quan)
                # Static
                else : 
                    output = model(input, 1, args.n_bits, args.acti_n_bits, args.acti_quan)

            elif args.method == 'ours_update':
                if (globals()['iterations'] + 1) % args.prune_freq == 0 : # and not DS:                 # if not DS:
                    if (epoch + 1) <= args.target_epoch:
                        target_sparsity = args.prune_rate - args.prune_rate * (
                                    1 - (globals()['iterations']) / (args.target_epoch * len(train_loader))) ** 3
                        if args.prune_type == 'structured':
                            filter_mask = pruning.get_filter_mask(model, target_sparsity, args)
                            pruning.filter_prune(model, filter_mask)
                        elif args.prune_type == 'unstructured':
                            threshold = pruning.get_weight_threshold(model, target_sparsity, args)
                            pruning.weight_prune(model, threshold, args)

                    else:
                        target_sparsity = args.prune_rate
                        if args.prune_type == 'structured':
                            filter_mask = pruning.get_filter_mask(model, target_sparsity, args)
                            pruning.filter_prune(model, filter_mask)
                        elif args.prune_type == 'unstructured':
                            threshold = pruning.get_weight_threshold(model, target_sparsity, args)
                            pruning.weight_prune(model, threshold, args)
                # warm up epoch FC + None LSQ
                if (epoch + 1) <= 10 : 
                    output = model(input, 3, args.n_bits, args.acti_n_bits, 0)
                # DPF
                elif DS == 0 : 
                    output = model(input, 0, args.n_bits, args.acti_n_bits, args.acti_quan)
                # Static
                else : 
                    output = model(input, 1, args.n_bits, args.acti_n_bits, args.acti_quan)


            elif args.method == 'static_freeze':
                if (globals()['iterations'] + 1) % args.prune_freq == 0 and (epoch + 1) <= args.target_epoch :   # and not DS:                 # 
                    if (epoch + 1) <= args.target_epoch:
                        target_sparsity = args.prune_rate - args.prune_rate * (
                                    1 - (globals()['iterations']) / (args.target_epoch * len(train_loader))) ** 3
                        if args.prune_type == 'structured':
                            filter_mask = pruning.get_filter_mask(model, target_sparsity, args)
                            pruning.filter_prune(model, filter_mask)
                        elif args.prune_type == 'unstructured':
                            threshold = pruning.get_weight_threshold(model, target_sparsity, args)
                            pruning.weight_prune(model, threshold, args)

                    else:
                        target_sparsity = args.prune_rate
                        if args.prune_type == 'structured':
                            filter_mask = pruning.get_filter_mask(model, target_sparsity, args)
                            pruning.filter_prune(model, filter_mask)
                        elif args.prune_type == 'unstructured':
                            threshold = pruning.get_weight_threshold(model, target_sparsity, args)
                            pruning.weight_prune(model, threshold, args)
    
                # # Static
                if (epoch + 1) <= 10 : 
                    output = model(input, 3, args.n_bits, args.acti_n_bits, 0)   
                else:
                    output = model(input, 1, args.n_bits, args.acti_n_bits, args.acti_quan)

            elif args.method == 'static_update':
                if (globals()['iterations'] + 1) % args.prune_freq == 0 : # and not DS:                 # if not DS:
                    if (epoch + 1) <= args.target_epoch:
                        target_sparsity = args.prune_rate - args.prune_rate * (
                                    1 - (globals()['iterations']) / (args.target_epoch * len(train_loader))) ** 3
                        if args.prune_type == 'structured':
                            filter_mask = pruning.get_filter_mask(model, target_sparsity, args)
                            pruning.filter_prune(model, filter_mask)
                        elif args.prune_type == 'unstructured':
                            threshold = pruning.get_weight_threshold(model, target_sparsity, args)
                            pruning.weight_prune(model, threshold, args)

                    else:
                        target_sparsity = args.prune_rate
                        if args.prune_type == 'structured':
                            filter_mask = pruning.get_filter_mask(model, target_sparsity, args)
                            pruning.filter_prune(model, filter_mask)
                        elif args.prune_type == 'unstructured':
                            threshold = pruning.get_weight_threshold(model, target_sparsity, args)
                            pruning.weight_prune(model, threshold, args)
                # # Static
                if (epoch + 1) <= 10 : 
                    output = model(input, 3, args.n_bits, args.acti_n_bits, 0)   
                else:
                    output = model(input, 1, args.n_bits, args.acti_n_bits, args.acti_quan)


            elif args.method == 'dynamic_freeze':
                if (globals()['iterations'] + 1) % args.prune_freq == 0 and (epoch + 1) <= args.target_epoch :        
                    if (epoch + 1) <= args.target_epoch:
                        target_sparsity = args.prune_rate - args.prune_rate * (
                                    1 - (globals()['iterations']) / (args.target_epoch * len(train_loader))) ** 3
                        if args.prune_type == 'structured':
                            filter_mask = pruning.get_filter_mask(model, target_sparsity, args)
                            pruning.filter_prune(model, filter_mask)
                        elif args.prune_type == 'unstructured':
                            threshold = pruning.get_weight_threshold(model, target_sparsity, args)
                            pruning.weight_prune(model, threshold, args)

                    else:
                        target_sparsity = args.prune_rate
                        if args.prune_type == 'structured':
                            filter_mask = pruning.get_filter_mask(model, target_sparsity, args)
                            pruning.filter_prune(model, filter_mask)
                        elif args.prune_type == 'unstructured':
                            threshold = pruning.get_weight_threshold(model, target_sparsity, args)
                            pruning.weight_prune(model, threshold, args)
  
       
                if (epoch + 1) <= 10 : 
                    output = model(input, 3, args.n_bits, args.acti_n_bits, 0)   
                else:
                    output = model(input, 0, args.n_bits, args.acti_n_bits, args.acti_quan)
            
            elif args.method == 'dynamic_update':
                if (globals()['iterations'] + 1) % args.prune_freq == 0 :        
                    if (epoch + 1) <= args.target_epoch:
                        target_sparsity = args.prune_rate - args.prune_rate * (
                                    1 - (globals()['iterations']) / (args.target_epoch * len(train_loader))) ** 3
                        if args.prune_type == 'structured':
                            filter_mask = pruning.get_filter_mask(model, target_sparsity, args)
                            pruning.filter_prune(model, filter_mask)
                        elif args.prune_type == 'unstructured':
                            threshold = pruning.get_weight_threshold(model, target_sparsity, args)
                            pruning.weight_prune(model, threshold, args)

                    else:
                        target_sparsity = args.prune_rate
                        if args.prune_type == 'structured':
                            filter_mask = pruning.get_filter_mask(model, target_sparsity, args)
                            pruning.filter_prune(model, filter_mask)
                        elif args.prune_type == 'unstructured':
                            threshold = pruning.get_weight_threshold(model, target_sparsity, args)
                            pruning.weight_prune(model, threshold, args)
  
       
                if (epoch + 1) <= 10 : 
                    output = model(input, 3, args.n_bits, args.acti_n_bits, 0)   
                else:
                    output = model(input, 0, args.n_bits, args.acti_n_bits, args.acti_quan)

        # original 
        # output = model(input)

        # static freeze
        # output = model(input, 1, args.n_bits, args.acti_n_bits, args.acti_quan)
        # loss = criterion(output.squeeze(), target)
        loss = criterion(output, target)

        
        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # measure accuracy and record loss
        # acc1 = 100. * correct / len(train_loader.dataset)
        # losses.update(loss.item(), input.size(0))
        # top1.update(acc1, input.size(0))
        # top5.update(acc5[0], input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            progress.print(i)

        end = time.time()

        # end of one mini-batch
        globals()['iterations'] += 1
        
        if i % 20 == 0:
            print(f"Train Epoch: {epoch} [{ i * len(input)}/{len(train_loader.dataset)} ({100. * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        # pbar.update(pbar_update)
        # record loss
        # losses.append(loss.item())
        

    # print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
    #       .format(top1=top1, top5=top5))
    # print(globals()['iterations'])
    # return 0
    print("acc : ",100. * correct / len(train_loader.dataset))
    return 100. * correct / len(train_loader.dataset)

def validate(args, val_loader, epoch, model, criterion):
    r"""Validate model each epoch and evaluation
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, 
                             prefix='Test: ')
    correct = 0
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.cuda:
                target = target.cuda(non_blocking=True)


            # output = model(input,0,args.n_bits,args.acti_n_bits,args.acti_quan)
            # output = model(input)
            output = model(input,0,args.n_bits,args.acti_n_bits,args.acti_quan)
            loss = criterion(output, target)
            
            # loss = F.nll_loss(output.squeeze(), target)
            
            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)

            # measure accuracy and record loss
            # acc1 = 100. * correct / len(val_loader.dataset)
            # losses.update(loss.item(), input.size(0))
            # top1.update(acc1, input.size(0))
            # top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)

            if i % args.print_freq == 0:
                progress.print(i)

            end = time.time()

        # print('====> Acc@1 :.{}'.format(top1))
        # print('====> Acc@1 {:.3f}'.format(top1.avg.item()))
        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        # pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(val_loader.dataset)} ({100. * correct / len(val_loader.dataset):.0f}%)\n")

    return 100. * correct / len(val_loader.dataset)


def validate_t(args, val_loader, epoch, model, criterion):
    r"""Validate model each epoch and evaluation
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1,
                             prefix='Test: ')
    correct = 0
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.cuda :
                target = target.cuda(non_blocking=True)
            # output = model(input)
            output = model(input,0,args.n_bits,args.acti_n_bits,args.acti_quan)
            # output = model(input, 0, args.n_bits,args.acti_n_bits,args.acti_quan)
            loss = criterion(output, target)
            # loss = F.nll_loss(output.squeeze(), target)

            # measure accuracy and record loss
            acc1 = 100. * correct / len(val_loader.dataset)
            losses.update(loss.item(), input.size(0))
            top1.update(acc1, input.size(0))
            # top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)

            if i % args.print_freq == 0:
                progress.print(i)

            end = time.time()
        # print('====> Acc@1 :.{}'.format(top1))
        # print('====> Acc@1 {top1.avg:.3f}'
        #       .format(top1=top1))
        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        # pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(val_loader.dataset)} ({100. * correct / len(val_loader.dataset):.0f}%)\n")

    return top1.avg


if __name__ == '__main__':
    start_time = time.time()
    args = hyperparam()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num
    main(args)
    elapsed_time = time.time() - start_time
    fff = open(args.txt_time, 'w')
    print('====> total time: {}h {}m {:.2f}s'.format(
        int(elapsed_time//3600), int((elapsed_time%3600)//60), elapsed_time%60))