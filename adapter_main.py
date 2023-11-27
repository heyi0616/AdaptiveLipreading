#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""" TCN for lipreading"""

import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from lipreading.utils import get_save_folder
from lipreading.utils import load_json, save2npz
from lipreading.utils import load_model, CheckpointSaver
from lipreading.utils import get_logger, update_logger_batch, get_one_logger
from lipreading.utils import showLR, calculateNorm2, AverageMeter, calculateFlopAndParam
from lipreading.model_for_udp import LipreadingForAdapter
from lipreading.mixup import mixup_data, mixup_criterion
from lipreading.optim_utils import get_optimizer, CosineScheduler, NoamOpt, WarmupCosineLR
from lipreading.dataloaders import get_data_loaders, get_preprocessing_pipelines, get_adp_data_loaders
from lipreading.models.lora import mark_only_lora_as_trainable


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Pytorch Lipreading ')
    # -- dataset config
    parser.add_argument('--dataset', default='lrw', help='dataset selection')
    parser.add_argument('--num-classes', type=int, default=500, help='Number of classes')
    parser.add_argument('--modality', default='video', choices=['video', 'audio'], help='choose the modality')
    # -- directory
    parser.add_argument('--data-dir', default='/data/common/lipread/LRW_visual_data_96/', help='Loaded data directory')
    parser.add_argument('--label-path', type=str, default='./labels/500WordsSortedList.txt',
                        help='Path to txt file with labels')
    parser.add_argument('--annonation-direc', default="/data/common/lipread/lipread_mp4/", help='Loaded data directory')
    # -- train
    parser.add_argument('--training-mode', default='tcn', help='tcn')
    parser.add_argument('--batch-size', type=int, default=32, help='Mini-batch size')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--init-epoch', default=0, type=int, help='epoch to start at')
    parser.add_argument('--epochs', default=80, type=int, help='number of epochs')
    parser.add_argument('--test', default=False, action='store_true', help='training mode')
    # -- mixup
    parser.add_argument('--alpha', default=0.4, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    # -- test
    parser.add_argument('--model-path', type=str, default=None, help='Pretrained model pathname')
    parser.add_argument('--allow-size-mismatch', default=False, action='store_true',
                        help='If True, allows to init from model with mismatching weight tensors. Useful to init from model with diff. number of classes')
    # -- feature extractor
    parser.add_argument('--extract-feats', default=False, action='store_true', help='Feature extractor')
    parser.add_argument('--mouth-patch-path', type=str, default=None,
                        help='Path to the mouth ROIs, assuming the file is saved as numpy.array')
    parser.add_argument('--mouth-embedding-out-path', type=str, default=None,
                        help='Save mouth embeddings to a specificed path')
    # -- json pathname
    parser.add_argument('--config-path', type=str, default="./configs/lrw_resnet18_dctcn.json",
                        help='Model configuration with json format')
    # -- other vars
    parser.add_argument('--interval', default=1000, type=int, help='display interval')
    parser.add_argument('--workers', default=5, type=int, help='number of data loading workers')
    # paths
    parser.add_argument('--logging-dir', type=str, default='/data/train_logs',
                        help='path to the directory in which to save the log file')
    parser.add_argument('-gpu', '--gpu', help='gpu id to use', default='2')

    parser.add_argument('--subject', default=-1, type=int, help='adapter speaker id (0-19)')
    parser.add_argument('--adapt_min', default=0, type=int, help='adapt speaking minutes (1,3,5)')
    parser.add_argument('--eval_epoch', default=1, type=int, help='num of epochs of eval interval')

    parser.add_argument("--fold", type=int, default=1)
    args = parser.parse_args()
    return args


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


args = load_args()
init_seeds(1, False)


def extract_feats(model):
    """
    :rtype: FloatTensor
    """
    model.eval()
    preprocessing_func = get_preprocessing_pipelines()['test']
    data = preprocessing_func(np.load(args.mouth_patch_path)['data'])  # data: TxHxW
    return model(torch.FloatTensor(data)[None, None, :, :, :].cuda(), lengths=[data.shape[0]])


def evaluate(model, dset_loader, criterion):
    model.eval()

    running_loss = 0.
    running_corrects = 0.

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dset_loader, disable=True)):
            if args.use_boundary:
                input, lengths, labels, boundaries = data
                boundaries = boundaries.cuda()
            else:
                input, lengths, labels = data
                boundaries = None
            logits = model(input.unsqueeze(1).cuda(), lengths=lengths, boundaries=boundaries)
            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()

            loss = criterion(logits, labels.cuda())
            running_loss += loss.item() * input.size(0)

    # print(f"{len(dset_loader.dataset)} in total\tCR: {running_corrects / len(dset_loader.dataset)}")
    return running_corrects / len(dset_loader.dataset), running_loss / len(dset_loader.dataset)


def train(model, dset_loader, criterion, epoch, optimizer, logger):
    data_time = AverageMeter()
    batch_time = AverageMeter()

    lr = showLR(optimizer)

    logger.info('-' * 10)
    logger.info(f"Epoch {epoch}/{args.epochs - 1}")
    logger.info(f"Current learning rate: {lr}")

    model.train()
    running_loss = 0.
    running_corrects = 0.
    running_all = 0.

    end = time.time()
    for batch_idx, data in enumerate(dset_loader):
        if args.use_boundary:
            input, lengths, labels, boundaries = data
            boundaries = boundaries.cuda()
        else:
            input, lengths, labels = data
            boundaries = None
        # measure data loading time
        data_time.update(time.time() - end)

        # --
        input, labels_a, labels_b, lam = mixup_data(input, labels, args.alpha)
        labels_a, labels_b = labels_a.cuda(), labels_b.cuda()

        optimizer.zero_grad()

        logits = model(input.unsqueeze(1).cuda(), lengths=lengths, boundaries=boundaries)

        loss_func = mixup_criterion(labels_a, labels_b, lam)
        loss = loss_func(criterion, logits)

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # -- compute running performance
        _, predicted = torch.max(F.softmax(logits, dim=1).data, dim=1)
        running_loss += loss.item() * input.size(0)
        running_corrects += lam * predicted.eq(labels_a.view_as(predicted)).sum().item() + (1 - lam) * predicted.eq(
            labels_b.view_as(predicted)).sum().item()
        running_all += input.size(0)
        # -- log intermediate results
        if batch_idx % args.interval == 0 or (batch_idx == len(dset_loader) - 1):
            update_logger_batch(args, logger, dset_loader, batch_idx, running_loss, running_corrects, running_all,
                                batch_time, data_time)

    return model


def get_model_from_json(logger, cal_params):
    assert args.config_path.endswith('.json') and os.path.isfile(args.config_path), \
        f"'.json' config path does not exist. Path input: {args.config_path}"
    args_loaded = load_json(args.config_path)
    args.backbone_type = args_loaded['backbone_type']
    args.width_mult = args_loaded['width_mult']
    args.relu_type = args_loaded['relu_type']
    args.use_boundary = args_loaded.get("use_boundary", False)

    if args_loaded.get('tcn_num_layers', ''):
        tcn_options = {'num_layers': args_loaded['tcn_num_layers'],
                       'kernel_size': args_loaded['tcn_kernel_size'],
                       'dropout': args_loaded['tcn_dropout'],
                       'dwpw': args_loaded['tcn_dwpw'],
                       'width_mult': args_loaded['tcn_width_mult'],
                       }
        logger.info("tcn options: " + str(tcn_options))
    else:
        tcn_options = {}
    if args_loaded.get('densetcn_block_config', ''):
        densetcn_options = {'block_config': args_loaded['densetcn_block_config'],
                            'growth_rate_set': args_loaded['densetcn_growth_rate_set'],
                            'reduced_size': args_loaded['densetcn_reduced_size'],
                            'kernel_size_set': args_loaded['densetcn_kernel_size_set'],
                            'dilation_size_set': args_loaded['densetcn_dilation_size_set'],
                            'squeeze_excitation': args_loaded['densetcn_se'],
                            'dropout': args_loaded['densetcn_dropout'],
                            }
        logger.info("dense tcn options: " + str(densetcn_options))
    else:
        densetcn_options = {}

    model = LipreadingForAdapter(modality=args.modality,
                                 num_classes=args.num_classes,
                                 tcn_options=tcn_options,
                                 densetcn_options=densetcn_options,
                                 # uniformer_options=uniformer_options,
                                 backbone_type=args.backbone_type,
                                 relu_type=args.relu_type,
                                 width_mult=args.width_mult,
                                 use_boundary=args.use_boundary,
                                 extract_feats=args.extract_feats).cuda()
    # params = calculateNorm2(model)
    # logger.info("model 2-norm params: " + str(params.item()))
    logger.info("training params: " + str(model.training_params))
    if cal_params:
        total = sum([param.nelement() for param in model.parameters()])
        print("parameter:%fM" % (total / 1e6))
    return model


def main(as_wrapper=False, cal_params=False):
    # -- logging
    save_path = get_save_folder(args, add_timestamp=not as_wrapper)
    print(f"Model and log being saved in: {save_path}")
    logger = get_logger(args, save_path)
    logger.disabled = as_wrapper
    ckpt_saver = CheckpointSaver(save_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # -- get model
    model = get_model_from_json(logger, cal_params)
    if cal_params:
        return
    # -- get dataset iterators
    if args.subject == -1:
        dset_loaders = get_data_loaders(args)
    else:
        logger.info("adapter subject id: {}, adapt min: {}, fold: {}".format(args.subject, args.adapt_min, args.fold))
        dset_loaders = get_adp_data_loaders(args)
    # -- get loss function
    criterion = nn.CrossEntropyLoss()
    logger.info("Basic info -> GPU: {}, batch size: {}, lr: {}, optimizer: {}, epochs: {}, pretrain model: {}"
                .format(args.gpu, args.batch_size, args.lr, args.optimizer, args.epochs, args.model_path))
    # -- get optimizer
    schedule_type = ""
    if "train" in dset_loaders:
        # args.lr = 0.0
        # optimizer = get_optimizer(args, optim_policies=model.parameters())
        optimizer = get_optimizer(args, optim_policies=model.training_tensors)
        if schedule_type == "cos":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        elif schedule_type == "noam":
            step_per_epoch = len(dset_loaders['train'])
            logger.info("step per epoch: " + str(step_per_epoch))
            warm_up_epochs, max_epochs = 5, args.epochs
            warm_up_steps = int(warm_up_epochs * step_per_epoch)
            max_steps = int(max_epochs * step_per_epoch)
            factor = 1
            optimizer = NoamOpt(model_size=512, factor=factor, warmup=warm_up_steps, optimizer=optimizer,
                                max_step=max_steps,
                                logger=logger)
            patient, cooldown = 4, 8
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.optimizer, factor=0.5, mode='max',
                                                                   patience=patient, cooldown=cooldown, min_lr=3e-7)
            logger.info(
                "noam opt info -> factor: {}, warm_up_epochs: {}, max_epochs: {}, patient: {}, cooldown: {}".format(
                    factor, warm_up_epochs, max_epochs, patient, cooldown))
        else:
            scheduler = None
    if args.model_path:
        assert args.model_path.endswith('.pth') and os.path.isfile(args.model_path), \
            f"'.pth' model path does not exist. Path input: {args.model_path}"
        # resume from checkpoint
        if args.init_epoch > 0:
            model, optimizer, epoch_idx, ckpt_dict = load_model(args.model_path, model, optimizer)
            args.init_epoch = epoch_idx
            ckpt_saver.set_best_from_ckpt(ckpt_dict)
            logger.info(f'Model and states have been successfully loaded from {args.model_path}')
        # init from trained model
        else:
            model = load_model(args.model_path, model, allow_size_mismatch=args.allow_size_mismatch)
            logger.info(f'Model has been successfully loaded from {args.model_path}')
        # feature extraction
        if args.mouth_patch_path:
            save2npz(args.mouth_embedding_out_path, data=extract_feats(model).cpu().detach().numpy())
            return
        # if test-time, performance on test partition and exit. Otherwise, performance on validation and continue (sanity check for reload)
        if args.test:
            acc_avg_test, loss_avg_test = evaluate(model, dset_loaders['test'], criterion)
            logger.info(
                f"Test-time performance on partition {'test'}: Loss: {loss_avg_test:.4f}\tAcc:{acc_avg_test:.4f}")
            return

    # -- fix learning rate after loading the ckeckpoint (latency)
    if args.model_path and args.init_epoch > 0:
        # scheduler.adjust_lr(optimizer, args.init_epoch-1)
        optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
        if scheduler:
            scheduler.load_state_dict((ckpt_dict["scheduler_state_dict"]))

    epoch = args.init_epoch
    eval_epoch = args.eval_epoch
    while epoch < args.epochs:
        model = train(model, dset_loaders['train'], criterion, epoch, optimizer, logger)
        if epoch % eval_epoch == 0:
            acc_avg_val, loss_avg_val = evaluate(model, dset_loaders['val'], criterion)
            logger.info(
                f"{'val'} Epoch:\t{epoch:2}\tLoss val: {loss_avg_val:.4f}\tAcc val:{acc_avg_val:.4f}, LR: {showLR(optimizer)}")
            # logger.info(model.tcn.avg_adapter.weight.cpu().detach().numpy())
            save_dict = {
                'epoch_idx': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            if scheduler:
                save_dict.update({'scheduler_state_dict': scheduler.state_dict()})
            ckpt_saver.save(save_dict, acc_avg_val)
            if ckpt_saver.is_best:
                logger.info("saving the best ckpt...")
        if schedule_type == "cos":
            # scheduler.adjust_lr(optimizer, epoch)
            scheduler.step()
        elif schedule_type == "noam":
            scheduler.step(acc_avg_val)
        epoch += 1

    # -- evaluate best-performing epoch on test partition
    best_fp = os.path.join(ckpt_saver.save_dir, ckpt_saver.best_fn)
    _ = load_model(best_fp, model)
    acc_avg_test, loss_avg_test = evaluate(model, dset_loaders['test'], criterion)
    logger.info(f"Test time performance of best epoch: {acc_avg_test} (loss: {loss_avg_test})")

    return acc_avg_test


if __name__ == '__main__':
    main(cal_params=False)

