import os
import json
import numpy as np

import datetime
import logging

import json
import torch
import shutil


def calculateNorm2(model):
    para_norm = 0.
    for p in model.parameters():
        para_norm += p.data.norm(2)
    # print('2-norm of the neural network: {:.4f}'.format(para_norm ** .5))
    return para_norm ** .5


def calculateFlopAndParam(model, inputs):
    from thop import profile, clever_format
    flops, params = profile(model, inputs=inputs)
    flops, params = clever_format([flops*2, params], "%.3f")
    return flops, params


def showLR(optimizer):
    return optimizer.param_groups[0]['lr']


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# -- IO utils
def read_txt_lines(filepath):
    assert os.path.isfile(filepath), "Error when trying to read txt file, path does not exist: {}".format(filepath)
    with open(filepath) as myfile:
        content = myfile.read().splitlines()
    return content


def save_as_json(d, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(d, outfile, indent=4, sort_keys=True)


def load_json(json_fp):
    assert os.path.isfile(json_fp), "Error loading JSON. File provided does not exist, cannot read: {}".format(json_fp)
    with open(json_fp, 'r') as f:
        json_content = json.load(f)
    return json_content


def save2npz(filename, data=None):
    assert data is not None, "data is {}".format(data)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.savez_compressed(filename, data=data)


# -- checkpoints
class CheckpointSaver:
    def __init__(self, save_dir, checkpoint_fn='ckpt.pth', best_fn='ckpt.best.pth', best_step_fn='ckpt.best.step{}.pth',
                 save_best_step=False, lr_steps=[]):
        """
        Only mandatory: save_dir
            Can configure naming of checkpoint files through checkpoint_fn, best_fn and best_stage_fn
            If you want to keep best-performing checkpoint per step
        """

        self.save_dir = save_dir

        # checkpoint names
        self.checkpoint_fn = checkpoint_fn
        self.best_fn = best_fn
        self.best_step_fn = best_step_fn

        # save best per step?
        self.save_best_step = save_best_step
        self.lr_steps = []

        # init var to keep track of best performing checkpoint
        self.current_best = 0
        self.is_best = False
        # save best at each step?
        if self.save_best_step:
            assert lr_steps != [], "Since save_best_step=True, need proper value for lr_steps. Current: {}".format(
                lr_steps)
            self.best_for_stage = [0] * (len(lr_steps) + 1)

    def save(self, save_dict, current_perf, epoch=-1):
        """
            Save checkpoint and keeps copy if current perf is best overall or [optional] best for current LR step
        """

        # save last checkpoint
        checkpoint_fp = os.path.join(self.save_dir, self.checkpoint_fn)

        # keep track of best model
        self.is_best = current_perf > self.current_best
        if self.is_best:
            self.current_best = current_perf
            best_fp = os.path.join(self.save_dir, self.best_fn)
        save_dict['best_prec'] = self.current_best

        # keep track of best-performing model per step [optional]
        if self.save_best_step:

            assert epoch >= 0, "Since save_best_step=True, need proper value for 'epoch'. Current: {}".format(epoch)
            s_idx = sum(epoch >= l for l in lr_steps)
            self.is_best_for_stage = current_perf > self.best_for_stage[s_idx]

            if self.is_best_for_stage:
                self.best_for_stage[s_idx] = current_perf
                best_stage_fp = os.path.join(self.save_dir, self.best_stage_fn.format(s_idx))
            save_dict['best_prec_per_stage'] = self.best_for_stage

        # save
        # torch.save(save_dict, checkpoint_fp)
        print("Checkpoint saved at {}".format(checkpoint_fp))
        if self.is_best:
            # shutil.copyfile(checkpoint_fp, best_fp)
            torch.save(save_dict, best_fp)

        if self.save_best_step and self.is_best_for_stage:
            shutil.copyfile(checkpoint_fp, best_stage_fp)

    def set_best_from_ckpt(self, ckpt_dict):
        self.current_best = ckpt_dict['best_prec']
        self.best_for_stage = ckpt_dict.get('best_prec_per_stage', None)


def load_model(load_path, model, optimizer=None, allow_size_mismatch=False):
    """
    Load model from file
    If optimizer is passed, then the loaded dictionary is expected to contain also the states of the optimizer.
    If optimizer not passed, only the model weights will be loaded
    """

    # -- load dictionary
    assert os.path.isfile(load_path), "Error when loading the model, provided path not found: {}".format(load_path)
    checkpoint = torch.load(load_path)
    loaded_state_dict = checkpoint['model_state_dict']

    if allow_size_mismatch:
        loaded_sizes = {k: v.shape for k, v in loaded_state_dict.items()}
        model_state_dict = model.state_dict()
        model_sizes = {k: v.shape for k, v in model_state_dict.items()}
        mismatched_params = []
        for k in loaded_sizes:
            if loaded_sizes[k] != model_sizes[k]:
                mismatched_params.append(k)
        for k in mismatched_params:
            del loaded_state_dict[k]

    # -- copy loaded state into current model and, optionally, optimizer
    model.load_state_dict(loaded_state_dict, strict=not allow_size_mismatch)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer, checkpoint['epoch_idx'], checkpoint
    return model


# -- logging utils
def get_logger(args, save_path):
    log_path = '{}/{}_{}_log.txt'.format(save_path, args.training_mode, args.lr)
    logger = logging.getLogger("mylog" + str(args.fold))
    logger.setLevel(logging.INFO)
    LOG_FORMAT = "%(asctime)s[%(levelname)s] %(message)s"
    DATE_FORMAT = "%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    # 输出到文件
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # 输出到console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def get_one_logger(save_path):
    training_mode = save_path.split("/")[-1]
    log_path = '{}/log_{}.txt'.format(save_path, training_mode)
    logger = logging.getLogger("mylog")
    logger.setLevel(logging.INFO)
    LOG_FORMAT = "%(asctime)s[%(levelname)s] %(message)s"
    DATE_FORMAT = "%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    # 输出到文件
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # 输出到console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def update_logger_batch(args, logger, dset_loader, batch_idx, running_loss, running_corrects, running_all, batch_time,
                        data_time):
    perc_epoch = 100. * batch_idx / (len(dset_loader) - 1)
    # logger.info(
    #     f"[{running_all:5.0f}/{len(dset_loader.dataset):5.0f} ({perc_epoch:.0f}%)]\tLoss: {running_loss / running_all:.4f}\tAcc:{running_corrects / running_all:.4f}\tCost time:{batch_time.val:1.3f} ({batch_time.avg:1.3f})s\tData time:{data_time.val:1.3f} ({data_time.avg:1.3f})\tInstances per second: {args.batch_size / batch_time.avg:.2f}")
    logger.info(
        f"[{running_all:5.0f}/{len(dset_loader.dataset):5.0f} ({perc_epoch:.0f}%)]\tLoss: {running_loss / running_all:.4f}\tAcc:{running_corrects / running_all:.4f}\tCost time:{batch_time.val:1.3f} ({batch_time.avg:1.3f})s")


def get_save_folder(args, add_timestamp=True):
    # create save and log folder
    if args.test:
        save_path = '{}/{}'.format(args.logging_dir, "test")
    else:
        save_path = '{}/{}'.format(args.logging_dir, args.training_mode)
        if add_timestamp:
            save_path += '/' + datetime.datetime.now().isoformat().split('.')[0]
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    return save_path

def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    return ~make_pad_mask(lengths, xs, length_dim)
