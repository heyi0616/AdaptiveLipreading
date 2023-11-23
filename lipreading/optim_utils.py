import math
import torch
import torch.optim as optim
import numpy as np


def change_lr_on_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class CosineScheduler(object):
    def __init__(self, lr_ori, epochs):
        self.lr_ori = lr_ori
        self.epochs = epochs

    def adjust_lr(self, optimizer, epoch):
        reduction_ratio = 0.5 * (1 + math.cos(math.pi * epoch / self.epochs))
        change_lr_on_optimizer(optimizer, self.lr_ori*reduction_ratio)


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, warm_up=0, T_max=10, start_ratio=0.1):
        """
        Description:
            - get warmup consine lr scheduler

        Arguments:
            - optimizer: (torch.optim.*), torch optimizer
            - lr_min: (float), minimum learning rate
            - lr_max: (float), maximum learning rate
            - warm_up: (int),  warm_up epoch or iteration
            - T_max: (int), maximum epoch or iteration
            - start_ratio: (float), to control epoch 0 lr, if ratio=0, then epoch 0 lr is lr_min

        Example:
            <<< epochs = 100
            <<< warm_up = 5
            <<< cosine_lr = WarmupCosineLR(optimizer, 1e-9, 1e-3, warm_up, epochs)
            <<< lrs = []
            <<< for epoch in range(epochs):
            <<<     optimizer.step()
            <<<     lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
            <<<     cosine_lr.step()
            <<< plt.plot(lrs, color='r')
            <<< plt.show()

        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warm_up = warm_up
        self.T_max = T_max
        self.start_ratio = start_ratio
        self.cur = 0  # current epoch or iteration

        super().__init__(optimizer, -1)

    def get_lr(self):
        if (self.warm_up == 0) & (self.cur == 0):
            lr = self.lr_max
        elif (self.warm_up != 0) & (self.cur <= self.warm_up):
            if self.cur == 0:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur + self.start_ratio) / self.warm_up
            else:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur) / self.warm_up
                # print(f'{self.cur} -> {lr}')
        else:
            # this works fine
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * \
                 (np.cos((self.cur - self.warm_up) / (self.T_max - self.warm_up) * np.pi) + 1)

        self.cur += 1

        return [lr for base_lr in self.base_lrs]


class NoamOpt(object):
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer, logger, max_step=90000):
        """Construct an NoamOpt object."""
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.max_step = max_step
        peak_lr = self.factor * self.model_size ** (-0.5) / math.sqrt(warmup)
        logger.info("peak lr: " + str(peak_lr))

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        if self._step <= self.max_step:  # 超过之后恒定学习率然后利用scheduler衰减
            rate = self.rate()
            for p in self.optimizer.param_groups:
                p['lr'] = rate
            self._rate = rate
        self.optimizer.step()

    def get_step(self):
        return self._step

    def set_step(self, step):
        self._step = step

    def rate(self, step=None):
        """Implement `lrate` above."""
        if step is None:
            step = self._step
        return (
            self.factor
            * self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def get_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def zero_grad(self):
        """Reset gradient."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state_dict."""
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)


def get_optimizer(args, optim_policies):
    # -- define optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(optim_policies, lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'adamw':
        # optimizer = optim.AdamW(optim_policies, lr=args.lr, weight_decay=1e-2)
        optimizer = optim.AdamW(optim_policies, lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(optim_policies, lr=args.lr, weight_decay=1e-4, momentum=0.9)
    else:
        raise NotImplementedError
    return optimizer
