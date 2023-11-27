import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

from lipreading.models.resnet_ud_adapter import ResNet, BasicBlock
from lipreading.models.resnet_ud_pad import ResNetUserPad, BasicBlockUserPad
from lipreading.models.shufflenetv2 import ShuffleNetV2
from lipreading.models.tcn import MultibranchTemporalConvNet, TemporalConvNet
from lipreading.models.densetcn_ud_adapter import DenseTemporalConvNet
from lipreading.models.swish import Swish
from lipreading.models.se_module import SELayer


# -- auxiliary functions
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


def _average_batch(x, lengths, B):
    # x: (B, C, T)
    return torch.stack([torch.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)], 0)


class MultiscaleMultibranchTCN(nn.Module):
    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(MultiscaleMultibranchTCN, self).__init__()

        self.kernel_sizes = tcn_options['kernel_size']
        self.num_kernels = len(self.kernel_sizes)

        self.mb_ms_tcn = MultibranchTemporalConvNet(input_size, num_channels, tcn_options, dropout=dropout,
                                                    relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        xtrans = x.transpose(1, 2)
        out = self.mb_ms_tcn(xtrans)
        out = self.consensus_func(out, lengths, B)
        return self.tcn_output(out)


class TCN(nn.Module):
    """Implements Temporal Convolutional Network (TCN)
    __https://arxiv.org/pdf/1803.01271.pdf
    """

    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(TCN, self).__init__()
        self.tcn_trunk = TemporalConvNet(input_size, num_channels, dropout=dropout, tcn_options=tcn_options,
                                         relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

        self.has_aux_losses = False

    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func(x, lengths, B)
        return self.tcn_output(x)


class DenseTCN(nn.Module):
    def __init__(self, block_config, growth_rate_set, input_size, reduced_size, num_classes,
                 kernel_size_set, dilation_size_set, dropout, relu_type, squeeze_excitation=False):
        super(DenseTCN, self).__init__()

        num_features = reduced_size + block_config[-1] * growth_rate_set[-1]
        self.tcn_trunk = DenseTemporalConvNet(block_config, growth_rate_set, input_size, reduced_size,
                                              kernel_size_set, dilation_size_set,
                                              dropout=dropout, relu_type=relu_type,
                                              squeeze_excitation=squeeze_excitation,
                                              )
        self.tcn_output = nn.Linear(num_features, num_classes)
        self.consensus_func = _average_batch

    def forward(self, x, lengths, B):
        x = self.tcn_trunk(x.transpose(1, 2))
        # return x.transpose(1, 2)
        x = self.consensus_func(x, lengths, B)
        return self.tcn_output(x)


class TransformerBackend(nn.Module):
    def __init__(self, input_size, num_channels, num_classes):
        super(TransformerBackend, self).__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=num_channels, nhead=8)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.tf_output = nn.Linear(num_channels, num_classes)

    def forward(self, x, lengths, B):
        # x: (B, 29, C)
        # masks = make_non_pad_mask(lengths)
        x = x.permute(1, 0, 2)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(0)
        x = self.tf_output(x)
        return x


class ResNetAdapter(nn.Module):
    def __init__(self, channel, reduction=2):
        super(ResNetAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        y = self.fc(x)
        return x + y


class LipreadingForAdapter(nn.Module):
    def __init__(self, modality='video', hidden_dim=256, backbone_type='resnet', num_classes=500,
                 relu_type='prelu', tcn_options={}, densetcn_options={}, width_mult=1.0,
                 use_boundary=False, extract_feats=False):
        super(LipreadingForAdapter, self).__init__()
        self.extract_feats = extract_feats
        self.backbone_type = backbone_type
        self.modality = modality
        self.use_boundary = use_boundary

        if self.modality == 'video':
            self.frontend_nout = 64
            # -- frontend3D
            if relu_type == 'relu':
                frontend_relu = nn.ReLU(True)
            elif relu_type == 'prelu':
                frontend_relu = nn.PReLU(self.frontend_nout)
            elif relu_type == 'swish':
                frontend_relu = Swish()

            self.frontend3D = nn.Sequential(
                nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3),
                          bias=False),
                nn.BatchNorm3d(self.frontend_nout),
                frontend_relu,
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            )

            if self.backbone_type == 'resnet':
                # self.frontend_nout = 64
                self.backend_out = 512
                self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

        else:
            raise NotImplementedError

        if tcn_options:
            tcn_class = TCN if len(tcn_options['kernel_size']) == 1 else MultiscaleMultibranchTCN
            self.tcn = tcn_class(input_size=self.backend_out,
                                 num_channels=[hidden_dim * len(tcn_options['kernel_size']) * tcn_options[
                                     'width_mult']] * tcn_options['num_layers'],
                                 num_classes=num_classes,
                                 tcn_options=tcn_options,
                                 dropout=tcn_options['dropout'],
                                 relu_type=relu_type,
                                 dwpw=tcn_options['dwpw'],
                                 )
        elif densetcn_options:
            self.tcn = DenseTCN(block_config=densetcn_options['block_config'],
                                growth_rate_set=densetcn_options['growth_rate_set'],
                                input_size=self.backend_out if not self.use_boundary else self.backend_out + 1,
                                reduced_size=densetcn_options['reduced_size'],
                                num_classes=num_classes,
                                kernel_size_set=densetcn_options['kernel_size_set'],
                                dilation_size_set=densetcn_options['dilation_size_set'],
                                dropout=densetcn_options['dropout'],
                                relu_type=relu_type,
                                squeeze_excitation=densetcn_options['squeeze_excitation'],
                                )
        else:
            self.tcn = TransformerBackend(input_size=self.backend_out, num_channels=512, num_classes=num_classes)
        self.adapter1 = ResNetAdapter(channel=512, reduction=32)
        self.training_params = []
        self.training_tensors = []
        for name, param in self.named_parameters():
            # if "adapter" not in name and "lora_" not in name and "transition1.norm" not in name and "transition2.norm" not in name and "transition3.norm" not in name:
            if "adapter" not in name and "lora_" not in name:
                param.requires_grad = False
            else:
                self.training_params.append(name)
                self.training_tensors.append(param)

    def forward(self, x, lengths, boundaries=None):
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]  # outpu should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor(x)

        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        x = self.adapter1(x)

        return self.tcn(x, lengths, B)

    def _initialize_weights_randomly(self):

        use_sqrt = True

        if use_sqrt:
            def f(n):
                return math.sqrt(2.0 / float(n))
        else:
            def f(n):
                return 2.0 / float(n)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod(m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))
