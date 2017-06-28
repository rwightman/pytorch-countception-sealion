""" C-Net Model (Count-Net)
A Pytorch model (inspired by U-net architecture) for object counting.

Inspired by: https://arxiv.org/abs/1505.04597
along with density counting ideas from:
https://www.robots.ox.ac.uk/~vgg/publications/2015/Xie15/weidi15.pdf
https://arxiv.org/pdf/1705.10118.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv_block(
        in_chan, out_chan, 
        ksize=3, stride=1, pad=0,
        activation=nn.ReLU(), use_bn=False, dropout=0.):
    layers = []
    layers += [nn.Conv2d(in_chan, out_chan, kernel_size=ksize, stride=stride, padding=pad)]
    if use_bn:
        layers += [nn.BatchNorm2d(out_chan)]
    layers += [activation]
    if dropout:
        layers += [nn.Dropout(p=dropout)]
    layers += [nn.Conv2d(out_chan, out_chan, kernel_size=ksize, stride=stride, padding=pad)]
    if use_bn:
        layers += [nn.BatchNorm2d(out_chan)]
    layers += [activation]
    if dropout:
        layers += [nn.Dropout(p=dropout)]
    return nn.Sequential(*layers)


def pool_layer():
    return nn.Sequential(nn.MaxPool2d(2))


def upsample_layer(in_chan, out_chan):
    return nn.Sequential(
        nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2))


class ModelCnet(nn.Module):

    def __init__(
            self,
            inplanes=3,
            outplanes=1,
            use_batch_norm=False,
            use_padding=False,
            target_size=(256, 256),
            debug=False):

        super(ModelCnet, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.activation = nn.LeakyReLU(0.1)
        self.use_batch_norm = use_batch_norm
        self.use_padding = use_padding
        self.debug = debug

        torch.LongTensor()

        pad = 1 if self.use_padding else 0

        self.enc1 = conv_block(inplanes, 64, 3, pad=pad, activation=self.activation, use_bn=self.use_batch_norm)
        self.enc2 = conv_block(64, 128, 3, pad=pad, activation=self.activation, use_bn=self.use_batch_norm)
        self.enc3 = conv_block(128, 256, 3, pad=pad, activation=self.activation, use_bn=self.use_batch_norm)
        self.enc4 = conv_block(256, 512, 3, pad=pad, activation=self.activation, use_bn=self.use_batch_norm)
        self.enc5 = conv_block(512, 1024, 3, pad=pad, activation=self.activation, use_bn=self.use_batch_norm)

        self.pool1 = pool_layer()
        self.pool2 = pool_layer()
        self.pool3 = pool_layer()
        self.pool4 = pool_layer()

        self.dec4 = conv_block(1024, 512, 3, pad=pad, activation=self.activation, use_bn=self.use_batch_norm)
        self.dec3 = conv_block(512, 256, 3, pad=pad, activation=self.activation, use_bn=self.use_batch_norm)
        self.dec2 = conv_block(256, 128, 3, pad=pad, activation=self.activation, use_bn=self.use_batch_norm)
        self.dec1 = conv_block(128, 64, 3, pad=pad, activation=self.activation, use_bn=self.use_batch_norm)

        self.upsample4 = upsample_layer(1024, 512)
        self.upsample3 = upsample_layer(512, 256)
        self.upsample2 = upsample_layer(256, 128)
        self.upsample1 = upsample_layer(128, 64)

        if self.use_padding:
            self.conv_final = nn.Sequential(
                nn.Conv2d(64, self.outplanes, kernel_size=1, stride=1),
                nn.ReLU())
        else:
            if not isinstance(target_size, tuple):
                target_size = tuple(target_size)
            self.conv_final = nn.Sequential(
                nn.Conv2d(64, self.outplanes, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.UpsamplingBilinear2d(size=target_size))

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _print(self, x, tag=[]):
        if isinstance(tag, str):
            tag = [tag]
        if self.debug:
            if tag:
                print('%s: %s' % (' '.join(filter(None, tag)), x.size()))
            else:
                print(x.size())

    def _crop_and_concat(self, upsampled, bypass, crop=False, tag=''):
        self._print(bypass, [tag, 'bypass'])
        self._print(upsampled, [tag, 'upsampled'])
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):

        enc1_out = self.enc1(x)  # 64
        self._print(enc1_out, 'enc1')
        enc2_out = self.enc2(self.pool1(enc1_out))  # 128
        self._print(enc2_out, 'enc2')
        enc3_out = self.enc3(self.pool2(enc2_out))  # 256
        self._print(enc3_out, 'enc3')
        enc4_out = self.enc4(self.pool3(enc3_out))  # 512
        self._print(enc4_out, 'enc4')
        enc5_out = self.enc5(self.pool4(enc4_out))  # 1024
        self._print(enc5_out, 'enc5')

        crop = False if self.use_padding else True
        dec4_out = self.dec4(self._crop_and_concat(self.upsample4(enc5_out), enc4_out, crop, 'dec4'))
        dec3_out = self.dec3(self._crop_and_concat(self.upsample3(dec4_out), enc3_out, crop, 'dec3'))
        dec2_out = self.dec2(self._crop_and_concat(self.upsample2(dec3_out), enc2_out, crop, 'dec2'))
        dec1_out = self.dec1(self._crop_and_concat(self.upsample1(dec2_out), enc1_out, crop, 'dec1'))
        conv_final_out = self.conv_final(dec1_out)
        self._print(conv_final_out, 'final')

        return conv_final_out

    def name(self):
        return 'cnet'
