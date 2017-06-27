import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv_block(
        in_chan, out_chan, ksize=3, stride=1, pad=1,
        activation=nn.ReLU(), use_bn=False, dropout=0.):

    layers = [nn.Conv2d(in_chan, out_chan, kernel_size=ksize, stride=stride, padding=pad)]
    if use_bn:
        layers += [nn.BatchNorm2d(out_chan)]
    layers += [activation]
    if dropout:
        layers += [nn.Dropout(p=dropout)]
    return nn.Sequential(*layers)


def unconv_block(
        in_chan, out_chan, ksize=3, stride=1, pad=1,
        activation=nn.ReLU(), use_bn=False, dropout=0.):

    layers = [
        nn.UpsamplingBilinear2d(scale_factor=2),
        activation,
        nn.Conv2d(in_chan, out_chan, kernel_size=ksize, stride=stride, padding=pad)]
    if use_bn:
        layers += [nn.BatchNorm2d(out_chan)]
    if dropout:
        layers += [nn.Dropout(p=dropout)]
    return nn.Sequential(*layers)


def pool_layer():
    return nn.Sequential(nn.MaxPool2d(2))


class ModelFcrn(nn.Module):
    def __init__(self, inplanes=3):
        super(ModelFcrn, self).__init__()
        # params
        self.inplanes = inplanes
        self.activation = nn.LeakyReLU(0.2)
        self.use_bn = True

        torch.LongTensor()

        self.conv1 = conv_block(inplanes, 32, 3, pad=1, activation=self.activation, use_bn=self.use_bn)
        self.conv2 = conv_block(32, 64, 3, pad=1, activation=self.activation, use_bn=self.use_bn)
        self.conv3 = conv_block(64, 128, 3, pad=1, activation=self.activation, use_bn=self.use_bn)
        self.conv4 = conv_block(128, 256, 5, pad=2, activation=self.activation, use_bn=self.use_bn)
        # self.conv5 = conv_block(256, 256, 5, activation=self.activation, use_batch_norm=True)
        self.fc1 = conv_block(256, 256, 1, pad=0, activation=self.activation, use_bn=self.use_bn)
        self.pool1 = pool_layer()
        self.pool2 = pool_layer()
        self.unconv1 = unconv_block(256, 256, 3, pad=1, activation=self.activation, use_bn=self.use_bn)
        self.unconv2 = unconv_block(256, 1, 5, pad=2, activation=self.activation, use_bn=self.use_bn)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        conv1_out = self.conv1(x)  # 32
        print(conv1_out.size())
        conv2_out = self.conv2(conv1_out)  # 64
        print(conv1_out.size())
        conv3_out = self.conv3(self.pool1(conv2_out))  # 128
        print(conv1_out.size())
        conv4_out = self.conv4(conv3_out)  # 256
        print(conv1_out.size())
        fc1_out = self.fc1(self.pool2(conv4_out))  # 256
        print(conv1_out.size())
        unconv1_out = self.unconv1(fc1_out)  # 256
        print(unconv1_out.size())
        unconv2_out = self.unconv2(unconv1_out)  # 1
        print(unconv2_out.size())
        return unconv2_out

    def name(self):
        return 'fcrn'
