import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


def conv_block(
        in_chan, out_chan, 
        ksize=3, stride=1,
        non_lin_fn=nn.LeakyReLU(0.1), use_batch_norm=False, dropout=0.):
    layers = []
    layers += [nn.Conv2d(in_chan, out_chan, kernel_size=ksize, stride=stride)]
    if use_batch_norm:
        layers += [nn.BatchNorm2d(out_chan)]
    layers += [non_lin_fn]
    if dropout:
        layers += [nn.Dropout(p=dropout)]
    layers += [nn.Conv2d(out_chan, out_chan, kernel_size=ksize, stride=stride)]
    if use_batch_norm:
        layers += [nn.BatchNorm2d(out_chan)]
    layers['act2'] = non_lin_fn
    if dropout:
        layers += [nn.Dropout(p=dropout)]
    return nn.Sequential(*layers)


def pool_layer():
    return nn.Sequential(nn.MaxPool2d(2))


def upsample_layer(in_chan, out_chan, deconv=False):
    #if deconv:
    return nn.Sequential(
        nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2))
    #else:
    #    return nn.Sequential(
    #        nn.modules.UpsamplingBilinear2d(scale_factor=2))


def concat(upsampled, bypass):
    return torch.cat((upsampled, bypass), 1)


class Model(nn.Module):

    def __init__(self, imsize):
        super(Model, self).__init__()
        self.imsize = imsize
        self.activation = nn.ReLU()  # nn.LeakyReLU(0.1)

        torch.LongTensor()
        
        self.enc1 = conv_block(self.imsize, 64, 3, non_lin_fn=self.activation, use_batch_norm=True)
        self.enc2 = conv_block(64, 128, 3, non_lin_fn=self.activation, use_batch_norm=True)
        self.enc3 = conv_block(128, 256, 3, non_lin_fn=self.activation, use_batch_norm=True)
        self.enc4 = conv_block(256, 512, 3, non_lin_fn=self.activation, use_batch_norm=True)
        self.enc5 = conv_block(512, 1024, 3, non_lin_fn=self.activation, use_batch_norm=True)

        self.pool1 = pool_layer()
        self.pool2 = pool_layer()
        self.pool3 = pool_layer()
        self.pool4 = pool_layer()

        self.dec4 = conv_block(1024, 512, 3, non_lin_fn=self.activation, use_batch_norm=True)
        self.dec3 = conv_block(512, 256, 3, non_lin_fn=self.activation, use_batch_norm=True)
        self.dec2 = conv_block(256, 128, 3, non_lin_fn=self.activation, use_batch_norm=True)
        self.dec1 = conv_block(128, 64, 3, non_lin_fn=self.activation, use_batch_norm=True)

        self.upsample4 = upsample_layer()
        self.upsample3 = upsample_layer()
        self.upsample2 = upsample_layer()
        self.upsample1 = upsample_layer()

        self.conv_final = nn.Sequential(
            nn.Conv2d(64, 5, kernel_size=1, stride=1),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):

        enc1_out = self.enc1(x) #64
        enc2_out = self.enc2(self.pool1(enc1_out)) #128
        enc3_out = self.enc3(self.pool2(enc2_out)) #256
        enc4_out = self.enc4(self.pool3(enc3_out)) #512
        enc5_out = self.enc5(self.pool4(enc4_out)) #1024
        dec4_out = self.dec4(concat(self.upsample4(enc5_out), enc4_out))
        dec3_out = self.dec3(concat(self.upsample3(dec4_out), enc3_out))
        dec2_out = self.dec2(concat(self.upsample2(dec3_out), enc2_out))
        dec1_out = self.dec1(concat(self.upsample1(dec2_out), enc1_out))
        conv_final_out = self.conv_final(dec1_out)

        return conv_final_out
