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


def upsample_layer(in_chan, out_chan, deconv=False):
    #if deconv:
    return nn.Sequential(
        nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2))
    #else:
    #    return nn.Sequential(
    #        nn.modules.UpsamplingBilinear2d(scale_factor=2))


def crop_and_concat(upsampled, bypass, crop=False):
    #print('pre', bypass.size())
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2])//2
        bypass = F.pad(bypass, (-c, -c, -c, -c))
    #print(upsampled.size(), bypass.size())
    return torch.cat((upsampled, bypass), 1)


class ModelCnet(nn.Module):

    def __init__(self, inplanes=3, outplanes=1):
        super(ModelCnet, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.activation = nn.LeakyReLU(0.2)
        self.use_bn = True
        self.use_pad = True

        torch.LongTensor()

        pad = 1 if self.use_pad else 0

        self.enc1 = conv_block(inplanes, 64, 3, pad=pad, activation=self.activation, use_bn=self.use_bn)
        self.enc2 = conv_block(64, 128, 3, pad=pad, activation=self.activation, use_bn=self.use_bn)
        self.enc3 = conv_block(128, 256, 3, pad=pad, activation=self.activation, use_bn=self.use_bn)
        self.enc4 = conv_block(256, 512, 3, pad=pad, activation=self.activation, use_bn=self.use_bn)
        self.enc5 = conv_block(512, 1024, 3, pad=pad, activation=self.activation, use_bn=self.use_bn)

        self.pool1 = pool_layer()
        self.pool2 = pool_layer()
        self.pool3 = pool_layer()
        self.pool4 = pool_layer()

        self.dec4 = conv_block(1024, 512, 3, pad=pad, activation=self.activation, use_bn=self.use_bn)
        self.dec3 = conv_block(512, 256, 3, pad=pad, activation=self.activation, use_bn=self.use_bn)
        self.dec2 = conv_block(256, 128, 3, pad=pad, activation=self.activation, use_bn=self.use_bn)
        self.dec1 = conv_block(128, 64, 3, pad=pad, activation=self.activation, use_bn=self.use_bn)

        self.upsample4 = upsample_layer(1024, 512)
        self.upsample3 = upsample_layer(512, 256)
        self.upsample2 = upsample_layer(256, 128)
        self.upsample1 = upsample_layer(128, 64)

        if self.use_pad:
            self.conv_final = nn.Sequential(
                nn.Conv2d(64, self.outplanes, kernel_size=1, stride=1),
                nn.ReLU())
        else:
            self.conv_final = nn.Sequential(
                nn.Conv2d(64, self.outplanes, kernel_size=1, stride=1),
                nn.UpsamplingBilinear2d(size=(284, 284)))

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
        #print(enc1_out.size())
        enc2_out = self.enc2(self.pool1(enc1_out)) #128
        #print(enc2_out.size())
        enc3_out = self.enc3(self.pool2(enc2_out)) #256
        #print(enc3_out.size())
        enc4_out = self.enc4(self.pool3(enc3_out)) #512
        #print(enc4_out.size())
        enc5_out = self.enc5(self.pool4(enc4_out)) #1024
        #print(enc5_out.size())

        crop = False if self.use_pad else True
        dec4_out = self.dec4(crop_and_concat(self.upsample4(enc5_out), enc4_out, crop=crop))
        dec3_out = self.dec3(crop_and_concat(self.upsample3(dec4_out), enc3_out, crop=crop))
        dec2_out = self.dec2(crop_and_concat(self.upsample2(dec3_out), enc2_out, crop=crop))
        dec1_out = self.dec1(crop_and_concat(self.upsample1(dec2_out), enc1_out, crop=crop))
        conv_final_out = self.conv_final(dec1_out)

        return conv_final_out
