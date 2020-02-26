import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, Conv2d, ReLU, LeakyReLU, BatchNorm1d, BatchNorm2d, Tanh
from torch.nn import MaxPool2d, UpsamplingNearest2d, ZeroPad2d

from torchvision.ops import RoIAlign, RoIPool


class Downward(Module):

    def __init__(self, in_ch, out_ch, normalise=True):

        super(Downward, self).__init__()

        self.pool = Sequential(
            ZeroPad2d((1, 2, 1, 2)),
            Conv2d(in_ch, out_ch, kernel_size=4, stride=2),
            LeakyReLU(negative_slope=0.2, inplace=True))

        if normalise:
            self.pool.add_module('batch_norm', BatchNorm2d(out_ch, momentum=0.8))

    def forward(self, x):
        x = self.pool(x)
        return x


class Upward(Module):

    def __init__(self, in_ch, out_ch):

        super(Upward, self).__init__()

        self.depool = Sequential(
            UpsamplingNearest2d(scale_factor=2),
            ZeroPad2d((1, 2, 1, 2)),
            Conv2d(in_ch, out_ch, kernel_size=4, stride=1),
            ReLU(inplace=True),
            BatchNorm2d(out_ch, momentum=0.8))

    def forward(self, x1, x2):

        x = self.depool(x1)
        x = torch.cat((x, x2), dim=1)

        return x


class FNet(Module):

    def __init__(self, nb_classes, base_filters=32):

        super(FNet, self).__init__()

        self.down1 = Downward(nb_classes, base_filters, normalise=False)
        self.down2 = Downward(base_filters, 2 * base_filters)
        self.down3 = Downward(2 * base_filters, 4 * base_filters)
        self.down4 = Downward(4 * base_filters, 8 * base_filters)
        self.down5 = Downward(8 * base_filters, 8 * base_filters)

        self.up1 = Upward(8 * base_filters, 8 * base_filters)
        self.up2 = Upward(16 * base_filters, 4 * base_filters)
        self.up3 = Upward(8 * base_filters, 2 * base_filters)
        self.up4 = Upward(4 * base_filters, base_filters)

        self.out_conv = Sequential(
            UpsamplingNearest2d(scale_factor=2),
            ZeroPad2d((1, 1, 1, 1)),
            Conv2d(2 * base_filters, 1, kernel_size=3, stride=1),
            Tanh())

    def forward(self, x):

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.out_conv(x)

        return x


class PatchGAN(Module):

    def __init__(self, nb_classes, base_filters=16):

        super(PatchGAN, self).__init__()

        self.down1 = Downward(nb_classes + 1, base_filters, normalise=False)
        self.down2 = Downward(base_filters, 2 * base_filters)

        self.padding = ZeroPad2d((1, 2, 1, 2))
        self.validity = Conv2d(2 * base_filters, 1, kernel_size=4, stride=1)

    def forward(self, x, y):

        x = torch.cat([x, y], axis=1)

        x = self.down1(x)
        x = self.down2(x)

        x = self.padding(x)
        x = self.validity(x)

        return x


class ConvBlock(Module):

    def __init__(self, in_ch, out_ch, normalise=True):

        super(ConvBlock, self).__init__()

        self.conv_block = Sequential(
            Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            LeakyReLU(negative_slope=0.2, inplace=True))

        if normalise:
            self.conv_block.add_module('batch_norm', BatchNorm2d(out_ch, momentum=0.8))

    def forward(self, x):

        return self.conv_block(x)


class RoIGAN(Module):

    def __init__(self, nb_classes, base_features=16):

        super(RoIGAN, self).__init__()

        self.base_features = base_features

        self.features = Sequential(
            ConvBlock(nb_classes + 1, base_features, normalise=False),
            ConvBlock(base_features, base_features),
            MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(base_features, 2 * base_features),
            ConvBlock(2 * base_features, 2 * base_features))

        self.roi_align = RoIPool(output_size=(7, 7), spatial_scale=1)

        self.classifier = Sequential(
            Linear(98 * base_features, 1))

    def forward(self, x, y, boxes):

        x = torch.cat([x, y], axis=1)

        features = self.features(x)
        # N.B. box coordinates correspond to features, not input_img
        roi = self.roi_align(features, boxes)

        roi_flat = roi.view(-1, 98 * self.base_features)
        valid = self.classifier(roi_flat)

        return valid
