import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.relu(x)
        return x


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.dw_separable_conv1 = DepthwiseSeparableConv(32, 64)
        self.dw_separable_conv2 = DepthwiseSeparableConv(64, 128)
        self.dw_separable_conv3 = DepthwiseSeparableConv(128, 128)
        self.dw_separable_conv4 = DepthwiseSeparableConv(128, 256)
        self.dw_separable_conv5 = DepthwiseSeparableConv(256, 256)
        self.dw_separable_conv6 = DepthwiseSeparableConv(256, 512)
        self.dw_separable_conv7 = DepthwiseSeparableConv(512, 512)
        self.dw_separable_conv8 = DepthwiseSeparableConv(512, 512)
        self.dw_separable_conv9 = DepthwiseSeparableConv(512, 512)
        self.dw_separable_conv10 = DepthwiseSeparableConv(512, 512)
        self.dw_separable_conv11 = DepthwiseSeparableConv(512, 512)
        self.dw_separable_conv12 = DepthwiseSeparableConv(512, 1024)
        self.dw_separable_conv13 = DepthwiseSeparableConv(1024, 1024)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.dw_separable_conv1(x)
        x = self.dw_separable_conv2(x)
        x = self.dw_separable_conv3(x)
        x = self.dw_separable_conv4(x)
        x = self.dw_separable_conv5(x)
        x = self.dw_separable_conv6(x)
        x = self.dw_separable_conv7(x)
        x = self.dw_separable_conv8(x)
        x = self.dw_separable_conv9(x)
        x = self.dw_separable_conv10(x)
        x = self.dw_separable_conv11(x)
        x = self.dw_separable_conv12(x)
        x = self.dw_separable_conv13(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def main():
    blk = DepthwiseSeparableConv(64, 128)
    tmp = torch.randn(2, 64, 224, 224)
    out = blk(tmp)
    print('block:', out.shape)


    model = MobileNetV1(5)
    tmp = torch.randn(2, 3, 224, 224)
    out = model(tmp)
    print('mobile:', out.shape)

    p = sum(map(lambda p:p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()