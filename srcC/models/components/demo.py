import torch
import torch.nn as nn
from torchvision.models._utils import _make_divisible

# 通道重排函数
# def channel_shuffle(x, groups):
#     batch_size, num_channels, height, width = x.size()
#     channels_per_group = num_channels // groups
#     x = x.view(batch_size, groups, channels_per_group, height, width)
#     x = torch.transpose(x, 1, 2).contiguous()
#     x = x.view(batch_size, -1, height, width)
#     return x

class ECAModule(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 为了适应 1D 卷积操作，我们将 channels 维度调整为 length 维度
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.gamma = gamma
        self.b = b

    def forward(self, x):
        # 对输入特征图进行全局平均池化，得到每个通道的平均值
        y = self.avg_pool(x)
        # 将维度调整为 [batch_size, channels, 1]，以适应 1D 卷积操作
        y = y.view(y.size(0), y.size(1), -1)
        # 通过 1D 卷积操作，学习到每个通道的重要性
        y = self.conv(y)
        # 对学习到的权重进行缩放和平移
        y = self.sigmoid(self.gamma * y + self.b)
        # 将权重广播到原始特征图维度
        y = y.unsqueeze(-1).expand_as(x)
        # 将特征图与权重相乘，实现通道注意力加权
        return x * y

# 倒残差结构
class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNLeakyReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, stride=stride, padding=1, groups=hidden_channel, bias=False),
            nn.BatchNorm2d(hidden_channel),
            ECAModule(hidden_channel),  # 添加ECA自注意力模块
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

# MobileNetV2模型
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        features.append(ConvBNLeakyReLU(3, input_channel, stride=2))
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNLeakyReLU(input_channel, last_channel, 1))
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def ConvBNLeakyReLU(in_channel, out_channel, kernel_size=3, stride=1, groups=1):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(inplace=True)  # 使用LeakyReLU函数
    )

def main():
    model = MobileNetV2(8)
    tmp = torch.randn(2, 3, 224, 224)
    out = model(tmp)
    print('mobilenetv2:', out.shape)

    p = sum(map(lambda p:p.numel(), model.parameters()))
    print('parameters size:', p)

if __name__ == '__main__':
    main()