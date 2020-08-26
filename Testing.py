import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroPaddingConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroPaddingConv, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=0),
            nn.ConvTranspose2d(out_channel, out_channel, 3, stride=1, padding=0)
        )

    def forward(self, input):
        return self.Conv(input)

class ZP_Conv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, factor=2):
        super(ZP_Conv, self).__init__()
        c_factor = factor**2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c*c_factor, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(out_c*c_factor),
            nn.ReLU()
        )
        self.shuffler = nn.PixelShuffle(factor)
        self.conv2  = nn.Sequential(
            nn.Conv2d(out_c * c_factor, out_c * c_factor, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(out_c * c_factor),
            nn.ReLU()
        )
    def forward(self, input):
        x = input
        H, W = input.shape[2:]
        avg_pool = nn.AdaptiveAvgPool2d((H//2, W//2))
        x = self.conv1(x)
        x = avg_pool(x)
        x = self.shuffler(x)
        return x


class ZP_downSample(nn.Module):
    def __init__(self, out_c, basic_channel=32):
        super(ZP_downSample, self).__init__()
        self.module= nn.Sequential(
            nn.Conv2d(basic_channel, basic_channel*2, 3, stride=1, padding=0),
            nn.BatchNorm2d(basic_channel*2),
            nn.ReLU(),
            nn.Conv2d(basic_channel*2, basic_channel*2, 3, stride=1, padding=0),
            nn.BatchNorm2d(basic_channel*2),
            nn.ReLU(),
            nn.Conv2d(basic_channel*2, out_c, 3, stride=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    def forward(self, input):
        H, W = input.shape[2:]
        pool = nn.AdaptiveAvgPool2d((H//2, W//2))
        x = self.module(input)
        x = pool(x)
        return x

class ZP_upSample(nn.Module):
    def __init__(self, in_c, out_c, factor=2):
        super(ZP_upSample, self).__init__()
        scale_factor = factor**2
        self.conv1 = ZP_Conv(in_c, out_c*scale_factor)
        self.shuffler = nn.PixelShuffle(factor)
    def forward(self, input):
        x = self.conv1(input)
        x = self.shuffler(x)
        return x

class TestNet(nn.Module):
    def __init__(self, input_c =3, output_c=3, basic_channel=32):
        super(TestNet, self).__init__()
        self.seq = nn.Sequential(
            ZP_Conv(input_c, basic_channel),
            ZP_downSample(basic_channel*4,basic_channel),
            ZP_downSample(basic_channel*16,basic_channel*4)
        )
        self.seq2 = nn.Sequential(
            ZP_upSample(basic_channel*16, basic_channel*4),
            ZP_upSample(basic_channel*4, basic_channel),
            ZP_Conv(basic_channel, output_c),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.seq(input)
        return self.seq2(x)

class IndexPooling(nn.Module):
    def __init__(self, basic_channel):
        super(IndexPooling, self).__init__()

    def forward(self, input):
        x=input
        return x

class IndexBlock(nn.Module):
    def __init__(self, c, factor=2):
        super(IndexBlock, self).__init__()
        f = factor**2
        self.module = nn.Sequential(
            nn.Conv2d(c, f*c, 3, stride=1, padding=0),
            nn.BatchNorm2d(f*c),
            nn.ReLU(),
            nn.Conv2d(f*c, f*c, 3, stride=1, padding=0),
            nn.BatchNorm2d(f * c),
            nn.ReLU(),
            nn.Conv2d(f * c, f, 3,  stride=1, padding=0),
            nn.BatchNorm2d(f),
            nn.ReLU()
        )
        self.shuffler = nn.PixelShuffle(factor)

    def forward(self, input):
        x=input
        H, W = input.shape[2:]
        avg_pool = nn.AdaptiveAvgPool2d((H//2, W//2))
        x = self.module(x)
        x = avg_pool(x)
        x = self.shuffler(x)
        return x

class AdaptiveConv(nn.Module):
    def __init__(self, input_c, output_c, activation=nn.ReLU()):
        super(AdaptiveConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_c, output_c, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(output_c),
            activation
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_c+input_c, output_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_c),
            activation
        )
    def forward(self, input):
        x = input
        H, W = input.shape[2:]
        pool = nn.AdaptiveAvgPool2d((H, W))
        x = self.conv1(x)
        x = torch.cat([pool(x), input], dim=1)
        x = self.conv2(x)# 만약 1x1 제거하고 싶으면 주석처리
        return x

class AdaptiveConv_Down(nn.Module):
    def __init__(self, input_c, output_c, activation=nn.ReLU()):
        super(AdaptiveConv_Down, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_c, output_c, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(output_c),
            activation
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_c, output_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_c),
            activation
        )
    def forward(self, input):
        x = input
        H, W = input.shape[2:]
        pool = nn.AdaptiveAvgPool2d((H//2, W//2))
        x = self.conv1(x)
        x = pool(x)
        x = self.conv2(x)
        return x

class AdaptiveConv_Up(nn.Module):
    def __init__(self, input_c, output_c, activation=nn.ReLU()):
        super(AdaptiveConv_Up, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_c, output_c*4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(output_c*4),
            activation
        )
        self.shuffler = nn.PixelShuffle(2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_c, output_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_c),
            activation
        )
    def forward(self, input):
        x = input
        H, W = input.shape[2:]
        pool = nn.AdaptiveAvgPool2d((H,W))
        x = self.conv1(x)
        x = pool(x)
        x = self.shuffler(x)
        x = self.conv2(x)
        return x

class ResBlock_Adaptive(nn.Module):
    def __init__(self, channel, res_num=3, activation = nn.ReLU(inplace=True)):
        super(ResBlock_Adaptive, self).__init__()
        self.list = nn.ModuleList([])
        for i in range(res_num):
            seq = nn.Sequential(
                AdaptiveConv(channel, channel),
                activation
            )
            self.list.append(seq)

    def forward(self, input):
        x = input
        for seq in self.list:
            x = seq(x)+x
        return x

class UNet_Adaptive(nn.Module):
    def __init__(self, basic_channel = 64, depth = 3):
        super(UNet_Adaptive, self).__init__()
        self.down_list = nn.ModuleList([])
        self.up_list = nn.ModuleList([])
        self.activation = nn.ReLU(inplace=True)
        current_c = basic_channel
        for d in range(depth):
            seq = nn.Sequential(
                AdaptiveConv_Down(current_c, current_c * 2),
                AdaptiveConv(current_c * 2, current_c * 2)
            )
            self.down_list.append(seq)
            current_c = current_c * 2
        self.resBlock = ResBlock_Adaptive(current_c, res_num=3)
        for d in range(depth):
            seq = nn.Sequential(
                AdaptiveConv_Up(current_c, current_c // 2),
                AdaptiveConv(current_c // 2, current_c // 2),
            )
            current_c = current_c // 2
            self.up_list.append(seq)
        self.lastConv = nn.Sequential(
            AdaptiveConv(current_c, current_c),
            nn.Tanh()
        )
    def forward(self, input):
        result_list = []
        x = input
        for seq in self.down_list:
            result_list.append(x)
            x = seq(x)
        x = self.resBlock(x)
        idx = 0
        for seq in self.up_list:
            x = seq(x)
            x = x + result_list[len(result_list) - (1 + idx)]
            x = self.activation(x)
            idx += 1
        self.lastConv(x)
        return x + input

class TestNet_Pool(nn.Module):
    def __init__(self):
        super(TestNet_Pool, self).__init__()
        self.conv1 = AdaptiveConv(3, 64)
        self.unet = UNet_Adaptive()
        self.lastConv = AdaptiveConv(64, 4, nn.Tanh())
        self.tempNet = nn.Sequential(nn.Conv2d(64, 4, 3, stride=1, padding=1),
                                     nn.Tanh())
    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.unet(x1)
        x3 = self.lastConv(x2)
        x2= self.tempNet(x2)
        return x3, x2

'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input = torch.randn((1, 3, 64, 64)).to(device)
#module = IndexBlock(3)
z_conv  = TestNet_Pool().to(device)#TestNet().to(device)#ZP_Conv(3, 64).to(device)
result = z_conv(input)#module(input)
print(result.shape)
'''
'''
avg = nn.AdaptiveAvgPool2d((15, 15))
result = avg(result)
print(result.shape)
'''