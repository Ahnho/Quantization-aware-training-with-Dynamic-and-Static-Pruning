"""ResNet/WideResNet in PyTorch.
See the paper "Deep Residual Learning for Image Recognition"
(https://arxiv.org/abs/1512.03385)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torchaudio.transforms import Spectrogram
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import MFCC

from .stft import STFT

 
def conv3x3(n_bit,in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return mnn.MaskConv2d(n_bit,in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x9(n_bit,in_planes, out_planes, kernel_size = (1,9), stride=(0, 4),padding =0, groups=1, dilation=1):
    """3x1 convolution with padding"""
    return mnn.MaskConv2d(n_bit,in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)


def conv1x1(n_bit,in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return mnn.MaskConv2d(n_bit,in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self,n_bit, inplanes, planes, stride=1, downsample_conv=None,downsample_p=None,downsample_f=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")


        # 0 -> part use, 1-> full use
        self.type_value = 0
        self.n_bits = 0
        self.acti_quan = 0
        self.acti_n_bits = 0

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(n_bit,inplanes, planes, stride)
        self.bn1_part = norm_layer(planes)
        self.bn1_full = norm_layer(planes)
        
        # self.relu = nn.ReLU(inplace=True)
        # self.relu = mnn.LsqActivation() if self.acti_quan else nn.ReLU(inplace=True)
        # print(self.relu)
        
        self.conv2 = conv3x3(n_bit,planes, planes)
        self.bn2_part = norm_layer(planes)
        self.bn2_full = norm_layer(planes)

        self.downsample_conv = downsample_conv
        self.downsample_p = downsample_p
        self.downsample_f = downsample_f

        self.stride = stride



    def forward(self, x):
        # print(self.relu)
        identity = x

        out = self.conv1(x)
        
        # print(self.relu)
        # switch the bn
        if self.type_value == 0 or self.type_value == 1:
            out = self.bn1_part(out)
        else:
            out = self.bn1_full(out)
        
        self.relu1 = mnn.LsqActivation(out,self.acti_n_bits) if self.acti_quan else nn.ReLU(inplace=True)
        out = self.relu1(out)
        out = self.conv2(out)

        # switch the bn
        if self.type_value == 0 or self.type_value == 1:
            out = self.bn2_part(out)
        else:
            out = self.bn2_full(out)


        if self.downsample_conv is not None:

            if self.type_value == 0 or self.type_value == 1:
                temp = self.downsample_conv(x)
                identity = self.downsample_p(temp)
            else:
                temp = self.downsample_conv(x)
                identity = self.downsample_f(temp)


        out += identity
        self.relu2 = mnn.LsqActivation(out,self.acti_n_bits) if self.acti_quan else nn.ReLU(inplace=True)
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self,n_bit, inplanes, planes, stride=1, downsample_conv=None,downsample_p=None,downsample_f=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # 0 -> part use, 1-> full use
        self.type_value = 0
        self.n_bits = 0
        self.acti_quan = 0
        self.acti_n_bits = 0

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv1x1(n_bit,inplanes, width)
        self.bn1_part = norm_layer(width)
        self.bn1_full = norm_layer(width)

        self.conv2 = conv3x3(n_bit,width, width, stride, groups, dilation)
        self.bn2_part = norm_layer(width)
        self.bn2_full = norm_layer(width)

        self.conv3 = conv1x1(n_bit,width, planes * self.expansion)
        self.bn3_part = norm_layer(planes * self.expansion)
        self.bn3_full = norm_layer(planes * self.expansion)

        # self.relu = nn.ReLU(inplace=True)
        # self.relu = mnn.LsqActivation()
        # self.relu = mnn.LsqActivation() if self.acti_quan else nn.ReLU(inplace=True)

        self.downsample_conv = downsample_conv
        self.downsample_p = downsample_p
        self.downsample_f = downsample_f


        self.stride = stride

    def forward(self, x):
        # self.relu = mnn.LsqActivation(x,self.acti_n_bits) if self.acti_quan else nn.ReLU(inplace=True)
        identity = x

        out = self.conv1(x)
        # switch the bn
        if self.type_value == 0 or self.type_value == 2:
            out = self.bn1_part(out)
        else:
            out = self.bn1_full(out)

        self.relu1 = mnn.LsqActivation(out,self.acti_n_bits) if self.acti_quan else nn.ReLU(inplace=True)
        out = self.relu1(out)


        out = self.conv2(out)
        # switch the bn
        if self.type_value == 0 or self.type_value == 2:
            out = self.bn2_part(out)
        else:
            out = self.bn2_full(out)
        self.relu2 = mnn.LsqActivation(out,self.acti_n_bits) if self.acti_quan else nn.ReLU(inplace=True)
        out = self.relu2(out)


        out = self.conv3(out)
        # switch the bn
        if self.type_value == 0 or self.type_value == 2:
            out = self.bn3_part(out)
        else:
            out = self.bn3_full(out)


        if self.downsample_conv is not None:

            if self.type_value == 0 or self.type_value == 2:
                temp = self.downsample_conv(x)
                identity = self.downsample_p(temp)
            else:
                temp = self.downsample_conv(x)
                identity = self.downsample_f(temp)

        out += identity
        self.relu3 = mnn.LsqActivation(out,self.acti_n_bits) if self.acti_quan else nn.ReLU(inplace=True)
        out = self.relu3(out)

        return out


class ResNet(nn.Module):
    def __init__(self,n_bit, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        self.block_name = str(block.__name__)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = mnn.MaskConv2d(n_bit,3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                    bias=False)

        self.bn1_part = norm_layer(self.inplanes)
        self.bn1_full = norm_layer(self.inplanes)

        self.n_bits = n_bit
        # self.relu = nn.ReLU(inplace=True)
        # self.relu = mnn.LsqActivation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_part = mnn.MaskLinear(n_bit,512 * block.expansion, num_classes)
        self.fc_full = mnn.MaskLinear(n_bit,512 * block.expansion, num_classes)
        
        # self.fc_part = nn.Linear(512 * block.expansion, num_classes)
        # self.fc_full = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, mnn.MaskConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer


        downsample_conv = None
        downsample_p = None
        downsample_f = None

        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_conv = conv1x1(self.n_bits, self.inplanes, planes * block.expansion, stride)
            downsample_p = norm_layer(planes * block.expansion)
            downsample_f = norm_layer(planes * block.expansion)




        layers = []
        layers.append(block(self.n_bits, self.inplanes, planes, stride, downsample_conv,downsample_p,downsample_f, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.n_bits, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self,x,type_value,n_bits,acti_n_bits,acti_quan):
        # See note [TorchScript super()]

        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.type_value = type_value
                m.n_bits = n_bits
                m.acti_quan = acti_quan
                m.acti_n_bits = acti_n_bits

            if isinstance(m, mnn.MaskConv2d):
                # mnn.MaskConv2d.init_bit(n_bits)
                m.type_value = type_value
                m.n_bits = n_bits
                m.acti_quan = acti_quan

            if isinstance(m, Bottleneck):
                m.type_value = type_value
                m.n_bits = n_bits
                m.acti_quan = acti_quan
                m.acti_n_bits = acti_n_bits
                      
            # if isinstance(m, mnn.LsqActivation):
            #     m.acti_n_bits = acti_n_bits
            #     # print("test", acti_n_bits)

            if isinstance(m, mnn.MaskLinear):
                # mnn.MaskLinear.init_bit(n_bits)
                m.type_value = type_value
                m.n_bits = n_bits
                m.acti_quan = acti_quan
                
        x = self.conv1(x)

        if type_value == 0 or type_value == 2:
            x = self.bn1_part(x)
        else:
            x = self.bn1_full(x)
        
        self.relu = mnn.LsqActivation(x,acti_n_bits) if acti_quan else nn.ReLU(inplace=True)
        
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # type 7 is sharing the fc
        if type_value == 0 or type_value == 2 or type_value == 7:
            x = self.fc_part(x)
        else:
            x = self.fc_full(x)

        return x
    def forward(self, x,type_value,n_bits,acti_n_bits,acti_quan):
        return self._forward_impl(x,type_value,n_bits,acti_n_bits,acti_quan)


class ResNet_CIFAR(nn.Module):
    def __init__(self,n_bit, block, layers, num_classes=10 ,zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_CIFAR, self).__init__()
        self.block_name = str(block.__name__)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d


        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = mnn.MaskConv2d(n_bit,3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                    bias=False)

        self.bn1_part = norm_layer(self.inplanes)
        self.bn1_full = norm_layer(self.inplanes)
        self.acti_quan = 0
        self.n_bits = n_bit
        
        # self.relu = mnn.LsqActivation()
        # self.relu = nn.ReLU()
        # self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_part = mnn.MaskLinear(n_bit,64 * block.expansion, num_classes)
        self.fc_full = mnn.MaskLinear(n_bit,64 * block.expansion, num_classes)
        
        # self.fc_part = nn.Linear(64 * block.expansion, num_classes)
        # self.fc_full = nn.Linear(64 * block.expansion, num_classes)



        for m in self.modules():
            if isinstance(m, mnn.MaskConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer


        downsample_conv = None
        downsample_p = None
        downsample_f = None

        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_conv = conv1x1(self.n_bits,self.inplanes, planes * block.expansion, stride)
            downsample_p = norm_layer(planes * block.expansion)
            downsample_f = norm_layer(planes * block.expansion)

        layers = []
        layers.append(block(self.n_bits,self.inplanes, planes, stride, downsample_conv,downsample_p,downsample_f, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.n_bits,self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)




    # type value 0 -> pruned and update only important
    # type value 0 -> pruned and update only important
    # type value 0 -> pruned and update only important


    def _forward_impl(self, x, type_value,n_bits,acti_n_bits,acti_quan):
        # See note [TorchScript super()]
        
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.type_value = type_value
                m.n_bits = n_bits
                m.acti_quan = acti_quan
                m.acti_n_bits = acti_n_bits

            if isinstance(m, mnn.MaskConv2d):
                # mnn.MaskConv2d.init_bit(n_bits)
                m.type_value = type_value
                m.n_bits = n_bits
                m.acti_quan = acti_quan

            if isinstance(m, Bottleneck):
                m.type_value = type_value
                m.n_bits = n_bits
                m.acti_quan = acti_quan
                m.acti_n_bits = acti_n_bits
                      
            # if isinstance(m, mnn.LsqActivation):
            #     m.acti_n_bits = acti_n_bits
            #     # print("test", acti_n_bits)

            if isinstance(m, mnn.MaskLinear):
                # mnn.MaskLinear.init_bit(n_bits)
                m.type_value = type_value
                m.n_bits = n_bits
                m.acti_quan = acti_quan
                
        x = self.conv1(x)        

        if type_value == 0 or type_value == 1:
            x = self.bn1_part(x)
        else:
            x = self.bn1_full(x)
        
            
        self.relu = mnn.LsqActivation(x,acti_n_bits) if acti_quan else nn.ReLU(inplace=True)

        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # type 7 is sharing the fc
        if type_value == 0 or type_value == 1:
            x = self.fc_part(x)
        else:
            x = self.fc_full(x)

        return x

    def forward(self, x,type_value,n_bits,acti_n_bits,acti_quan):
        return self._forward_impl(x,type_value,n_bits,acti_n_bits,acti_quan)

    
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        if in_channels != out_channels: # 스트라이드가 2인 경우
            stride = 2
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()) 
        else: # 스트라이드가 1인 경우
            stride = 1
            self.residual = nn.Sequential() 

        if in_channels != out_channels: # 스트라이드가 2인 경우
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size = (1, 9), stride = stride, padding = (0, 4), bias = False)
        else: # 스트라이드가 1인 경우
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size = (1, 9), stride = stride, padding = (0, 4), bias = False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size = (1, 9), stride = 1, padding = (0, 4), bias = False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU()


    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        res = self.residual(inputs)
        out = self.relu(out + res)
        return out



class TCResNet(nn.Module):
    def __init__(self, bins, n_channels, n_class):
        super(TCResNet, self).__init__()
        """
        Args:
            bin: frequency bin or feature bin
        """
        self.conv = nn.Conv2d(
            bins, n_channels[0], kernel_size = (1, 3), padding = (0, 1), bias = False)
        
        layers = []
        for in_channels, out_channels in zip(n_channels[0:-1], n_channels[1:]):
            layers.append(Residual(in_channels, out_channels))
        self.layers = nn.Sequential(*layers)

        # Average Pooling -> FC -> Softmax로 이어지는 분류기
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(n_channels[-1], n_class)
        

    def forward(self, inputs):
        """
        Args:
            input
            [B, 1, H, W] ~ [B, 1, freq, time]
            reshape -> [B, freq, 1, time]
        """
        B, C, H, W = inputs.shape
        inputs = rearrange(inputs, "b c f t -> b f c t", c = C, f = H)
        out = self.conv(inputs)
        out = self.layers(out)
        
        # 분류기
        out = self.pool(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out
    
class STFT_TCResnet(nn.Module):
    def __init__(self, filter_length, hop_length, bins, channels, channel_scale, num_classes):
        super(STFT_TCResnet, self).__init__()
        sampling_rate      = 16000
        self.filter_length = filter_length
        self.hop_length    = hop_length
        self.bins          = bins
        self.channels      = channels
        self.channel_scale = channel_scale
        self.num_classes   = num_classes
        
        self.stft_layer = STFT(self.filter_length, self.hop_length)
        self.tc_resnet  = TCResNet(
            self.bins, [int(cha * self.channel_scale) for cha in self.channels], self.num_classes)
        
    def __spectrogram__(self, real, imag):
        spectrogram = torch.sqrt(real ** 2 + imag ** 2)
        return spectrogram
    
    def forward(self, waveform):
        real, imag  = self.stft_layer(waveform)
        spectrogram = self.__spectrogram__(real, imag)
        logits      = self.tc_resnet(spectrogram)
        return logits

class MFCC_TCResnet(nn.Module):
    def __init__(self, bins: int, channels, channel_scale: int, num_classes = 12):
        super(MFCC_TCResnet, self).__init__()
        self.sampling_rate = 16000
        self.bins          = bins
        self.channels      = channels
        self.channel_scale = channel_scale
        self.num_classes   = num_classes
        
        self.mfcc_layer = MFCC(
            sample_rate = self.sampling_rate, n_mfcc = self.bins, log_mels = True)
        self.tc_resnet  = TCResNet(
            self.bins, [int(cha * self.channel_scale) for cha in self.channels], self.num_classes)
        
    def forward(self, waveform):
        mel_sepctogram = self.mfcc_layer(waveform)
        logits      = self.tc_resnet(mel_sepctogram)
        return logits

    
    
# class audio_Block(nn.Module):
#     def __init__(self, in_channels, out_channels, conv_kernel, downsample=False):
#         super(audio_Block, self).__init__()
#         self.downsample = downsample
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel, stride=(2 if downsample else 1), padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=conv_kernel, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         if downsample:
#             self.downsample_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
#             self.downsample_bn = nn.BatchNorm2d(out_channels)
    
#     def forward(self, x):
#         identity = x
        
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
        
#         if self.downsample:
#             identity = self.downsample_bn(self.downsample_layer(x))
        
#         out += identity
#         return F.relu(out)


# class TCResNet(nn.Module):
#     def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
#         super().__init__()
#         self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
#         self.bn1 = nn.BatchNorm1d(n_channel)
#         self.pool1 = nn.MaxPool1d(4)
#         self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
#         self.bn2 = nn.BatchNorm1d(n_channel)
#         self.pool2 = nn.MaxPool1d(4)
#         self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
#         self.bn3 = nn.BatchNorm1d(2 * n_channel)
#         self.pool3 = nn.MaxPool1d(4)
#         self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
#         self.bn4 = nn.BatchNorm1d(2 * n_channel)
#         self.pool4 = nn.MaxPool1d(4)
#         self.fc1 = nn.Linear(2 * n_channel, n_output)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(self.bn1(x))
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = F.relu(self.bn2(x))
#         x = self.pool2(x)
#         x = self.conv3(x)
#         x = F.relu(self.bn3(x))
#         x = self.pool3(x)
#         x = self.conv4(x)
#         x = F.relu(self.bn4(x))
#         x = self.pool4(x)
#         x = F.avg_pool1d(x, x.shape[-1])
#         x = x.permute(0, 2, 1)
#         x = self.fc1(x)
#         return F.log_softmax(x, dim=2)


class our_Residual(nn.Module):
    def __init__(self,n_bit, in_channels, out_channels):
        super(our_Residual, self).__init__()
        
        self.type_value = 0
        self.n_bits = 0
        self.acti_quan = 0
        self.acti_n_bits = 0
        
        if in_channels != out_channels: # 스트라이드가 2인 경우
            stride = 2
            self.residual = nn.Sequential(
                conv1x9(n_bit,in_channels, out_channels, kernel_size = 1, stride = stride),
                
                nn.BatchNorm2d(out_channels)) 
        else: # 스트라이드가 1인 경우
            stride = 1
            self.residual = nn.Sequential() 

        if in_channels != out_channels: # 스트라이드가 2인 경우
            self.conv1 = conv1x9(n_bit,in_channels, out_channels, stride = stride, padding = (0, 4))
        else: # 스트라이드가 1인 경우
            self.conv1 = conv1x9(n_bit,in_channels, out_channels, stride = stride, padding = (0, 4))
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = conv1x9(n_bit,out_channels, out_channels , stride = 1, padding = (0, 4))
        self.bn2   = nn.BatchNorm2d(out_channels)
        # self.relu  = mnn.LsqActivation(out,self.acti_n_bits)


    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        self.relu0 = mnn.LsqActivation(out,self.acti_n_bits)
        out = self.relu0(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        res = self.residual(inputs)
        self.relu1 = mnn.LsqActivation(res,self.acti_n_bits)
        res = self.relu1(res)
        self.relu2 = mnn.LsqActivation(out,self.acti_n_bits)
        out = self.relu2(out + res)
        return out



class our_TCResNet(nn.Module):
    def __init__(self, n_bit,  bins, n_channels, n_class):
        super(our_TCResNet, self).__init__()
        """
        Args:
            bin: frequency bin or feature bin
        """
        self.n_bits = n_bit
        self.conv = mnn.MaskConv2d(n_bit ,bins, n_channels[0], kernel_size = (1, 3), padding = (0, 1), bias = False)
#         self.conv = nn.Conv2d(
#             bins, n_channels[0], kernel_size = (1, 3), padding = (0, 1), bias = False)
        
        layers = []
        for in_channels, out_channels in zip(n_channels[0:-1], n_channels[1:]):
            layers.append(our_Residual(self.n_bits,in_channels, out_channels))
        self.layers = nn.Sequential(*layers)

        # Average Pooling -> FC -> Softmax로 이어지는 분류기
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.linear = mnn.MaskLinear(n_bit ,n_channels[-1], n_class)
        # self.linear = nn.Linear(n_channels[-1], n_class)
 

    def forward(self, inputs,type_value,n_bits,acti_n_bits,acti_quan):
        for m in self.modules():
            if isinstance(m, our_Residual):
                m.type_value = type_value
                m.n_bits = n_bits
                m.acti_quan = acti_quan
                m.acti_n_bits = acti_n_bits

            if isinstance(m, mnn.MaskConv2d):
                # mnn.MaskConv2d.init_bit(n_bits)
                m.type_value = type_value
                m.n_bits = n_bits
                m.acti_quan = acti_quan
                      
            # if isinstance(m, mnn.LsqActivation):
            #     m.acti_n_bits = acti_n_bits
            #     # print("test", acti_n_bits)

            if isinstance(m, mnn.MaskLinear):
                # mnn.MaskLinear.init_bit(n_bits)
                m.type_value = type_value
                m.n_bits = n_bits
                m.acti_quan = acti_quan
                
        """
        Args:
            input
            [B, 1, H, W] ~ [B, 1, freq, time]
            reshape -> [B, freq, 1, time]
        """
        B, C, H, W = inputs.shape
        inputs = rearrange(inputs, "b c f t -> b f c t", c = C, f = H)
        out = self.conv(inputs)
        out = self.layers(out)
        
        # 분류기
        out = self.pool(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out
    
class our_STFT_TCResnet(nn.Module):
    def __init__(self,n_bit, filter_length, hop_length, bins, channels, channel_scale, num_classes):
        super(our_STFT_TCResnet, self).__init__()
        sampling_rate      = 16000
        self.filter_length = filter_length
        self.hop_length    = hop_length
        self.bins          = bins
        self.channels      = channels
        self.channel_scale = channel_scale
        self.num_classes   = num_classes
        self.n_bit = n_bit
        
        self.stft_layer = STFT(self.filter_length, self.hop_length)
        self.tc_resnet  = our_TCResNet(n_bit,
            self.bins, [int(cha * self.channel_scale) for cha in self.channels], self.num_classes)
        
    def __spectrogram__(self, real, imag):
        spectrogram = torch.sqrt(real ** 2 + imag ** 2)
        return spectrogram
    
    def forward(self, waveform, type_value,n_bits,acti_n_bits,acti_quan):
        real, imag  = self.stft_layer(waveform)
        spectrogram = self.__spectrogram__(real, imag)
        logits      = self.tc_resnet(spectrogram,type_value,n_bits,acti_n_bits,acti_quan)
        return logits

class our_MFCC_TCResnet(nn.Module):
    def __init__(self,n_bit, bins: int, channels, channel_scale: int, num_classes = 12):
        super(our_MFCC_TCResnet, self).__init__()
        self.sampling_rate = 16000
        self.bins          = bins
        self.channels      = channels
        self.channel_scale = channel_scale
        self.num_classes   = num_classes
        self.n_bit = n_bit
        
        self.mfcc_layer = MFCC(
            sample_rate = self.sampling_rate, n_mfcc = self.bins, log_mels = True)
        self.tc_resnet  = our_TCResNet(n_bit,
            self.bins, [int(cha * self.channel_scale) for cha in self.channels], self.num_classes)
        
    def forward(self,  waveform, type_value,n_bits,acti_n_bits,acti_quan):
        mel_sepctogram = self.mfcc_layer(waveform)
        logits      = self.tc_resnet(mel_sepctogram, type_value,n_bits,acti_n_bits,acti_quan)
        return logits



# Model configurations
cfgs = {
    '18':  (BasicBlock, [2, 2, 2, 2]),
    '34':  (BasicBlock, [3, 4, 6, 3]),
    '50':  (Bottleneck, [3, 4, 6, 3]),
    '101': (Bottleneck, [3, 4, 23, 3]),
    '152': (Bottleneck, [3, 8, 36, 3]),
}
cfgs_cifar = {
    '20':  [3, 3, 3],
    '32':  [5, 5, 5],
    '44':  [7, 7, 7],
    '56':  [9, 9, 9],
    '110': [18, 18, 18],
}


def resnet(data='cifar10', **kwargs):
    r"""ResNet models from "[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)"
    Args:
        data (str): the name of datasets
    """
    num_layers = str(kwargs.get('num_layers'))
    # in_num = int(kwargs.get('in_num'))
    # out_num = int(kwargs.get('out_num'))
    
    # set pruner
    global mnn
    mnn = kwargs.get('mnn')
    assert mnn is not None, "Please specify proper pruning method"
    n_bits = kwargs.get('n_bit')
    if data in ['cifar10', 'cifar100']:
        if num_layers in cfgs_cifar.keys():
            if int(num_layers) >= 100:
                model = ResNet_CIFAR(n_bits, Bottleneck, cfgs_cifar[num_layers], int(data[5:]))
            else:
                model = ResNet_CIFAR(n_bits,BasicBlock, cfgs_cifar[num_layers], int(data[5:]))
        else:
            model = None
        image_size = 32
    elif data == 'imagenet':
        if num_layers in cfgs.keys():
            block, layers = cfgs[num_layers]
            model = ResNet(n_bits, block, layers, 1000)
        else:
            model = None
        image_size = 224
    elif data == 'audio':
        n_blocks = 3
        n_channels = [16, 24, 32, 48] # 
        # n_blocks = 6
        # n_channels = [16, 24, 24, 32, 32, 48, 48]
        # # n_channels =  [16, 24, 24, 32, 32, 48, 48] # 14
        n_channels = [int(x * 1.0) for x in n_channels]
        # model = TCResNet(in_num,out_num)
        # model = MFCC_TCResnet(bins=40, channels=n_channels, channel_scale=3, num_classes=12)
        model = our_MFCC_TCResnet(n_bits, bins=40, channels=n_channels, channel_scale=3, num_classes=35)
        # model = STFT_TCResnet(256,128,bins=129, channels=n_channels, channel_scale=3, num_classes=12)
        # model = our_STFT_TCResnet(n_bits,256,128,bins=129, channels=n_channels, channel_scale=3, num_classes=12)
        # model = STFT_TCResnet(256,40,bins=40, channels=n_channels, channel_scale=3, num_classes=12)
        # model = ResNet_CIFAR(n_bits,BasicBlock, cfgs_cifar[num_layers])
        image_size = 32 
    else:
        model = None
        image_size = None

    return model, image_size

def TCResNet8(inputs, num_classes, width_multiplier=1.0):
    n_blocks = 3
    n_channels = [16, 24, 32, 48]
    n_channels = [int(x * width_multiplier) for x in n_channels]
    return TCResNet(inputs.shape, num_classes, n_blocks, n_channels)

def TCResNet14(inputs, num_classes, width_multiplier=1.0):
    n_blocks = 6
    n_channels = [16, 24, 24, 32, 32, 48, 48]
    n_channels = [int(x * width_multiplier) for x in n_channels]
    return TCResNet(inputs.shape, num_classes, n_blocks, n_channels)

