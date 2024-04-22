# based on:
# https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/backbone/model_irse.py
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import Conv2d
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import PReLU
from torch.nn import Dropout
from torch.nn import MaxPool2d
from torch.nn import Sequential
from torch.nn import Module

def initialize_weights(modules):
    """ Weight initilize, conv2d and linear is initialized with kaiming_normal
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()


class Flatten(Module):
    """ Flat tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class LinearBlock(Module):
    """ Convolution block without no-linear activation layer
    """
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GNAP(Module):
    """ Global Norm-Aware Pooling block
    """
    def __init__(self, in_c):
        super(GNAP, self).__init__()
        self.bn1 = BatchNorm2d(in_c, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = BatchNorm1d(in_c, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature


class GDC(Module):
    """ Global Depthwise Convolution block
    """
    def __init__(self, in_c, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = LinearBlock(in_c, in_c,
                                     groups=in_c,
                                     kernel=(7, 7),
                                     stride=(1, 1),
                                     padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(in_c, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size, affine=False)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


class SEModule(Module):
    """ SE block
    """
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction,
                          kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels,
                          kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x

class BasicBlockIR(Module):
    """ BasicBlock for IRNet
    """
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BottleneckIR(Module):
    """ BasicBlock with bottleneck for IRNet
    """
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        reduction_channel = depth // 4
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, reduction_channel, (1, 1), (1, 1), 0, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, reduction_channel, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, depth, (1, 1), stride, 0, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BasicBlockIRSE(BasicBlockIR):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class BottleneckIRSE(BottleneckIR):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] +\
           [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            # 重建中间的特征
            get_block(in_channel=128, depth=256, num_units=2),
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    elif num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=6),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=8),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]
    elif num_layers == 200:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=24),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]

    return blocks

### CVPR 2024 ###
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class XCosAttention(nn.Module):
    def __init__(self, group_size, use_softmax=True, softmax_t=1, chw2hwc=True):
        super(XCosAttention, self).__init__()
        self.embedding_net = nn.Sequential(
            nn.Conv2d(group_size, group_size , 3, padding=1),
            nn.BatchNorm2d(group_size ),
            nn.PReLU())
        self.attention = nn.Sequential(
            nn.Conv2d(group_size, group_size , 3, padding=1),
            nn.BatchNorm2d(group_size),
            nn.PReLU(),
            nn.Conv2d(group_size, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )
        self.USE_SOFTMAX = use_softmax
        self.SOFTMAX_T = softmax_t
        self.chw2hwc = chw2hwc
    def softmax(self, x, T=1):
        x /= T
        return F.softmax(x.reshape(x.size(0), x.size(1), -1), 2).view_as(x)
    def divByNorm(self, x):
        '''
            attention_weights.size(): [bs, 1, 7, 7]
        '''
        x -= x.view(x.size(0),
                    x.size(1), -1).min(dim=2)[0].repeat(1,
                                                        1,
                                                        x.size(2) * x.size(3)).view(x.size(0),
                                                                                    x.size(1),
                                                                                    x.size(2),
                                                                                    x.size(3))
        x /= x.view(x.size(0),
                    x.size(1), -1).sum(dim=2).repeat(1,
                                                     1,
                                                     x.size(2) * x.size(3)).view(x.size(0),
                                                                                 x.size(1),
                                                                                 x.size(2),
                                                                                 x.size(3))
        return x

    def forward(self, feat_grid):
        '''
            feat_grid_1.size(): [bs, group_size, 7, 7]
            attention_weights.size(): [bs, 1, 7, 7]
        '''
        proj_embed = self.embedding_net(feat_grid)
        attention_weights = self.attention(proj_embed)
        # To Normalize attention
        if self.USE_SOFTMAX:
            attention_weights = self.softmax(attention_weights, self.SOFTMAX_T)
        else:
            attention_weights = self.divByNorm(attention_weights)
        if self.chw2hwc:
            attention_weights = attention_weights.permute(0, 2, 3, 1)
        return attention_weights

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class SlerpFace(Module):
    def __init__(self, input_size, num_layers, mode='ir', group_size = 8):
        ### IR Model ###
        """ Args:
            input_size: input_size of backbone
            num_layers: num_layers of backbone
            mode: support ir or irse
        """
        super(SlerpFace, self).__init__()
        assert input_size[0] in [112, 224], \
            "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [18, 34, 50, 100, 152, 200], \
            "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], \
            "mode should be ir or ir_se"
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64), PReLU(64))
        blocks = get_blocks(num_layers)
        if num_layers <= 100:
            if mode == 'ir':
                unit_module = BasicBlockIR
            elif mode == 'ir_se':
                unit_module = BasicBlockIRSE
            output_channel = 512
        else:
            if mode == 'ir':
                unit_module = BottleneckIR
            elif mode == 'ir_se':
                unit_module = BottleneckIRSE
            output_channel = 2048

        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(output_channel),
                                           Dropout(0.4), Flatten(),
                                           Linear(output_channel * 7 * 7, 512),
                                           BatchNorm1d(512, affine=False))
        else:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel), Dropout(0.4), Flatten(),

                Linear(output_channel * 14 * 14, 512),
                BatchNorm1d(512, affine=False))
        # fc
        total_output_layer = sum([param.nelement() for param in self.output_layer.parameters()])
        print("The number of output_layer's parameter: %.2fM" % (total_output_layer/1e6))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
        total_body = sum([param.nelement() for param in self.body.parameters()])
        
        ### Xcos module ###
        self.group_cov = Sequential(Conv2d(512, group_size, (1, 1), 1, 0),
                                  BatchNorm2d(group_size),
                                  PReLU(group_size))
        # group_cov
        total_group_cov = sum([param.nelement() for param in self.group_cov.parameters()])
        
        self.group_attention = XCosAttention(group_size = group_size, use_softmax=True, softmax_t=1, chw2hwc=True)
        # group_attention
        total_group_attention = sum([param.nelement() for param in self.group_attention.parameters()])
        
        print("Total(without output_layer):%.2fM" % ((total_body + total_group_cov + total_group_attention)/1e6)) 
        initialize_weights(self.modules())

    ### 生成识别 ID Vector ###
    def forward(self, x):
        x = self.input_layer(x)
        # (512, 7, 7)
        x = self.body(x)
        x = self.output_layer(x)
        return x
    
    ### 生成解耦的组特征 ### 
    def gen_group_feature(self, x):
        # 翻转输入图像生成对称特征
        x_flip = torch.flip(x, dims=[3])
        x = self.input_layer(x)
        x_flip = self.input_layer(x_flip)
        # (512, 7, 7)
        x = self.body(x)
        x_flip = self.body(x_flip)
        # (group_size, 7, 7)
        x = self.group_cov(x) 
        x_flip = self.group_cov(x_flip) 
        return x + x_flip
    
    ### 生成组特征的cos权重矩阵 ###
    def gen_weight_map(self,x):
        map = self.group_attention(x)
        return map
