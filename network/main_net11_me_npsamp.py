import torch
import torch.nn as nn
import torch.nn.functional as F
from .rev_blocks_ca import RevSequential, RevGroupBlock
from .rev_utils import rev_sequential_backward_wrapper


def ConvBnAct(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation),
                         nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9),
                         act)


def BnActConv(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1, dilation=1):
    return nn.Sequential(nn.BatchNorm2d(in_ch, eps=1e-8, momentum=0.9),
                         act,
                         nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation))


def DeConvBnAct(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1, dilation=1, out_pad=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, ker_sz, stride, pad, output_padding=out_pad, groups=group, bias=False, dilation=dilation),
        nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9),
        act)


def DenseBnAct(in_ch, out_ch, act=nn.Identity()):
    return nn.Sequential(nn.Linear(in_ch, out_ch, bias=False),
                         nn.BatchNorm1d(out_ch, eps=1e-8, momentum=0.9),
                         act)


def PsConvBnAct(upscale, in_ch, out_ch, ker_sz, stride, pad, group=1, dilation=1):
    return nn.Sequential(
        nn.PixelShuffle(upscale),
        nn.Conv2d(in_ch//(upscale**2), out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation),
        nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9))


class SEBlock(nn.Module):
    def __init__(self, in_ch, act):
        super().__init__()
        inter_ch = in_ch
        self.conv1 = ConvBnAct(in_ch, inter_ch, 1, 1, 0, act)
        self.conv2 = nn.Conv2d(inter_ch, in_ch, 1, 1, 0)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.conv1(y)
        y = self.conv2(y)
        y = torch.softmax(y, 1)
        y = y * x
        return y

class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()


        self.avg_pool_x = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, None))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
        b, c, h, w = x.shape
        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, h], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out

class MultiScalePyramid(nn.Module):
    def __init__(self, in_ch, act):
        super().__init__()
        self.v_conv1 = BnActConv(in_ch, in_ch, 1, 1, 0, act)
        self.conv1 = BnActConv(in_ch, in_ch//2, 3, 1, 7, act, dilation=7)
        self.conv2 = BnActConv(in_ch, in_ch//2, 3, 1, 14, act, dilation=14)

    def forward(self, x):
        v = self.v_conv1(x)
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y = torch.cat([y1, y2], 1)
        y = torch.softmax(y, 1)
        y = v * y
        y = x + y
        return y


class ResBlockA(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act):
        super().__init__()
        assert stride in (1, 2)
        if stride == 1 and in_ch == out_ch:
            self.skip = nn.Identity()
        else:
            skip_layer = []
            if stride > 1:
                skip_layer.append(nn.AvgPool2d(2, 2, 0, ceil_mode=True, count_include_pad=False))
            if in_ch != out_ch:
                skip_layer.append(nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False))
                skip_layer.append(nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9))
            self.skip = nn.Sequential(*skip_layer)

        inter_ch = out_ch // 1
        self.conv1 = BnActConv(in_ch, inter_ch, ker_sz=3, stride=stride, pad=1, act=act)
        self.conv2 = BnActConv(inter_ch, out_ch, ker_sz=3, stride=1, pad=1, act=act)
        self.a = nn.Parameter(torch.zeros(1), True)

    def forward(self, x):
        skip_y = self.skip(x)
        y = self.conv1(x)
        y = self.conv2(y)
        y = skip_y + y * self.a
        return y


class ResBlockB(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act):
        super().__init__()
        assert stride in (1, 2)
        if stride == 1 and in_ch == out_ch:
            self.skip = nn.Identity()
        else:
            skip_layer = []
            if stride > 1:
                skip_layer.append(nn.UpsamplingBilinear2d(scale_factor=2.))
            if in_ch != out_ch:
                skip_layer.append(nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False))
                skip_layer.append(nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9))
            self.skip = nn.Sequential(*skip_layer)

        inter_ch = out_ch // 2

        if stride == 1:
            self.conv1 = BnActConv(in_ch, inter_ch, ker_sz=3, stride=stride, pad=1, act=act)
        else:
            # self.conv1 = DeConvBnAct(in_ch, inter_ch, 2, stride, 0, act)
            self.conv1 = nn.Sequential(BnActConv(in_ch, inter_ch, 3, 1, 1, act),
                                       nn.UpsamplingBilinear2d(scale_factor=2.))
        self.conv2 = BnActConv(inter_ch, out_ch, ker_sz=3, stride=1, pad=1, act=act)
        self.a = nn.Parameter(torch.zeros(1), True)

    def forward(self, x):
        skip_y = self.skip(x)
        y = self.conv1(x)
        y = self.conv2(y)
        y = skip_y + y * self.a
        return y


class RevBlockC(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act,h, **kwargs):
        super().__init__()
        assert in_ch == out_ch
        assert stride == 1
        inter_ch = in_ch // 1
        self.conv1 = BnActConv(in_ch, inter_ch, ker_sz=3, stride=1, pad=1, act=act)
        self.conv2 = BnActConv(inter_ch, out_ch, ker_sz=3, stride=1, pad=1, act=act)
        # self.se = SEBlock(in_ch, act)
        self.se = CA_Block(in_ch)
        self.a = nn.Parameter(torch.zeros(1), True)

    def func(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.se(y)
        y = x + y * self.a
        return y

    def forward(self, x1, x2):
        y = x1 + self.func(x2)
        return x2, y

    def invert(self, y1, y2):
        x2, y = y1, y2
        x1 = y - self.func(x2)
        return x1, x2


class GroupBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act, block_type, blocks, **kwargs):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        layers = []
        for i in range(blocks):
            if i == 0:
                block = block_type(in_ch=in_ch, out_ch=out_ch, stride=stride, act=act, **kwargs)
            else:
                block = block_type(in_ch=out_ch, out_ch=out_ch, stride=1, act=act, **kwargs)
            layers.append(block)
        self.layers = nn.Sequential(*layers)

    # @torch.jit.script_method
    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class upsamp(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, scare, last=True):
        super().__init__()
        act = nn.LeakyReLU(0.02, inplace=True)

        self.conv1 = nn.Sequential(
            ConvBnAct(in_ch, mid_ch, 3, 1, 1, act),
            PsConvBnAct(scare, mid_ch, mid_ch, scare+1, 1, scare//2))

        self.conv2 = nn.Sequential(
            ConvBnAct(mid_ch*2, mid_ch, 3, 1, 1, act),
            ConvBnAct(mid_ch, mid_ch, 3, 1, 1, act),
        )
        if last:
            self.last = nn.Conv2d(mid_ch, out_ch, 1, 1, 0, bias=True)
        else:
            self.last = nn.Sequential()

    # @torch.jit.script_method
    def forward(self, x1, x2):
        x = self.conv1(x1)
        x = torch.cat((x, x2),dim=1)
        x = self.conv2(x)
        outputs = self.last(x)
        return outputs


class MainNet(nn.Module):
    model_id = 11

    def __init__(self, in_dim=3, b1_out_dim=2, b2_out_dim=2, b3_out_dim=4):
        super().__init__()
        act = nn.LeakyReLU(0.02, inplace=True)

        self._bm1 = nn.ModuleList()     # 第一段，检测粗略输出
        self._bm2 = nn.ModuleList()     # 第二段，检测细化输出
        self._bm3 = nn.ModuleList()     # 第三段，纯分类输出

        # 168x
        # 检测粗略分支
        self.conv1 = ConvBnAct(in_dim, 64, 3, 1, 1, act)
        self.rvb1 = RevGroupBlock(64, 64, 1, act, RevBlockC, 8,64)


        self.rb2 = ResBlockA(64, 128, 2, act)
        self.rvb2 = RevGroupBlock(128, 128, 1, act, RevBlockC, 8,32)

        # self.b1_conv1 = nn.Sequential(
        #     ConvBnAct(128, 128, 1, 1, 0, act),
        #     PsConvBnAct(2, 128, 128, 3, 1, 1),
        #     nn.Conv2d(128, b1_out_dim, 1, 1, 0, bias=True)
        # )
        self.b1_conv1 = upsamp(128,64,b1_out_dim,2)

        self.rb3 = ResBlockA(128, 256, 2, act)
        self.rvb3 = RevGroupBlock(256, 256, 1, act, RevBlockC, 8,16)

        # self.b2_conv1 = nn.Sequential(
        #     ConvBnAct(256, 256, 3, 1, 1, act),
        #     PsConvBnAct(4, 256, 256, 5, 1, 2),
        #     nn.Conv2d(256, b1_out_dim, 1, 1, 0, bias=True)
        # )
        self.b2_conv1 = upsamp(256, 128, 128, 2, False)
        self.b2_conv2 = upsamp(128, 64, 1, 2)

        self._bm1.extend([self.conv1, self.rvb1, self.rb2, self.rvb2, self.rb3, self.rvb3, self.b1_conv1, self.b2_conv1, self.b2_conv2])

        # 检测细化分支
        # self.rb4 = ResBlockA(256, 512, 2, act)
        self.rvb4 = RevGroupBlock(256, 256, 1, act, RevBlockC, 9,16)

        self.b4_conv1 = nn.Sequential(
            ConvBnAct(256, 256, 1, 1, 0, act),
            PsConvBnAct(4, 256, 256, 5, 1, 2),
            nn.Conv2d(256, b2_out_dim, 1, 1, 0, bias=True)
        )

        self._bm2.extend([self.rvb4, self.b4_conv1])

        # 纯分类分支
        self.rvb5 = RevGroupBlock(256, 256, 1, act, RevBlockC, 9,16)

        self.b5_conv1 = nn.Sequential(
            ConvBnAct(256, 256, 1, 1, 0, act),
            PsConvBnAct(4, 256, 256, 5, 1, 2),
            nn.Conv2d(256, b3_out_dim, 1, 1, 0, bias=True)
        )
        # self.b5_conv1 = upsamp(512, 256, 256, 2, False)
        # self.b5_conv2 = upsamp(256, 128, 128, 2, False)
        # self.b5_conv3 = upsamp(128, 64, b3_out_dim, 2)

        self._bm3.extend([self.rvb5, self.b5_conv1])

        self.enabled_b2_branch = True
        self.enabled_b3_branch = True
        self.is_freeze_seg1 = False
        self.is_freeze_seg2 = False
        self.is_freeze_seg3 = False

    def set_freeze_seg1(self, b):
        assert isinstance(b, bool)
        self.is_freeze_seg1 = b
        self._bm1.train(not b)
        for p in self._bm1.parameters():
            p.requires_grad = not b

    def set_freeze_seg2(self, b):
        assert isinstance(b, bool)
        self.is_freeze_seg2 = b
        self._bm2.train(not b)
        for p in self._bm2.parameters():
            p.requires_grad = not b

    def set_freeze_seg3(self, b):
        assert isinstance(b, bool)
        self.is_freeze_seg3 = b
        self._bm3.train(not b)
        for p in self._bm3.parameters():
            p.requires_grad = not b

    def seg1_state_dict(self):
        return dict(self._bm1.state_dict())

    def load_seg1_state_dict(self, *args, **kwargs):
        self._bm1.load_state_dict(*args, **kwargs)

    def seg2_state_dict(self):
        return dict(self._bm2.state_dict())

    def load_seg2_state_dict(self, *args, **kwargs):
        self._bm2.load_state_dict(*args, **kwargs)

    def seg3_state_dict(self):
        return dict(self._bm3.state_dict())

    def load_seg3_state_dict(self, *args, **kwargs):
        self._bm3.load_state_dict(*args, **kwargs)

    def rev_warp(self, rvb, x):
        if self.training:
            y1, y2 = rev_sequential_backward_wrapper(rvb, x, x, preserve_rng_state=False)
        else:
            y1, y2 = rvb(x, x)
        y = (y1 + y2) / 2
        # y = y1
        return y

    def forward(self, x):
        # 168
        y_1 = self.conv1(x)
        y_1 = self.rev_warp(self.rvb1, y_1)

        y_2 = self.rb2(y_1)
        y_2 = self.rev_warp(self.rvb2, y_2)

        y_det_rough_b1 = self.b1_conv1(y_2, y_1)

        y_3 = self.rb3(y_2)
        y_3 = self.rev_warp(self.rvb3, y_3)

        y_det_rough_b2 = self.b2_conv1(y_3, y_2)
        y_det_rough_b2 = self.b2_conv2(y_det_rough_b2, y_1)

        y_det_rough = y_det_rough_b1 + y_det_rough_b2

        y_det_refine = None
        y_cla = None

        # y_4 = self.rb4(y_3)
        if self.enabled_b2_branch:
            yb2 = y_3
            if self.is_freeze_seg1 and torch.is_grad_enabled():
                yb2 = yb2 + torch.zeros(1, dtype=yb2.dtype, device=yb2.device, requires_grad=True)
            yb2 = self.rev_warp(self.rvb4, yb2)
            y_det_refine = self.b4_conv1(yb2)

        if self.enabled_b3_branch:
            yb3 = y_3
            if self.is_freeze_seg1 and torch.is_grad_enabled():
                yb3 = yb3 + torch.zeros(1, dtype=yb3.dtype, device=yb3.device, requires_grad=True)
            yb3 = self.rev_warp(self.rvb5, yb3)
            y_cla = self.b5_conv1(yb3)

        return y_det_rough, y_det_refine, y_cla


if __name__ == '__main__':
    # import model_utils_torch
    from torchsummary import summary
    a = torch.zeros(8, 3, 64, 64)

    # a = torch.zeros(2, 512, 512, 3).cuda(0).permute(0, 3, 1, 2)
    net = MainNet(3, 2, 1, 5)
    summary(net, input_size=(3,64,64),batch_size=1, device='cpu')
    # net.load_state_dict(torch.load('./train_2_10/finish_hed_me_model_big_channel_au/11_model_best_p3.pt', 'cpu'))
    # print('success')