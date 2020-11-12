# This file contains modules common to various models

import math

import numpy as np
import torch
import torch.nn as nn

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    img_size = 640  # inference size (pixels)
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model

    def forward(self, x, size=640, augment=False, profile=False):
        # supports inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   opencv:     x = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:        x = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:      x = np.zeros((720,1280,3))  # HWC
        #   torch:      x = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:   x = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        p = next(self.model.parameters())  # for device and type
        if isinstance(x, torch.Tensor):  # torch
            return self.model(x.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        if not isinstance(x, list):
            x = [x]
        shape0, shape1 = [], []  # image and inference shapes
        batch = range(len(x))  # batch size
        for i in batch:
            x[i] = np.array(x[i])  # to numpy
            x[i] = x[i][:, :, :3] if x[i].ndim == 3 else np.tile(x[i][:, :, None], 3)  # enforce 3ch input
            s = x[i].shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(x[i], new_shape=shape1, auto=False)[0] for i in batch]  # pad
        x = np.stack(x, 0) if batch[-1] else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32

        # Inference
        x = self.model(x, augment, profile)  # forward
        x = non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS

        # Post-process
        for i in batch:
            if x[i] is not None:
                x[i][:, :4] = scale_coords(shape1, x[i][:, :4], shape0[i])
        return x


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


class BatchRenorm2D(nn.Module):

    def __init__(self, num_features,  gamma=None, beta=None,
                 running_mean=None, running_var=None, eps=1e-05,
                 momentum=0.01, r_d_max_inc_step=0.0001, r_max=1.0,
                 d_max=0.0, max_r_max=3.0, max_d_max=5.0):
        super(BatchRenorm2D, self).__init__()

        self.eps = eps
        self.num_features = num_features
        self.momentum = torch.tensor(momentum, requires_grad=False)

        if gamma is None:
            self.gamma = torch.nn.Parameter(
                torch.ones((1, num_features, 1, 1)), requires_grad=True)
        else:
            self.gamma = torch.nn.Parameter(gamma.view(1, -1, 1, 1), requires_grad=True)
        if beta is None:
            self.beta = torch.nn.Parameter(
                torch.zeros((1, num_features, 1, 1)), requires_grad=True)
        else:
            self.beta = torch.nn.Parameter(beta.view(1, -1, 1, 1), requires_grad=True)

        if running_mean is None:
            self.register_buffer("running_avg_mean", torch.ones((1, num_features, 1, 1)))
            self.register_buffer("running_avg_std", torch.zeros((1, num_features, 1, 1)))
        else:
            self.register_buffer("running_avg_mean", running_mean.view(1, -1, 1, 1))
            self.register_buffer("running_avg_std", torch.sqrt(running_var.view(1, -1, 1, 1)))

        self.max_r_max = max_r_max
        self.max_d_max = max_d_max

        self.r_max_inc_step = r_d_max_inc_step
        self.d_max_inc_step = r_d_max_inc_step

        self.r_max = r_max
        self.d_max = d_max

    def forward(self, x):

        device = self.gamma.device

        batch_ch_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True).to(device)
        batch_ch_std = torch.sqrt(torch.var(
            x, dim=(0, 2, 3), keepdim=True, unbiased=False) + self.eps)
        batch_ch_std = batch_ch_std.to(device)

        self.running_avg_std = self.running_avg_std.to(device)
        self.running_avg_mean = self.running_avg_mean.to(device)
        self.momentum = self.momentum.to(device)

        if self.training:
            r = torch.clamp(batch_ch_std / self.running_avg_std, 1.0 /
                            self.r_max, self.r_max).to(device).data.to(device)
            d = torch.clamp((batch_ch_mean - self.running_avg_mean) /
                            self.running_avg_std, -self.d_max, self.d_max)\
                .to(device).data.to(device)

            x = ((x - batch_ch_mean) * r) / batch_ch_std + d
            x = self.gamma * x + self.beta

            if self.r_max < self.max_r_max:
                self.r_max += self.r_max_inc_step * x.shape[0]

            if self.d_max < self.max_d_max:
                self.d_max += self.d_max_inc_step * x.shape[0]

            self.running_avg_mean = self.running_avg_mean + self.momentum * \
                (batch_ch_mean.data.to(device) - self.running_avg_mean)
            self.running_avg_std = self.running_avg_std + self.momentum * \
                (batch_ch_std.data.to(device) - self.running_avg_std)

        else:

            x = (x - self.running_avg_mean) / self.running_avg_std
            x = self.gamma * x + self.beta

        return x
