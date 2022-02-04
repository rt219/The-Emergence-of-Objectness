import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.ops import resize

from ..utils.warp_utils import flow_warp
#from .correlation_package.correlation import Correlation
from .correlation_native import Correlation
import pdb

def _resize(in_img, shape):
    output = resize(
        in_img,
        size=shape,
        mode='bilinear',
        align_corners=True,
        warning=False)
    return output

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, isBias=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=isBias)
        )


class FeatureExtractor(nn.Module):
    def __init__(self, num_chs):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out)
            )
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class FlowEstimatorReduce(nn.Module):
    # can reduce 25% of training time.
    def __init__(self, ch_in, mask_layer=1):
        super(FlowEstimatorReduce, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(128, 128)
        self.conv3 = conv(128 + 128, 96)
        self.conv4 = conv(128 + 96, 64)
        self.conv5 = conv(96 + 64, 32)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.feat_dim = 32
        self.predict_flow1 = conv(64 + 32, 64, isReLU=True, kernel_size=1, isBias=True)
        self.predict_flow2 = conv(64, 2, isReLU=False, kernel_size=1, isBias=True)
        self.mask_layer = mask_layer

    def forward(self, x, mask):
        _b, _, _h, _w = x.shape
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x3, x4], dim=1))
        
        __feat__ = torch.cat([x4, x5], dim=1)
        flow_direct = self.predict_flow2(self.predict_flow1(__feat__))
        flow_group=[]
        
        def get_averge_feat(_mask, in_feat):
            _bb, _cc, _hh, _ww = in_feat.shape
            cur_feat = torch.sum((in_feat * _mask).view(_bb, _cc, -1), 2) / torch.sum(_mask.view(_mask.shape[0], _mask.shape[1], -1), 2)
            cur_feat = torch.unsqueeze(torch.unsqueeze(cur_feat, 2), 2)
            cur_feat = cur_feat.repeat(1, 1, _hh, _ww)
            return cur_feat
        
        for _i in range(self.mask_layer):
            _mask = mask[:, _i:_i+1, :, :]
            cur_feat = get_averge_feat(_mask, __feat__)
            cur_flow = self.predict_flow2(self.predict_flow1(cur_feat))
            flow_group.append(cur_flow)

        flow_group = [-1] + flow_group
        
        
        for _i_ in range(self.mask_layer):
            if _i_ == 0:
                _flow = mask[:, _i_:_i_+1, : , :] * flow_group[_i_ + 1]
            else:
                _flow = _flow + mask[:, _i_:_i_+1, : , :] * flow_group[_i_ + 1]
        
        return x5, flow_group, flow_direct


class PWCLite(nn.Module):
    def __init__(self, mask_layer):
        super(PWCLite, self).__init__()
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 192]
        self.num_levels = 7
        self.output_level = 4
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)
        self.mask_layer = mask_layer
        
        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)

        self.upsample = True
        self.n_frames = 2
        self.reduce_dense = True

        self.corr = Correlation(pad_size=self.search_range, kernel_size=1,
                                max_displacement=self.search_range, stride1=1,
                                stride2=1, corr_multiply=1)

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = 32 + (self.dim_corr + 2) * (self.n_frames - 1) 

        if self.reduce_dense:
            self.flow_estimators = FlowEstimatorReduce(self.num_ch_in, mask_layer)
        else:
            raise NotImplementedError

        self.conv_1x1 = nn.ModuleList([conv(192, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1),
                                     ])

    def num_parameters(self):
        return sum(
            [p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward_2_frames(self, x1_pyramid, x2_pyramid, mask):
        mask_layer = mask.shape[1]
        assert mask_layer == self.mask_layer
        
        # outputs
        flows = []
        flows_all = []

        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype,
                           device=init_device).float()
        flow_all = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype,
                           device=init_device).float()
        sum_group = [torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype,
                           device=init_device).float() for it in range(mask_layer + 1)]

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            if l >= 1:
                flow = F.interpolate(flow * 2, scale_factor=2,
                                     mode='bilinear', align_corners=True)
                flow_all = F.interpolate(flow_all * 2, scale_factor=2,
                                     mode='bilinear', align_corners=True)
                sum_group = [F.interpolate(it * 2, scale_factor=2,
                                     mode='bilinear', align_corners=True) for it in sum_group]

            if x2.shape[-2:] != sum_group[0].shape[-2:]: 
                assert 1==0
                _, _, _h, _w = x2.shape
                sum_flow_bg = sum_flow_bg[:, :, :_h, :_w]
                sum_flow_fg = sum_flow_fg[:, :, :_h, :_w]

            # warping
            if l == 0:
                x2_warp = x2
            else:
                x2_warp = flow_warp(x2, flow)

            # correlation
            out_corr = self.corr(x1, x2_warp)
            out_corr_relu = self.leakyRELU(out_corr)

            # concat and estimate flow
            x1_1by1 = self.conv_1x1[l](x1)
            mask_resize = _resize(mask, sum_group[0].shape[-2:])
            
            input_tensor = torch.cat([out_corr_relu, x1_1by1, flow], dim=1)
            x5, flow_group, flow_all_res = self.flow_estimators(input_tensor, mask_resize)
            
            assert len(flow_group) == mask_layer + 1
            for _i_ in range(mask_layer + 1):
                sum_group[_i_] = sum_group[_i_] + flow_group[_i_]
            
            for _i_ in range(mask_layer):
                if _i_ == 0:
                    flow = mask_resize[:, _i_:_i_+1, : , :] * sum_group[_i_ + 1]
                else:
                    flow = flow + mask_resize[:, _i_:_i_+1, : , :] * sum_group[_i_ + 1]

            flow_all = flow_all + flow_all_res

            flows.append(flow)
            flows_all.append(flow_all)

            if l == self.output_level:
                break
        if self.upsample:
            flows = [F.interpolate(flow * 4, scale_factor=4, mode='bilinear', align_corners=True) for flow in flows]
            flows_all = [F.interpolate(flow * 4, scale_factor=4, mode='bilinear', align_corners=True) for flow in flows_all]
            sum_group = [F.interpolate(it * 4, scale_factor=4, mode='bilinear', align_corners=True) for it in sum_group]
        return flows[::-1], flows_all[::-1], sum_group


    def forward(self, x, mask, with_bk=True):
        n_frames = x.size(1) / 3

        imgs = [x[:, 3 * i: 3 * i + 3] for i in range(int(n_frames))]

        x = [self.feature_pyramid_extractor(img) + [img] for img in imgs]

        res_dict = {}
        
        if n_frames == 2:
            res_dict['flows_fw'], res_dict['flows_fw_all'], res_dict['flows_fw_group'] = self.forward_2_frames(x[0], x[1], mask[1])
            if with_bk:
                res_dict['flows_bw'], res_dict['flows_bw_all'], res_dict['flows_bw_group'] = self.forward_2_frames(x[1], x[0], mask[0])
        else:
            raise NotImplementedError
        return res_dict

