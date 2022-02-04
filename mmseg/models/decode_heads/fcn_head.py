import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F
from mmseg.ops import resize

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .models.pwclite import PWCLite
from .losses.flow_loss import unFlowLoss
import math

def _resize(in_img, shape):
    output = resize(
        in_img,
        size=shape,
        mode='bilinear',
        align_corners=True,
        warning=False)
    return output


class Objectview(object):
    def __init__(self, d):
        self.__dict__ = d
    def keys(self):
        return self.__dict__.keys()


@HEADS.register_module()
class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 ssim_sz = 1,
                 load_flownet = False,
                 freeze_flownet=False,
                 flow_model_path = '', 
                 mask_layer=1,
                 create_flownet = False,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs > 0
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.mask_layer = mask_layer
        super(FCNHead, self).__init__(**kwargs)
        print("[info] ssim_sz={}".format(ssim_sz))
        if create_flownet:
            self.flownet = PWCLite(mask_layer)
            cfg =  {"alpha": 10,
                    "ssim_sz": ssim_sz,
                    "occ_from_back": True,
                    "type": "unflow",
                    "w_l1": 0.15,
                    "w_scales": [1.0, 1.0, 1.0, 1.0, 0.0],
                    "w_sm_scales": [1.0, 0.0, 0.0, 0.0, 0.0],
                    "w_smooth": 50.0,
                    "w_ssim": 0.85,
                    "w_ternary": 0.0,
                    "warp_pad": "border",
                    "with_bk": True}
            cfg = Objectview(cfg)
            self.loss_func = unFlowLoss(cfg)
            if load_flownet:
                print('[Flownet] Load weights ')
                pth = torch.load(flow_model_path)
                load_result = self.flownet.load_state_dict(pth['state_dict'], strict=False)
                print('Load weights done!', load_result)
            print('[freeze]', freeze_flownet) 
            if freeze_flownet:
                for param in self.flownet.parameters():
                    param.requires_grad = False
            return
        else:
            self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
            convs = []
            convs.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    dilation=dilation,
                    kernel_size=kernel_size,
                    padding=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            for i in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        self.channels,
                        self.channels,
                        dilation=dilation,
                        kernel_size=kernel_size,
                        padding=dilation,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.convs = nn.Sequential(*convs)
            if self.concat_input:
                self.conv_cat = ConvModule(
                    self.in_channels + self.channels,
                    self.channels,
                    dilation=dilation,
                    kernel_size=kernel_size,
                    padding=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
    
    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output
    
    def flow_forward(self, imgs, masks):
        flow_loss = {'seg':0.0, 'whole':0.0}
        #flows = []
        flows = {'seg':[], 'whole':[]}
        flows_bg = []; flows_fg = []
        groups = []
        batch_size, im_num, _, im_h, im_w = imgs.shape
        for i in range(1, im_num):
            im1 = imgs[:, i-1, :, :, :]
            im2 = imgs[:, i, :, :, :]
            _mean=[123.675, 116.28, 103.53]
            _std=[58.395, 57.12, 57.375]
            im1 = torch.cat([im1[:, 0:1] * _std[0] + _mean[0], im1[:, 1:2] * _std[1] + _mean[1], im1[:, 2:3] * _std[2] + _mean[2]], 1) / 255.0
            im2 = torch.cat([im2[:, 0:1] * _std[0] + _mean[0], im2[:, 1:2] * _std[1] + _mean[1], im2[:, 2:3] * _std[2] + _mean[2]], 1) / 255.0
            im1 = _resize(im1, (384, 640))
            im2 = _resize(im2, (384, 640))
            mask1 = masks[:, i-1, :, :, :]
            mask2 = masks[:, i, :, :, :]
            two_frame = torch.cat([im1, im2], 1)
            res_dict = self.flownet(two_frame, [mask1, mask2], with_bk=True)

            def get_loss(flows_12, flows_21):
                concat_flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in zip(flows_12, flows_21)]
                loss, l_ph, l_sm, flow_mean = self.loss_func(concat_flows, two_frame)
                return loss

            loss = get_loss(res_dict['flows_fw'], res_dict['flows_bw'])
            loss_all = get_loss(res_dict['flows_fw_all'], res_dict['flows_bw_all'])

            flow_loss['seg'] += loss
            flow_loss['whole'] += loss_all
            
            def get_norm_flow(lis1=res_dict['flows_fw'], lis2=res_dict['flows_bw']):
                flow = lis1[0]
                _, _, _h, _w = flow.shape
                flow = torch.cat([flow[:, 0:1] / (_h/2.0), flow[:, 1:2] / (_w/2.0)], 1 )
                flow2 = lis2[0]
                _, _, _h, _w = flow2.shape
                flow2 = torch.cat([flow2[:, 0:1] / (_h/2.0), flow2[:, 1:2] / (_w/2.0)], 1 )
                return _h, _w, flow, flow2
            
            _h, _w, flow, flow2 = get_norm_flow(lis1=res_dict['flows_fw'], lis2=res_dict['flows_bw'])
            flows['seg'].append(torch.cat([flow, flow2], 1))
            _, _, flow, flow2 = get_norm_flow(lis1=res_dict['flows_fw_all'], lis2=res_dict['flows_bw_all'])
            flows['whole'].append(torch.cat([flow, flow2], 1))
            
            
            flow_bg, flow_fg = -1, -1
            flows_bg.append(flow_bg) ; flows_fg.append(flow_fg)
            groups.append(res_dict['flows_fw_group'])
        return flows, flow_loss, flows_bg, flows_fg, groups 

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
