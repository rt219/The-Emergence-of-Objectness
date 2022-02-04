import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random
import os
import scipy.ndimage as ndimage

import numpy as np
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
import flow_vis


@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone2,
                 decode_head,
                 decode_head2,
                 w_seg = 2.0,
                 mask_layer = 1,
                 train_iter = 0,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoder, self).__init__()
        self.backbone2 = builder.build_backbone(backbone2)
        self.mask_layer = mask_layer
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.decode_head = self._init_decode_head(decode_head)
        self.decode_head2 = self._init_decode_head(decode_head2)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)
        
        assert self.with_decode_head

        self.train_iter = train_iter
        self.train_iter_per_log = 50

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.L1loss = nn.MSELoss()
        self.L1 = nn.L1Loss()

        tgt_dir = os.environ['PT_OUTPUT_DIR']
        os.system('mkdir {}/train'.format(tgt_dir))
        os.system('mkdir {}/train_good'.format(tgt_dir))

        self.has_dir = set([])
        
        self.w_seg = w_seg


    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        decode_head = builder.build_head(decode_head)
        self.align_corners = decode_head.align_corners
        self.num_classes = decode_head.num_classes
        return decode_head

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained)
        self.backbone2.init_weights(pretrained=None)
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()


    def extract_feat(self, img, net):
        """Extract features from images."""
        x = net(img)
        if self.with_neck:
            raise NotImplemented
        return x

    def _decode_head_forward_with_img(self, x, img, decode_head, mask):
        """Run forward function and calculate loss for decode head in
        training."""
        pred_flows, loss_flow, flow_bg, flow_fg, flow_group = decode_head.flow_forward(img, mask)
        return -1, pred_flows, loss_flow, flow_bg, flow_fg, flow_group

    def _decode_head_forward(self, x, decode_head):
        """Run forward function and calculate loss for decode head in
        training."""
        pred = decode_head.forward(x)
        return pred

    
    def numpy_to_tensor(self, flowrgb):
        output= torch.from_numpy(flowrgb).cuda()
        output = torch.unsqueeze(output,0).transpose(3, 2).transpose(2,1)
        return output

    
    def resize(self, in_img, shape):
        output = resize(
            in_img,
            size=shape,
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)
        return output
    

    def let_tensor_vis(self, t, setmax=False):
        t = t.clone()
        if setmax:
            _th_ = 0.2
            t[:,0, 0, 0] = _th_
            t[:,1, 0, 0] = _th_
            assert torch.max(t[:,0, :, :]) == _th_
            assert torch.max(t[:,1, :, :]) == _th_
        t = torch.squeeze(t)
        t = t.transpose(0,1).transpose(1,2).data.cpu().numpy()
        flow_color = flow_vis.flow_to_color(t, convert_to_bgr=False)
        flow_color = self.numpy_to_tensor(flow_color)
        return flow_color


    def forward_for_train_test(self, img, img_metas, is_train, state=-1):
        tgt_dir = os.environ['PT_OUTPUT_DIR']
        
        if is_train:
            losses = dict()
            log_interval = 60
        else:
            losses = dict()
            log_interval = 1
            _batch, im_num, _channel, _h, _w = img.shape
            img_3 = img.view(_batch * im_num, _channel, _h, _w)
            all_feat = self.extract_feat(img_3, self.backbone2)
            all_pred_mask = self._decode_head_forward(all_feat, self.decode_head2)
            _, _, _feat_h, _feat_w = all_pred_mask.shape
            all_pred_mask = all_pred_mask.view(_batch, im_num, self.mask_layer, _feat_h, _feat_w)
            all_pred_mask = F.softmax(all_pred_mask, dim = 2)
            
            ith_img = img[:, 0, :, :, :]
            ith_img = ith_img.detach()
            toH, toW = all_pred_mask.shape[-2:]
            ith_img_resize = self.resize(ith_img, (toH * 2, toW * 2))
            ith_img_resize = (ith_img_resize + 2.0) / 4.0
            
            pred_masks = all_pred_mask[:, 0, 1:2, :, :]
            pred_vis_list = []
            
            for _i_ in range(min(self.mask_layer,pred_masks.shape[1])):
                _pred_mask_resize = self.resize(pred_masks[:, _i_:_i_+1, :, :], (toH * 2, toW * 2)).repeat(1,3,1,1)
                pred_vis_list.append(_pred_mask_resize)
            tosave = torch.cat([ith_img_resize] + pred_vis_list, 2)
            
            try:
                prefix, _fnlist = img_metas[0]['filename']
                prefix = prefix.split('/')[-2]
                _fnlist = _fnlist[0][:-4]
                input_fn = prefix + '_' + _fnlist
                dir_path = '{}/eval_{}_{}'.format(tgt_dir, state, self.train_iter)
                if not dir_path in self.has_dir:
                    os.system('mkdir {}'.format(dir_path))
                    self.has_dir.add(dir_path)
                fn_name = '{}/{}_iter{:07d}.jpg'.format(dir_path, input_fn, self.train_iter)
                torchvision.utils.save_image(tosave, fn_name)
            except:
                print("[error in save]", fn_name)
            
            losses = {}
            losses['pred'] = pred_masks
            return losses

        _batch, im_num, _channel, _h, _w = img.shape

        img_3 = img.view(_batch * im_num, _channel, _h, _w)
        all_feat = self.extract_feat(img_3, self.backbone2)
        all_pred_mask = self._decode_head_forward(all_feat, self.decode_head2)
        _, _, _feat_h, _feat_w = all_pred_mask.shape
        all_pred_mask = all_pred_mask.view(_batch, im_num, self.mask_layer, _feat_h, _feat_w)
        all_pred_mask = F.softmax(all_pred_mask, dim = 2)
        
        _, pred_flows, loss_flow, flow_bg, flow_fg, flow_group = self._decode_head_forward_with_img(-1, img, self.decode_head, all_pred_mask)
        assert len(pred_flows['seg']) == 1 and len(pred_flows['whole']) == 1
        pred_flows_seg   = [self.resize(it, all_pred_mask.shape[-2:]) for it in pred_flows['seg']][0]
        pred_flows_whole = [self.resize(it, all_pred_mask.shape[-2:]) for it in pred_flows['whole']][0]


        
        if self.train_iter % log_interval == 0:
            ith_imgs=[]
            ith_flowxy_ori = []
            ith_flowxy_res = []
            
            mask_mat   = [[all_pred_mask[:, it, iit:iit+1, :, :].repeat(1,3,1,1) for it in range(im_num)] for iit in range(self.mask_layer)]
            pic_mask_mat = [torch.cat(it, 0) for it in mask_mat]

        loss_input_warp_seg = loss_flow['seg']


        line_mean_loss = 0.0
        for _i in range(im_num):
            ith_img = img[:, _i, :, :, :]
            ith_img = ith_img.detach()
            pred_mask = all_pred_mask[:, _i, :, :, :]
            
            if _i == 0:
                cur_flow_seg = pred_flows_seg[:, :2, :, :]
                cur_flow_whole = pred_flows_whole[:, :2, :, :]
            else:
                cur_flow_seg = pred_flows_seg[:, 2:, :, :]
                cur_flow_whole = pred_flows_whole[:, 2:, :, :]
                assert _i == 1
                
            if self.train_iter % log_interval == 0:
                for __i in range(_batch):
                    flow_color_3 = self.let_tensor_vis(cur_flow_seg[__i:__i+1])
                    flow_color_3_whole = self.let_tensor_vis(cur_flow_whole[__i:__i+1])
                    flow_color_3 = torch.cat([flow_color_3, flow_color_3_whole], 2)
                    ith_flowxy_ori.append(flow_color_3 / 255.0)
                ith_img_resize = self.resize(ith_img, pred_mask.shape[-2:])
                ith_imgs.append((ith_img_resize+2)/4.0)


        if self.train_iter % log_interval == 0:
            video_fn = img_metas[0]['filename'][0].split('/')[-1].split('_')[-1]
            pic_ith_imgs = torch.cat(ith_imgs, 0)
            pic_ith_flowxy_ori = torch.cat(ith_flowxy_ori, 0)
            
            tosave = torch.cat(pic_mask_mat + [pic_ith_imgs, pic_ith_flowxy_ori], 2)
            
            try:
                if is_train:
                    fn_name = '{}/train_good/{}_img_pred_recons_iter{:07d}.jpg'.format(tgt_dir, video_fn, self.train_iter)
                else:
                    input_fn = img_metas[0]['filename'][0].replace('./','').replace('/', '_')
                    if not self.train_iter in self.has_dir:
                        os.system('mkdir {}/eval_{}'.format(tgt_dir, self.train_iter))
                        self.has_dir.add(self.train_iter)
                    fn_name = '{}/eval_{}/{}_iter{:07d}.jpg'.format(tgt_dir, self.train_iter, input_fn, self.train_iter)
                torchvision.utils.save_image(tosave, fn_name)
            except:
                print("[error in save]", fn_name)

        if is_train:
            losses['loss_input_warp_seg']   = loss_input_warp_seg * self.w_seg
            return losses
        else:
            raise NotImplemented
            return losses

    def forward_train(self, img, img_metas, flow_x, flow_y):
        losses = self.forward_for_train_test(img, img_metas, is_train=True)
        self.train_iter += 1
        return losses


    def simple_test(self, img, img_meta, rescale=True, state='val'):
        seg_logit = self.forward_for_train_test(img, img_meta, is_train=False, state=state)
        return seg_logit
