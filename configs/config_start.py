ssim_sz=1
step_limit=1

w_seg=1.0

batch_size = 2
load_num = 2
mask_layer = 5

flow_model_path = ''
load_flownet = False
freeze_flownet = False

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'amd_pretrained.pth'

resume_from = None

workflow = [('train', 1)]
cudnn_benchmark = True

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    w_seg = w_seg,
    mask_layer = mask_layer,
    backbone2=dict(
        dilations=(1, 1, 1, 2),
        strides=(1, 2, 1, 1),
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=False
        ),
    decode_head=dict(
        ssim_sz=ssim_sz,
        create_flownet=True,
        flow_model_path = flow_model_path,
        freeze_flownet = freeze_flownet, 
        load_flownet = load_flownet, 
        mask_layer = mask_layer,
        concat_input=False,
        dilation=6,
        channels=128,
        type='FCNHead',
        in_channels=2048,
        in_index=3,
        num_convs=8,
        dropout_ratio=0.1,
        num_classes=24,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    decode_head2=dict(
        concat_input=False,
        dilation=6,
        channels=256,
        type='FCNHead',
        in_channels=2048,
        in_index=3,
        num_convs=2,
        dropout_ratio=0.1,
        num_classes=mask_layer,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    )
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')

# dataset settings
dataset_type = 'PascalVOCDataset'
#data_root = ""
data_root = 'data_example/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
sz=400
crop_size = (384, 384)
train_pipeline = [
    dict(type='LoadImageFromFile', load_num=load_num, step_limit=step_limit),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(9999, sz), ratio_range=(0.96, 1.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'flow_x', 'flow_y']),
]

test_pipeline = [
    dict(type='LoadImageFromFile', is_train=False),
    dict(type='Resize', img_scale=(9999, sz), ratio_range=(0.98, 0.98)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]


#DATA

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='',
        split='example_train.txt',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='',
        split='example_inf.txt',
        pipeline=test_pipeline)
    )


optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0001 * 0.01)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
total_iters = 0
checkpoint_config = dict(by_epoch=False, interval=2000, create_symlink=False)
test_evaluation = dict(interval=990000, metric='mIoU', state='test')
find_unused_parameters = False

