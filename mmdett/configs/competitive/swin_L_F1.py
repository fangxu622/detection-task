_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    #'../_base_/datasets/coco_instance.py',
    #'../_base_/schedules/schedule_1x.py', '../_base_/default_runtime_det.py'
]

custom_imports = dict(
    imports=['swin.swin_transformer'], allow_failed_imports=False)
fp16 = dict(loss_scale=dict(init_scale=512))

model = dict(
    type='CascadeRCNN',
    pretrained=None,#'./pretrain/swin/swin_tiny_patch4_window7_224.pth',
    backbone=dict(
        _delete_=True,
        # SwinTransformer is registered in the MMCV MODELS registry
        type='mmcv.SwinTransformer',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    neck=dict(in_channels=[96, 192, 384, 768]),
    
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.55),
            max_per_img=100))
    )

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

## dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/coco3/'
classes= ('Crack','Manhole','Net','Pothole','Patch-Crack','Patch-Net','Patch-Pothole',
        'Other')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[{
            'type':
            'Resize',
            'img_scale': [(1184, 1600), (1000, 1600), (960, 1600), 
                          (900, 1600),  (860, 1600), (800, 1600),
                          (672, 1600),  (704, 1600), (736, 1600), 
                          (768, 1600),  (600, 1600)],
            'multiscale_mode':
            'value',
            'keep_ratio':
            True
        }],
                  [{
                      'type': 'Resize',
                      'img_scale': [(800, 4200), (900, 4200), (1000, 4200)],
                      'multiscale_mode': 'value',
                      'keep_ratio': True
                  }, {
                      'type': 'RandomCrop',
                      'crop_type': 'absolute_range',
                      'crop_size': (960, 1500),
                      'allow_negative_crop': True
                  }, {
                      'type':
                      'Resize',
                    'img_scale': [(1184, 1600), (1000, 1600), (960, 1600), 
                          (900, 1600),  (860, 1600), (800, 1600),
                          (672, 1600),  (704, 1600), (736, 1600), 
                          (768, 1600),  (600, 1600)],
                      'multiscale_mode':
                      'value',
                      'override':
                      True,
                      'keep_ratio':
                      True
                  }]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', ]),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1600, 1184),(1600, 600 )],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),

    val=dict(
        #samples_per_gpu=5,
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),    
    
    test=dict(
        #samples_per_gpu=5,
        type=dataset_type,
        classes=classes,
        ann_file= '/data/coco2/test.json',
        #img_prefix=data_root + 'val/',
        #ann_file='/media/fangxu/SSD247/All-data/DeepSlamData/road_data/images/1-test-json/annotations3.json',
        img_prefix='/data/coco2/test_A/',
        pipeline=test_pipeline))

optimizer = dict(
    #_delete_=True,
    type='AdamW',
    lr=0.0002,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='step', warmup='linear', warmup_iters=1000, warmup_ratio=0.001, step=[27, 33])

runner = dict(type='EpochBasedRunner',max_epochs=36)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
#custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from =None#"/data/weight/cascade_mask_rcnn_swin_small_patch4_window7.pth"
resume_from = None
workflow = [('train', 1)]

work_dir = '/data/Res/cas_swin_S_F1'