
dataset_type = 'CocoDataset'
data_root = '/data/coco2/'
classes= ('Crack','Manhole','Net','Pothole','Patch-Crack','Patch-Net','Patch-Pothole',
        'Other')

custom_imports = dict(
    imports=['mmdet.datasets.pipelines.self_transform'],
    allow_failed_imports=False)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='FixCrop', crop_size=(900, 1600)),
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
                      'crop_size': (800, 1300),
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
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='FixCrop', crop_size=(900, 1600)),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1600, 1000)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            #dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='FixCrop', crop_size=(900, 1600)),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1600, 1000)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
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
        pipeline=val_pipeline),    
    
    test=dict(
        #samples_per_gpu=5,
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        #img_prefix=data_root + 'val/',
        #ann_file='/media/fangxu/SSD247/All-data/DeepSlamData/road_data/images/1-test-json/annotations3.json',
        img_prefix=data_root + 'test_A/',
        pipeline=test_pipeline))


evaluation = dict(interval=1, metric='bbox')
checkpoint_config = dict(interval=1)

log_config = dict(interval=5, 
            hooks=[dict(type='TextLoggerHook'),
            dict(type='TensorboardLoggerHook')
            ])

custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "/data/weight/dedetr_ep91.pth"
resume_from = None
workflow = [('train', 1)]

model = dict(
    type='DeformableDETR',
    pretrained=None,#'torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg= dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='DeformableDETRHead',
        num_query=300,
        num_classes=8,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=True,
        transformer=dict(
            type='DeformableDetrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        with_box_refine=True),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100))

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))))

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

lr_config = dict(policy='step', step=[20,40, 52])

runner = dict(type='EpochBasedRunner', max_epochs=60)

work_dir = '/data/Res/dedetr_F1'

gpu_ids = range(0, 4)

