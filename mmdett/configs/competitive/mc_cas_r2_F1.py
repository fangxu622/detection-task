
# _base_ = [
#     '../_base_/models/cascade_rcnn_r50_fpn.py',
#     #'../_base_/datasets/coco_detection.py',
#     #'../_base_/schedules/schedule_1x.py', 
#     #'../_base_/default_runtime.py'
# ]


norm_cfg = dict(type='SyncBN', requires_grad=True)

# custom_imports = dict(
#     imports=['mmdet.datasets.pipelines.self_transform'],
#     allow_failed_imports=False)

model = dict(
    type='CascadeRCNN',
    pretrained=None,#'open-mmlab://resnest101',
    backbone=dict(
        type='Res2Net', depth=101, scales=4, base_width=26,
        #type='ResNet',
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg= norm_cfg,
        norm_eval=False ,
        style='pytorch'),
        
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),

        bbox_head=[
            dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                norm_cfg=norm_cfg,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                norm_cfg=norm_cfg,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                norm_cfg=norm_cfg,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],),
        # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.003,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))

# custom_imports = dict(
#     imports=['mmdet.datasets.pipelines.self_transform'],
#     allow_failed_imports=False)

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/coco3/'
classes= ('Crack','Manhole','Net','Pothole','Patch-Crack','Patch-Net','Patch-Pothole',
        'Other')


# # use ResNeSt img_norm
img_norm_cfg = dict(
    mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True)



train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False, poly2mask=False),
    dict(type='FixCrop', crop_size=(960, 1600)),
    dict( type='Mosaic_test', size=(1184, 1600),  min_offset=0.2 ,probability = 0.7 ),
    dict(
        type='Resize',
        img_scale=[(1600, 960), (1600, 600)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='FixCrop', crop_size=(901, 1566)),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1600, 1184), (1600, 600)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            #dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]


val_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='FixCrop', crop_size=(900, 1600)),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1600, 1100), (1600, 600)],
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
    samples_per_gpu=5,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),

    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'val/',
        pipeline=val_pipeline),
    
    test=dict(
        #samples_per_gpu=2,
        type=dataset_type,
        classes=classes,
        ann_file='/data/coco2/' + 'test.json',
        #img_prefix=data_root + 'val/',
        #ann_file='/media/fangxu/SSD247/All-data/DeepSlamData/road_data/images/1-test-json/annotations3.json',
        img_prefix='/data/coco2/' + 'test_A/',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')#,save_best ="bbox_mAP")

# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,   
    warmup_ratio=0.001,
    step=[ 28 , 32])
runner = dict(type='EpochBasedRunner', max_epochs=38)

# runtime
checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from =None#"/data/weight/cascade_rcnn_r2_101_fpn_20e_coco-f4b7b7db.pth" #"/data/Res/cast_F1/epoch_30.pth" #"/data/weight/cast_ep39.pth" #"/media/fangxu/Disk4T/fangxuPrj/PHD-det/mmdetection/weight/cascade_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco_20201005_113242-b9459f8f.pth"
#load_from
resume_from ="/data/Res/mc_cas_r2_F1/epoch_18.pth"
workflow = [('train', 1)]

work_dir ="/data/Res/mc_cas_r2_F2"