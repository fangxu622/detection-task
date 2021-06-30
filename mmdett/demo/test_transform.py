import copy
import mmcv
import numpy as np
import os.path as osp
import pytest
import torch
from mmcv.utils import build_from_cfg
from unittest.mock import MagicMock, patch

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets import DATASETS
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.self_transform import Mosaic,Mosaic_test

@patch('mmdet.datasets.CustomDataset.load_annotations', MagicMock)
@patch('mmdet.datasets.CustomDataset._filter_imgs', MagicMock)
@patch('mmdet.datasets.CustomDataset.__len__', MagicMock(return_value=4))
def test_mosaic():
    results = dict()
    #img = mmcv.imread(
    #    osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    img = mmcv.imread("/workspace/mmdett/tests/data/color.jpg",'color')
    results['img'] = img

    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # TODO: add img_fields test
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    def create_random_bboxes(num_bboxes, img_w, img_h):
        bboxes_left_top = np.random.uniform(0, 0.5, size=(num_bboxes, 2))
        bboxes_right_bottom = np.random.uniform(0.5, 1, size=(num_bboxes, 2))
        bboxes = np.concatenate((bboxes_left_top, bboxes_right_bottom), 1)
        bboxes = (bboxes * np.array([img_w, img_h, img_w, img_h])).astype(
            np.int)
        return bboxes

    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_bboxes'] = gt_bboxes

    results['gt_labels'] = np.zeros(8, dtype=np.int64)
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    dataset = 'CustomDataset'
    dataset_class = DATASETS.get(dataset)
    img_norm_cfg = dict(
                mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True)
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[dict(type='Normalize', **img_norm_cfg),],
        classes=None,
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')
    custom_dataset.__getitem__ = MagicMock(return_value=results)
    
    print(results['img'].shape)
    print(custom_dataset.__getitem__())
    transform = dict(
        type='Mosaic_test', size=(320, 320), dataset=custom_dataset, min_offset=0.2)

    mosaic_module = build_from_cfg(transform, PIPELINES)

    results = mosaic_module(results)

    img = results['img']
    bboxes = results['gt_bboxes']
    #print(bboxes)
    #print(img.shape)
    # mmcv.imshow_bboxes(img, bboxes, show=True, out_file='img.png')
    # mmcv.imshow_bboxes(img, gt_bboxes, show=True)
    mmcv.imwrite(img, "img2.png")

def test2_mosaic(i):
    img_norm_cfg = dict(
                mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True)
    class_name =('Crack','Manhole','Net','Pothole','Patch-Crack','Patch-Net','Patch-Pothole', 'Other')
    dataset_class = DATASETS.get("CocoDataset")
    mc_dataset = dataset_class(
        ann_file='/data/coco3/train.json',
        pipeline=[ dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=False, poly2mask=False),
                dict(type='FixCrop', crop_size=(900, 1600)),
                # dict(type='Resize',
                #     img_scale=[(1600, 1000)],
                #     multiscale_mode='range',
                #     keep_ratio=True),
                #dict(type='RandomFlip', flip_ratio=0.0),
                #dict(type='Normalize', **img_norm_cfg),
                #dict(type='Pad', size_divisor=1),
                #dict(type='Collect', keys=['img','gt_bboxes', 'gt_bboxes_ignore', 'gt_labels' ]),    
                                    ],
            classes=('Crack','Manhole','Net','Pothole','Patch-Crack','Patch-Net','Patch-Pothole', 'Other'),
            test_mode=False,
            img_prefix= '/data/coco3/train/' )

    #print(results['img'].shape)
    results = mc_dataset.__getitem__(7)
    #print(results['img'].shape)
    #print(results['img_info']['file_name'])
    transform = dict(
        type='Mosaic_test', size=(1280, 1280), dataset=mc_dataset, min_offset=0.3)

    mosaic_module = build_from_cfg(transform, PIPELINES)
    results = mosaic_module(results)
    img = results['img']
    bboxes = results['gt_bboxes']
    labels = results['gt_labels']
    #print(bboxes)
    print(labels)
    #print(img.shape)
    #mmcv.imshow_bboxes(img, bboxes, show=False, out_file='img22.png')
    mmcv.imshow_det_bboxes(img, bboxes,labels,class_names=class_name, show=False,font_scale=1.5, out_file='img22.png')
    #mmcv.imshow_det_bboxes(img, bboxes,labels,class_names=class_name, show=False,font_scale=1.5, out_file='./images/'+str(x)+'.png')
    #mmcv.imshow_bboxes(img, gt_bboxes, show=True)
    #mmcv.imwrite(img, "img2.png")
def test3_mosaic(i):
    img_norm_cfg = dict(
                mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True)
    class_name =('Crack','Manhole','Net','Pothole','Patch-Crack','Patch-Net','Patch-Pothole', 'Other')
    dataset_class = DATASETS.get("CocoDataset")
    mc_dataset = dataset_class(
        ann_file='/data/coco3/train.json',
        pipeline=[ dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=False, poly2mask=False),
                dict(type='FixCrop', crop_size=(900, 1600)),
                # dict(type='Resize',
                #     img_scale=[(1600, 1000)],
                #     multiscale_mode='range',
                #     keep_ratio=True),
                #dict(type='RandomFlip', flip_ratio=0.0),
                #dict(type='Normalize', **img_norm_cfg),
                #dict(type='Pad', size_divisor=1),
                #dict(type='Collect', keys=['img','gt_bboxes', 'gt_bboxes_ignore', 'gt_labels' ]),    
                                    ],
            classes=('Crack','Manhole','Net','Pothole','Patch-Crack','Patch-Net','Patch-Pothole', 'Other'),
            test_mode=False,
            img_prefix= '/data/coco3/train/' )

    #print(results['img'].shape)
    results = mc_dataset.__getitem__(7)
    #print(results['img'].shape)
    #print(results['img_info']['file_name'])
    transform = dict(
        type='Mosaic_test', size=(1280, 1280), min_offset=0.3)

    mosaic_module = build_from_cfg(transform, PIPELINES)
    results = mosaic_module(results)
    img = results['img']
    bboxes = results['gt_bboxes']
    labels = results['gt_labels']
    #print(bboxes)
    print(labels)
    #print(img.shape)
    #mmcv.imshow_bboxes(img, bboxes, show=False, out_file='img22.png')
    mmcv.imshow_det_bboxes(img, bboxes,labels,class_names=class_name, show=False,font_scale=1.5, out_file='img22.png')
    #mmcv.imshow_det_bboxes(img, bboxes,labels,class_names=class_name, show=False,font_scale=1.5, out_file='./images/'+str(x)+'.png')
    #mmcv.imshow_bboxes(img, gt_bboxes, show=True)
    #mmcv.imwrite(img, "img2.png")

def test_resize():
    mg_norm_cfg = dict(
                mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True)
    class_name =('Crack','Manhole','Net','Pothole','Patch-Crack','Patch-Net','Patch-Pothole', 'Other')
    dataset_class = DATASETS.get("CocoDataset")
    mc_dataset = dataset_class(
        ann_file='/data/coco3/train.json',
        pipeline=[ dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=False, poly2mask=False),
                dict(type='FixCrop', crop_size=(960, 1600)),
                # dict(type='Resize',
                #     img_scale=[(1600, 1000)],
                #     multiscale_mode='range',
                #     keep_ratio=True),
                #dict(type='RandomFlip', flip_ratio=0.0),
                #dict(type='Normalize', **img_norm_cfg),
                #dict(type='Pad', size_divisor=1),
                #dict(type='Collect', keys=['img','gt_bboxes', 'gt_bboxes_ignore', 'gt_labels' ]),    
                                    ],
            classes=('Crack','Manhole','Net','Pothole','Patch-Crack','Patch-Net','Patch-Pothole', 'Other'),
            test_mode=False,
            img_prefix= '/data/coco3/train/' )

    #print(results['img'].shape)
    results = mc_dataset.__getitem__(7)
    # results = dict(
    #     img_prefix=osp.join(osp.dirname(__file__), '../../../data'),
    #     img_info=dict(filename='color.jpg'))

    #load = dict(type='LoadImageFromFile')
    #load = build_from_cfg(load, PIPELINES)
    transform =  dict(type='Resize',
                     img_scale=[(1600, 960),(1600, 660),(1600, 450)],
                     multiscale_mode='value',
                     keep_ratio=True)

    load= build_from_cfg(transform, PIPELINES)
    results = load(results)
    #results['scale'] = (1333, 800)
    #results['scale_factor'] = 1.0
    #results = transform(results)
    print(results['img'].shape)

def test2_random():
    img_norm_cfg = dict(
                mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True)

    dataset_class = DATASETS.get("CocoDataset")
    mc_dataset = dataset_class(
        ann_file='/data/coco3/train.json',
        pipeline=[ dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=False, poly2mask=False),
                dict(type='FixCrop', crop_size=(900, 1600)),
                    # test crop_type "absolute"
                dict(
                type='RandomCrop',
                crop_type='absolute',
                crop_size=(654, 724),
                allow_negative_crop=False)
                # dict(type='Resize',
                #     img_scale=[(1600, 1000)],
                #     multiscale_mode='range',
                #     keep_ratio=True),
                #dict(type='RandomFlip', flip_ratio=0.0),
                #dict(type='Normalize', **img_norm_cfg),
                #dict(type='Pad', size_divisor=1),
                #dict(type='Collect', keys=['img','gt_bboxes', 'gt_bboxes_ignore', 'gt_labels' ]),    
                                    ],
            classes=('Crack','Manhole','Net','Pothole','Patch-Crack','Patch-Net','Patch-Pothole', 'Other'),
            test_mode=False,
            img_prefix= '/data/coco3/train/' )

    #print(results['img'].shape)
    results = mc_dataset.__getitem__(1470)
    #print(results['img'].shape)
    #print(results)
    # transform = dict(
    #     type='Mosaic_test', size=(960, 960), dataset=mc_dataset, min_offset=0.3)

    # mosaic_module = build_from_cfg(transform, PIPELINES)
    # results = mosaic_module(results)
    img = results['img']
    bboxes = results['gt_bboxes']
    #print(results['img_info']['file_name'])
    #print(img.shape)
    mmcv.imshow_bboxes(img, bboxes, show=False, out_file='test_random.png')
    #mmcv.imshow_bboxes(img, gt_bboxes, show=True)
    #mmcv.imwrite(img, "img2.png")
# import time
# for x in range(100):
#     time.sleep(2)
#     test2_mosaic(x)
#test2_random()
#test_resize()
test3_mosaic(1)




