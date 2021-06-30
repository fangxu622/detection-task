import copy
import inspect

import mmcv
import numpy as np
from numpy import random

from mmdet.core import PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from numpy.core.arrayprint import _leading_trailing
from ..builder import PIPELINES
from .transforms import RandomCrop
from mmdet.datasets import DATASETS
from mmdet.datasets.builder import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

def remove_min_bbox(result_x, min_area = 200, length_ratio = 0.005 ):
    result_a =result_x.copy()
    a1 = result_a['gt_bboxes'] 
    area_params  = (a1[:,2] - a1[:,0]) * (a1[:,3] - a1[:,1])
    hw_ratio_1 =  (a1[:,2] - a1[:,0]) / (a1[:,3] - a1[:,1])
    #hw_ratio_2 =  (a1[:,3] - a1[:,1]) / (a1[:,2] - a1[:,0]) 
    tm1_bool = area_params > min_area
    tm2_bool = (hw_ratio_1 < 1/length_ratio) & (hw_ratio_1 >= length_ratio)
    #print(tm1_bool)
    tm3_bool = tm1_bool & tm2_bool
    #print(tm2_bool)
    #print(tm3_bool)
    result_a['gt_bboxes'] = result_a['gt_bboxes'][tm3_bool]
    result_a['gt_labels'] = result_a['gt_labels'][tm3_bool]
    return result_a

@PIPELINES.register_module()
class FixCrop(object):
    """Random crop the image & bboxes & masks.

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        crop_type (str, optional): one of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])]. Default "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 bbox_clip_border=True):
        if crop_type not in [
                'relative_range', 'relative', 'absolute', 'absolute_range'
        ]:
            raise ValueError(f'Invalid crop_type {crop_type}.')
        if crop_type in ['absolute', 'absolute_range']:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(
                crop_size[1], int)
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)

            # offset_h = np.random.randint(0, margin_h + 1)
            # offset_w = np.random.randint(0, margin_w + 1)
            offset_h = margin_h 
            offset_w = margin_w 

            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results

    def _get_crop_size(self, image_size):
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.

        Args:
            image_size (tuple): (h, w).

        Returns:
            crop_size (tuple): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == 'absolute':
            #return (min(self.crop_size[0], h), min(self.crop_size[1], w))
            return ( self.crop_size[0] , self.crop_size[1] )

        elif self.crop_type == 'absolute_range':
            assert self.crop_size[0] <= self.crop_size[1]
            crop_h = np.random.randint(
                min(h, self.crop_size[0]),
                min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(
                min(w, self.crop_size[0]),
                min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w
        elif self.crop_type == 'relative':
            crop_h, crop_w = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        elif self.crop_type == 'relative_range':
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        image_size = results['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str

@PIPELINES.register_module()
class Mosaic(object):
    """Mosaic augmentation.

    Given 4 images, Mosaic augmentation randomly crop a patch on each image
    and combine them into one output image. The output image is composed of
    the parts from each sub-image.

                        output image
                                cut_x
               +-----------------------------+
               |     image 0      | image 1  |
               |                  |          |
        cut_y  |------------------+----------|
               |     image 2      | image 3  |
               |                  |          |
               |                  |          |
               |                  |          |
               |                  |          |
               |                  |          |
               +-----------------------------+

    Args:
        size (tuple[int]): output image size in (h,w).
        min_offset (float | tuple[float]): Volume of the offset
            of the cropping window. If float, both height and width are
        dataset (torch.nn.Dataset): Dataset with augmentation pipeline.
    """

    def __init__(self, size=(640, 640), min_offset=0.2, dataset=None):

        assert isinstance(size, tuple)
        assert isinstance(size[0], int) and isinstance(size[1], int)
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError('image size must > 0 in train mode')

        if isinstance(min_offset, float):
            assert 0 <= min_offset <= 1
            self.min_offset = (min_offset, min_offset)
        elif isinstance(min_offset, tuple):
            assert isinstance(min_offset[0], float) \
                   and isinstance(min_offset[1], float)
            assert 0 <= min_offset[0] <= 1 and 0 <= min_offset[1] <= 1
            self.min_offset = min_offset
        else:
            raise TypeError('Unsupported type for min_offset, '
                            'should be either float or tuple[float]')

        self.size = size
        img_norm_cfg = dict(
                mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True)
        dataset_class = DATASETS.get("CocoDataset")
        mc_dataset = dataset_class(
            ann_file='/data/coco3/train.json',
            pipeline= [ dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=False, poly2mask=False),
                #dict(type='FixCrop', crop_size=(900, 1600)),
                dict(type='Resize',
                    img_scale=[(1600, 1000)],
                    multiscale_mode='range',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.0),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=1),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels','ann_info']),    
                                    ],
            classes=('Crack','Manhole','Net','Pothole','Patch-Crack','Patch-Net','Patch-Pothole', 'Other'),
            test_mode=False,
            img_prefix= '/data/coco3/train/' )
        
        self.dataset =  mc_dataset

        self.cropper = RandomCrop(crop_size=size, allow_negative_crop=True)
        self.num_sample = len(mc_dataset)

    def __call__(self, results):
        """Call the function to mix 4 images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images and bounding boxes cropped.
        """
        # Generate the Mosaic coordinate
        cut_y = random.randint(
            int(self.size[0] * self.min_offset[0]),
            int(self.size[0] * (1 - self.min_offset[0])))
        cut_x = random.randint(
            int(self.size[1] * self.min_offset[1]),
            int(self.size[1] * (1 - self.min_offset[1])))

        cut_position = (cut_y, cut_x)
        tmp_result = copy.deepcopy(results)
        # create the image buffer and mask buffer
        tmp_result['img'] = np.zeros(
            (self.size[0], self.size[1], *tmp_result['img'].shape[2:]),
            dtype=tmp_result['img'].dtype)
        for key in tmp_result.get('seg_fields', []):
            tmp_result[key] = np.zeros(
                (self.size[0], self.size[1], *tmp_result[key].shape[2:]),
                dtype=tmp_result[key].dtype)
        tmp_result['img_shape'] = self.size

        out_bboxes = []
        out_labels = []
        out_ignores = []

        for loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right'):
            if loc == 'top_left':
                # use the current image
                results_i = copy.deepcopy(results)
            else:
                # randomly sample a new image from the dataset
                index = random.randint(self.num_sample)
                results_i = copy.deepcopy(self.dataset.__getitem__(index))
                #results_i['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']

            # compute the crop parameters
            crop_size, img_slices, paste_position = self._mosiac_combine(
                loc, cut_position)

            # randomly crop the image and segmentation mask
            self.cropper.crop_size = crop_size
            results_i = self.cropper(results_i)
            # paste to the buffer image
            
            tmp_result['img'][img_slices] = results_i['img'].copy()
            for key in tmp_result.get('seg_fields', []):
                tmp_result[key][img_slices] = results_i[key].copy()

            results_i = self._adjust_coordinate(results_i, paste_position)

            out_bboxes.append(results_i['gt_bboxes'])
            out_labels.append(results_i['gt_labels'])
            out_ignores.append(results_i['gt_bboxes_ignore'])

        out_bboxes = np.concatenate(out_bboxes, axis=0)
        out_labels = np.concatenate(out_labels, axis=0)
        out_ignores = np.concatenate(out_ignores, axis=0)

        tmp_result['gt_bboxes'] = out_bboxes
        tmp_result['gt_labels'] = out_labels
        tmp_result['gt_bboxes_ignore'] = out_ignores

        return tmp_result

    def _mosiac_combine(self, loc, cut_position):
        """Crop the subimage, change the label and mix the image.

        Args:
            loc (str): Index for the subimage, loc in ('top_left',
                'top_right', 'bottom_left', 'bottom_right').
            results (dict): Result dict from loading pipeline.
            img (numpy array): buffer for mosiac image, (H x W x 3).
            cut_position (tuple[int]): mixing center for 4 images, (y, x).

        Returns:
            bboxes: Result dict with images and bounding boxes cropped.
        """
        if loc == 'top_left':
            # Image 0: top left
            crop_size = cut_position
            img_slices = (slice(0, cut_position[0]), slice(0, cut_position[1]))
            paste_position = (0, 0)
        elif loc == 'top_right':
            # Image 1: top right
            crop_size = (cut_position[0], self.size[1] - cut_position[1])
            img_slices = (slice(0, cut_position[0]),
                          slice(cut_position[1], self.size[1]))
            paste_position = (0, cut_position[1])
        elif loc == 'bottom_left':
            # Image 2: bottom left
            crop_size = (self.size[0] - cut_position[0], cut_position[1])
            img_slices = (slice(cut_position[0],
                                self.size[0]), slice(0, cut_position[1]))
            paste_position = (cut_position[0], 0)
        elif loc == 'bottom_right':
            # Image 3: bottom right
            crop_size = (self.size[0] - cut_position[0],
                         self.size[1] - cut_position[1])
            img_slices = (slice(cut_position[0], self.size[0]),
                          slice(cut_position[1], self.size[1]))
            paste_position = cut_position

        return crop_size, img_slices, paste_position

    def _adjust_coordinate(self, results, paste_position):
        """Convert subimage coordinate to mosaic image coordinate.

         Args:
            results (dict): Result dict from :obj:`dataset`.
            paste_position (tuple[int]): paste up-left corner
                coordinate (y, x) in mosaic image.

        Returns:
            results (dict): Result dict with corrected bbox
                and mask coordinate.
        """

        for key in results.get('bbox_fields', []):
            box = results[key]
            box[:, 0::2] += paste_position[1]
            box[:, 1::3] += paste_position[0]
            results[key] = box
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'size={self.size}, '
        repr_str += f'min_offset={self.min_offset})'
        return repr_str

@PIPELINES.register_module()
class Mosaic_test(object):
    """Mosaic augmentation.
    Given 4 images, Mosaic augmentation randomly crop a patch on each image
    and combine them into one output image. The output image is composed of
    the parts from each sub-image.
                        output image
                                cut_x
               +-----------------------------+
               |     image 0      | image 1  |
               |                  |          |
        cut_y  |------------------+----------|
               |     image 2      | image 3  |
               |                  |          |
               |                  |          |
               |                  |          |
               |                  |          |
               |                  |          |
               +-----------------------------+
    Args:
        size (tuple[int]): output image size in (h,w).
        min_offset (float | tuple[float]): Volume of the offset
            of the cropping window. If float, both height and width are
        dataset (torch.nn.Dataset): Dataset with augmentation pipeline.
    """

    def __init__(self, size=(640, 640), min_offset=0.2, min_area = 3000, length_ratio=0.01, probability = 0.6, dataset=None ,area_rm = True ):

        assert isinstance(size, tuple)
        assert isinstance(size[0], int) and isinstance(size[1], int)
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError('image size must > 0 in train mode')

        if isinstance(min_offset, float):
            assert 0 <= min_offset <= 1
            self.min_offset = (min_offset, min_offset)
        elif isinstance(min_offset, tuple):
            assert isinstance(min_offset[0], float) \
                   and isinstance(min_offset[1], float)
            assert 0 <= min_offset[0] <= 1 and 0 <= min_offset[1] <= 1
            self.min_offset = min_offset
        else:
            raise TypeError('Unsupported type for min_offset, '
                            'should be either float or tuple[float]')

        self.size = size
        #self.dataset = dataset
        self.cropper = RandomCrop(crop_size=size, allow_negative_crop=False,bbox_clip_border= True)

        self.min_area = min_area
        self.length_ratio = length_ratio
        self.area_rm = area_rm
        self.probability =probability
        #img_norm_cfg = dict(
        #        mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True)
        #class_name =('Crack','Manhole','Net','Pothole','Patch-Crack','Patch-Net','Patch-Pothole', 'Other')
        dataset_class = DATASETS.get("CocoDataset")
        self.dataset = dataset_class(
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
        self.num_sample = len(self.dataset)

    def __call__(self, results):
        """Call the function to mix 4 images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images and bounding boxes cropped.
        """
        if np.random.rand() > self.probability:
            return results

        # Generate the Mosaic coordinate
        cut_y = random.randint(
            int(self.size[0] * self.min_offset[0]),
            int(self.size[0] * (1 - self.min_offset[0])))
        cut_x = random.randint(
            int(self.size[1] * self.min_offset[1]),
            int(self.size[1] * (1 - self.min_offset[1])))

        cut_position =  (cut_y, cut_x)#(478, 785)#
        tmp_result = copy.deepcopy(results)
        # create the image buffer and mask buffer
        tmp_result['img'] = np.zeros(
            (self.size[0], self.size[1], *tmp_result['img'].shape[2:]),
            dtype=tmp_result['img'].dtype)
        for key in tmp_result.get('seg_fields', []):
            tmp_result[key] = np.zeros(
                (self.size[0], self.size[1], *tmp_result[key].shape[2:]),
                dtype=tmp_result[key].dtype)
        tmp_result['img_shape'] = self.size
        self.i=0
        out_bboxes = []
        out_labels = []
        out_ignores = []

        for loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right'):
            #print(k,"******************")
            #k=k+1
            if loc == 'top_left':
                # use the current image
                results_i = copy.deepcopy(results)
            else:
                # randomly sample a new image from the dataset
                index = random.randint(self.num_sample)
                #print("index",index)# 1470
                results_i = copy.deepcopy(self.dataset.__getitem__(index))
                #print(results_i['img_info']['file_name'])
            # compute the crop parameters
            #print(results_i['img_info']['file_name'])
            crop_size, img_slices, paste_position = self._mosiac_combine(
                loc, cut_position)

            # randomly crop the image and segmentation mask
            self.cropper.crop_size = crop_size

            #mmcv.imshow_bboxes(results_i['img'], results_i['gt_bboxes'], show=False, out_file='before_crop_'+str(k)+'.png')
            ## make sure crop with label
            #results_i = self.cropper(results_i)
            for x in range(15):
                results_tmp = results_i.copy()
                results_tmp = self.cropper(results_tmp)
                if results_tmp != None:
                    results_i = results_tmp#.copy()
                    break
            if results_tmp == None:
                cropper_maybe_no_label = RandomCrop(crop_size=crop_size, allow_negative_crop=True)
                results_tmp = cropper_maybe_no_label(results_i)
                results_i = results_tmp
            
            if self.area_rm:
                results_i = remove_min_bbox(results_i, min_area=self.min_area , length_ratio= self.length_ratio)

            tmp_result['img'][img_slices] = results_i['img'].copy()
            for key in tmp_result.get('seg_fields', []):
                tmp_result[key][img_slices] = results_i[key].copy()

            results_i = self._adjust_coordinate(results_i, paste_position)

            out_bboxes.append(results_i['gt_bboxes'])
            out_labels.append(results_i['gt_labels'])
            out_ignores.append(results_i['gt_bboxes_ignore'])

        out_bboxes = np.concatenate(out_bboxes, axis=0)
        out_labels = np.concatenate(out_labels, axis=0)
        out_ignores = np.concatenate(out_ignores, axis=0)

        tmp_result['gt_bboxes'] = out_bboxes
        tmp_result['gt_labels'] = out_labels
        tmp_result['gt_bboxes_ignore'] = out_ignores

        return tmp_result

    def _mosiac_combine(self, loc, cut_position):
        """Crop the subimage, change the label and mix the image.
        Args:
            loc (str): Index for the subimage, loc in ('top_left',
                'top_right', 'bottom_left', 'bottom_right').
            results (dict): Result dict from loading pipeline.
            img (numpy array): buffer for mosiac image, (H x W x 3).
            cut_position (tuple[int]): mixing center for 4 images, (y, x).
        Returns:
            bboxes: Result dict with images and bounding boxes cropped.
        """
        if loc == 'top_left':
            # Image 0: top left
            crop_size = cut_position
            img_slices = (slice(0, cut_position[0]), slice(0, cut_position[1]))
            paste_position = (0, 0)
        elif loc == 'top_right':
            # Image 1: top right
            crop_size = (cut_position[0], self.size[1] - cut_position[1])
            img_slices = (slice(0, cut_position[0]),
                          slice(cut_position[1], self.size[1]))
            paste_position = (0, cut_position[1])
        elif loc == 'bottom_left':
            # Image 2: bottom left
            crop_size = (self.size[0] - cut_position[0], cut_position[1])
            img_slices = (slice(cut_position[0],
                                self.size[0]), slice(0, cut_position[1]))
            paste_position = (cut_position[0], 0)
        elif loc == 'bottom_right':
            # Image 3: bottom right
            crop_size = (self.size[0] - cut_position[0],
                         self.size[1] - cut_position[1])
            img_slices = (slice(cut_position[0], self.size[0]),
                          slice(cut_position[1], self.size[1]))
            paste_position = cut_position

        return crop_size, img_slices, paste_position

    def _adjust_coordinate(self, results, paste_position):
        """Convert subimage coordinate to mosaic image coordinate.
         Args:
            results (dict): Result dict from :obj:`dataset`.
            paste_position (tuple[int]): paste up-left corner
                coordinate (y, x) in mosaic image.
        Returns:
            results (dict): Result dict with corrected bbox
                and mask coordinate.
        """
        for key in results.get('bbox_fields', []):
            box = results[key]
            #print(key,box)
            #mmcv.imshow_bboxes(results['img'], box, show=False, out_file=str(self.i)+'.png')
            box[:, 0::2] += paste_position[1]
            box[:, 1::2] += paste_position[0]
            results[key] = box
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'size={self.size}, '
        repr_str += f'min_offset={self.min_offset})'
        return repr_str

