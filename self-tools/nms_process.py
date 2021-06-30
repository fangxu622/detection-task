
from unicodedata import category
import torch
import time
import torchvision
import json
from mmcv.ops.nms import nms ,soft_nms
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import argparse

def xywh2xyxy(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]   # x center
    y[:, 1] = x[:, 1]    # y center
    y[:, 2] = x[:, 0] + x[:, 2]  # width
    y[:, 3] = x[:, 1] + x[:, 3] # height
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]   # x center
    y[:, 1] = x[:, 1]    # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1] # height
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        #box = xywh2xyxy(x[:, :4])
        box = x[:, :4]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def apply_nms(bbox_list, score_list, iou_threshold =0.5, score_thresh = 0.3):
    #inp 1 : bbox , (n,4) list
    #inp 2 : score , (n,) list
    #print(len(bbox_list))
    boxes = xywh2xyxy(torch.from_numpy(np.vstack(bbox_list)).float())
    scores = torch.tensor(score_list).float()
    bbox_res, _ = nms(boxes, scores, iou_threshold, offset=0)
    #bbox_res, _ = soft_nms(boxes, scores, iou_threshold, offset=0)

    return bbox_res


def nms2res(bbox_score, cat_id, image_id):
    # input1: res_nms (bbox, score), tensor [n,4] , [n,] 
    # input2: cat_id
    # input3: image_id 
    out = []
    bbox = xyxy2xywh( bbox_score[:,:4] )
    #bbox_score = torch.cat( [bbox, score.unsqueeze(1) ], dim = 1 )
    for indx in range( bbox_score.size(0) ):

        tem_dict = {}
        tem_dict["image_id"] = image_id
        tem_dict["bbox"] = bbox[indx,:].numpy().tolist()
        tem_dict["score"] = bbox_score[indx,4].numpy().tolist()
        tem_dict["category_id"] = cat_id

        out.append(tem_dict)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description='use nms ')
    parser.add_argument('input_json', help='train config file path')
    parser.add_argument('--save_dir', required=True, help='the dir to save logs and models')
    parser.add_argument(
        '--iou_thresh', type=float, help='iou_thresh')
    parser.add_argument(
        '--score_thresh', type=float, help='score')
    parser.add_argument(
        '--number', type=int, default= 2000, help='the checkpoint file to resume from')

    #### example
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    args = parser.parse_args()

    return args

## 
def main_nms():
    args = parse_args()
    json_path = args.input_json
    save_dir = args.save_dir
    total_number = args.number
    score_thresh = 
    #json_path = "/media/fangxu/Disk4T/fangxuPrj/PHD-det/self-tools/Res/fx_detr_ep50.bbox.json"
    #json_path = "/media/fangxu/Disk4T/fangxuPrj/PHD-det/self-tools/Res/fx_reteded_0.02ep91.bbox.json"
    #json_path = "/media/fangxu/Disk4T/fangxuPrj/PHD-det/self-tools/Res/fx_reteded_0.01ep91.bbox.json"
    # res_path = "/media/fangxu/Disk4T/fangxuPrj/PHD-det/self-tools/lizhen-ori-91-nms0.5.json"
    with open(json_path,"r") as f:
        load_dict = json.load(f)
        print("加载入文件完成...")
    category_id = [1, 2, 3, 4, 5, 6, 7, 8]
    
    print(len(load_dict))
    res_list = []
    for ii in tqdm(range(total_number)):

        img_id_list = []

        for bx in load_dict:  
            if bx["image_id"] ==ii:
                img_id_list.append(bx)
        
        for idx in category_id:
            res_bbox = []
            res_score = []
            for obj_x in img_id_list:

                if obj_x["category_id"] == idx:
                    res_bbox.append(obj_x["bbox"])
                    res_score.append(obj_x["score"])
            if len(res_bbox)==0:
                continue
            bbox_score = apply_nms(res_bbox, res_score)
            res_dict = nms2res( bbox_score, idx, ii)
            #print(res_dict)
            res_list.extend(res_dict)

    #print(load_dict[0]["image_id"])
    res_name = json_path.split("/")[-1].split(".")[0] + 
    json_file = open(res_path , 'w+')
    print(len(res_list))
    json.dump(res_list, json_file, indent=4)
    json_file.close()
    #print("x")

#usage
#

if __name__ == "__main__":
    main_nms()