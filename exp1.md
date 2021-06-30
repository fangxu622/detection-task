
第一个策略：

coco1 : 

train_files_ids =img_ids[:4800] + img_ids[5500:6000]
val_files_ids = img_ids[4500:5500]

coco2:

train_files_ids =img_ids[100:6000]
val_files_ids = img_ids[0:600]


CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh configs/competitive/dedetr_F1.py 4


CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29510 ./tools/dist_train.sh configs/competitive/cast_F1.py 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29510 ./tools/dist_train.sh configs/detr competitive/crop_cast_F1.py 4
cast CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29511 ./tools/dist_train.sh configs/competitive/mc_cast_F1.py 4

infer phase

python tools/test.py configs/competitive/cast_B1.py /data/Res/cast_B1/epoch_20.pth  --format-only --options "jsonfile_prefix=/data/fangxu/cast_B1/cast_B1_ep20"

CUDA_VISIBLE_DEVICES=1 python tools/test.py configs/competitive/cast_F1.py /data/Res/cast_F1/epoch_26.pth  --format-only --options "jsonfile_prefix=/data/Res/cast_F1/cast_F1_ep26"

CUDA_VISIBLE_DEVICES=1 python tools/test.py configs/competitive/cast_F1.py /data/Res/cast_F1/epoch_24.pth  --format-only --options "jsonfile_prefix=/data/fangxu/cast_F1/cast_F1_ep24

CUDA_VISIBLE_DEVICES=2 PORT=29511 python tools/test.py configs/competitive/dedetr_B1.py /data/Res/dedetr_B1/epoch_58.pth  --format-only --options "jsonfile_prefix=/data/fangxu/dedetr_B1/dedetr_B1_ep58"

# 验证过拟合问题 
CUDA_VISIBLE_DEVICES=1 python tools/test.py configs/competitive/cast_F1.py /data/Res/cast_F1/epoch_7.pth  --format-only --options "jsonfile_prefix=/data/fangxu/cast_F1/cast_F1_ep7"

# 验证 NMS 参数问题 

CUDA_VISIBLE_DEVICES=1 python tools/test.py configs/competitive/cast_F1.py /data/Res/cast_F1/epoch_20.pth  --format-only --options "jsonfile_prefix=/data/fangxu/cast_F1/cast_F1_ep20_n0.5"

# crop_cast_F1 
## infer 
source deactivate det
cd /workspace/mmdett

CUDA_VISIBLE_DEVICES=3 python tools/test.py configs/competitive/crop_cast_F1.py /data/Res/crop_cast_F1/epoch_20.pth  --format-only --options "jsonfile_prefix=/data/fangxu/crop_cast_F1/crop_cast_F1_ep20" 

CUDA_VISIBLE_DEVICES=3 python tools/test.py configs/competitive/crop_cast_F1.py /data/Res/crop_cast_F1/epoch_20.pth  --format-only --options "jsonfile_prefix=/data/fangxu/crop_cast_F1/crop_cast_F1_ep20" 
score 0.23430

CUDA_VISIBLE_DEVICES=6 python tools/test.py configs/competitive/crop_cast_F1.py /data/Res/crop_cast_F1/epoch_20.pth  --format-only --options "jsonfile_prefix=/data/fangxu/crop_cast_F1/crop_cast_F1_ep20_rph0.6_rcnn0.5" 
score  0.23487

CUDA_VISIBLE_DEVICES=7 python tools/test.py configs/competitive/crop_cast_F1.py /data/Res/crop_cast_F1/epoch_40.pth  --format-only --options "jsonfile_prefix=/data/fangxu/crop_cast_F1/crop_cast_F1_ep40_rph0.6_rcnn0.5" 
score 0.2303

CUDA_VISIBLE_DEVICES=7 python tools/test.py configs/competitive/crop_cast_F1.py /data/Res/crop_cast_F1/epoch_30.pth  --format-only --options "jsonfile_prefix=/data/fangxu/crop_cast_F1/crop_cast_F1_ep30_rph0.6_rcnn0.5" 
score 0.2325

# cast_F1 
## infer 

CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/competitive/mc_cast_F1.py /data/Res/cast_F1/epoch_22.pth  --format-only --options "jsonfile_prefix=/data/fangxu/cast_F1/cast_F1_ep22" 

CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/competitive/mc_cast_F1.py /data/Res/cast_F1/epoch_26.pth  --format-only --options "jsonfile_prefix=/data/fangxu/cast_F1/cast_F1_ep26_sot_rpn_0.7_rcnn_0.5" 
## train
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29511 ./tools/dist_train.sh configs/competitive/mc_cast_F1.py 4

# swin transformer
source activate det
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=27519 PYTHONPATH='/opt/conda/bin/python':$PYTHONPATH mim train mmdet configs/swin_mask_rcnn/swin_L_F1.py --gpus 4 --gpus-per-node 4 

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=27519 PYTHONPATH='/opt/conda/bin/python':$PYTHONPATH mim train mmdet configs/swin_mask_rcnn/swin_L_F1.py --work-dir /data/Res/cas_swin_S_F1  --gpus 4 --gpus-per-node 4  launcher=pytorch



# cas r2 F1
## train
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29512 ./tools/dist_train.sh configs/competitive/cas_r2_F1.py 4
## infer 
CUDA_VISIBLE_DEVICES=6 python tools/test.py configs/competitive/cas_r2_F1.py /data/Res/cas_r2_F1/epoch_19.pth  --format-only --options "jsonfile_prefix=/data/fangxu/cas_r2_F1/cas_r2_F1_ep19_rpn0.7_rcnn0.6"  
score 0.23126

CUDA_VISIBLE_DEVICES=6 python tools/test.py configs/competitive/cas_r2_F1.py /data/Res/cas_r2_F1/epoch_15.pth  --format-only --options "jsonfile_prefix=/data/fangxu/cas_r2_F1/cas_r2_F1_ep15_rpn0.7_rcnn0.6"
score 0.20196

CUDA_VISIBLE_DEVICES=6 python tools/test.py configs/competitive/cas_r2_F1.py /data/Res/cas_r2_F1/epoch_10.pth  --format-only --options "jsonfile_prefix=/data/fangxu/cas_r2_F1/cas_r2_F1_ep10_rpn0.7_rcnn0.5"
score 0.20598

CUDA_VISIBLE_DEVICES=6 python tools/test.py configs/competitive/cas_r2_F1.py /data/Res/cas_r2_F1/epoch_25.pth  --format-only --options "jsonfile_prefix=/data/fangxu/cas_r2_F1/cas_r2_F1_ep25_rpn0.7_rcnn0.5"
score 0.2324



# mc cas r2 F1
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29512 ./tools/dist_train.sh configs/competitive/mc_cas_r2_F1.py 4

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=27512 ./tools/dist_train.sh configs/competitive/mc_cas_r2_F1.py 4
## infer
CUDA_VISIBLE_DEVICES=6 python tools/test.py configs/competitive/mc_cas_r2_F1.py /data/Res/cas_r2_F1/epoch_20.pth  --format-only --options "jsonfile_prefix=/data/fangxu/mc_cas_r2_F1/mc_cas_r2_F1_ep20_rpn0.7_rcnn0.5"
score 0.2247

CUDA_VISIBLE_DEVICES=6 python tools/test.py configs/competitive/mc_cas_r2_F1.py /data/Res/mc_cas_r2_F1/epoch_30.pth  --format-only --options "jsonfile_prefix=/data/fangxu/mc_cas_r2_F1/mc_cas_r2_F1_ep30_rpn0.7_rcnn0.5"
score 0.2238

# alb mc cast F1
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=39512 ./tools/dist_train.sh configs/competitive/alb_mc_cast_F1.py 4
## infer
CUDA_VISIBLE_DEVICES=6 python tools/test.py configs/competitive/alb_mc_cast_F1.py /data/Res/alb_mc_cast_F1/epoch_35.pth  --format-only --options "jsonfile_prefix=/data/fangxu/alb_mc_cast_F1/alb_mc_cast_F1_ep35_rpn0.7_rcnn0.55"
`


# test_B

CUDA_VISIBLE_DEVICES=1 python tools/test.py configs/competitive/cast_F1.py /data/Res/cast_F1/epoch_26.pth  --format-only --options "jsonfile_prefix=/data/Res/cast_F1/cast_F1_ep26_test_B"

CUDA_VISIBLE_DEVICES=6 python tools/test.py configs/competitive/cas_r2_F1.py /data/Res/cas_r2_F1/epoch_25.pth  --format-only --options "jsonfile_prefix=/data/fangxu/cas_r2_F1/cas_r2_F1_ep25_rpn0.7_rcnn0.5_test_B"
score 0.2324