import time
import cv2
import numpy as np
import logging
from multiprocessing import Lock

from tqdm import tqdm
from mmcv.runner.fp16_utils import wrap_fp16_model
from mmdet.apis import init_detector
from detect.utils_torchvision import inference

logging.basicConfig(format='%(levelname)s:%(name)s:%(asctime)s %(message)s', level=logging.DEBUG)

image_path = 'demo/demo.jpg'
config_path = 'configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'
checkpoint_path = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
model = init_detector(config_path, checkpoint_path, device='cuda:0')
wrap_fp16_model(model)

lock = Lock()
pipeline = model.cfg.data.test.pipeline[1]

img = cv2.imread(image_path)
img = cv2.resize(img, (1333, 800))
logging.info(f'img_shape: {img.shape}')

# warmup
inference(model, cfg=pipeline, img=img, lock=lock)

inference_time_list = []
for i in tqdm(range(20)):
    start = time.time()
    result = inference(model, cfg=pipeline, img=img, lock=lock)
    inference_time = time.time() - start
    inference_time_list.append(inference_time)
    logging.info(f'Inference time: {inference_time * 1000: .2f}ms')

logging.info(f'Mean Inference time: {np.mean(inference_time_list) * 1000: .2f}ms')

logging.root.setLevel(logging.INFO)
model.show_result(
    img, result=result, score_thr=0.5,
    font_size=13, thickness=1, show=True, win_name='model_result'
)
