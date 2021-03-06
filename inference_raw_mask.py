import logging
import time
from multiprocessing import Lock

import cv2
import numpy as np

from PIL import Image
from mmcv.runner.fp16_utils import wrap_fp16_model
from tqdm import tqdm

from detect.utils_model_result import post_process_result
from detect.utils_torchvision import inference
from detect.utils_visualize import Visualizer
from mmdet.apis import init_detector

logging.basicConfig(format='%(levelname)s:%(name)s:%(asctime)s %(message)s', level=logging.DEBUG)

image_path = 'demo/demo.jpg'
config_path = 'configs/mask_rcnn/mask_rcnn_r50_fpn_test.py'
checkpoint_path = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
model = init_detector(config_path, checkpoint_path, device='cuda:0')
wrap_fp16_model(model)

lock = Lock()
pipeline = model.cfg.data.test.pipeline[1]

target_shape = (1333, 800)
img = cv2.imread(image_path)
img = cv2.resize(img, target_shape)
model.cfg.data.test.pipeline[1].img_scale = target_shape
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

logging.info(
    f'Mean Inference time: {np.mean(inference_time_list) * 1000: .2f}ms')

logging.info(f'FPS: {1 / np.mean(inference_time_list): .2f}')

result_data = post_process_result(model, result)

visualizer = Visualizer()
vis_img = visualizer.visualize(img, result_data)
cv2.imwrite('test.jpg', vis_img)
Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)).show()
