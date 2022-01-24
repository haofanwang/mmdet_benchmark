import logging
import time

import cv2
import numpy as np
import torch

from tqdm import tqdm
from mmdeploy.utils import get_backend, get_input_shape, load_config

from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils.utils import get_root_logger

logging = get_root_logger(log_level=logging.DEBUG)

deploy_cfg = 'configs/mmdet/instance-seg/instance-seg_tensorrt-fp16_dynamic-320x320-1344x1344.py'
model_cfg = 'configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'
checkpoint_path = 'work_dirs/mask_rcnn_coco_trt/end2end.engine'
image_path = 'demo/demo.jpg'
output_path = 'demo/output_trt.jpg'
device = 'cuda'

img = cv2.imread(image_path)
logging.info(f'img_shape: {img.shape}')

backend = get_backend(deploy_cfg)
model = [checkpoint_path]
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
input_shape = get_input_shape(deploy_cfg)
logging.info(f'input_shape: {input_shape}')

task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.init_backend_model(model)

# warmup
with torch.no_grad():
    model_inputs, _ = task_processor.create_input(img, input_shape)
    logging.info(f'model_inputs: {model_inputs["img"][0].shape}')
    result = task_processor.run_inference(model, model_inputs)[0]

inference_time_list = []
with torch.no_grad():
    for i in range(20):
        start = time.time()

        a = time.time()
        model_inputs, _ = task_processor.create_input(img, input_shape)
        b = time.time()
        logging.debug(f'pre-processing: {(b - a) * 1000:.2f}ms')

        result = task_processor.run_inference(model, model_inputs)[0]
        inference_time = time.time() - start
        inference_time_list.append(inference_time)
        logging.info(f'Inference time: {inference_time * 1000: .2f}ms')

logging.info(
    f'Mean Inference time: {np.mean(inference_time_list) * 1000: .2f}ms')

logging.info(f'FPS: {1 / np.mean(inference_time_list): .2f}ms')

task_processor.visualize(
    image=img,
    model=model,
    result=result,
    output_file=output_path,
    window_name=backend.value,
    show_result=False)
