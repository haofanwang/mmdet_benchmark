import logging
import time

import cv2
import numpy as np
import torch

from PIL import Image
from tqdm import tqdm
from detect.utils_model_result import post_process_result
from detect.utils_visualize import Visualizer

from mmdeploy.utils import get_backend, get_input_shape, load_config

from mmdeploy.apis.utils import build_task_processor

logging.basicConfig(
    format='%(levelname)s:%(name)s:%(asctime)s %(message)s',
    level=logging.DEBUG)

deploy_cfg = '/data/ypw/mmdet_benchmark/configs/mmdet/instance-seg/instance-seg_tensorrt-fp16_dynamic-5000_test.py'
model_cfg = '/data/ypw/mmdeploy/configs/mmdet/instance-seg/mask_rcnn_test.py'
checkpoint_path = '/data/ypw/mmdeploy/work_dir/mask_rcnn_trt/end2end.engine'
output_path = '/data/ypw/mmdeploy/work_dir/mask_rcnn_trt/test.jpg'
image_path = '/data/ypw/detection/sample/5.jpg'
device = 'cuda'

img = cv2.imread(image_path)
logging.info(f'img_shape: {img.shape}')

backend = get_backend(deploy_cfg)
model = [checkpoint_path]
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
input_shape = get_input_shape(deploy_cfg)

task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.init_backend_model(model)
model.cfg = model_cfg
model_inputs, _ = task_processor.create_input(img, input_shape)

# warmup
with torch.no_grad():
    result = task_processor.run_inference(model, model_inputs)[0]

inference_time_list = []
with torch.no_grad():
    for i in tqdm(range(1000)):
        start = time.time()
        result = task_processor.run_inference(model, model_inputs)[0]
        inference_time = time.time() - start
        inference_time_list.append(inference_time)
        logging.info(f'Inference time: {inference_time * 1000: .2f}ms')
logging.info(
    f'Mean Inference time: {np.mean(inference_time_list) * 1000: .2f}ms')

result_data = post_process_result(model, result)

visualizer = Visualizer()
vis_img = visualizer.visualize(img, result_data)
cv2.imwrite('mmdeploy_trt_optimized.jpg', vis_img)
Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)).show()
