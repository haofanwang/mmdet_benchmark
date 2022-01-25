# -- coding: utf-8 --
import logging
import time

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as functional
from PIL import Image
from mmcv.image.geometric import rescale_size


def resize(image, scale, img_meta):
    n, c, h, w = image.shape
    new_size, scale_factor = rescale_size((w, h), scale=scale, return_scale=True)
    new_image = functional.resize(image, size=[new_size[1], new_size[0]], interpolation=Image.BILINEAR)
    new_h, new_w = new_image.shape[-2:]
    w_scale = new_w / w
    h_scale = new_h / h
    img_meta['img_shape'] = (new_h, new_w, c)
    img_meta['ori_shape'] = (h, w, c)
    img_meta['scale_factor'] = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    return new_image


def normalize(img, transform):
    img = img.float()
    mean = torch.tensor([[transform.mean]]).T
    std = torch.tensor([[transform.std]]).T
    if torch.cuda.is_available():
        mean = mean.cuda()
        std = std.cuda()
    img -= mean
    img /= std
    return img


def pad(img, transform):
    n = transform.size_divisor
    h, w = img.shape[2:]
    x = w % n
    if x > 0:
        x = n - x
    y = h % n
    if y > 0:
        y = n - y
    img_pad = functional.pad(img, [0, 0, x, y])
    return img_pad


def preprocess(cfg, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(np.expand_dims(img.transpose(2, 0, 1), 0))
    if torch.cuda.is_available():
        img = img.cuda()

    # MultiScaleFlipAug
    if 'img_scale' in cfg:
        tta_scales = cfg.img_scale
        if isinstance(tta_scales, tuple):
            tta_scales = [tta_scales]
    elif 'scale_factor' in cfg:
        tta_scales = cfg.scale_factor
    else:
        logging.error(f'没有找到 img_scale 也没有找到 scale_factor，请检查 cfg 是否正确：\n{cfg}')
        raise Exception('模型配置错误')

    img_metas = []
    imgs = []
    for scale in tta_scales:
        x = img
        img_meta = {
            'flip': False, 'flip_direction': None,
        }
        for transform in cfg.transforms:
            if transform.type == 'Resize':
                x = resize(x, scale, img_meta)
                logging.info(f'Resize: {x.shape}')
            elif transform.type == 'Normalize':
                x = normalize(x, transform)
            elif transform.type == 'Pad':
                x = pad(x, transform)

        imgs.append(x)
        img_metas.append([img_meta])
    return {
        'img': imgs,
        'img_metas': img_metas
    }


def inference(model, cfg, img, lock, set_device=True):
    logging.debug(f'开始预处理')

    if set_device:
        device = next(model.parameters()).device
        if device.type == 'cuda':
            torch.cuda.set_device(device)
    with torch.no_grad(), lock:
        start = time.time()
        data = preprocess(cfg, img)
        torch.cuda.synchronize()
        logging.debug(f'预处理完成，耗时：{(time.time() - start) * 1000:.2f}ms')
        logging.debug(f'预处理后的矩阵尺寸：{[x.shape for x in data["img"]]}')

        logging.debug(f'开始 GPU 预测')
        start = time.time()
        results = model(return_loss=False, rescale=True, **data)
    logging.debug(f'GPU 预测完成，耗时：{(time.time() - start) * 1000:.2f}ms')

    return results[0]
