# -- coding: utf-8 --
import logging
import random
import platform
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from skimage.transform import resize


class Visualizer:

    def __init__(self, font_size=24):
        font_path = None
        if platform.system() == 'Windows':
            font_path = 'msyh.ttc'
        elif platform.system() == 'Linux':
            font_path = 'NotoSansCJK-Regular.ttc'
        elif platform.system() == 'Darwin':
            font_path = 'Hiragino Sans GB.ttc'
        logging.info(f'字体路径：{font_path}')
        self.font = ImageFont.truetype(font_path, font_size)

    def visualize(self, img_array, result, detail=False):
        logging.debug(f'开始可视化'.center(64, '='))

        for r in result:
            if 'mask' in r:
                if 'bbox' not in r:
                    continue
                x, y, w, h = r['bbox']
                w, h = int(w), int(h)
                rand_color = (random.randint(0, 255), random.randint(0, 255),
                              random.randint(0, 255))

                mask = np.array(r['mask'], dtype=np.float32)
                # mask_resize = cv2.resize(mask, (w, h))  # mode edge is not correct, do not use opencv to resize
                mask_resize = resize(mask, (h, w), mode='constant', cval=0)
                mask_resize[mask_resize < 0.5] = 0
                mask_resize[mask_resize >= 0.5] = 255

                mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
                mask_colored[np.where(mask_resize)] = rand_color

                if 'raw_bbox' in r:
                    x, y, _, _ = r['raw_bbox']
                x, y = max(int(x), 0), max(int(y), 0)
                small_image = img_array[y:y + h, x:x + w]
                img_array[y:y + h,
                          x:x + w] = cv2.addWeighted(small_image, 1,
                                                     mask_colored, 0.25, 0)

        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)

        for r in result:
            if 'bbox' not in r:
                continue
            x, y, w, h = r['bbox']
            rand_color = (random.randint(0, 255), random.randint(0, 255),
                          random.randint(0, 255))
            draw.rectangle(((int(x), int(y)), (int(x + w), int(y + h))),
                           outline=rand_color,
                           width=2)
            if 'category_name' in r:
                if 'score' in r:
                    draw_text = f"{r['category_name']}-{r['score']:.2f}\n"
                else:
                    draw_text = r['category_name']
                draw.multiline_text((int(x), max(int(y) - 30, 0)),
                                    draw_text,
                                    font=self.font,
                                    fill=(255, 0, 0))

        new_img = np.array(img)
        logging.debug(f'可视化完成')
        return new_img
