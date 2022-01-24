from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core import bbox2result
from mmdeploy.codebase.mmdet.deploy.object_detection_model import __BACKEND_MODEL, End2EndModel


@__BACKEND_MODEL.register_module('end2end_optimized')
class End2EndModelOptimized(End2EndModel):

    @staticmethod
    def clear_outputs(
        test_outputs: List[Union[torch.Tensor, np.ndarray]]
    ) -> List[Union[List[torch.Tensor], List[np.ndarray]]]:
        batch_size = len(test_outputs[0])

        num_outputs = len(test_outputs)
        outputs = [[None for _ in range(batch_size)]
                   for _ in range(num_outputs)]

        for i in range(batch_size):
            inds = test_outputs[0][i, :, 4] > 0.0
            for output_id in range(num_outputs):
                outputs[output_id][i] = test_outputs[output_id][i, inds, ...]
        return outputs

    def forward(self, img: Sequence[torch.Tensor], img_metas: Sequence[dict],
                *args, **kwargs):
        import time
        import logging
        from mmdeploy.utils.utils import get_root_logger

        logging = get_root_logger(log_level=logging.DEBUG)

        a = time.time()
        input_img = img[0].contiguous()
        outputs = self.forward_test(input_img, img_metas, *args, **kwargs)
        outputs = self.clear_outputs(outputs)
        torch.cuda.synchronize()
        b = time.time()
        logging.debug(f'forward: {(b - a) * 1000:.2f}ms')

        a = time.time()
        batch_dets, batch_labels = outputs[:2]
        batch_masks = outputs[2] if len(outputs) == 3 else None
        batch_size = input_img.shape[0]
        img_metas = img_metas[0]
        results = []
        rescale = kwargs.get('rescale', True)
        for i in range(batch_size):
            dets, labels = batch_dets[i], batch_labels[i]
            if rescale:
                scale_factor = img_metas[i]['scale_factor']

                if isinstance(scale_factor, (list, tuple, np.ndarray)):
                    assert len(scale_factor) == 4
                    scale_factor = np.array(scale_factor)[None, :]  # [1,4]
                dets[:, :4] /= scale_factor

            if 'border' in img_metas[i]:
                # offset pixel of the top-left corners between original image
                # and padded/enlarged image, 'border' is used when exporting
                # CornerNet and CentripetalNet to onnx
                x_off = img_metas[i]['border'][2]
                y_off = img_metas[i]['border'][0]
                dets[:, [0, 2]] -= x_off
                dets[:, [1, 3]] -= y_off
                dets[:, :4] *= (dets[:, :4] > 0).astype(dets.dtype)

            dets_results = bbox2result(dets, labels, len(self.CLASSES))

            if batch_masks is not None:
                masks = batch_masks[i]
                segms_results = [[] for _ in range(len(self.CLASSES))]
                for j in range(len(dets)):
                    segms_results[labels[j]].append(masks[j])
                results.append((dets_results, segms_results))
            else:
                results.append(dets_results)
        b = time.time()
        logging.debug(f'post-processing: {(b - a) * 1000:.2f}ms')
        return results
