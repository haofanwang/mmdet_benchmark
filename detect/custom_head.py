import torch

from mmdet.models.roi_heads.mask_heads.fcn_mask_head import FCNMaskHead
from mmdet.models.builder import HEADS


@HEADS.register_module()
class FCNMaskHeadWithRawMask(FCNMaskHead):
    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_pred = det_bboxes.new_tensor(mask_pred)

        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        labels = det_labels

        for i in range(len(mask_pred)):
            cls_segms[labels[i]].append(mask_pred[i, labels[i]].detach().cpu().numpy())
        return cls_segms
