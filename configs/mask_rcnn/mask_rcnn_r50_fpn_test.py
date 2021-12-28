_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['detect.custom_head'],
    allow_failed_imports=False)

model = dict(
    roi_head=dict(
        mask_head=dict(
            type="FCNMaskHeadWithRawMask"
        )
    )
)
