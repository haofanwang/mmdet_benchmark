_base_ = [
    '../_base_/base_instance-seg_dynamic.py',
    '../../_base_/backends/tensorrt-fp16.py'
]

backend_config = dict(
    common_config=dict(max_workspace_size=10000000000),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 320, 320],
                    opt_shape=[1, 3, 800, 1344],
                    max_shape=[1, 3, 5000, 5000])))
    ])
codebase_config = dict(model_type='end2end_optimized')

custom_imports = dict(
    imports=['detect.e2e_optimized'], allow_failed_imports=False)
