import os

CUDA_DEVICE_NAME = os.environ.get('CUDA_DEVICE_NAME', 'cuda:0')   # Whatever device is given via CUDA_VISIBLE_DEVICES will be seen as 'cuda:0' at run time
