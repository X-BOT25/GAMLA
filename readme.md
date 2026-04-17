# Environment
```bash
# we use cuda=11.8, python=3.10 and torch=2.1.2+cu118

# For Image Classification
conda create -n gamla python=3.10
conda activate gamla
pip install torch==2.1.2 torchvision==0.16.2 -i https://download.pytorch.org/whl/cu118
pip install timm

# For Object Detection and Segmentation
pip install -U openmim
mim install mmengine==0.10.1 mmcv==2.1.0 mmdet==3.3.0 mmsegmentation==1.2.2
```
