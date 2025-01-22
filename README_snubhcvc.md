## Train
Run DINOv2 training on 4 A100-80GB nodes (32 GPUs) in a SLURM cluster environment with submitit:
- train with 1 machine and multiple GPUs: https://github.com/facebookresearch/dinov2/issues/134
- initialize weight using pretrained DINOv2 model and pretraining: https://github.com/facebookresearch/dinov2/issues/339

```shell
python dinov2/run/train/train.py --nodes 1 --ngpus 2 --config-file dinov2/configs/train/vitl16_short.yaml --output-dir ./outputs train.dataset_path=ImageNet:split=TRAIN:root=/mnt/nas/external/public/raw/imagenet-1k:extra=/mnt/nas/external/public/raw/imagenet-1k
# 1-machine, multi-GPU 학습 시 아래 명령어로 실행시켜야 한다. https://github.com/facebookresearch/dinov2/issues/161#issuecomment-1689542308
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 dinov2/train/train.py --config-file=dinov2/configs/train/vitl16_short.yaml --output-dir=./outputs train.dataset_path=ImageNet:split=TRAIN:root=/mnt/nas/external/public/raw/imagenet-1k:extra=/mnt/nas/external/public/raw/imagenet-1k

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 dinov2/train/train.py --config-file=dinov2/configs/train/vitb14_short.yaml --output-dir=./outputs train.dataset_path=CAG:split=TRAIN:root=./data/cagfm
```

### Continual Pre-training
https://github.com/facebookresearch/dinov2/issues/431
https://github.com/cpheidelberg/tools_dinov2

## Dataset
### ImageNet
https://github.com/facebookresearch/dinov2/issues/460 에서 labels.txt 다운로드 후 아래 코드 실행하면 필요한 extra 파일이 생성된다.

```python
from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="<ROOT>", extra="<EXTRA>")
    dataset.dump_extra()
```

### CAG
`extended.ExtendedVisionDataset`을 상속 받아서 `get_image_data` 매서드를 구현해주면 된다. classification을 하려면 `get_target`가 필요한데, 이 부분은 넘어가도 된다.

## Pretrain
README.md 확인해서 pretrained checkpoint 다운로드 받기 

1 machine, multiple GPUs 조건에서 학습하기 위해서 아래 명령어 실행. https://github.com/facebookresearch/dinov2/issues/161#issuecomment-1689542308
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 dinov2/train/train.py --config-file=dinov2/configs/train/vitl16_short.yaml --output-dir=./outputs train.dataset_path=ImageNet:split=TRAIN:root=<ROOT>:extra=<EXTRA>
```
CAG 데이터 학습 시 `train.dataset_path=CAG:split=TRAIN`으로 변경해주면 된다.

DINOv2 pretrained weight로 initialize 한 뒤에 pretrain 하고 싶은 경우, student, teacher를 학습된 checkpoint로 initialize 해주면 된다.
`ssl_meta_arch.py` 파일 내에서 `teacher_backbone`도 initialize 해주면 될 듯하다.
https://github.com/facebookresearch/dinov2/issues/339

pretrained weight 크기 기준이 518 이므로 이에 맞춰서 설정하고 돌려야 한다. https://github.com/facebookresearch/dinov2/issues/316
V100 32GB 기준 batch size 16이 최대로 빅데이터서버 기준 128이 최대

## Downstream Task

### Checkpoint 
- 사용하지 않는 parameter 제거, checkpoint parameter 이름, pos_embed 차원 interpolation 문제를 해결해야 load 할 수 있다.

### Angle prediction
angle 별 100개 정도 모아서 평가 데이터셋으로 구축 후 테스트 진행 
- scratch 학습, DINOv2 pretrained 학습 성능 비교하기

## TroubleShooting
- `mask_token` downstream task에서 사용하지 않아서 `find_unused_parameters` 에러가 생길 수 있다. `requires_grad` 를 False로 설정해주면 해결된다.
    - https://github.com/facebookresearch/dinov2/issues/210
    - mmpretrain에서 알 수 없는 버그로 성능이 loss가 이상하게 나와서 별도의 학습 스크립트 구현.
- ViT-S에서 `drop_path_rate > 0`인 경우 에러 발생 
    - https://github.com/facebookresearch/dinov2/issues/160
    - 해결에 대한 PR이 있긴 한데, 일단 `drop_path_rate: 0`으로 설정한 뒤에 학습 