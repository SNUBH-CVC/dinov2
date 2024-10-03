## Installation

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

## TODO
- [ ] 1 machine 학습 시 `fsdp.FSDPCheckpointer`를 변경해줘야 하는지 확인 필요, https://github.com/facebookresearch/dinov2/issues/134