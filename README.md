# E-NeRV: Expedite Neural Video Representation with Disentangled Spatial-Temporal Context  (ECCV 2022)

### [Paper](https://arxiv.org/abs/2207.08132)


[Zizhang Li](https://kyleleey.github.io),
Mengmeng Wang,
Huaijin Pi,
Kechun Xu,
Jianbiao Mei,
Yong Liu

This is the official implementation of the paper "E-NeRV: Expedite Neural Video Representation with Disentangled Spatial-Temporal Context".

## Abstract
Recently, the image-wise implicit neural representation of videos, NeRV, has gained popularity for its promising results and swift speed compared to regular pixel-wise implicit representations. However, the redundant parameters within the network structure can cause a large model size when scaling up for desirable performance. The key reason of this phenomenon is the coupled formulation of NeRV, which outputs the spatial and temporal information of video frames directly from the frame index input. In this paper, we propose E-NeRV, which dramatically expedites NeRV by decomposing the image-wise implicit neural representation into separate spatial and temporal context. Under the guidance of this new formulation, our model greatly reduces the redundant model parameters, while retaining the representation ability. We experimentally find that our method can improve the performance to a large extent with fewer parameters, resulting in a more than 8$\times$ faster speed on convergence

## Method overview
<img src=./assets/method.png width="800"  />

## Get started
We run with Python 3.8, you can set up a conda environment with all dependencies like so:

```shell
conda create -n enerv python=3.8
conda activate enerv
pip install -r requirements.txt
```

## Data
The paper's main experiments were conducted on:
* [data/bunny](./data/bunny) directory contains big buck bunny video sequence
* [UVG](http://ultravideo.fi/#testsequences) video sequences can be found in official website

## Train & Evaluate & Test

### Train
For now, we provide training and evaluation on original NeRV and our E-NeRV, by running following command with different configs and specified ```EXP_NAME```:

```python
python train.py --gpus 0 --cfg_path ./cfgs/E-NeRV-bunny.yaml --exp_name ENeRV/train
--save_image

--resume_path outputs/ENeRV/demo/checkpoint.pth
```

```shell
bash scripts/run.sh ./cfgs/E-NeRV-bunny.yaml EXP_DEMO 29500
```
### Evaluate
```python
python eval.py --gpus 0 --cfg_path ./cfgs/E-NeRV-bunny.yaml --exp_name ENeRV/eval --ckp_path ./outputs/ENeRV/demo/checkpoint.pth
```

### Test
```python
python test.py --gpus 0 --cfg_path ./cfgs/E-NeRV-bunny.yaml --exp_name ENeRV/test --ckp_path ./outputs/ENeRV/demo/checkpoint.pth --frame_idx 'list(range(0,264))'
--vid_path ./data/bunny --eval
```

The logs and other outputs will be automaically saved to the output directory ```./outputs/EXP_NAME```

## Acknowledgement
Thanks to Hao Chen for his excellent work and implementation [NeRV](https://github.com/haochen-rye/NeRV).

## Contact
If you have any questions, please feel free to email the authors.
