# SimVP: Simpler yet Better Video Prediction
![GitHub stars](https://img.shields.io/github/stars/gaozhangyang/SimVP-Simpler-yet-Better-Video-Prediction)  ![GitHub forks](https://img.shields.io/github/forks/gaozhangyang/SimVP-Simpler-yet-Better-Video-Prediction?color=green) 

**In the example, the default epoch is 50. Please read our paper, and train 1000~2000 epochs for repruducing this work!** I will not respond to such a lowly mistake.

**SimVPv2 is available on https://github.com/chengtan9907/SimVPv2, which performs better than SimVP and is in the review process.** If our work is helpful for your research, we would hope you give us a star and citation. Thanks!

This repository contains the implementation code for paper:

**SimVP: Simpler yet Better Video Prediction**  
[Zhangyang Gao](https://westlake-drug-discovery.github.io/zhangyang_gao.html), [Cheng Tan](https://westlake-drug-discovery.github.io/cheng_tan.html), [Lirong Wu](https://lirongwu.github.io/), [Stan Z. Li](https://scholar.google.com/citations?user=Y-nyLGIAAAAJ&hl). In [CVPR](), 2022.
## Introduction

<p align="center">
    <img src="./readme_figures/overall_framework.png" width="600"> <br>
</p>

From CNN, RNN, to ViT, we have witnessed remarkable advancements in video prediction, incorporating auxiliary inputs, elaborate neural architectures, and sophisticated training strategies. We admire these progresses but are confused about the necessity: is there a simple method that can perform comparably well? This paper proposes SimVP, a simple video prediction model that is completely built upon CNN and trained by MSE loss in an end-to-end fashion. Without introducing any additional tricks and complicated strategies, we can achieve state-of-the-art performance on five benchmark datasets. Through extended experiments, we demonstrate that SimVP has strong generalization and extensibility on real-world datasets. The significant reduction of training cost makes it easier to scale to complex scenarios. We believe SimVP can serve as a solid baseline to stimulate the further development of video prediction.

## Dependencies
* torch
* scikit-image=0.16.2
* numpy
* argparse
* tqdm

## Overview

* `API/` contains dataloaders and metrics.
* `main.py` is the executable python file with possible arguments.
* `model.py` contains the SimVP model.
* `exp.py` is the core file for training, validating, and testing pipelines.

## Install

Poetryでインストールできます。

```
  git clone git@github.com:NIBB-Neurophysiology-Lab/SimVP-Simpler-yet-Better-Video-Prediction.git
  cd SimVP-Simpler-yet-Better-Video-Prediction
  poetry install
```

### Dataset
サンプルのデータセットを`data/image_list`に保存しています。


```
  cd ./data/image_list
  tree data/
data/
├── frame_00000.jpg
├── frame_00001.jpg
├── frame_00002.jpg
├── frame_00003.jpg
├── frame_00004.jpg
├── frame_00005.jpg
├── frame_00006.jpg
├── frame_00007.jpg
├── frame_00008.jpg
├── frame_00009.jpg
├── frame_00010.jpg
├── frame_00011.jpg
├── frame_00012.jpg
├── frame_00013.jpg
├── frame_00014.jpg
├── frame_00015.jpg
├── frame_00016.jpg
├── frame_00017.jpg
├── frame_00018.jpg
├── frame_00019.jpg
├── frame_00020.jpg
├── test_list.txt
├── train_list.txt
├── train_list_x.txt
└── train_list_xt.txt

```

以下のコマンドで実行します。

```
python main.py
```

実行すると`results/images`に画像が保存されます。
画像は`<イテレーション数>_<フレーム数>x.jpg`で入力画像が保存され、`<イテレーション数>_<フレーム数>y.jpg`で予測画像が保存されます。
本アルゴリズムでは入力フレーム数が出力フレーム数となり、入力フレーム数と同じ数の予測画像が出力される形になっています。
フレーム数を変えて実行する方法は以下のようになります。

```
python main.py --input_len 20
```

画像サイズの変更は以下のように行います。

```
python main.py --size 320,240
```

デフォルトでは500エポックで計算しており、100エポック毎にチェックポイントを`results/Debug/checkpoints`に保存しています。
保存したチェックポイントから学習を開始する際は以下のように実行します。

```
python main.py --modelfile results/Debug/checkpoints/100.pth
```

テストのみ実行する場合は以下のように実行します。

```
python main.py --modelfile results/Debug/checkpoints/100.pth --test
```

## Citation

If you are interested in our repository and our paper, please cite the following paper:

```
@InProceedings{Gao_2022_CVPR,
    author    = {Gao, Zhangyang and Tan, Cheng and Wu, Lirong and Li, Stan Z.},
    title     = {SimVP: Simpler Yet Better Video Prediction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {3170-3180}
}
```

## Contact

If you have any questions, feel free to contact us through email (tancheng@westlake.edu.cn, gaozhangyang@westlake.edu.cn). Enjoy!
