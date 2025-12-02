# MSRC-KAN-PAM: A new interpretable rotating machinery fault diagnosis framework
* Core codes for the paper:
<br> MSRC-KAN-PAM: A new interpretable rotating machinery fault diagnosis framework

* This repository uses an efficient implementation of KAN: https://github.com/Blealtan/efficient-kan

<div align="center">
<img src="https://github.com/WHX0119/MSRC-KAN-PAM/blob/main/framework.jpg" width="600" />
</div>

## Datasets
* [Case study 1: HUST bearing](https://github.com/CHAOZHAO-1/HUSTbearing-dataset)
* [Case study 2: WT-Planetary gearbox](https://github.com/Liudd-BJUT/WT-planetary-gearbox-dataset)

## Pakages
* `data` needs loading the Datasets in above links
* `datasets` contians the pre-processing process for the data
* `models` contians 8 methods including the proposed method
* `utils` contians logger & train_val_test_visualize process
