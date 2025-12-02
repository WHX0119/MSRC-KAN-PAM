# MSRC-KAN-PAM: A new interpretable rotating machinery fault diagnosis framework
* Core codes for the paper:
<br> MSRC-KAN-PAM: A new interpretable rotating machinery fault diagnosis framework

<div align="center">
<img src="https://github.com/WHX0119/MSRC-KAN-PAM/blob/main/framework.jpg" width="600" />
</div>

## Operating environment
* Python 3.8.13
* PyTorch 1.13.1
* numpy  1.22.0 (If you get an error when saving data, try lowering your numpy version!)
* and other necessary libs

## Datasets
* [Case study 1: HUST bearing](https://github.com/CHAOZHAO-1/HUSTbearing-dataset)
* [Case study 2: WT-Planetary gearbox](https://github.com/Liudd-BJUT/WT-planetary-gearbox-dataset)

## Pakages
* `data` contians the datasets
* `datasets` contians the pre-processing process for the data
* `efficient_kan` contians an efficient implementation of KAN: https://github.com/Blealtan/efficient-kan
* `models` contians 8 methods including the proposed method
* `utils` contians logger and train&val&test&visualize process

## Guide
* You need to download the datasets in above link at first, and put them in the `data` folder.
<br> Pay attention to that if you want to run the data pre-processing and save the results of data preprocessing, set --save_dataset to True;
<br> or you can only load the results of data preprocessing, set --save_dataset to False, Then run in `args_diagnosis.py`.
* You can also choose the modules or adjust the parameters of the model to suit your needs.
