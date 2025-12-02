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
* `data` needs loading the Datasets in above links
* `datasets` contians the pre-processing process for the data
* `efficient_kan` an efficient implementation of KAN: https://github.com/Blealtan/efficient-kan
* `models` contians 8 methods including the proposed method
* `utils` contians logger and train&val&test&visualize process

## Guide 
* `train_val_test_visualize.py` is the train&val&test&visualize process of all methods.
* You need to load the data in above Datasets link at first, and put them in the `data` folder. Then run in `args_diagnosis.py`
<br> Pay attention to that if you want to run the data pre-process, you need to load [Case study 1] and [Case study 2] in Datasets,
<br> and set --save_dataset (in `args_diagnosis.py`) to True; or you can just load the [save_dataset], and set --save_dataset to False.
* You can also choose the modules or adjust the parameters of the model to suit your needs.
