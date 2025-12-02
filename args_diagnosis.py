#!/usr/bin/python
# -*- coding:utf-8 -*-
from datasets.HUST_bearing import dataset_save as HUST_Bearing
from datasets.WT_Planetary_gearbox import dataset_save as WT_Planetary_Gearbox

from utils.train_val_test_visualize import train_val_test_visualize
from utils.logger import setlogger

import numpy as np

import os
from datetime import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # dataset parameters
    parser.add_argument('--dataset_name', type=str, default='HUST_Bearing', help='dataset name', choices=['HUST_Bearing', 'WT_Planetary_Gearbox'])
    parser.add_argument('--sample_length', type=int, default=1024, help='sample_length')
    parser.add_argument('--snr_noise', type=bool, default=False, help='Add snr noise')
    parser.add_argument('--snr', type=int, default=-6, help='SNR, the level of noise under noise task', choices=[-6, -4, -2, 0])
    parser.add_argument('--D_num', type=str, default='D1', help='D1-D4', choices=['D1', 'D2', 'D3', 'D4'])
    parser.add_argument('--save_dataset', type=bool, default=True, help='whether saving the dataset')

    # basic parameters
    parser.add_argument('--model_name', type=str, default='MSRC_KAN_PAM', help='the name of the model',
                        choices=['MSRC_KAN_PAM', 'QCNN', 'LaplaceWaveletNet', 'WDCNN',
                                 'CNN_BiLSTM', 'ResNet18', 'AlexNet', 'LeNet'])
    parser.add_argument('--batch_size', type=int, default=64, help='the number of samples for each batch')

    # optimization information
    parser.add_argument('--lr', type=float, default=0.01, help='the initial learning rate')
    parser.add_argument('--patience', type=int, default=5, help='the para of lr scheduler')
    parser.add_argument('--min_lr', type=int, default=1e-5, help='the para of lr scheduler')
    parser.add_argument('--epoch', type=int, default=100, help='the max number of epoch')

    # saving results
    parser.add_argument('--operation_num', type=int, default=5, help='the repeat operation of model')
    parser.add_argument('--only_test', type=bool, default=False, help='loading the trained model if only test')
    parser.add_argument('--save_result', type=bool, default=False, help='whether saving the results')
    parser.add_argument('--PSD_PAM', type=bool, default=False, help='whether visual interpretability analysis')
    args = parser.parse_args()
    return args

args = parse_args()

if args.snr_noise:
    save_name = args.dataset_name + '_' + args.D_num + '_snr' + str(args.snr)
else:
    save_name = args.dataset_name + '_' + args.D_num

if args.save_dataset:
    if args.dataset_name == 'HUST_Bearing':
        HUST_Bearing(args, save_name)
    elif args.dataset_name == 'WT_Planetary_Gearbox':
        WT_Planetary_Gearbox(args, save_name)
else:
    # create the result dir
    save_dir = os.path.join(os.getcwd() + '/results/{}'.format(save_name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # set the logger
    logger = setlogger(os.path.join(save_dir, args.model_name + '.log'))
    logger.info('***********************Start*****************************')
    logger.info('**********Results Folder: {}**********'.format(save_dir))
    logger.info('**********Parameter configuration**********')
    time_now = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    logger.info('Time: {}'.format(time_now))
    for k, v in args.__dict__.items():
        logger.info("{}: {}".format(k, v))

    Accuracy, Precision, Recall, F1, J = [], [], [], [], []
    operation = train_val_test_visualize(args)
    for op_num in range(args.operation_num):
        logger.info('\n**********Operation: {}**********'.format(op_num))
        if args.only_test:
            operation.setup(logger, op_num, save_name)
        else:
            operation.setup(logger, op_num, save_name)
            operation.train_val(logger, op_num, save_name)
        acc, precision, recall, f1 = operation.test(logger, op_num, save_name)
        Accuracy.append(acc)
        Precision.append(precision)
        Recall.append(recall)
        F1.append(f1)
        if args.PSD_PAM and 'PAM' in args.model_name:
            true_states, pred_states = operation.visualize(logger, op_num, save_name)

        if op_num == (args.operation_num - 1):
            logger.info('\n**********Result analysis**********')
            Accuracy = np.array(Accuracy) * 100
            Precision = np.array(Precision) * 100
            Recall = np.array(Recall) * 100
            F1 = np.array(F1) * 100

            Accuracy_list = ', '.join(['{:.2f}'.format(acc) for acc in Accuracy])
            Precision_list = ', '.join(['{:.2f}'.format(precision) for precision in Precision])
            Recall_list = ', '.join(['{:.2f}'.format(recall) for recall in Recall])
            F1_list = ', '.join(['{:.2f}'.format(f1) for f1 in F1])
            logger.info('All acc: {}, \nMean acc: {:.2f}, Std acc: {:.2f}, Max acc: {:.2f}, Min acc: {:.2f}'.format(
                Accuracy_list, Accuracy.mean(), Accuracy.std(), Accuracy.max(), Accuracy.min()))
            logger.info('All precision: {}, \nMean precision: {:.2f}, Std precision: {:.2f}, Max precision: {:.2f}, Min precision: {:.2f}'.format(
                Precision_list, Precision.mean(), Precision.std(), Precision.max(), Precision.min()))
            logger.info('All recall: {}, \nMean recall: {:.2f}, Std recall: {:.2f}, Max recall: {:.2f}, Min recall: {:.2f}'.format(
                Recall_list, Recall.mean(), Recall.std(), Recall.max(), Recall.min()))
            logger.info('All f1: {}, \nMean f1: {:.2f}, Std f1: {:.2f}, Max f1: {:.2f}, Min f1: {:.2f}'.format(
                F1_list, F1.mean(), F1.std(), F1.max(), F1.min()))
            Accuracy, Precision, Recall, F1 = [], [], [], []
    logger.info('***********************End*****************************\n')