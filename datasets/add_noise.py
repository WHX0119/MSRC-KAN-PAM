#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np

def add_snr_noise(x, snr):
    noise = np.random.randn(len(x))
    signal_power = np.sum(np.power(x, 2)) / x.shape[0]
    noise_power = signal_power / np.power(10, (snr/10))
    noise = np.sqrt(noise_power / np.std(noise)) * noise
    noise_signal = x + noise
    return noise_signal