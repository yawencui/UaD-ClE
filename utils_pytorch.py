#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.init as init
import os
import os.path as osp

import subprocess
import pickle
import numpy as np


def get_data_file(filename, data_dir, label2id, unlabel=False):
    data = []
    targets = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(os.path.join(data_dir, line.strip()))
            targets.append(label2id[line.strip().split("/")[1]])
    if unlabel:
        return np.array(data)

    return np.array(data), np.array(targets)

def get_data_file_unlabeled(filename, data_dir, label2id, unlabel=False):
    data = []
    targets = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(os.path.join(data_dir, line.strip()))
            targets.append(label2id[line.strip().split("/")[1]])
    if unlabel:
        return np.array(data)

    return np.array(data), np.array(targets)


def get_label2id(filename):
    label_set = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if line not in label_set.keys():
                label_set[line] = len(label_set)
    return label_set



def savepickle(data, file_path):
    mkdir_p(osp.dirname(file_path), delete=False)
    print('pickle into', file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def mkdir_p(path, delete=False, print_info=True):
    if path == '': return

    if delete:
        subprocess.call(('rm -r ' + path).split())
    if not osp.exists(path):
        if print_info:
            print('mkdir -p  ' + path)
        subprocess.call(('mkdir -p ' + path).split())


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)
