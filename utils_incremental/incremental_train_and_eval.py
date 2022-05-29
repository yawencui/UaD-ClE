#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
from utils_pytorch import *
from dataloder import BaseDataset, UnlabelDataset
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pdb
import math
def incremental_train_and_eval(base_lamda, adapt_lamda, u_t, label2id, uncertainty_distillation, prototypes, prototypes_flag, prototypes_on_flag, update_unlabeled, epochs, method, unlabeled_num, unlabeled_iteration, unlabeled_num_selected, train_batch_size, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
                               trainloader, testloader, \
                               iteration, start_iteration, \
                               T, beta, unlabeled_data, unlabeled_gt, nb_cl, trainset, image_size,
                               fix_bn=False, weight_per_class=None, device=None):


    if iteration > start_iteration:
        unlabeled_trainset = UnlabelDataset(image_size)
        unlabeled_trainset.data = unlabeled_data
        unlabeled_trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=train_batch_size,
                                                            shuffle=False, num_workers=4)


    if iteration > start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features

    # train the model with labeled data
    for epoch in range(epochs):
        # train
        tg_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    # m.weight.requires_grad = False
                    # m.bias.requires_grad = False
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        correct = 0
        total = 0
        tg_lr_scheduler.step()
        if epoch % 40 == 0:
            print('\nEpoch: %d, LR: ' % epoch, end='')
            print(tg_lr_scheduler.get_lr())
        #print('the number of total training data: {}'.format(trainloader.df()))
        for batch_idx, (inputs, targets, flags, on_flags) in enumerate(trainloader):
            inputs, targets, flags, on_flags = inputs.to(device), targets.to(device), flags.to(device), on_flags.to(device)
            tg_optimizer.zero_grad()
            outputs = tg_model(inputs)
            # ref_outputs = ref_model(inputs)
            if iteration == start_iteration:
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets.long())
            else:
                ref_outputs = ref_model(inputs)
                if uncertainty_distillation:
                    loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets.long())
                    #uncertainty-aware distillation
                    ###############################
                    out_prob = []
                    for _ in range(10):
                        #Gaussian noise
                        noise = torch.clamp(torch.randn_like(inputs) * 0.01, -0.02, 0.02)
                        inputs_noise = inputs + noise.to(device)
                        outputs_noise = ref_model(inputs_noise)
                        out_prob.append(F.softmax(outputs_noise, dim=1))
                    out_prob = torch.stack(out_prob)
                    out_std = torch.std(out_prob, dim=0)
                    out_prob = torch.mean(out_prob, dim=0)
                    max_value, max_idx = torch.max(out_prob, dim=1)
                    max_std = out_std.gather(1, max_idx.view(-1, 1))
                    max_std_sorted, std_indices = torch.sort(max_std, descending=False)
                    max_std = max_std.squeeze(1).detach().cpu().numpy()

                    outputs_cp = outputs
                    outputs = outputs.detach().cpu().numpy()
                    ref_outputs = ref_outputs.detach().cpu().numpy()
                    idx_del = []
                    for idx in range(len(max_std)):
                        if max_std[idx] > max_std_sorted[int(u_t * len(max_std))]:
                            if flags[idx] == 0:
                                idx_del.append(idx)
                    outputs = np.delete(outputs, idx_del, axis = 0)
                    ref_outputs = np.delete(ref_outputs, idx_del, axis = 0)
                    outputs = torch.from_numpy(outputs)
                    ref_outputs = torch.from_numpy(ref_outputs)
                    if adapt_lamda:
                        cur_lamda = base_lamda * 1 / u_t *  math.sqrt(num_old_classes / nb_cl)
                    else:
                        cur_lamda = base_lamda

                    loss1 = cur_lamda * nn.KLDivLoss()(F.log_softmax(outputs[:, :num_old_classes] / T, dim=1),
                                       F.softmax(ref_outputs.detach() / T, dim=1)) * T * T * beta * num_old_classes

                else:
                    loss1 = nn.KLDivLoss()(F.log_softmax(outputs[:, :num_old_classes] / T, dim=1),
                                       F.softmax(ref_outputs.detach() / T, dim=1)) * T * T * beta * num_old_classes
                    loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets.long())

                loss = loss1 + loss2

            loss.backward()
            tg_optimizer.step()

            train_loss += loss.item()
            if iteration > start_iteration:
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
            if uncertainty_distillation and iteration > start_iteration:
                _, predicted = outputs_cp.max(1)
            else:
                _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        if epoch % 40 == 0:
            if iteration == start_iteration:
                print('Train set: {}, Train Loss: {:.4f} Acc: {:.4f}'.format( \
                    len(trainloader), train_loss / (batch_idx + 1), 100. * correct / total))
            else:
                print('Train set: {}, Train Loss1: {:.4f}, Train Loss2: {:.4f},\
                    Train Loss: {:.4f} Acc: {:.4f}'.format(len(trainloader), \
                                                            train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                                                            train_loss / (batch_idx + 1), 100. * correct / total))

    # if add unlabeled data, start unlabeled iteration.
    total_unlabeled_selected = 0  # total number of unlabeled data selected so far.
    if iteration > start_iteration and unlabeled_data is not None:
        for u_i in range(unlabeled_iteration):
            if total_unlabeled_selected < unlabeled_num_selected:
                num_unlabeled = 10  # number of unlabeled data selected from every epoch.
                num_unlabeled = min(num_unlabeled, unlabeled_data.shape[0],
                                    unlabeled_num_selected - total_unlabeled_selected)
                if num_unlabeled < nb_cl:
                    break

                unlabeled_selected = []
                unlabeled_selected_l = []
                selected_idx = []
                #total max_values and max_stds
                max_values = []
                max_indices = []
                max_indices_all = []
                max_stds = []
                outputs_unlabeled = []
                # for class-balance self-train
                if method == "self_train":
                    for batch_idx, inputs in enumerate(unlabeled_trainloader):
                        inputs = inputs.to(device)

                        outputs = tg_model(inputs)
                        out_prob = F.softmax(outputs, dim=1)
                        # [[session1],[session2],[session3],.....]
                        outputs_new = out_prob[:, iteration * nb_cl: (iteration + 1) * nb_cl]
                        max_value, max_idx = torch.max(outputs_new, dim=1)
                        max_value_all, max_idx_all = torch.max(out_prob, dim=1)
                        if batch_idx == 0:

                            max_values = max_value
                            max_indices = max_idx
                            max_indices_all = max_idx_all
                            outputs_unlabeled = outputs
                        else:

                            max_values = torch.cat((max_values, max_value), 0)
                            max_indices = torch.cat((max_indices, max_idx), 0)
                            max_indices_all = torch.cat((max_indices_all, max_idx_all), 0)
                            outputs_unlabeled = torch.cat((outputs_unlabeled, outputs), 0)

                    print('for class-balance selection')
                    for c_i in range(nb_cl):
                        idx_cl = [i for (i, value) in enumerate(max_indices) if value == c_i]
                        max_values_cl = max_values[idx_cl]
                        if len(idx_cl) <= int(num_unlabeled/nb_cl):
                            if c_i == 0:
                                same_indices = idx_cl
                            else:
                                same_indices = np.concatenate((same_indices, idx_cl), axis=0)
                        else:
                            idx_cl = np.array(idx_cl)
                            max_values_cl_sorted_idx = np.argsort(-max_values_cl.detach().cpu().numpy())  # descending order
                            selected_cl_idx = idx_cl[max_values_cl_sorted_idx[:int(num_unlabeled/nb_cl)]]
                            if c_i == 0:
                                same_indices = selected_cl_idx
                            else:
                                same_indices = np.concatenate((same_indices, selected_cl_idx), axis=0)

                    same_indices = same_indices.astype(int)
                    unlabeled_selected = unlabeled_data[same_indices]
                    unlabeled_selected_l = iteration * nb_cl + max_indices[same_indices]
                    num_unlabeled = len(same_indices)
                    selected_idx = same_indices


                if num_unlabeled > 0:
                    total_unlabeled_selected += num_unlabeled
                    print('the total number of unlabeled data selected is {}'.format(total_unlabeled_selected))
                    unlabeled_data = np.delete(unlabeled_data, selected_idx, axis=0)
                    unlabeled_gt = np.delete(unlabeled_gt, selected_idx, axis=0)
                    unlabeled_selected = np.array(unlabeled_selected)
                    unlabeled_selected_l = np.array(unlabeled_selected_l.cpu().numpy())

                    # add unlabeled data to prototypes and prototypes_flag for computing class-means
                    if update_unlabeled:
                        for i in range(len(unlabeled_selected_l)):
                            prototypes[unlabeled_selected_l[i]] = np.append(prototypes[unlabeled_selected_l[i]], unlabeled_selected[i])
                            prototypes_flag[unlabeled_selected_l[i]] = np.append(prototypes_flag[unlabeled_selected_l[i]], 0)
                            prototypes_on_flag[unlabeled_selected_l[i]] = np.append(prototypes_on_flag[unlabeled_selected_l[i]], 0)

                    # add unlabeled data to trainset
                    ################################

                    trainset_1 = BaseDataset("train", 224, label2id)
                    trainset_1.data = np.concatenate([trainset.data, unlabeled_selected])
                    trainset_1.targets = np.concatenate([trainset.targets, unlabeled_selected_l])
                    trainloader = torch.utils.data.DataLoader(trainset_1, batch_size=train_batch_size, shuffle=True, num_workers=4)

                    for epoch in range(10):
                        tg_model.train()
                        if fix_bn:
                            for m in tg_model.modules():
                                if isinstance(m, nn.BatchNorm2d):
                                    m.eval()
                        tg_lr_scheduler.step()
                        for batch_idx, (inputs, targets) in enumerate(trainloader):
                            inputs, targets = inputs.to(device), targets.to(device)
                            tg_optimizer.zero_grad()
                            outputs = tg_model(inputs)
                            if iteration == start_iteration:

                                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets.long())
                            else:
                                ref_outputs = ref_model(inputs)
                                loss1 = nn.KLDivLoss()(F.log_softmax(outputs[:, :num_old_classes] / T, dim=1),
                                                     F.softmax(ref_outputs.detach() / T, dim=1)) * T * T * beta * num_old_classes
                                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets.long())
                                loss = loss1 + loss2
                            loss.backward()
                            tg_optimizer.step()

                            train_loss += loss.item()
                            if iteration > start_iteration:
                                train_loss1 += loss1.item()
                                train_loss2 += loss2.item()
                            _, predicted = outputs.max(1)
                            total += targets.size(0)
                            correct += predicted.eq(targets).sum().item()

                    if epoch % 40 == 0:
                        if iteration == start_iteration:
                            print('Train set: {}, Train Loss: {:.4f} Acc: {:.4f}'.format( \
                                len(trainloader), train_loss / (batch_idx + 1), 100. * correct / total))
                        else:
                            print('Train set: {}, Train Loss1: {:.4f}, Train Loss2: {:.4f},\
                                Train Loss: {:.4f} Acc: {:.4f}'.format(len(trainloader), \
                                                                       train_loss1 / (batch_idx + 1),
                                                                       train_loss2 / (batch_idx + 1),
                                                                       train_loss / (batch_idx + 1),
                                                                       100. * correct / total))

                if unlabeled_data.shape[0] < 1:
                    unlabeled_data = None
                else:
                    unlabeled_trainset = UnlabelDataset(image_size)
                    unlabeled_trainset.data = unlabeled_data
                    unlabeled_trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=train_batch_size,
                                                                    shuffle=False, num_workers=4)

        # eval
        tg_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = tg_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        if epoch % 10 == 0:
            print('Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format( \
                len(testloader), test_loss / (batch_idx + 1), 100. * correct / total))
    return tg_model


