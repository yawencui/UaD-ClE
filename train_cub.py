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
import sys
import copy
import argparse
from PIL import Image
import torch
import utils_pytorch
from utils_incremental.compute_features import compute_features
from utils_incremental.compute_accuracy import compute_accuracy
from utils_incremental.compute_confusion_matrix import compute_confusion_matrix
from utils_incremental.incremental_train_and_eval import incremental_train_and_eval
from resnet import resnet18
import pickle
import os
import random
import pdb
from dataloder import BaseDataset, BaseDataset_flag, BaseDataset_flip, UnlabelDataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='dataset', type=str)
parser.add_argument('--dataset', default='cub', type=str)
parser.add_argument('--num_classes', default=200, type=int)
parser.add_argument('--nb_cl_fg', default=100, type=int,
                    help='the number of classes in first session')
parser.add_argument('--nb_cl', default=10, type=int,
                    help='Classes per group')
parser.add_argument('--nb_protos', default=20, type=int,
                    help='Number of prototypes per class at the end')
parser.add_argument('--k_shot', default=5, type=int,
                    help='')
parser.add_argument('--ckp_prefix', default=os.path.basename(sys.argv[0])[:-3], type=str,
                    help='Checkpoint prefix')
parser.add_argument('--epochs', default=160, type=int,
                    help='Epochs for first sesssion')
parser.add_argument('--T', default=2, type=float,
                    help='Temperature for distialltion')
parser.add_argument('--beta', default=0.25, type=float,
                    help='Beta for distialltion')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume from checkpoint')
parser.add_argument('--rs_ratio', default=0.0, type=float,
                    help='The ratio for resample')

parser.add_argument('--unlabeled_iteration', default=100, type=int,
                    help='the total iteration to add unlabeled data')
parser.add_argument('--update_unlabeled', action='store_true', default=True,
                    help='if using selected unlabled data to update the class_mean')
parser.add_argument('--use_nearest_mean', action='store_true', default=True,
                    help='if using nearest-mean-of-examplars classification for selecting unlabeled data')
parser.add_argument('--unlabeled_num', default=300, type=int,
                    help='The total number for resample')
parser.add_argument('--unlabeled_num_selected', default=160, type=int,
                    help='The number of selected unlabeled data')
parser.add_argument('--random_seed', default=1993, type=int,
                    help='random seed')

parser.add_argument('--method', default='self_train', type=str,
                    choices=['self_train', 'random'],
                    help='the method for adding unlabeled data')

parser.add_argument('--uncertainty_distillation', action='store_true', default=False,
                    help='if uncertainty distillation')

parser.add_argument('--flip_on_means', action='store_true', default=False,
                    help='if flip when computing class-means')

parser.add_argument('--base_lamda', default=2, type=int,
                    help='the base weight for distillation loss')

parser.add_argument('--u_t', default=3/5, type=int,
                    help='the threshold in uncertainty estimation')

parser.add_argument('--adapt_lamda', action='store_true', default = False,
                    help='adaptive weight for distillation loss')

parser.add_argument('--frozen_backbone_part', action='store_true', default = False,
                    help='if freeze part of the backbone')

args = parser.parse_args()
assert (args.nb_cl_fg % args.nb_cl == 0)
assert (args.nb_cl_fg >= args.nb_cl)
train_batch_size = 32  # Batch size for train
test_batch_size = 50  # Batch size for test (original 100)
eval_batch_size = 32  # Batch size for eval
base_lr = 1e-3 # Initial learning rate
lr_strat = [80, 120]  # Epochs where learning rate gets decreased
lr_factor = 0.1 # Learning rate decrease factor
custom_weight_decay = 1e-4  # Weight Decay
custom_momentum = 0.9  # Momentum
args.ckp_prefix = '{}_nb_cl_fg_{}_nb_cl_{}_nb_protos_{}'.format(args.ckp_prefix, args.nb_cl_fg, args.nb_cl, args.nb_protos)
np.random.seed(args.random_seed)  # Fix the random seed
print(args)

#device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device = torch.cuda.current_device()

dictionary_size = 30

label2id = utils_pytorch.get_label2id("./dataset/cub/split/label_name.txt")

trainset_data, trainset_targets = utils_pytorch.get_data_file("./dataset/cub/split/train.txt",
                                                              "./dataset/cub/",
                                                              label2id)

testset_data, testset_targets = utils_pytorch.get_data_file("./dataset/cub/split/test.txt",
                                                            "./dataset/cub/",
                                                            label2id)
X_train_total = np.array(trainset_data)
Y_train_total = np.array(trainset_targets)

X_valid_total = np.array(testset_data)
Y_valid_total = np.array(testset_targets)

order_name = "./checkpoint/seed_{}_{}_order_run.pkl".format(args.random_seed, args.dataset)
id2label = {index: la for la, index in label2id.items()}
print("Order name:{}".format(order_name))
order = np.array([i for i in id2label.keys()])
print(order)
# np.random.shuffle(order)
order_list = list(order)
print(order_list)

X_valid_cumuls = []
X_protoset_cumuls = []
X_train_cumuls = []
Y_valid_cumuls = []
Y_protoset_cumuls = []
Y_train_cumuls = []

X_valid_cumuls_base = []
Y_valid_cumuls_base = []
X_valid_cumuls_novel = []
Y_valid_cumuls_novel = []

# alpha_dr_herding = np.zeros((int(args.num_classes / args.nb_cl), dictionary_size, args.nb_cl), np.float32)

# The following contains all the training samples of the different classes
# because we want to compare our method with the theoretical case where all the training samples are stored
prototypes = [[] for i in range(args.num_classes)]
prototypes_flag = [[] for i in range(args.num_classes)]
prototypes_on_flag = [[] for i in range(args.num_classes)]
for orde in range(args.num_classes):
    prototypes[orde] = X_train_total[np.where(Y_train_total == order[orde])]
    prototypes_flag[orde] = np.ones(len(prototypes[orde]), dtype = int)
    if orde < 100:
        prototypes_on_flag[orde] = np.ones(len(prototypes[orde]), dtype=int)
    else:
        prototypes_on_flag[orde] = np.zeros(len(prototypes[orde]), dtype=int)

# prototypes = np.array(prototypes)

start_session = int(args.nb_cl_fg / args.nb_cl) - 1

alpha_dr_herding = []
#for i in range(int(args.num_classes / args.nb_cl)):
#    if i > start_session:
#        alpha_dr_herding.append(np.zeros((args.nb_cl, args.k_shot), np.float32))
#    else:
#        alpha_dr_herding.append(np.zeros((args.nb_cl, dictionary_size), np.float32))

for session in range(start_session, int(args.num_classes / args.nb_cl)):

    if session == start_session:
        #args.rs_ratio = 0.2
        ############################################################
        last_iter = 0
        ############################################################
        if args.resume:
            print('resume the results of first session')
            ckp_name = './checkpoint/{}_epochs_{}_iteration_{}_model.pth'.format(args.dataset, args.epochs, session)
            tg_model = torch.load(ckp_name)
            ref_model = None
            args.epochs = 0
        else:
            tg_model = resnet18(num_classes=args.nb_cl_fg, pretrained=True)
            ref_model = None
    else:
        #args.rs_ratio = 0.99
        last_iter = session
        ############################################################
        # increment classes
        ref_model = copy.deepcopy(tg_model)
        in_features = tg_model.fc.in_features
        out_features = tg_model.fc.out_features
        new_fc = nn.Linear(in_features, out_features + args.nb_cl)
        new_fc.weight.data[:out_features] = tg_model.fc.weight.data
        new_fc.bias.data[:out_features] = tg_model.fc.bias.data
        tg_model.fc = new_fc

    train_file = os.path.join(args.data_dir, args.dataset, "split", "session_{}.txt".format(session - start_session+1))
    test_file = os.path.join(args.data_dir, args.dataset, "split", "test_{}.txt".format(session - start_session+1))

    unlabeled_data = None
    unlabeled_gt = None
    if session > start_session:
        args.epochs = 60
        base_lr = 0.0005
        print('the learning rate is {}'.format(base_lr))
        unlabeled_file = os.path.join(args.data_dir, args.dataset, "split", "unlabeled_{}.txt".format(session - start_session+1))
        unlabeled_data, unlabeled_gt = utils_pytorch.get_data_file(unlabeled_file, "./dataset/cub/", label2id, unlabel=False)  #unlabeled=True
        random.shuffle(unlabeled_data)
        if args.unlabeled_num == 0:
            unlabeled_data=None
            unlabeled_gt=None
        elif args.unlabeled_num == -1:
            unlabeled_data=unlabeled_data
            unlabeled_gt=unlabeled_gt
        else:
            try:
                unlabeled_data = unlabeled_data[:args.unlabeled_num]
                unlabeled_gt = unlabeled_gt[:args.unlabeled_num]
            except:
                unlabeled_data = unlabeled_data
                unlabeled_gt == unlabeled_gt

    X_train, Y_train = utils_pytorch.get_data_file(train_file, "./dataset/cub/", label2id)
    X_valid,  Y_valid = utils_pytorch.get_data_file(test_file, "./dataset/cub/", label2id)

    X_valid_cumuls.append(X_valid)
    X_train_cumuls.append(X_train)
    X_valid_cumul = np.concatenate(X_valid_cumuls)
    X_train_cumul = np.concatenate(X_train_cumuls)

    Y_valid_cumuls.append(Y_valid)
    Y_train_cumuls.append(Y_train)
    Y_valid_cumul = np.concatenate(Y_valid_cumuls)
    Y_train_cumul = np.concatenate(Y_train_cumuls)

    if session == start_session:
        X_valid_ori = X_valid
        Y_valid_ori = Y_valid
        X_flag = []
        X_on_flag = []
        for cls_id in range(0, (session + 1) * args.nb_cl):
            X_flag = np.append(X_flag, prototypes_flag[cls_id])
            X_on_flag = np.append(X_on_flag, prototypes_on_flag[cls_id])

        X_valid_cumuls_base = X_valid
        Y_valid_cumuls_base = Y_valid
    else:

        X_protoset = np.concatenate(X_protoset_cumuls)
        Y_protoset = np.concatenate(Y_protoset_cumuls)
        X_protoset_flag = np.concatenate(X_protoset_cumuls_flag)
        X_protoset_on_flag = np.concatenate(X_protoset_cumuls_on_flag)
        X_current_flag = []
        X_current_on_flag = []
        for cls_id in range(session * args.nb_cl, (session + 1) * args.nb_cl):
            X_current_flag = np.append(X_current_flag, prototypes_flag[cls_id])
            X_current_on_flag = np.append(X_current_on_flag, prototypes_on_flag[cls_id])
        X_current_flag = np.array(X_current_flag)
        X_current_on_flag = np.array(X_current_on_flag)

        if args.rs_ratio > 0:
            # 1/rs_ratio = (len(X_train)+len(X_protoset)*scale_factor)/(len(X_protoset)*scale_factor)
            scale_factor = (len(X_train) * args.rs_ratio) / (len(X_protoset) * (1 - args.rs_ratio))
            rs_sample_weights = np.concatenate((np.ones(len(X_train)), np.ones(len(X_protoset)) * scale_factor))
            # number of samples per epoch, undersample on the new classes
            # rs_num_samples = len(X_train) + len(X_protoset)
            rs_num_samples = int(len(X_train) / (1 - args.rs_ratio))
            print("X_train:{}, X_protoset:{}, rs_num_samples:{}".format(len(X_train), len(X_protoset), rs_num_samples))

        X_train = np.concatenate((X_train, X_protoset), axis=0)
        Y_train = np.concatenate((Y_train, Y_protoset))
        X_flag = np.concatenate((X_protoset_flag, X_current_flag))
        X_on_flag = np.concatenate((X_protoset_on_flag, X_current_on_flag))


        X_valid_cumuls_novel.append(X_valid)
        Y_valid_cumuls_novel.append(Y_valid)
        X_valid_cumul_novel = np.concatenate(X_valid_cumuls_novel)
        Y_valid_cumul_novel = np.concatenate(Y_valid_cumuls_novel)
        ###################

    print('Batch of classes number {0} arrives ...'.format(session))

    ############################################################

    trainset = BaseDataset_flag("train", 224, label2id)
    trainset.data = X_train
    trainset.targets = Y_train
    trainset.flags = X_flag
    trainset.on_flags = X_on_flag

    if session > start_session and args.rs_ratio > 0 and scale_factor > 1:

        index1 = np.where(rs_sample_weights > 1)[0]
        index2 = np.where(Y_train < session * args.nb_cl)[0]
        assert ((index1 == index2).all())
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(rs_sample_weights, rs_num_samples)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                                  shuffle=False, sampler=train_sampler, num_workers=4)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                                  shuffle=True, num_workers=4)
    testset = BaseDataset("test", 224, label2id)
    testset.data = X_valid_cumul
    testset.targets = Y_valid_cumul
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                             shuffle=False, num_workers=4)
    print('Max and Min of train labels: {}, {}'.format(min(Y_train), max(Y_train)))
    print('Max and Min of valid labels: {}, {}'.format(min(Y_valid_cumul), max(Y_valid_cumul)))

    ##############################################################
    ckp_name = './checkpoint/{}_iteration_{}_model.pth'.format(args.ckp_prefix, session)
    print('ckp_name', ckp_name)

    if args.frozen_backbone_part:
        print('freeze part of the backbone')
        for name, param in tg_model.named_parameters():
            if name == 'conv1.weight' or name == 'bn1.weight' or name == 'bn1.bias':
                param.requires_grad = False
            else:
                if name[0:6] == 'layer1' or name[0:6] == 'layer2' or name[0:6] == 'layer3':
                    param.requires_grad = False
                else:
                    print(name)
        tg_params = filter(lambda p: p.requires_grad, tg_model.parameters())
    else:
        tg_params = tg_model.parameters()

    tg_model = tg_model.to(device)
    if session > start_session:
        ref_model = ref_model.to(device)
        #base_lr = 0.01
        print('the learning rate is {}'.format(base_lr))

    tg_optimizer = optim.SGD(tg_params, lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
    tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)
    tg_model = incremental_train_and_eval(args.base_lamda, args.adapt_lamda, args.u_t, label2id, args.uncertainty_distillation, prototypes, prototypes_flag, prototypes_on_flag, args.update_unlabeled, args.epochs, args.method, args.unlabeled_num, args.unlabeled_iteration, args.unlabeled_num_selected, train_batch_size, tg_model, ref_model, tg_optimizer, tg_lr_scheduler,
                                          trainloader, testloader,
                                          session, start_session,
                                          args.T, args.beta, unlabeled_data, unlabeled_gt, args.nb_cl, trainset, 224,
                                          device=device)
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(tg_model, ckp_name)
    if args.resume == False:
        if session == start_session:
            ckp_name = './checkpoint/{}_epochs_{}_iteration_{}_model.pth'.format(args.dataset, args.epochs, session)
            torch.save(tg_model, ckp_name)

    nb_protos_cl = args.nb_protos
    tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
    num_features = tg_model.fc.in_features
    # Herding
    print('Updating exemplar set...')
    dr_herding = []
    for iter_dico in range(last_iter * args.nb_cl, (session + 1) * args.nb_cl):
        # Possible exemplars in the feature space and projected on the L2 sphere

        evalset = BaseDataset("test", 224, label2id)
        evalset.data = prototypes[iter_dico]
        evalset.targets = np.zeros(len(evalset))  # zero labels
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                 shuffle=False, num_workers=4)
        num_samples = len(evalset)
        mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features, device=device)
        D = mapped_prototypes.T
        D = D / np.linalg.norm(D, axis=0)

        '''
        # Herding procedure : ranking of the potential exemplars      
        mu = np.mean(D, axis=1)
        index1 = int(iter_dico / args.nb_cl)
        index2 = iter_dico % args.nb_cl
        alpha_dr_herding[index1][index2] = alpha_dr_herding[index1][index2] * 0
        w_t = mu
        iter_herding = 0
        iter_herding_eff = 0
        while not (np.sum(alpha_dr_herding[index1][index2] != 0) == min(nb_protos_cl, 500)) and iter_herding_eff < 1000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if alpha_dr_herding[index1][index2][ind_max] == 0:
                alpha_dr_herding[index1][index2][ind_max]= 1 + iter_herding
                iter_herding += 1
            w_t = w_t + mu - D[:, ind_max]
        '''
        herding = np.zeros(len(prototypes[iter_dico]), np.float32)
        dr_herding.append(herding)
        # Herding procedure : ranking of the potential exemplars
        mu = np.mean(D, axis=1)
        index1 = int(iter_dico / args.nb_cl)
        index2 = iter_dico % args.nb_cl
        dr_herding[index2] = dr_herding[index2] * 0
        w_t = mu
        iter_herding = 0
        iter_herding_eff = 0
        while not (np.sum(dr_herding[index2] != 0) == min(nb_protos_cl, 500)) and iter_herding_eff < 1000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if dr_herding[index2][ind_max] == 0:
                dr_herding[index2][ind_max] = 1 + iter_herding
                iter_herding += 1
            w_t = w_t + mu - D[:, ind_max]

        if (iter_dico + 1) % args.nb_cl == 0:
            alpha_dr_herding.append(np.array(dr_herding))
            dr_herding = []

    X_protoset_cumuls = []
    X_protoset_cumuls_flag = []
    X_protoset_cumuls_on_flag = []
    Y_protoset_cumuls = []

        # Class means for iCaRL and NCM + Storing the selected exemplars in the protoset
    print('Computing mean-of-exemplars...')
    class_means = np.zeros((512, 200, 3))
    for iteration2 in range(session+1):
        for iter_dico in range(args.nb_cl):
            current_cl = order[range(iteration2*args.nb_cl, (iteration2+1)*args.nb_cl)]

            # Collect data in the feature space for each class
            evalset = BaseDataset("test", 224, label2id)
            evalset.data = prototypes[iteration2*args.nb_cl+iter_dico]
            evalset.targets = np.zeros(evalset.data.shape[0]) #zero labels
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                    shuffle=False, num_workers=4)
            num_samples = evalset.data.shape[0]
            mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features, device=device)
            D = mapped_prototypes.T
            D = D/np.linalg.norm(D,axis=0)
            # Flipped version also
            evalset.data = prototypes[iteration2*args.nb_cl+iter_dico]
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                    shuffle=False, num_workers=4)
            mapped_prototypes2 = compute_features(tg_feature_model, evalloader, num_samples, num_features,device=device)
            D2 = mapped_prototypes2.T
            D2 = D2/np.linalg.norm(D2,axis=0)

            # iCaRL
            alph = alpha_dr_herding[iteration2][iter_dico]
            alph = (alph>0)*(alph<nb_protos_cl+1)*1.
            X_protoset_cumuls.append(prototypes[iteration2*args.nb_cl+iter_dico][np.where(alph==1)[0]])
            X_protoset_cumuls_flag.append(prototypes_flag[iteration2 * args.nb_cl + iter_dico][np.where(alph == 1)[0]])
            X_protoset_cumuls_on_flag.append(prototypes_on_flag[iteration2 * args.nb_cl + iter_dico][np.where(alph == 1)[0]])
            Y_protoset_cumuls.append(order[iteration2*args.nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0])))
            alph = alph/np.sum(alph)
            class_means[:,current_cl[iter_dico],0] = (np.dot(D,alph)+np.dot(D2,alph))/2
            class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico],0])

            # Normal NCM

            if iteration2 > start_session:
                alph = np.ones(len(prototypes[iteration2*args.nb_cl+iter_dico])) / len(prototypes[iteration2*args.nb_cl+iter_dico])
            else:
                alph = np.ones(dictionary_size) / dictionary_size

            class_means[:,current_cl[iter_dico],1] = (np.dot(D,alph)+np.dot(D2,alph))/2
            class_means[:,current_cl[iter_dico],1] /= np.linalg.norm(class_means[:,current_cl[iter_dico],1])

            # dividing labeled and unlabeled and compute class-means
            if iteration2 > start_session:
                alph = np.zeros(len(prototypes[iteration2*args.nb_cl+iter_dico]))
                num_labeled = np.sum(prototypes_flag[iteration2*args.nb_cl+iter_dico], axis=0)
                num_unlabeled = len(prototypes[iteration2*args.nb_cl+iter_dico]) - num_labeled
                alph_labeled = 2 / (2 * num_labeled + num_unlabeled)
                alph_unlabeled = 1 / (2 * num_labeled + num_unlabeled)
                for i in range(len(prototypes[iteration2*args.nb_cl+iter_dico])):
                    if prototypes_flag == 1:
                        alph[i] = alph_labeled
                    else:
                        alph[i] = alph_unlabeled
            else:
                alph = np.ones(dictionary_size) / dictionary_size

            class_means[:, current_cl[iter_dico], 2] = (np.dot(D, alph) + np.dot(D2, alph)) / 2
            class_means[:, current_cl[iter_dico], 2] /= np.linalg.norm(class_means[:, current_cl[iter_dico], 0])

    torch.save(class_means, './checkpoint/{}_run_iteration_{}_class_means.pth'.format(args.ckp_prefix, session))

    current_means = class_means[:, order[range(0, (session+1)*args.nb_cl)]]

    print('Computing cumulative accuracy...')
    evalset = BaseDataset("test", 224, label2id)
    evalset.data = X_valid_cumul
    evalset.targets = Y_valid_cumul
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                shuffle=False, num_workers=4)
    cumul_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader, device=device)

    if session > start_session:

        print('Computing the accuracy of base classes...')
        evalset = BaseDataset("test", 224, label2id)
        evalset.data = X_valid_cumuls_base
        evalset.targets = Y_valid_cumuls_base
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                 shuffle=False, num_workers=4)
        cumul_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader, device=device)

        print('Computing the accuracy of novel classes...')
        evalset = BaseDataset("test", 224, label2id)
        evalset.data = X_valid_cumul_novel
        evalset.targets = Y_valid_cumul_novel
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                 shuffle=False, num_workers=4)
        cumul_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader, device=device)
