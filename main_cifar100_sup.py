import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import os
import os.path
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random
import pdb
import argparse
import time
import math
from copy import deepcopy

from scipy.stats import wasserstein_distance

def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))


class Conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(Conv2d, self).__init__(in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     bias=bias)

    def forward(self, input, task_id, p, epoch):
        if p is not None:
            if epoch == 1:
                sz = self.weight.grad.data.size(0)
                norm_project = torch.mm(p, p.transpose(1, 0))
                proj_weight = torch.mm(self.weight.view(sz, -1),
                                       norm_project).view(self.weight.size())
                masked_weight = self.weight - proj_weight
            else:
                masked_weight = self.weight
        else:
            masked_weight = self.weight
        return F.conv2d(input, masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)

    def forward(self, input, task_id, p, epoch):
        if p is not None:
            if epoch == 1:
                norm_project = torch.mm(p, p.transpose(1, 0))
                proj_weight = torch.mm(self.weight, norm_project)

                masked_weight = self.weight - proj_weight
            else:
                masked_weight = self.weight
        else:
            masked_weight = self.weight
        return F.linear(input, masked_weight, self.bias)


class LeNet(nn.Module):
    def __init__(self, taskcla):
        super(LeNet, self).__init__()
        self.act = OrderedDict()
        self.map = []
        self.ksize = []
        self.in_channel = []

        self.map.append(32)
        self.conv1 = Conv2d(3, 20, 5, bias=False, padding=2)

        s = compute_conv_output_size(32, 5, 1, 2)
        s = compute_conv_output_size(s, 3, 2, 1)
        self.ksize.append(5)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = Conv2d(20, 50, 5, bias=False, padding=2)

        s = compute_conv_output_size(s, 5, 1, 2)
        s = compute_conv_output_size(s, 3, 2, 1)
        self.ksize.append(5)
        self.in_channel.append(20)
        self.smid = s
        self.map.append(50*self.smid*self.smid)
        self.maxpool = torch.nn.MaxPool2d(3, 2, padding=1)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0)
        self.drop2 = torch.nn.Dropout(0)
        self.lrn = torch.nn.LocalResponseNorm(4, 0.001/9.0, 0.75, 1)

        self.fc1 = Linear(50*self.smid*self.smid, 800, bias=False)
        self.fc2 = Linear(800, 500, bias=False)
        self.map.extend([800])

        self.taskcla = taskcla
        self.fc3 = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.fc3.append(torch.nn.Linear(500, n, bias=False))

    def forward(self, x, t, p, epoch):
        if p is None:
            bsz = deepcopy(x.size(0))
            self.act['conv1'] = x
            x = self.conv1(x, t, None, epoch)
            x = self.maxpool(self.drop1(self.lrn(self.relu(x))))

            self.act['conv2'] = x
            x = self.conv2(x, t, None, epoch)
            x = self.maxpool(self.drop1(self.lrn(self.relu(x))))

            x = x.reshape(bsz, -1)
            self.act['fc1'] = x
            x = self.fc1(x, t, None, epoch)
            x = self.drop2(self.relu(x))

            self.act['fc2'] = x
            x = self.fc2(x, t, None, epoch)
            x = self.drop2(self.relu(x))

            y = []
            for t, i in self.taskcla:
                y.append(self.fc3[t](x))
        else:
            bsz = deepcopy(x.size(0))
            self.act['conv1'] = x
            x = self.conv1(x, t, p[0], epoch)
            x = self.maxpool(self.drop1(self.lrn(self.relu(x))))

            self.act['conv2'] = x
            x = self.conv2(x, t, p[1], epoch)
            x = self.maxpool(self.drop1(self.lrn(self.relu(x))))

            x = x.reshape(bsz, -1)
            self.act['fc1'] = x
            x = self.fc1(x, t, p[2], epoch)
            x = self.drop2(self.relu(x))

            self.act['fc2'] = x
            x = self.fc2(x, t, p[3], epoch)
            x = self.drop2(self.relu(x))

            y = []
            for t, i in self.taskcla:
                y.append(self.fc3[t](x))
        return y


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == Conv2d or type(m) == Linear:
        torch.nn.init.kaiming_uniform_(
            m.weight, mode='fan_in', nonlinearity='relu')


def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return


def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if (epoch == 1):
            param_group['lr'] = args.lr
        else:
            param_group['lr'] /= args.lr_factor


def train(args, epoch, task_id, model, device, x, y, optimizer, criterion):
    model.train()
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0, len(r), args.batch_size_train):
        if i+args.batch_size_train <= len(r):
            b = r[i:i+args.batch_size_train]
        else:
            b = r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data, task_id, None, -1)
        loss = criterion(output[task_id], target)
        loss.backward()
        optimizer.step()


def contrast_cls(every_task_base, sim_tasks, sim_scores, model, task_id, device, criterion):
    l2 = 0
    cnt = 0
    list_keys = list(model.act.keys())
    for k, (m, params) in enumerate(model.named_parameters()):

        if k < 4 and len(params.size()) != 1:

            sim = []

            for tt in range(task_id):
                sz = params.grad.data.size(0)
                tmp = torch.FloatTensor(every_task_base[tt][cnt]).to(device)
                norm_project = torch.mm(tmp, tmp.transpose(1, 0)).to(device)
                if "conv" in m:
                    proj_weight = torch.mm(params.view(sz, -1),
                                           norm_project).view(params.size())
                else:
                    proj_weight = torch.mm(params, norm_project)
                sim.append(proj_weight.data)
                dis = list(set(range(task_id)) - set(sim_tasks[cnt]))
                tt = random.sample(dis, 1)[0]
                tmp = torch.FloatTensor(every_task_base[tt][cnt]).to(device)
                norm_project = torch.mm(tmp, tmp.transpose(1, 0))
                proj_weight = torch.mm(params.view(sz, -1),
                                       norm_project).view(params.size())
                sim.append(proj_weight.data)
                if len(sim) >= 4:
                    break

            sim = torch.stack(sim).view(4, -1)
            if sum(sim_scores[cnt]) != 2:
                idxs = torch.arange(0, sim.shape[0], device=device)
                y_true = idxs + 1 - idxs % 2 * 2
                similarities = F.cosine_similarity(sim.unsqueeze(1), sim.unsqueeze(0), dim=2)

                similarities = similarities - torch.eye(sim.shape[0], device=device) * 1e12
                similarities = similarities / 0.05

                loss = F.cross_entropy(similarities, y_true)
                l2 += torch.mean(loss)

            cnt += 1

    return l2


def train_projected(args, p, model, device, x, y, optimizer, criterion, feature_mat, task_id, epoch, sim_tasks, sim_scores, every_task_base):
    '''Train for one epoch on the training set'''
    model.train()
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0, len(r), args.batch_size_train):
        if i+args.batch_size_train <= len(r):
            b = r[i:i+args.batch_size_train]
        else:
            b = r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data, task_id, p, i + epoch)
        loss = criterion(output[task_id], target)

        if len(sim_tasks) != 0:
            l2 = contrast_cls(every_task_base, sim_tasks,
                              sim_scores, model, task_id, device, criterion)
            loss += l2

        loss.backward()
        # Gradient Projections
        kk = 0
        for k, (m, params) in enumerate(model.named_parameters()):
            if k < 4 and len(params.size()) != 1:
                sz = params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1),
                                                               feature_mat[kk]).view(params.size())
                kk += 1
            elif (k < 4 and len(params.size()) == 1) and task_id != 0:
                params.grad.data.fill_(0)

        optimizer.step()


def test(args, model, device, x, y, criterion, task_id):
    '''Evaluate the model on the test set'''
    model.eval()
    total_loss = 0
    total_num = 0
    correct = 0
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0, len(r), args.batch_size_test):
            if i+args.batch_size_test <= len(r):
                b = r[i:i+args.batch_size_test]
            else:
                b = r[i:]
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = model(data, task_id, None, -1)
            loss = criterion(output[task_id], target)
            pred = output[task_id].argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc


def get_representation_matrix(task_id, net, device, x, y, old_task_distribution):
    '''Get the representation matrix for the current task'''
    net.eval()
    example_data = []
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    un = torch.unique(y)
    idx = 0
    for _ in range(25):
        b = []
        for i in un:
            while y[idx] != i:
                idx += 1
            b.append(idx)
        assert len(b) == 5
        tmp_data = x[b].to(device)
        target = y[b]
        example_data.append(tmp_data)

    example_data = torch.cat(example_data, dim=0)
    example_data = example_data.to(device)
    example_out = net(example_data, task_id, None, -1)

    batch_list = [2*12, 100, 125, 125]
    pad = 2
    p1d = (2, 2, 2, 2)
    mat_list = []
    act_key = list(net.act.keys())
    # pdb.set_trace()
    for i in range(len(net.map)):
        bsz = batch_list[i]
        k = 0
        if i < 2:
            ksz = net.ksize[i]
            s = compute_conv_output_size(net.map[i], net.ksize[i], 1, pad)
            mat = np.zeros((net.ksize[i]*net.ksize[i]
                           * net.in_channel[i], s*s*bsz))
            act = F.pad(net.act[act_key[i]], p1d,
                        "constant", 0).detach().cpu().numpy()

            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:, k] = act[kk, :, ii:ksz+ii,
                                        jj:ksz+jj].reshape(-1)  # ?
                        k += 1
            mat_list.append(mat)
            old_task_distribution[task_id][i].append(deepcopy(mat.flatten()))
        else:
            act = net.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)
            old_task_distribution[task_id][i].append(deepcopy(activation.flatten()))

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_list)):
        print('Layer {} : {}'.format(i+1, mat_list[i].shape))
    print('-'*30)
    return mat_list


def update_GPM(task_id, model, mat_list, threshold, feature_list=[], proj=None, every_task_base=None):
    '''Update the GPM'''
    print('Threshold: ', threshold)
    if not feature_list:
        # After First Task
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1
            feature_list.append(U[:, 0:r])
            proj[task_id][i] = U[:, 0:r]
            every_task_base[task_id][i] = U[:, 0:r]
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            sval_ratio = (S1**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1
            every_task_base[task_id][i] = U1[:, 0:r]

            act_hat = activation - \
                np.dot(
                    np.dot(feature_list[i], feature_list[i].transpose()), activation)
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)

            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            accumulated_sval = (sval_total-sval_hat)/sval_total

            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r != 0:
                print('Not Skip Updating GPM for layer: {}'.format(i + 1))

                # update GPM
                Ui = np.hstack((feature_list[i], U[:, 0:r]))
                if Ui.shape[1] > Ui.shape[0]:
                    feature_list[i] = Ui[:, 0:Ui.shape[0]]
                else:
                    feature_list[i] = Ui
            if r == 0:
                proj[task_id][i] = proj[task_id-1][i]
            else:
                proj[task_id][i] = U[:, 0:r]

    print('-'*40)
    print('Gradient Constraints Summary')
    print('-'*40)
    for i in range(len(feature_list)):
        print('Layer {} : {}/{}'.format(i+1,
              feature_list[i].shape[1], feature_list[i].shape[0]))
    print('-'*40)
    return feature_list


def main(args):
    tstart = time.time()
    # Device Setting
    device = torch.device("cuda:{}".format(args.cuda)
                          if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Choose any task order - ref {yoon et al. ICLR 2020}
    task_order = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                  np.array([15, 12, 5, 9, 7, 16, 18, 17, 1, 0,
                           3, 8, 11, 14, 10, 6, 2, 4, 13, 19]),
                  np.array([17, 1, 19, 18, 12, 7, 6, 0, 11, 15,
                           10, 5, 13, 3, 9, 16, 4, 14, 2, 8]),
                  np.array([11, 9, 6, 5, 12, 4, 0, 10, 13, 7,
                           14, 3, 15, 16, 8, 1, 2, 19, 18, 17]),
                  np.array([6, 14, 0, 11, 12, 17, 13, 4, 9, 1, 7, 19, 8, 10, 3, 15, 18, 5, 2, 16])]

    # Load CIFAR100_SUPERCLASS DATASET
    from dataloader import cifar100_superclass as data_loader
    data, taskcla = data_loader.cifar100_superclass_python(
        task_order[args.t_order], group=5, validation=True)
    test_data, _ = data_loader.cifar100_superclass_python(
        task_order[args.t_order], group=5)
    print(taskcla)
    n_task = 20
    acc_matrix = np.zeros((20, 20))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0)

    task_id = 0
    task_list = []

    model = LeNet(taskcla).to(device)
    model.apply(init_weights)
    for k_t, (m, param) in enumerate(model.named_parameters()):
        print(k_t, m, param.shape)
    print('-'*40)

    pre_task_distribution = [[[] for j in range(4)] for i in range(n_task)]
    old_task_distribution = [[[] for j in range(4)] for i in range(n_task)]

    task_id = 0
    print("*" * 100)
    print("Get Init Distribution.")
    for k, ncla in taskcla:
        xtrain = data[k]['train']['x']
        ytrain = data[k]['train']['y']
        mat_list = get_representation_matrix(
            task_id, model, device, xtrain, ytrain, pre_task_distribution)
        task_id += 1
    print("*" * 100)
    del model

    proj = {}
    every_task_base = {}
    task_id = 0
    task_list = []

    for k, ncla in taskcla:
        # specify threshold hyperparameter
        threshold = np.array([0.98] * 5) + task_id*np.array([0.001] * 5)

        print('*'*100)
        print('Task {:2d} ({:s})'.format(k, data[k]['name']))
        print('*'*100)
        xtrain = data[k]['train']['x']
        ytrain = data[k]['train']['y']
        xvalid = data[k]['valid']['x']
        yvalid = data[k]['valid']['y']
        xtest = test_data[k]['test']['x']
        ytest = test_data[k]['test']['y']
        task_list.append(k)

        lr = args.lr
        best_loss = np.inf
        print('-'*40)
        print('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        print('-'*40)

        proj[task_id] = {}
        every_task_base[task_id] = {}

        if task_id == 0:
            # Initialize model
            model = LeNet(taskcla).to(device)
            print('Model parameters ---')
            for k_t, (m, param) in enumerate(model.named_parameters()):
                print(k_t, m, param.shape)
            print('-'*40)
            # Initialize model
            model.apply(init_weights)

            best_model = get_model(model)
            feature_list = []
            optimizer = optim.SGD(model.parameters(),
                                  lr=lr, momentum=args.momentum)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.n_epochs)

            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0 = time.time()
                train(args, epoch, task_id, model, device,
                      xtrain, ytrain, optimizer, criterion)
                clock1 = time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,
                                       criterion, k)
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,
                                                                                                 tr_loss, tr_acc, 1000*(clock1-clock0)), end='')
                # Validate
                valid_loss, valid_acc = test(args, model, device, xvalid,
                                             yvalid, criterion, k)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(
                    valid_loss, valid_acc), end='')
                # Adapt lr
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = get_model(model)
                    patience = args.lr_patience
                    print(' *', end='')
                scheduler.step()
                print()

            # Test
            print('-'*40)
            test_loss, test_acc = test(
                args, model, device, xtest, ytest,  criterion, k)
            print('Test: loss={:.3f} , acc={:5.1f}%'.format(
                test_loss, test_acc))
            # Memory Update
            mat_list = get_representation_matrix(
                task_id, model, device, xtrain, ytrain, old_task_distribution)
            feature_list = update_GPM(
                task_id, model, mat_list, threshold, feature_list, proj, every_task_base)

        else:

            sim_tasks = []
            sim_scores = []

            # find top-2 tasks
            if task_id >= 3:
                sim_tasks = [i for i in range(4)]
                sim_scores = [i for i in range(4)]
                # Calculate the distribution of each layer of the current task
                _ = get_representation_matrix(
                    task_id, model, device, xtrain, ytrain, old_task_distribution)
                distribution = [[[]
                                 for j in range(task_id)] for i in range(4)]
                # Calculate the distance between the current task and the previous task
                cnt = 0
                for kk, (m, params) in enumerate(model.named_parameters()):
                    if len(params.size()) != 1 and kk < 4:
                        for tt in range(task_id):
                            # Calculate the distance distribution for each task
                            distribution[cnt][tt] = wasserstein_distance(
                                old_task_distribution[tt][cnt][0], old_task_distribution[task_id][cnt][0])
                        cnt += 1

                cnt = 0
                for kk, (m, params) in enumerate(model.named_parameters()):
                    if len(params.size()) != 1 and kk < 4:
                        for tt in range(task_id):
                            # Calculate the distance distribution for each task
                            t = wasserstein_distance(
                                pre_task_distribution[tt][cnt][0], pre_task_distribution[task_id][cnt][0])
                            if t < distribution[cnt][tt]:
                                distribution[cnt][tt] = 1000
                        cnt += 1

                print(distribution)

                for idx, ii in enumerate(distribution):
                    sim_tasks[idx] = sorted(
                        range(len(ii)), key=lambda i: ii[i])[:2]
                print(sim_tasks)

                # Calculate the similarity score
                for idx, ii in enumerate(distribution):
                    if ii.count(ii[0]) == len(ii) and ii[0] == 1000:
                        tmp = [1, 1]
                    else:
                        tmp = sorted(ii)[:2]
                        t = [sum(tmp) - i for i in tmp]
                        tmp = [i/sum(t) for i in t]
                    sim_scores[idx] = tmp
                print(sim_scores)

                print("*" * 40)
                print("Task {} has sim Tasks".format(task_id), end="")
                cnt = 0
                for kk, (m, params) in enumerate(model.named_parameters()):
                    if len(params.size()) != 1 and kk < 4:
                        print("Layer: {}".format(cnt))
                        print(sim_tasks[cnt], sim_scores[cnt])
                        cnt += 1
                print("*" * 40)

            optimizer = optim.SGD(model.parameters(),
                                  lr=lr, momentum=args.momentum)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.n_epochs)

            feature_mat = []
            # Projection Matrix Precomputation
            for i in range(len(model.act)):
                Uf = torch.Tensor(
                    np.dot(feature_list[i], feature_list[i].transpose())).to(device)
                print('Layer {} - Projection Matrix shape: {}'.format(i+1, Uf.shape))
                feature_mat.append(Uf)
            print('-'*40)

            p = [None, None, None, None]
            if task_id >= 1:
                # Calculate the orthogonal direction of previous task subspace
                for i in range(4):
                    p[i] = torch.FloatTensor(proj[task_id-1][i]).to(device)

            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0 = time.time()
                train_projected(args, p, model, device, xtrain,
                                ytrain, optimizer, criterion, feature_mat, k, epoch, sim_tasks, sim_scores, every_task_base)
                clock1 = time.time()
                tr_loss, tr_acc = test(
                    args, model, device, xtrain, ytrain, criterion, k)
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,
                                                                                                 tr_loss, tr_acc, 1000*(clock1-clock0)), end='')
                # Validate
                valid_loss, valid_acc = test(
                    args, model, device, xvalid, yvalid, criterion, k)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(
                    valid_loss, valid_acc), end='')
                # Adapt lr
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = get_model(model)
                    patience = args.lr_patience
                    print(' *', end='')
                scheduler.step()
                print()

            # Test
            test_loss, test_acc = test(
                args, model, device, xtest, ytest,  criterion, k)
            print('Test: loss={:.3f} , acc={:5.1f}%'.format(
                test_loss, test_acc))
            # Memory Update
            mat_list = get_representation_matrix(
                task_id, model, device, xtrain, ytrain, old_task_distribution)
            feature_list = update_GPM(
                task_id, model, mat_list, threshold, feature_list, proj, every_task_base)

        # save accuracy
        jj = 0
        for ii in task_order[args.t_order][0:task_id+1]:
            xtest = test_data[ii]['test']['x']
            ytest = test_data[ii]['test']['y']
            _, acc_matrix[task_id, jj] = test(
                args, model, device, xtest, ytest, criterion, ii)
            jj += 1
        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t', end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a, j_a]), end='')
            print()
        # update task id
        task_id += 1
    print('-'*50)
    # Simulation Results
    print('Task Order : {}'.format(task_order[args.t_order]))
    print('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean()))
    bwt = np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1])
    print('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    print('-'*50)
    # Plots
    array = acc_matrix
    df_cm = pd.DataFrame(array, index=[i for i in ["1", "2", "3", "4", "5", "6", "7",
                                                   "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]],
                         columns=[i for i in ["1", "2", "3", "4", "5", "6", "7",
                                              "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]])
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid', default=0.05, type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--t_order', type=int, default=0, metavar='TOD',
                        help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=0, metavar='id',
                        help='(default: 0)')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')

    args = parser.parse_args()
    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':', getattr(args, arg))
    print('='*100)

    main(args)
