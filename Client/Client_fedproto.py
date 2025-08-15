#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy as np

def train(args, dataset, ann, global_protos, idx_user, idx_pos):
    ann.train()
    ldr_train = DataLoader(dataset, batch_size=args.local_bs, shuffle=True)
    loss_function = nn.CrossEntropyLoss().to(args.device)
    mse_function = nn.MSELoss().to(args.device)
    optimizer = torch.optim.SGD(ann.parameters(), lr=args.lr, momentum=args.momentum)
    lr_step = StepLR(optimizer, step_size=10, gamma=0.1)
    epoch_loss = []
    epoch_acc = []
    for iter in range(args.local_ep):
        batch_loss = []
        correct = 0
        agg_protos_label = {}
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.squeeze().long().to(args.device)
            log_probs, protos = ann(images, return_features=True)

            # proto normalization
            # protos = F.normalize(protos, p=2, dim=1)

            if log_probs.shape[0] == 1: labels = labels.unsqueeze(0)
            loss_classifier = loss_function(log_probs, labels)

            if len(global_protos) == 0:
                loss_protos = 0 * loss_classifier
            else:
                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                loss_protos = mse_function(proto_new, protos)

            loss = loss_classifier + args.ld * loss_protos
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss_classifier.item())

            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()

            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label:
                    agg_protos_label[labels[i].item()].append(protos[i,:])
                else:
                    agg_protos_label[labels[i].item()] = [protos[i,:]]

        accuracy = 100.00 * correct / len(ldr_train.dataset)
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        epoch_acc.append(accuracy)

        lr_step.step()

        print(f'Device {idx_user} of Position {idx_pos}, Update Epoch: {iter} [{iter}/{args.local_ep}]\tLoss: {sum(batch_loss) / len(batch_loss)}, Acc: {accuracy}% ({correct}/{len(ldr_train.dataset)})')
    torch.save(ann, f"./decentsei_fedproto_frac{args.frac}_epoch{args.epochs}_subepoch{args.local_ep}_ld{args.ld}/localmodel_pos{idx_pos}_user{idx_user}.pth")

    return ann, sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc), agg_protos_label

def test(args, ann, dataset):
    ann.eval()
    test_loss = 0
    correct = 0
    y_preds = []
    y_reals = []
    data_loader = DataLoader(dataset, batch_size=args.bs)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.squeeze().long().to(args.device)
        log_probs = ann(data)
        if log_probs.shape[0] == 1: target = target.unsqueeze(0)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        y_preds.append(y_pred.cpu())
        y_reals.append(target.cpu())

    y_preds = np.concatenate(y_preds)
    y_reals = np.concatenate(y_reals)

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)

    return test_loss, accuracy, y_preds, y_reals

def obtain_feature_map(args, ann, dataset):
    ann.eval()
    y_reals = []
    protos = []
    data_loader = DataLoader(dataset, batch_size=args.bs)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.squeeze().long().to(args.device)
        _, proto = ann(data, return_features=True)
        if _.shape[0] == 1: target = target.unsqueeze(0)

        protos.append(proto.detach().cpu())
        y_reals.append(target.cpu())

    protos = np.concatenate(protos)
    y_reals = np.concatenate(y_reals)

    return protos, y_reals


