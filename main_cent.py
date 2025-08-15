#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
import numpy as np
matplotlib.use('Agg')
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils.options import args_parser
from utils.base import FedBase
from utils.network import ShuffleNetV2
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch import nn

class Cent:
    def __init__(self, args):
        self.args = args
        self.nn = ShuffleNetV2().to(self.args.device)
        self.fedbase = FedBase()
        self.local_datasets, self.test_datasets = self.fedbase.create_signal_datasets(num_clients=8, num_pos=3)
        self.loss = []
        self.acc = []

    def server(self):
        # communication
        data, label = self.dispatch()
        for t in tqdm(range(self.args.epochs)):
            print('round', t + 1, ":")
            # server updating
            self.nn, loss_epoch, acc_epoch = self.train(data, label)
            self.loss.append(loss_epoch)
            self.acc.append(acc_epoch)

        np.savetxt(f"./save/cent_loss.csv", np.array(self.loss))
        np.savetxt(f"./save/cent_acc.csv", np.array(self.acc))
        torch.save(self.nn, f"./centsei/centmodel.pth")

    def dispatch(self):
        data = []
        label = []
        for idx_pos in range(self.args.num_pos):
            for idx_user in range(self.args.num_users):
                data.append(self.local_datasets[idx_pos * self.args.num_users + idx_user].signals)
                label.append(self.local_datasets[idx_pos * self.args.num_users + idx_user].labels)
        data = np.concatenate(data)
        label = np.concatenate(label)
        return data, label

    def train(self, data, label):
        self.nn.train()
        cent_dataset = TensorDataset(torch.Tensor(data), torch.Tensor(label))
        ldr_train = DataLoader(cent_dataset, batch_size=self.args.local_bs, shuffle=True)
        loss_function = nn.CrossEntropyLoss().to(self.args.device)
        optimizer = torch.optim.SGD(self.nn.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        lr_step = StepLR(optimizer, step_size=10, gamma=0.1)
        epoch_loss = []
        epoch_acc = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            correct = 0
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.squeeze().long().to(self.args.device)
                log_probs = self.nn(images)
                if log_probs.shape[0] == 1: labels = labels.unsqueeze(0)
                loss = loss_function(log_probs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
            accuracy = 100.00 * correct / len(ldr_train.dataset)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(accuracy)

            lr_step.step()

            print(f'Update Epoch: {iter} [{iter}/{self.args.local_ep}]\tLoss: {sum(batch_loss) / len(batch_loss)}, Acc: {accuracy}% ({correct}/{len(ldr_train.dataset)})')

        return self.nn, sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)

    def test(self):
        # model = self.nn
        model = torch.load("./centsei/centmodel.pth")
        model.eval()
        for idx_pos in range(0, self.args.num_pos):
            data_loader = DataLoader(self.test_datasets[idx_pos],batch_size=self.args.bs)
            test_loss = 0
            correct = 0
            for idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.args.device), target.squeeze().long().to(self.args.device)
                log_probs = model(data)
                if log_probs.shape[0] == 1: target = target.unsqueeze(0)
                test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            test_loss /= len(data_loader.dataset)
            accuracy = 100.00 * correct / len(data_loader.dataset)

            print("Testing accuracy in pos {}: {:.2f}".format(idx_pos, accuracy))


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(args.seed)

    cent = Cent(args)
    cent.server()
    cent.test()