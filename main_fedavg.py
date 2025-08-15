#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import copy
import numpy as np
from collections import OrderedDict
import torch
from utils.options import args_parser
from utils.base import FedBase
from utils.network import ShuffleNetV2
from Client.Client_fedavg import train, test
from tqdm import tqdm

class FedAvg:
    def __init__(self, args):
        self.args = args
        self.nn = ShuffleNetV2().to(self.args.device)
        self.fedbase = FedBase()
        self.local_datasets, self.test_datasets = self.fedbase.create_signal_datasets(num_clients=self.args.num_users, num_pos=self.args.num_pos)
        self.loss = []
        self.acc = []
        self.nns = []
        for i in range(self.args.num_pos * self.args.num_users):
            temp = copy.deepcopy(self.nn)
            self.nns.append(temp)

    def server(self):
        for t in tqdm(range(self.args.epochs)):
            print('round', t + 1, ":")

            # sampling
            m = max(int(self.args.frac * self.args.num_users), 1)
            idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)

            # communication
            self.dispatch(idxs_users)

            # local updating
            loss_epoch, acc_epoch = self.client_update(idxs_users)
            self.loss.append(loss_epoch)
            self.acc.append(acc_epoch)

            # aggregation
            self.aggregation(idxs_users)

            if t % 10 == 0:
                self.globalmodel_test()
                # self.finetune_test()

        np.savetxt(f"./save/decent_fedavg_frac{self.args.frac}_epoch{self.args.epochs}_subepoch{self.args.local_ep}_loss.csv", np.array(self.loss))
        np.savetxt(f"./save/decent_fedavg_frac{self.args.frac}_epoch{self.args.epochs}_subepoch{self.args.local_ep}_acc.csv", np.array(self.acc))

        return self.nn

    def aggregation(self, idxs_users):
        nns_weights = []
        for i in range(self.args.num_pos):
            for j in idxs_users:
                nns_weights.append(copy.deepcopy(self.nns[i * self.args.num_users + j].state_dict()))

        nn_weight = copy.deepcopy(nns_weights[0])
        for k in nn_weight.keys():
            for i in range(1, len(nns_weights)):
                nn_weight[k] += nns_weights[i][k]
            nn_weight[k] = torch.div(nn_weight[k], len(nns_weights))

        self.nn.load_state_dict(nn_weight)

        torch.save(self.nn, f"./decentsei_fedavg_frac{self.args.frac}_epoch{self.args.epochs}_subepoch{self.args.local_ep}/globalmodel.pth")

    def dispatch(self, idxs_users):
        for i in range(self.args.num_pos):
            for j in idxs_users:
                self.nns[i * self.args.num_users + j] = copy.deepcopy(self.nn)

    def client_update(self, idxs_users):
        loss_clients = []
        acc_clients = []
        for i in range(self.args.num_pos):
            for j in idxs_users:
                print(f"idxs_users is {idxs_users} and current training submodel is {i * self.args.num_users + j}")
                self.nns[i * self.args.num_users + j], loss, acc = train(self.args,
                                                                         self.local_datasets[i * self.args.num_users + j],
                                                                         self.nns[i * self.args.num_users + j],
                                                                         j,
                                                                         i)
                loss_clients.append(copy.deepcopy(loss))
                acc_clients.append(copy.deepcopy(acc))
        loss_avg = sum(loss_clients) / len(loss_clients)
        acc_avg = sum(acc_clients) / len(acc_clients)
        print('Average loss is {:.3f}'.format(loss_avg))
        print('Average acc is {:.3f}'.format(acc_avg))

        return loss_avg, acc_avg

    def final_dispatch(self,):
        for i in range(self.args.num_pos):
            for j in range(self.args.num_users):
                self.nns[i * self.args.num_users + j] = copy.deepcopy(self.nn)

    def client_adaption(self,):
        for i in range(self.args.num_pos):
            for j in range(self.args.num_users):
                self.nns[i * self.args.num_users + j], loss, acc = train(self.args,
                                                                         self.local_datasets[i * self.args.num_users + j],
                                                                         self.nns[i * self.args.num_users + j],
                                                                         j,
                                                                         i)

    def globalmodel_test(self):
        for i in range(0, self.args.num_pos):
            # model = copy.deepcopy(self.nn)
            model = torch.load(f"./decentsei_fedavg_frac{self.args.frac}_epoch{self.args.epochs}_subepoch{self.args.local_ep}/globalmodel.pth")
            model.eval()
            loss, acc = test(self.args, model, self.test_datasets[i])
            print("Testing accuracy of global model in pos {}: {:.2f}".format(i, acc))
        print("-----------------------------------------------------------------------------------------------------------------------")

    def finetune_test(self):
        for i in range(0, self.args.num_pos):
            for j in range(0, self.args.num_users):
                # model = copy.deepcopy(self.nns[i * self.args.num_users + j])
                model = torch.load(f"./decentsei_fedavg_frac{self.args.frac}_epoch{self.args.epochs}_subepoch{self.args.local_ep}/localmodel_pos{i}_user{j}.pth")
                model.eval()
                loss, acc = test(self.args, model, self.test_datasets[i])
                print("Testing accuracy of finetuning model in dev {} of pos {}: {:.2f}".format(j, i, acc))
            print("-----------------------------------------------------------------------------------------------------------------------")

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    fedavg = FedAvg(args)
    fedavg.server()
    fedavg.final_dispatch()
    fedavg.client_adaption()
    fedavg.globalmodel_test()
    fedavg.finetune_test()