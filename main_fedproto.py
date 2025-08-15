#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import copy
import numpy as np
from scipy.io import savemat
import torch
from utils.options import args_parser
from utils.base import FedBase
from utils.network_ShuffleNetV2 import ShuffleNetV2
from Client.Client_fedproto import train, test, obtain_feature_map
from tqdm import tqdm

class FedProto:
    def __init__(self, args):
        self.args = args
        self.nn = ShuffleNetV2().to(self.args.device)
        self.fedbase = FedBase()
        self.local_datasets, self.test_datasets = self.fedbase.create_signal_datasets(num_clients=self.args.num_users, num_pos=self.args.num_pos)
        self.loss = []
        self.acc = []
        self.local_protos = {}
        self.global_protos = dict()
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
            self.protos_aggregation(idxs_users)
            self.nns_aggregation(idxs_users)

            if t % 10 == 0:
                self.globalmodel_test()
                self.finetune_test()

        np.savetxt(f"./save/decent_fedproto_frac{self.args.frac}_epoch{self.args.epochs}_subepoch{self.args.local_ep}_ld{self.args.ld}_loss.csv", np.array(self.loss))
        np.savetxt(f"./save/decent_fedproto_frac{self.args.frac}_epoch{self.args.epochs}_subepoch{self.args.local_ep}_ld{self.args.ld}_acc.csv", np.array(self.acc))

        return self.nn

    def protos_aggregation(self, idxs_users):
        for i in range(self.args.num_pos):
            for j in idxs_users:
                local_protos = self.local_protos[i*len(idxs_users)+j]
                for label in local_protos.keys():
                    if label in self.global_protos:
                        self.global_protos[label].append(local_protos[label])
                    else:
                        self.global_protos[label] = [local_protos[label]]

        for [label, proto_list] in self.global_protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                self.global_protos[label] = [proto / len(proto_list)]
            else:
                self.global_protos[label] = [proto_list[0].data]

    def nns_aggregation(self, idxs_users):
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

        torch.save(self.nn, f"./decentsei_fedproto_frac{self.args.frac}_epoch{self.args.epochs}_subepoch{self.args.local_ep}_ld{self.args.ld}/globalmodel.pth")


    def dispatch(self, idxs_users):
        for i in range(self.args.num_pos):
            for j in idxs_users:
                self.nns[i * self.args.num_users + j] = copy.deepcopy(self.nn)

    def client_update(self, idxs_users):
        loss_clients = []
        acc_clients = []
        for i in range(args.num_pos):
            for k in idxs_users:
                self.nns[i * self.args.num_users + k], loss, acc, protos = train(self.args,
                                                                                 self.local_datasets[i * self.args.num_users + k],
                                                                                 self.nns[i * self.args.num_users + k],
                                                                                 self.global_protos,
                                                                                 k,
                                                                                 i)
                agg_protos = self.agg_protos(protos)
                self.local_protos[i * len(idxs_users) + k] = agg_protos
                loss_clients.append(copy.deepcopy(loss))
                acc_clients.append(copy.deepcopy(acc))
        loss_avg = sum(loss_clients) / len(loss_clients)
        acc_avg = sum(acc_clients) / len(acc_clients)
        print('Average loss is {:.3f}'.format(loss_avg))
        print('Average acc is {:.3f}'.format(acc_avg))

        return loss_avg, acc_avg

    def agg_protos(self, protos):
        """
        Returns the average of the weights.
        """

        for [label, proto_list] in protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                protos[label] = proto / len(proto_list)
            else:
                protos[label] = proto_list[0]

        return protos

    def final_dispatch(self,):
        for i in range(self.args.num_pos):
            for j in range(self.args.num_users):
                self.nn = torch.load(f"./decentsei_fedproto_frac{self.args.frac}_epoch{self.args.epochs}_subepoch{self.args.local_ep}_ld{self.args.ld}/globalmodel.pth")
                self.nns[i * self.args.num_users + j] = copy.deepcopy(self.nn)

    def client_adaption(self,):
        for i in range(self.args.num_pos):
            for j in range(self.args.num_users):
                self.nns[i * self.args.num_users + j], loss, acc, protos = train(self.args,
                                                                                 self.local_datasets[i * self.args.num_users + j],
                                                                                 self.nns[i * self.args.num_users + j],
                                                                                 self.global_protos,
                                                                                 j,
                                                                                 i)

    def globalmodel_test(self):
        for i in range(0, self.args.num_pos):
            # model = copy.deepcopy(self.nn)
            model = torch.load(f"./decentsei_fedproto_frac{self.args.frac}_epoch{self.args.epochs}_subepoch{self.args.local_ep}_ld{self.args.ld}/globalmodel.pth")
            model.eval()
            loss, acc, y_preds, y_reals = test(self.args, model, self.test_datasets[i])

            np.savetxt(f"./results/fedproto/globalmodel_pos{i}_preds.csv", y_preds.astype(int))
            np.savetxt(f"./results/fedproto/globalmodel_pos{i}_reals.csv", y_reals.astype(int))

            print("Testing accuracy of global model in pos {}: {:.2f}".format(i, acc))
        print("-----------------------------------------------------------------------------------------------------------------------")

    def finetune_test(self):
        for i in range(0, self.args.num_pos):
            for j in range(0, self.args.num_users):
                # model = copy.deepcopy(self.nns[i * self.args.num_users + j])
                model = torch.load(f"./decentsei_fedproto_frac{self.args.frac}_epoch{self.args.epochs}_subepoch{self.args.local_ep}_ld{self.args.ld}/localmodel_pos{i}_user{j}.pth")
                model.eval()
                loss, acc, y_preds, y_reals = test(self.args, model, self.test_datasets[i])

                np.savetxt(f"./results/fedproto/localmodel_pos{i}_user{j}_preds.csv", y_preds.astype(int))
                np.savetxt(f"./results/fedproto/localmodel_pos{i}_user{j}_reals.csv", y_reals.astype(int))

                print("Testing accuracy of finetuning model in dev {} of pos {}: {:.2f}".format(j, i, acc))
            print("-----------------------------------------------------------------------------------------------------------------------")

    def feature_maps_from_training_samples(self):
        for i in range(0, self.args.num_pos):
            for j in range(0, self.args.num_users):
                model = torch.load(f"./decentsei_fedproto_wonorm_frac{self.args.frac}_epoch{self.args.epochs}_subepoch{self.args.local_ep}_ld{self.args.ld}/globalmodel.pth")
                protos, labels = obtain_feature_map(self.args, model, self.local_datasets[i * self.args.num_users + j])
                savemat(f"./protos/fedproto/trainingsamples_pos{i}_user{j}_protos.mat", {'protos': protos})
                savemat(f"./protos/fedproto/trainingsamples_pos{i}_user{j}_labels.mat", {'labels': labels})

    def feature_maps_from_testing_samples(self):
        for i in range(0, self.args.num_pos):
            model = torch.load(f"./decentsei_fedproto_wonorm_frac{self.args.frac}_epoch{self.args.epochs}_subepoch{self.args.local_ep}_ld{self.args.ld}/globalmodel.pth")
            protos, labels = obtain_feature_map(self.args, model, self.test_datasets[i])
            savemat(f"./protos/fedproto/testingsamples_pos{i}_protos.mat", {'protos': protos})
            savemat(f"./protos/fedproto/testingsamples_pos{i}_labels.mat", {'labels': labels})


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    fedproto = FedProto(args)
    fedproto.server()
    fedproto.globalmodel_test()