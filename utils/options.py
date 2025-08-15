#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=300, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=8, help="number of users: K")
    parser.add_argument('--num_pos', type=int, default=3, help='number of position: P')
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=16, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # FedProx
    parser.add_argument('--mu', type=float, default=0.1, help='weight of proximal term')

    # Decorr
    parser.add_argument('--feddecorr_coef', type=float, default=0.04, help='feddecorr_coef')

    # FedAtt
    parser.add_argument('--epsilon', type=float, default=1.4, help='stepsize')
    parser.add_argument('--ord', type=int, default=2, help='similarity metric')
    parser.add_argument('--dp', type=float, default=0.001, help='differential privacy')


    # q-FedAvg
    parser.add_argument('--q', type=float, default=0.001, help='reweighting factor')  # no weighting, the same as fedavg

    # ICFA
    parser.add_argument('--num_clusters', type=int, default=3, help='number of clusters')

    # FedPer
    parser.add_argument('--Kp', type=int, default=1, help='number of personalized layers')

    # FedProto
    parser.add_argument('--ld', type=float, default=0.2, help='weight of proto term')

    # FedALA
    parser.add_argument('-eta', "--eta", type=float, default=1.0)
    parser.add_argument('-rand_percent', "--rand_percent", type=int, default=50)
    parser.add_argument('-layer_idx', "--layer_idx", type=int, default=40, help="More fine-graind than its original paper.")

    # model arguments
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='drones', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    args = parser.parse_args()
    return args
