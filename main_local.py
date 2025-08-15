import matplotlib
import numpy as np
matplotlib.use('Agg')
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.options import args_parser
from utils.base import FedBase
from utils.network_ResNet18 import ResNet18
from utils.triplet_loss import TripletLoss
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from pandas import DataFrame
import pandas as pd
from scipy.io import savemat
from sklearn.model_selection import train_test_split

class Local:
    def __init__(self, args, idx_pos, idx_user):
        self.args = args
        self.fedbase = FedBase()
        self.local_datasets, self.test_datasets = self.fedbase.create_signal_datasets(num_clients=8, num_pos=3)
        self.idx_pos = idx_pos
        self.idx_user = idx_user
        self.loss = []
        self.acc = []
        self.min_val_loss = 1e6
        self.nn = ResNet18().to(self.args.device)
        self.network_name = 'ResNet18'

    def client(self):
        for t in tqdm(range(self.args.epochs)):
            print('round', t + 1, ":")
            train_dataset, val_dataset = train_test_split(self.local_datasets[self.idx_pos * self.args.num_users + self.idx_user], test_size=0.3, random_state=30)
            self.nn, loss_epoch, acc_epoch = self.train(train_dataset, val_dataset)
            self.loss.append(loss_epoch)
            self.acc.append(acc_epoch)

        np.savetxt(f"./save/{self.network_name}/local_loss_pos{self.idx_pos}_user{self.idx_user}.csv", np.array(self.loss))
        np.savetxt(f"./save/{self.network_name}/local_acc_pos{self.idx_pos}_user{self.idx_user}.csv", np.array(self.acc))

    def train(self, train_dataset, val_dataset):
        self.nn.train()
        ldr_train = DataLoader(train_dataset, batch_size=self.args.local_bs, shuffle=True)
        ldr_val = DataLoader(val_dataset, batch_size=self.args.local_bs, shuffle=True)
        loss_function_ce = nn.CrossEntropyLoss().to(self.args.device)
        loss_function_tri = TripletLoss(margin=5).to(self.args.device)
        optimizer = torch.optim.SGD(self.nn.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        lr_step = StepLR(optimizer, step_size=10, gamma=0.1)
        epoch_train_loss = []
        epoch_train_acc = []
        epoch_val_loss = []
        epoch_val_acc = []
        for iter in range(self.args.local_ep):
            batch_train_loss = []
            batch_val_loss = []
            train_correct = 0
            val_correct = 0
            # train
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.squeeze().long().to(self.args.device)
                log_probs, features = self.nn(images, return_features=True)
                if log_probs.shape[0] == 1: labels = labels.unsqueeze(0)
                loss_ce = loss_function_ce(log_probs, labels)
                loss_tri = loss_function_tri(features, labels)
                loss = loss_ce + 0.01 * loss_tri[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_train_loss.append(loss.item())

                y_pred = log_probs.data.max(1, keepdim=True)[1]
                train_correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
            train_accuracy = 100.00 * train_correct / len(ldr_train.dataset)
            epoch_train_loss.append(sum(batch_train_loss) / len(batch_train_loss))
            epoch_train_acc.append(train_accuracy)

            # val
            for batch_idx, (images, labels) in enumerate(ldr_val):
                images, labels = images.to(self.args.device), labels.squeeze().long().to(self.args.device)
                log_probs = self.nn(images)
                if log_probs.shape[0] == 1: labels = labels.unsqueeze(0)
                loss = loss_function_ce(log_probs, labels)
                batch_val_loss.append(loss.item())

                y_pred = log_probs.data.max(1, keepdim=True)[1]
                val_correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
            val_accuracy = 100.00 * val_correct / len(ldr_val.dataset)
            epoch_val_loss.append(sum(batch_val_loss) / len(batch_val_loss))
            epoch_val_acc.append(val_accuracy)

            if sum(batch_val_loss) / len(batch_val_loss) < self.min_val_loss:
                print(f"the min val loss is improved from {self.min_val_loss} to {sum(batch_val_loss) / len(batch_val_loss)}, and model are saved")
                torch.save(self.nn, f"./localsei/{self.network_name}/localmodel_pos{self.idx_pos}_user{self.idx_user}.pth")
                self.min_val_loss = sum(batch_val_loss) / len(batch_val_loss)
            else:
                print(f"the min val loss is not improved from {self.min_val_loss}")

            lr_step.step()

            print(f'Device {self.idx_user} of Position {self.idx_pos}, '
                  f'Update Epoch: {iter} [{iter}/{args.local_ep}]\t'
                  f'Train Loss: {sum(batch_train_loss) / len(batch_train_loss)}, '
                  f'Train Acc: {train_accuracy}% ({train_correct}/{len(ldr_train.dataset)})'
                  f'Val Loss: {sum(batch_val_loss) / len(batch_val_loss)},'
                  f'Val Acc: {val_accuracy}% ({val_correct}/{len(ldr_val.dataset)})')

        return self.nn, sum(epoch_train_loss) / len(epoch_train_loss), sum(epoch_train_acc) / len(epoch_train_acc)

    def test(self):
        model = torch.load(f"./localsei/{self.network_name}/localmodel_pos{self.idx_pos}_user{self.idx_user}.pth")
        model.eval()
        data_loader = DataLoader(self.test_datasets[self.idx_pos], batch_size=self.args.bs)
        correct = 0
        y_preds = []
        y_reals = []
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.args.device), target.squeeze().long().to(self.args.device)
            log_probs = model(data)
            if log_probs.shape[0] == 1: target = target.unsqueeze(0)
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            y_preds.append(y_pred.cpu())
            y_reals.append(target.cpu())

        accuracy = 100.00 * correct / len(data_loader.dataset)

        y_preds = np.concatenate(y_preds)
        y_reals = np.concatenate(y_reals)

        np.savetxt(f"./results/localsei/{self.network_name}/localmodel_pos{self.idx_pos}_user{self.idx_user}_preds.csv", y_preds.astype(int))
        np.savetxt(f"./results/localsei/{self.network_name}/localmodel_pos{self.idx_pos}_user{self.idx_user}_reals.csv", y_reals.astype(int))

        print("Testing accuracy in dev {} of pos {}: {:.2f}".format(self.idx_user, self.idx_pos, accuracy))


    def test_cross_environment(self):
        model = torch.load(f"./localsei/{self.network_name}/localmodel_pos{self.idx_pos}_user{self.idx_user}.pth")
        model.eval()
        for idx_pos_cross in range(self.args.num_pos):
            data_loader = DataLoader(self.test_datasets[idx_pos_cross], batch_size=self.args.bs)
            correct = 0
            y_preds = []
            y_reals = []
            for idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.args.device), target.squeeze().long().to(self.args.device)
                log_probs = model(data)
                if log_probs.shape[0] == 1: target = target.unsqueeze(0)
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

                y_preds.append(y_pred.cpu())
                y_reals.append(target.cpu())

            accuracy = 100.00 * correct / len(data_loader.dataset)

            y_preds = np.concatenate(y_preds)
            y_reals = np.concatenate(y_reals)

            np.savetxt(f"./results/localsei/{self.network_name}/localmodel_pos{self.idx_pos}_user{self.idx_user}_testonpos{idx_pos_cross}_preds.csv", y_preds.astype(int))
            np.savetxt(f"./results/localsei/{self.network_name}/localmodel_pos{self.idx_pos}_user{self.idx_user}_testonpos{idx_pos_cross}_reals.csv", y_reals.astype(int))

            print("localmodel_pos{}_user{}， testing accuracy in pos {}: {:.2f}".format(self.idx_pos, self.idx_user, idx_pos_cross, accuracy))

    def obtain_feature_map(self, ann, dataset):
        ann.eval()
        y_reals = []
        protos = []
        data_loader = DataLoader(dataset, batch_size=self.args.bs)
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.squeeze().long().to(args.device)
            _, proto = ann(data, return_features=True)
            if _.shape[0] == 1: target = target.unsqueeze(0)

            protos.append(proto.detach().cpu())
            y_reals.append(target.cpu())

        protos = np.concatenate(protos)
        y_reals = np.concatenate(y_reals)

        return protos, y_reals

    def feature_maps_from_training_samples(self):
        model = torch.load(f"./localsei/{self.network_name}/localmodel_pos{self.idx_pos}_user{self.idx_user}.pth")
        protos, labels = self.obtain_feature_map(model, self.local_datasets[self.idx_pos * self.args.num_users + self.idx_user])
        savemat(f"./protos/localsei/{self.network_name}/trainingsamples_pos{self.idx_pos}_user{self.idx_user}_protos.mat", {'protos': protos})
        savemat(f"./protos/localsei/{self.network_name}/trainingsamples_pos{self.idx_pos}_user{self.idx_user}_labels.mat", {'labels': labels})

    def feature_maps_from_testing_samples(self):
        model = torch.load(f"./localsei/{self.network_name}/localmodel_pos{self.idx_pos}_user{self.idx_user}.pth")
        for idx_pos_cross in range(self.args.num_pos):
            protos, labels = self.obtain_feature_map(model, self.test_datasets[idx_pos_cross])
            savemat(f"./protos/localsei/{self.network_name}/testingsamples_pos{self.idx_pos}_user{self.idx_user}_testonpos{idx_pos_cross}_protos.mat", {'protos': protos})
            savemat(f"./protos/localsei/{self.network_name}/testingsamples_pos{self.idx_pos}_user{self.idx_user}_testonpos{idx_pos_cross}_labels.mat", {'labels': labels})

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(args.seed)

    for idx_pos in [2]: # range(args.num_pos):
        for idx_user in [5,6,7]: # range(args.num_users):
            local = Local(args, idx_pos, idx_user)
            local.client()

    for idx_pos in range(args.num_pos):
        for idx_user in range(args.num_users):
            local = Local(args, idx_pos, idx_user)
            local.test()
            local.test_cross_environment()
            # local.feature_maps_from_training_samples()
            # local.feature_maps_from_testing_samples()