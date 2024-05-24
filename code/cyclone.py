import os
import shutil
import numpy as np
import pickle
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

from torchdiffeq import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.metrics import accuracy_score


def load_datasets(filename="dataset.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["datasets"], data["time_points"]


def dataset_preparation(num_tasks=10):
    datasets, time_points = load_datasets('../data/CTD/TCIR/dataset_2016_d.pkl')
    X = [np.stack([np.asarray(img) for img in dataset[0]], axis=0) for dataset in datasets]
    Y = [np.array(dataset[1]) for dataset in datasets]
    # X = X / 255
    X = [np.transpose(x, (0, 3, 1, 2)) for x in X]

    time_points=time_points/365*50

    dataloaders = []

    for i in range(train_tasks):
        temp_X = X[i]
        temp_Y = Y[i].reshape((-1, 1))
        domain_dataset = TensorDataset(torch.Tensor(temp_X).to(device), torch.Tensor(temp_Y).to(device))
        train_dataloader = DataLoader(domain_dataset, batch_size=batch_size, shuffle=True)

        dataloaders.append(train_dataloader)

    for i in range(train_tasks, num_tasks):
        temp_X = X[i]
        temp_Y = Y[i].reshape((-1, 1))
        domain_dataset = TensorDataset(torch.Tensor(temp_X).to(device), torch.Tensor(temp_Y).to(device))
        test_dataloader = DataLoader(domain_dataset, batch_size=temp_X.shape[0], shuffle=False)
        dataloaders.append(test_dataloader)

    return dataloaders, time_points


class ODEF2(nn.Module):
    def __init__(self):
        super(ODEF2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )

    def forward(self, t, y):
        return self.net(y)

def manual_forward(input, weights, biases):
    x = F.linear(
        F.relu(F.linear(
        F.relu(F.linear(
            input,
            weights['0.weight'], biases['0.bias'])),
        weights['3.weight'], biases['3.bias'])),
            weights['6.weight'], biases['6.bias'])
    return x


class RNN1(nn.Module):
    def __init__(self, ode_func, time_point, device):
        super(RNN1, self).__init__()
        self.n = len(time_point)
        self.time_point = time_point
        self.odefunc = ode_func()

        self.shared_conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.7),
        )

        self.pred_model = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(32, 1),
            ) for _ in range(self.n)
        ])

        self.encoder = nn.Sequential(
            nn.Linear(E_shape, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim))

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, E_shape))


    def init_model(self, X, s, t):
        num = len(t)
        pred, param = [], []
        for i in range(num):
            x = X[i]
            pred.append(self.pred_model[s + i](x))
            param.append(torch.cat([p.flatten() for p in self.pred_model[s + i].parameters()]))
        pred, param = torch.cat(pred), torch.stack(param)
        return param, pred

    def get_domain_pred(self, x, param):
        weights = {}
        biases = {}
        start_idx = 0
        for name, p in self.pred_model[0].state_dict().items():
            end_idx = start_idx + p.numel()
            if 'bias' in name:
                biases[name] = param[start_idx:end_idx].view(p.shape)
            else:
                weights[name] = param[start_idx:end_idx].view(p.shape)
            start_idx = end_idx

        pred = manual_forward(x, weights, biases)
        return pred

    def get_pred(self, X, E):
        pred = []
        for d in range(E.shape[0]):
            param, x = E[d], X[d]
            y = self.get_domain_pred(x, param)
            pred.append(y)
        return torch.cat(pred)

    def get_param(self, t, start):
        Param = []
        num = len(t)
        for i in range(start + 1, start + num):
            param = torch.cat([p.flatten() for p in self.pred_model[i].parameters()])
            Param.append(param)
        return torch.stack(Param)

    def forward(self, X, continous_time=None, s=0):
        X = [self.shared_conv(x) for x in X]
        init_param, init_pred = self.init_model(X, s, continous_time)
        init_embed = self.encoder(init_param)
        E_embed = odeint(self.odefunc, init_embed[0], continous_time, method='rk4', options={'step_size': rk_step})
        E = self.decoder(E_embed)
        init_debed = self.decoder(init_embed)
        pred = self.get_pred(X, E)
        return E, pred, init_param, init_pred, init_embed, E_embed, init_debed


def trainModel(loaders, time_point, repeat):
    domain_num = len(loaders)
    all_data = []
    for s in range(domain_num - 1):
        l = min(domain_num-s, batch_time)
        batch_y = loaders[s:s + l]
        batch_t = time_point[s:s + l]
        X, Y = [], []
        for loader in batch_y:
            for x, y in loader:
                X.append(x), Y.append(y)
        # X, Y = torch.stack(X), torch.stack(Y)
        all_data.append([batch_t, X, Y])

    model = RNN1(ODEF2, time_point, device).to(device)
    optimizer = torch.optim.Adam([{'params': model.shared_conv.parameters(), 'lr': pred_learning_rate},
                                  {'params': model.pred_model.parameters(), 'lr': pred_learning_rate},
                                  {'params': model.encoder.parameters(), 'lr': coder_learning_rate},
                                  {'params': model.decoder.parameters(), 'lr': coder_learning_rate},
                                  {'params': model.odefunc.parameters(), 'lr': ode_learning_rate}])
    min_val_loss = np.inf
    # train_num = int(len(all_data)*train_rate)
    train_num = 45
    idx_train = np.arange(0, train_num, dtype=np.int64)
    idx_val = np.arange(train_num, len(all_data), dtype=np.int64)
    for epoch in range(epoch_num):
        # if epoch ==150:
        #     optimizer.param_groups[0]['lr'] = 0.
        model.train()
        # np.random.shuffle(idx_train)
        acc1, acc2 = [], []
        epoch_loss, loss1, loss2, loss3, loss4, loss5 = 0, 0, 0, 0, 0, 0

        for batch in np.array_split(idx_train, len(idx_train)):
        # for batch in np.array_split(idx_train, 10):
            loss = 0

            for s in batch:
                batch_t, X, Y = all_data[s]
                batch_t = torch.Tensor(batch_t).to(device)

                New_E, pred, init_param, init_pred, init_embed, New_E_embed, init_debed = model(X, batch_t, s)

                Y = torch.cat(Y)
                loss1 = loss1 + F.l1_loss(init_pred, Y)
                loss2 = loss2 + F.l1_loss(pred, Y)
                loss3 = loss3 + F.mse_loss(New_E, init_param)
                loss4 = loss4 + F.mse_loss(init_embed, New_E_embed)
                loss5 = loss5 + F.mse_loss(init_param, init_debed)
                loss = loss + \
                       F.l1_loss(init_pred, Y) + \
                       F.l1_loss(pred, Y) + \
                       model_lambda * F.mse_loss(New_E, init_param) + \
                       10 * F.mse_loss(init_embed, New_E_embed) + \
                       model_lambda * F.mse_loss(init_param, init_debed)

                accuracy_init = F.l1_loss(init_pred, Y)
                accuracy_gene = F.l1_loss(pred, Y)
                acc1.append(accuracy_init.item()), acc2.append(accuracy_gene.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        with torch.no_grad():
            model.eval()
            val_loss = 0
            for v in idx_val:
                batch_t, X, Y = all_data[v]
                batch_t = torch.Tensor(batch_t).to(device)
                New_E, pred, init_param, init_pred, init_embed, New_E_embed, init_debed = model(X, batch_t, v)
                Y = torch.cat(Y)
                val_loss = val_loss + F.l1_loss(pred, Y)

            if val_loss < min_val_loss:
                # min_val_loss = val_loss
                if epoch > epoch_num*0.7:
                    min_val_loss = val_loss
                    torch.save(model.state_dict(), PATH + '/model_{}.pt'.format(repeat))

        print(
            "Epoch: {}\tLoss: {:.5f}\tLoss1: {:.3f}\tLoss2: {:.3f}\tLoss3: {:.3f}\tLoss4: {:.3f}\tLoss5: {:.3f}\tAcc init: {:.3f}\tAcc gene: {:.3f}\tVal: {:.3f}".format(
                epoch, epoch_loss, loss1, loss2, loss3, loss4, loss5, np.mean(acc1), np.mean(acc2), val_loss))

        with open(PATH + '/log_{}.txt'.format(repeat), 'a') as f:
            f.write("{},{},{},{},{},{},{},{},{},{}\n".format(epoch, epoch_loss, loss1, loss2, loss3, loss4, loss5,
                                                             np.mean(acc1), np.mean(acc2), val_loss))

    model.load_state_dict(torch.load(PATH + '/model_{}.pt'.format(repeat)))
    return model



def testModel(loaders, time_point, model, repeat):
    model.eval()
    with torch.no_grad():
        X, Y = [], []
        for loader in loaders:
            for x, y in loader:
                X.append(x), Y.append(y)
        # X, Y = torch.stack(X), torch.stack(Y)
        Y = torch.cat(Y)


        init_param = torch.cat([p.flatten() for p in model.pred_model[-1].parameters()])
        init_t = model.time_point[-1]
        time_point = torch.Tensor(np.insert(time_point, 0, init_t)).to(device)

        init_embed = model.encoder(init_param)
        test_embed = odeint(model.odefunc, init_embed, time_point, method='rk4', options={'step_size': rk_step})[1:]
        test_param = model.decoder(test_embed)
        X = [model.shared_conv(x) for x in X]
        test_pred = model.get_pred(X, test_param)

    f = open(PATH + '/scores.txt', 'a')
    f.write('Repeat {} ###########################################################\n'.format(repeat))
    accuracy = F.l1_loss(test_pred, Y)
    print('All Accuracy {:.4}'.format(accuracy.item()))
    f.write('All Accuracy {:.4}\n'.format(accuracy.item()))

    start = 0
    for i in range(len(loaders)):
        len_domain = len(X[i])
        test_pred_step, Y_step = test_pred[start:start+len_domain], Y[start:start+len_domain]
        start+=len_domain
        accuracy = F.l1_loss(test_pred_step, Y_step)
        print('Step {}, Accuracy {:.4}'.format(i + 1, accuracy.item()))
        f.write('Step {}, Accuracy {:.4}\n'.format(i + 1, accuracy.item()))


def main(repeat):

    dataloaders, time_points = dataset_preparation(num_tasks)
    train_loaders, test_loaders = dataloaders[:-test_tasks], dataloaders[-test_tasks:]
    train_time, test_time = time_points[:-test_tasks], time_points[-test_tasks:]

    model = trainModel(train_loaders, train_time, repeat)
    testModel(test_loaders, test_time, model, repeat)


DATANAME = 'TCIR'
MODELNAME = 'ODE'
KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = './save_{}/'.format(MODELNAME) + KEYWORD
#######################################
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
num_tasks = 72
data_size = 2
E_shape = 135361
pred_learning_rate = 1e-3
coder_learning_rate = 1e-3
ode_learning_rate = 1e-3
embed_dim = 32

batch_size = 200
epoch_num = 300

test_tasks = 22
train_tasks = num_tasks - test_tasks
batch_time = 10
train_rate = 0.8
rk_step = 1/365*50
model_lambda = 100
embed_method = 'tsne'

if __name__ == '__main__':
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)

    for repeat in range(1, 6):
        main(repeat)
