import sys
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

from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

from torchdiffeq import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap


def load_datasets(filename="dataset.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["datasets"], data["time_points"]


def dataset_preparation(num_tasks=10):
    datasets, time_points = load_datasets('../data/CTD/Influenza/dataset.pkl')
    X = np.array([item[0] for item in datasets])
    Y = np.array([item[1] for item in datasets])

    dataloaders = []

    time_points = (time_points / 213)*50

    for i in range(num_tasks - 1):
        temp_X = X[i]
        temp_Y = Y[i].reshape((-1, 1))
        domain_dataset = TensorDataset(torch.Tensor(temp_X).to(device), torch.Tensor(temp_Y).to(device))
        temp_dataloader = DataLoader(domain_dataset, batch_size=batch_size, shuffle=True)
        dataloaders.append(temp_dataloader)

    temp_X = X[-1]
    temp_Y = Y[-1].reshape((-1, 1))
    domain_dataset = TensorDataset(torch.Tensor(temp_X).to(device), torch.Tensor(temp_Y).to(device))
    temp_dataloader = DataLoader(domain_dataset, batch_size=temp_X.shape[0], shuffle=False)
    dataloaders.append(temp_dataloader)

    return dataloaders, time_points


class ODEF2(nn.Module):
    def __init__(self, latent_dim):
        super(ODEF2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )

    def forward(self, t, y):
        return self.net(y)


class EmbeddingNet(nn.Module):
    def __init__(self, discrete_value=57, embedding_dim=10):
        super(EmbeddingNet, self).__init__()
        self.embedding = nn.Embedding(discrete_value, embedding_dim)
        self.fc1 = nn.Linear(535, 32)
        self.relu = nn.ReLU()

    def forward(self, x):
        discrete_feature = x[:, 0].long()
        continuous_features = x[:, 1:]
        embedded_feature = self.embedding(discrete_feature)
        x = torch.cat((embedded_feature, continuous_features), dim=1)
        x = self.relu(self.fc1(x))
        return x


class RNN1(nn.Module):
    def __init__(self, ode_func, time_point, device):
        super(RNN1, self).__init__()
        self.n = len(time_point)
        self.time_point = time_point
        self.odefunc = ode_func(latent_dim)

        self.shared_conv = EmbeddingNet()

        self.pred_model = nn.ModuleList([nn.Sequential(
            nn.Linear(data_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()) for i in range(self.n)])

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

    def get_pred(self, X, E):
        pred = []
        for d in range(E.shape[0]):
            param, x = E[d], X[d]
            m_1 = param[:data_size * 128]
            b_1 = param[data_size * 128:data_size * 128 + 128]
            m_2 = param[data_size * 128 + 128:data_size * 128 + 128 + 128 * 32]
            b_2 = param[data_size * 128 + 128 + 128 * 32:data_size * 128 + 128 + 128 * 32 + 32]
            m_3 = param[data_size * 128 + 128 + 128 * 32 + 32:data_size * 128 + 128 + 128 * 32 + 32 + 32]
            b_3 = param[-1]

            y = torch.sigmoid(F.linear(torch.relu(F.linear(torch.relu(F.linear(x, m_1.reshape((128, data_size)), b_1)), m_2.reshape((32, 128)), b_2)),m_3.reshape((1, 32)), b_3))
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
        X = torch.stack([self.shared_conv(x) for x in X], dim=0)
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
                X.append(x[:51]), Y.append(y[:51])
        X, Y = torch.stack(X), torch.stack(Y)
        all_data.append([batch_t, X, Y])

    model = RNN1(ODEF2, time_point, device).to(device)
    optimizer = torch.optim.Adam([{'params': model.shared_conv.parameters(), 'lr': pred_learning_rate},
                                  {'params': model.pred_model.parameters(), 'lr': pred_learning_rate},
                                  {'params': model.encoder.parameters(), 'lr': coder_learning_rate},
                                  {'params': model.decoder.parameters(), 'lr': coder_learning_rate},
                                  {'params': model.odefunc.parameters(), 'lr': ode_learning_rate}])
    min_val_loss = np.inf
    # train_num = int(len(all_data)*train_rate)
    train_num = 31
    idx_train = np.arange(0, train_num, dtype=np.int64)
    idx_val = np.arange(train_num, len(all_data), dtype=np.int64)
    for epoch in range(epoch_num):
        if epoch ==100:
            optimizer.param_groups[0]['lr'] = 0.

        model.train()
        # np.random.shuffle(idx_train)
        acc1, acc2 = [], []
        epoch_loss, loss1, loss2, loss3, loss4, loss5 = 0, 0, 0, 0, 0, 0

        for batch in np.array_split(idx_train, 20):
            loss = 0

            for s in batch:
                batch_t, X, Y = all_data[s]
                batch_t = torch.Tensor(batch_t).to(device)

                New_E, pred, init_param, init_pred, init_embed, New_E_embed, init_debed = model(X, batch_t, s)

                pred, init_pred, Y = pred.view(-1, 1), init_pred.view(-1, 1), Y.view(-1, 1)
                loss1 = loss1 + F.binary_cross_entropy(init_pred, Y)
                loss2 = loss2 + F.binary_cross_entropy(pred, Y)
                loss3 = loss3 + F.mse_loss(New_E, init_param)
                loss4 = loss4 + F.mse_loss(init_embed, New_E_embed)
                loss5 = loss5 + F.mse_loss(init_param, init_debed)
                loss = loss + \
                       F.binary_cross_entropy(init_pred, Y) + \
                       F.binary_cross_entropy(pred, Y) + \
                       model_lambda * F.mse_loss(New_E, init_param) + \
                       10 * F.mse_loss(init_embed, New_E_embed) + \
                       model_lambda * F.mse_loss(init_param, init_debed)

                # regularization_loss = 0
                # for param in model.odefunc.parameters():
                #     regularization_loss += torch.sum(abs(param))
                # loss = loss + regularization_loss

                init_prediction = torch.as_tensor((init_pred.detach() - 0.5) > 0).float()
                prediction = torch.as_tensor((pred.detach() - 0.5) > 0).float()

                accuracy_init = (init_prediction.squeeze(-1) == Y.squeeze(-1)).float().sum() / \
                                init_prediction.shape[0]
                accuracy_gene = (prediction.squeeze(-1) == Y.squeeze(-1)).float().sum() / prediction.shape[0]
                acc1.append(accuracy_init.item()), acc2.append(accuracy_gene.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_loss = 0
        for v in idx_val:
            batch_t, X, Y = all_data[v]
            batch_t = torch.Tensor(batch_t).to(device)
            New_E, pred, init_param, init_pred, init_embed, New_E_embed, init_debed = model(X, batch_t, v)
            pred, init_pred, Y = pred.view(-1, 1), init_pred.view(-1, 1), Y.view(-1, 1)
            val_loss = val_loss + F.binary_cross_entropy(pred, Y)

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
                X.append(x[:51]), Y.append(y[:51])
        X, Y = torch.stack(X), torch.stack(Y)


        init_param = torch.cat([p.flatten() for p in model.pred_model[-1].parameters()])
        init_t = model.time_point[-1]
        time_point = torch.Tensor(np.insert(time_point, 0, init_t)).to(device)

        init_embed = model.encoder(init_param)
        test_embed = odeint(model.odefunc, init_embed, time_point, method='rk4', options={'step_size': rk_step})[1:]
        test_param = model.decoder(test_embed)
        X = torch.stack([model.shared_conv(x) for x in X], dim=0)
        test_pred = model.get_pred(X, test_param).reshape_as(Y)

    f = open(PATH + '/scores.txt', 'a')
    f.write('Repeat {} ###########################################################\n'.format(repeat))
    test_pred_flatten, Y_flatten = test_pred.view(-1, 1), Y.view(-1, 1)
    prediction = torch.as_tensor((test_pred_flatten.detach() - 0.5) > 0).float()
    accuracy = (prediction.squeeze(-1) == Y_flatten.squeeze(-1)).float().sum() / prediction.shape[0]

    gene_auc = roc_auc_score(Y.view(-1, 1).cpu().numpy(), test_pred.view(-1, 1).detach().cpu().numpy(), labels=[0, 1])

    print('All Accuracy {:.2%}'.format(accuracy))
    print('All AUC {:.4}'.format(gene_auc))
    f.write('All Accuracy {:.2%}\n'.format(accuracy))
    f.write('All AUC {:.4}\n'.format(gene_auc))

    for i in range(len(loaders)):
        test_pred_step, Y_step = test_pred[i], Y[i]
        prediction = torch.as_tensor((test_pred_step.detach() - 0.5) > 0).float()
        accuracy = (prediction.squeeze(-1) == Y_step.squeeze(-1)).float().sum() / prediction.shape[0]
        print('Step {}, Accuracy {:.2%}'.format(i + 1, accuracy))
        f.write('Step {}, Accuracy {:.2%}\n'.format(i + 1, accuracy))


def main(repeat):

    dataloaders, time_points = dataset_preparation(num_tasks)
    train_loaders, test_loaders = dataloaders[:-test_tasks], dataloaders[-test_tasks:]
    train_time, test_time = time_points[:-test_tasks], time_points[-test_tasks:]

    model = trainModel(train_loaders, train_time, repeat)
    testModel(test_loaders, test_time, model, repeat)


DATANAME = 'Twitter'
MODELNAME = 'ODE'
KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = './save_{}/'.format(MODELNAME) + KEYWORD
#######################################
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
num_tasks = 50
data_size = 32
latent_dim = 128
E_shape = data_size * 128 + 128 * 32 + 32 + 128 + 32 + 1
pred_learning_rate = 1e-3
coder_learning_rate = 1e-3
ode_learning_rate = 1e-3
embed_dim = 32

batch_size = 55
epoch_num = 150

test_tasks = 15
train_tasks = num_tasks - test_tasks
batch_time = 10
train_rate = 0.7
rk_step = 1/213*50
model_lambda = 100
embed_method = 'tsne'

if __name__ == '__main__':
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)

    for repeat in range(1, 6):
        main(repeat)
