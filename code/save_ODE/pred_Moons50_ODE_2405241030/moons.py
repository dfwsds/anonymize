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
    datasets, time_points = load_datasets('../data/CTD/Moons50/dataset.pkl')
    X = np.array([item[0] for item in datasets])
    Y = np.array([item[1] for item in datasets])

    dataloaders = []

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
            nn.Linear(embed_dim, embed_dim, bias=False))

    def forward(self, t, y):
        return self.net(y)


class RNN1(nn.Module):
    def __init__(self, ode_func, time_point, device):
        super(RNN1, self).__init__()
        self.n = len(time_point)
        self.time_point = time_point
        self.odefunc = ode_func(latent_dim)

        self.pred_model = nn.ModuleList([nn.Sequential(
            nn.Linear(data_size, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
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
            m_1 = param[:data_size * 50]
            b_1 = param[data_size * 50:data_size * 50 + 50]
            m_2 = param[data_size * 50 + 50:data_size * 50 + 50 + 50 * 50]
            b_2 = param[data_size * 50 + 50 + 50 * 50:data_size * 50 + 50 + 50 * 50 + 50]
            m_3 = param[data_size * 50 + 50 + 50 * 50 + 50:data_size * 50 + 50 + 50 * 50 + 50 + 50]
            b_3 = param[-1]

            y = torch.sigmoid(F.linear(torch.relu(F.linear(torch.relu(F.linear(x, m_1.reshape((50, data_size)), b_1)), m_2.reshape((50, 50)), b_2)),m_3.reshape((1, 50)), b_3))
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
        X, Y = torch.stack(X), torch.stack(Y)
        all_data.append([batch_t, X, Y])

    model = RNN1(ODEF2, time_point, device).to(device)
    optimizer = torch.optim.Adam([{'params': model.pred_model.parameters(), 'lr': pred_learning_rate},
                                  {'params': model.encoder.parameters(), 'lr': coder_learning_rate},
                                  {'params': model.decoder.parameters(), 'lr': coder_learning_rate},
                                  {'params': model.odefunc.parameters(), 'lr': ode_learning_rate}])
    min_val_loss = np.inf
    # train_num = int(len(all_data)*train_rate)
    train_num = 31
    idx_train = np.arange(0, train_num, dtype=np.int64)
    idx_val = np.arange(train_num, len(all_data), dtype=np.int64)
    for epoch in range(epoch_num):
        model.train()
        # np.random.shuffle(idx_train)
        acc1, acc2 = [], []
        epoch_loss, loss1, loss2, loss3, loss4, loss5 = 0, 0, 0, 0, 0, 0

        for batch in np.array_split(idx_train, 4):
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
            if epoch > 200:
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
    X, Y = [], []
    for loader in loaders:
        for x, y in loader:
            X.append(x), Y.append(y)
    X, Y = torch.stack(X), torch.stack(Y)


    init_param = torch.cat([p.flatten() for p in model.pred_model[-1].parameters()])
    init_t = model.time_point[-1]
    time_point = torch.Tensor(np.insert(time_point, 0, init_t)).to(device)

    init_embed = model.encoder(init_param)
    test_embed = odeint(model.odefunc, init_embed, time_point, method='rk4', options={'step_size': rk_step})[1:]
    test_param = model.decoder(test_embed)
    test_pred = model.get_pred(X, test_param).reshape_as(Y)

    f = open(PATH + '/scores.txt', 'a')
    f.write('Repeat {} ###########################################################\n'.format(repeat))
    test_pred_flatten, Y_flatten = test_pred.view(-1, 1), Y.view(-1, 1)
    prediction = torch.as_tensor((test_pred_flatten.detach() - 0.5) > 0).float()
    accuracy = (prediction.squeeze(-1) == Y_flatten.squeeze(-1)).float().sum() / prediction.shape[0]
    print('All Accuracy {:.2%}'.format(accuracy))
    f.write('All Accuracy {:.2%}\n'.format(accuracy))

    for i in range(len(loaders)):
        test_pred_step, Y_step = test_pred[i], Y[i]
        prediction = torch.as_tensor((test_pred_step.detach() - 0.5) > 0).float()
        accuracy = (prediction.squeeze(-1) == Y_step.squeeze(-1)).float().sum() / prediction.shape[0]
        print('Step {}, Accuracy {:.2%}'.format(i + 1, accuracy))
        f.write('Step {}, Accuracy {:.2%}\n'.format(i + 1, accuracy))

def plot_decision_boundary(model, ax, param, data=None, t=0, ani=False):
    h = .02
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2.5, 2.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.get_pred(torch.tensor([np.c_[xx.ravel(), yy.ravel()]], dtype=torch.float32).to(device), param)
    Z = Z.detach().cpu().numpy()
    Z = (Z > 0.5).astype(int)
    Z = Z.reshape(xx.shape)

    if ani:
        im1 = ax.contourf(xx, yy, Z, alpha=0.3)
        im2 = ax.text(1, -2, 'scalability = {:.2f}'.format(t))
        return im1.collections + [im2]
    else:
        ax.contourf(xx, yy, Z, alpha=0.3)
        if data:
            X, Y = data
            scatter = ax.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.coolwarm, alpha=0.2)


def visModel(loaders, time_point, model, repeat):
    # Figure
    ## Train
    train_loader, train_time = loaders[:train_tasks], torch.Tensor(time_point[:train_tasks]).to(device)
    train_data = []

    for loader in train_loader:
        X, Y = [], []
        for x, y in loader:
            X.append(x), Y.append(y)
        X, Y = torch.cat(X), torch.cat(Y)
        train_data.append([X.cpu().numpy(), Y.cpu().numpy()])

    train_model_param = []
    for i in range(train_tasks):
        train_model_param.append(torch.cat([p.flatten() for p in model.pred_model[i].parameters()]))
    train_model_param = torch.stack(train_model_param)

    fig, axs = plt.subplots(train_tasks // 5, 5, figsize=(15, 3 * train_tasks // 5))
    axs = axs.ravel()
    for i in range(train_tasks):
        plot_decision_boundary(model, axs[i], train_model_param[i:i+1], train_data[i])
        axs[i].set_title('t = {:.4f}'.format(train_time[i]))
    plt.tight_layout()
    plt.savefig(PATH + '/figure_{}_train_model.png'.format(repeat))

    ## Test
    test_loader, test_time = loaders[train_tasks:], torch.Tensor(time_point[train_tasks-1:]).to(device)
    test_data = []

    for loader in test_loader:
        X, Y = [], []
        for x, y in loader:
            X.append(x), Y.append(y)
        X, Y = torch.cat(X), torch.cat(Y)
        test_data.append([X.cpu().numpy(), Y.cpu().numpy()])

    final_train_model_param = torch.cat([p.flatten() for p in model.pred_model[-1].parameters()])
    final_train_model_embed = model.encoder(final_train_model_param)
    test_model_embed = odeint(model.odefunc, final_train_model_embed, test_time, method='rk4', options={'step_size': rk_step})[1:]
    test_model_param = model.decoder(test_model_embed)

    fig, axs = plt.subplots(test_tasks // 5, 5, figsize=(15, 3 * test_tasks // 5))
    axs = axs.ravel()
    for i in range(test_tasks):
        plot_decision_boundary(model, axs[i], test_model_param[i:i+1], test_data[i])
        axs[i].set_title('t = {:.4f}'.format(test_time[i+1]))
    plt.tight_layout()
    plt.savefig(PATH + '/figure_{}_test_model.png'.format(repeat))


    ####################################################### Animation
    inter_time_seg = []
    tmp = 0
    for idx, i in enumerate(train_time[:-1]):
        seg = [train_time[idx].item()]
        while tmp + rk_step <= train_time[idx + 1]:
            tmp += rk_step
            seg.append(tmp)
        if len(seg) > 1:
            inter_time_seg.append([idx, seg])

    inter_param, inter_time = [], []
    for s, time_ in inter_time_seg:
        time_ = torch.Tensor(time_).to(device)
        s_param = torch.cat([p.flatten() for p in model.pred_model[s].parameters()])
        s_embed = model.encoder(s_param)
        tmp_param = odeint(model.odefunc, s_embed, time_, method='rk4', options={'step_size': rk_step})[1:]
        tmp_param = model.decoder(tmp_param)
        inter_param.append(tmp_param)
        inter_time.append(time_[1:].cpu().numpy())
    inter_param, inter_time = torch.cat(inter_param), np.concatenate(inter_time)

    ## Extrapolation
    extra_time_seg = []
    seg = [train_time[-1].item()]
    for i in np.arange(train_time[-1].item()//rk_step*rk_step+rk_step, train_tasks*2, rk_step):
        seg.append(i)
    extra_time_seg.append([train_tasks - 1, seg])

    extra_param, extra_time = [], []
    for s, time_ in extra_time_seg:
        time_ = torch.Tensor(time_).to(device)
        s_param = torch.cat([p.flatten() for p in model.pred_model[s].parameters()])
        s_embed = model.encoder(s_param)
        tmp_param = odeint(model.odefunc, s_embed, time_, method='rk4', options={'step_size': rk_step})[1:]
        tmp_param = model.decoder(tmp_param)
        extra_param.append(tmp_param)
        extra_time.append(time_[1:].cpu().numpy())
    extra_param, extra_time = torch.cat(extra_param), np.concatenate(extra_time)

    extra_time_seg2 = torch.Tensor([train_time[-1].item()] + [i for i in range(int(train_time[-1].item()//1+1), train_tasks*2)]).to(device)
    s_param = torch.cat([p.flatten() for p in model.pred_model[-1].parameters()])
    s_embed = model.encoder(s_param)
    tmp_param = odeint(model.odefunc, s_embed, extra_time_seg2, method='rk4', options={'step_size': rk_step})[1:]
    extra_param2 = model.decoder(tmp_param)
    extra_time2 = extra_time_seg2[1:].cpu().numpy()

    ## Plot
    inter_figures = []
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    for i in range(len(inter_param)):
        img = plot_decision_boundary(model, ax, inter_param[i:i + 1], t=inter_time[i], ani=True)
        inter_figures.append(img)
    ani = animation.ArtistAnimation(fig, inter_figures, interval=100, blit=True, repeat_delay=5000)
    ani.save(PATH + '/figure_{}_interpolation.gif'.format(repeat), writer='pillow')

    extra_figures = []
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    for i in range(len(extra_param)):
        img = plot_decision_boundary(model, ax, extra_param[i:i + 1], t=extra_time[i], ani=True)
        extra_figures.append(img)
    ani = animation.ArtistAnimation(fig, extra_figures, interval=100, blit=True, repeat_delay=5000)
    ani.save(PATH + '/figure_{}_extrapolation.gif'.format(repeat), writer='pillow')

    extra_tasks = len(extra_param2)
    fig, axs = plt.subplots(math.ceil(extra_tasks / 5), 5, figsize=(15, 3 * math.ceil(extra_tasks / 5)))
    axs = axs.ravel()
    for i in range(extra_tasks):
        plot_decision_boundary(model, axs[i], extra_param2[i:i+1])
        axs[i].set_title('t = {:.2f}'.format(extra_time2[i]))
    plt.tight_layout()
    plt.savefig(PATH + '/figure_{}_extrapolation.png'.format(repeat))

    ##################################### Traj
    inter_traj = np.concatenate([train_model_param.detach().cpu().numpy(), inter_param.detach().cpu().numpy()])
    extra_traj = np.concatenate([inter_param.detach().cpu().numpy(), extra_param.detach().cpu().numpy()])
    if embed_method == 'tsne':
        inter_traj_embedded = TSNE(n_components=2, init='random', random_state=2023).fit_transform(inter_traj)
        extra_traj_embedded = TSNE(n_components=2, init='random', random_state=2023).fit_transform(extra_traj)
    elif embed_method == 'umap':
        inter_traj_embedded_model = umap.UMAP(n_components=2, init='random', random_state=2023).fit(inter_traj)
        inter_traj_embedded = inter_traj_embedded_model.transform(inter_traj)
        extra_traj_embedded = inter_traj_embedded_model.transform(extra_traj)
    elif embed_method == 'pca':
        inter_traj_embedded_model = PCA(n_components=2, random_state=2023).fit(inter_traj)
        inter_traj_embedded = inter_traj_embedded_model.transform(inter_traj)
        extra_traj_embedded = inter_traj_embedded_model.transform(extra_traj)
    plt.figure(figsize=(5, 5))
    plt.scatter(x=inter_traj_embedded[train_tasks:, 0], y=inter_traj_embedded[train_tasks:, 1], c='red')
    plt.plot(inter_traj_embedded[train_tasks:, 0], inter_traj_embedded[train_tasks:, 1], c='red', label='Interpolation')
    plt.scatter(x=inter_traj_embedded[:train_tasks, 0], y=inter_traj_embedded[:train_tasks, 1], c='blue')
    plt.plot(inter_traj_embedded[:train_tasks, 0], inter_traj_embedded[:train_tasks, 1], c='blue', label='Train')
    plt.legend()
    plt.savefig(PATH + '/figure_{}_inter_traj.png'.format(repeat))

    plt.figure(figsize=(5, 5))
    # plt.scatter(x=extra_traj_embedded[:len(inter_param), 0], y=extra_traj_embedded[:len(inter_param), 1], c='red')
    plt.plot(extra_traj_embedded[:len(inter_param), 0], extra_traj_embedded[:len(inter_param), 1], c='red', label='Interpolation')
    # plt.scatter(x=extra_traj_embedded[len(inter_param):, 0], y=extra_traj_embedded[len(inter_param):, 1], c='blue')
    plt.plot(extra_traj_embedded[len(inter_param):, 0], extra_traj_embedded[len(inter_param):, 1], c='blue', label='Extrapolation')
    plt.legend()
    plt.savefig(PATH + '/figure_{}_extra_traj.png'.format(repeat))



def main(repeat):

    dataloaders, time_points = dataset_preparation(num_tasks)
    train_loaders, test_loaders = dataloaders[:-test_tasks], dataloaders[-test_tasks:]
    train_time, test_time = time_points[:-test_tasks], time_points[-test_tasks:]

    model = trainModel(train_loaders, train_time, repeat)
    testModel(test_loaders, test_time, model, repeat)
    # visModel(dataloaders, time_points, model, repeat)


DATANAME = 'Moons50'
MODELNAME = 'ODE'
KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = './save_{}/'.format(MODELNAME) + KEYWORD
#######################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_tasks = 50
data_size = 2
latent_dim = 128
E_shape = data_size * 50 + 50 * 50 + 50 + 50 + 50 + 1
pred_learning_rate = 1e-2
coder_learning_rate = 1e-3
ode_learning_rate = 1e-3
embed_dim = 32

batch_size = 1000
epoch_num = 300

test_tasks = 15
train_tasks = num_tasks - test_tasks
batch_time = 10
train_rate = 0.7
rk_step = 0.2
model_lambda = 100
embed_method = 'tsne'

if __name__ == '__main__':
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)

    for repeat in range(1, 6):
        main(repeat)
