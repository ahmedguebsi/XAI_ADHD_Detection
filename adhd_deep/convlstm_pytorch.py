import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from adhd_classification import data_load
from sklearn.model_selection import train_test_split
import numpy as np
import math
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from torch.autograd import Variable
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        self.lstm1 = nn.Sequential(
            nn.LSTM(input_size=16, hidden_size=10, num_layers=1, batch_first=True, bidirectional=True),
            nn.Dropout(0.5)
        )

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(4 * 2 * 7, 1)

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
        x = x.reshape(-1, 4 * 2 * 7)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        x = F.sigmoid(self.fc1(x))
        return x


net = EEGNet()
print(net)
print(net.forward(Variable(torch.Tensor(np.random.rand(1, 1, 120, 64)))))
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters())


def evaluate(model, X, Y, params=["acc"]):
    results = []
    batch_size = 100

    predicted = []

    for i in range(len(X) / batch_size):
        s = i * batch_size
        e = i * batch_size + batch_size

        inputs = Variable(torch.from_numpy(X[s:e]).cuda(0))
        pred = model(inputs)

        predicted.append(pred.data.cpu().numpy())

    inputs = Variable(torch.from_numpy(X).cuda(0))
    predicted = model(inputs)

    predicted = predicted.data.cpu().numpy()

    for param in params:
        if param == 'acc':
            results.append(accuracy_score(Y, np.round(predicted)))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted)))
        if param == "precision":
            results.append(precision_score(Y, np.round(predicted)))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted))
            recall = recall_score(Y, np.round(predicted))
            results.append(2 * precision * recall / (precision + recall))
    return results

#Datatype - float32 (both X and Y)
#X.shape - (#samples, 1, #timepoints, #channels)
#Y.shape - (#samples)
