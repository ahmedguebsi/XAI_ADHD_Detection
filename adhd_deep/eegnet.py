#
# (EEGNet)
# V.J.Lawhern,A.J.Solon,N.R.Waytowich,S.M.Gordon,C.P.Hung,
# and B. J. Lance, “Eegnet: a compact convolutional neural
# network for eeg-based brain–computer interfaces,” J. Neural
# Eng., vol. 15, no. 5, p. 056013, 2018.
#
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

PATH_DATASET_MAT= r"C:\Users\Ahmed Guebsi\Downloads\ADHD_part1"


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()

        self.ch = 19
        self.sf = 128
        self.n_class = 1
        hidden_size = 64
        self.half_sf = math.floor(self.sf / 2)
        print(
              f'\nch:{self.ch}',
              f'\nsf:{self.sf}',
              f'\nclasses:{self.n_class}'
              )

        self.F1 = 8
        self.F2 = 16
        self.D = 2  # spatial filters

        self.conv1 = nn.Sequential(
            # temporal kernel size(1, floor(sf*0.5)) means 500ms EEG at sf/2
            # padding=(0, floor(sf*0.5)/2) maintain raw data shape
            #nn.Conv2d(64, (1, self.half_sf), bias=False),  # 62,32
            nn.Conv2d(1, 16, kernel_size=(1, 19), bias=False),
            #nn.Conv2d(1, self.F1, (self.ch, 1), bias=True),
            #nn.Conv2d(in_channels=19, out_channels=32, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(self.F1)
        )
        print(self.conv1)

        self.conv2 = nn.Sequential(
            # spatial kernel size (n_ch, 1)
            nn.Conv2d(self.F1, self.D * self.F1, (self.ch, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D * self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # reduce the sf to sf/4
            nn.Dropout(0.5)  # 0.25 in cross-subject classification beacuse the training size are larger
        )
        print(self.conv2)

        self.Conv3 = nn.Sequential(
            # kernel size=(1, floor((sf/4))*0.5) means 500ms EEG at sf/4 Hz
            nn.Conv2d(self.D * self.F1, self.D * self.F1, (1, math.floor(self.half_sf / 4)), padding=(0, 8),
                      groups=self.D * self.F1, bias=False),
            nn.Conv2d(self.D * self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # dim reduction
            nn.Dropout(0.5)
        )
        print(self.Conv3)

        self.lstm1 = nn.Sequential(
            nn.LSTM(input_size=16, hidden_size=10, num_layers=1, batch_first=True, bidirectional=True),
            nn.Dropout(0.5)
        )
        print(self.lstm1)

        # (floor((sf/4))/2 * timepoint//32, n_class)
        #self.classifier = nn.Linear(math.ceil(self.half_sf / 4) * math.ceil(self.tp // 32), self.n_class, bias=True)
        self.fc = nn.Linear(10, num_classes)
        print(self.fc)

    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.Conv3(x)
        print(x.shape)
        # (-1, sf/8* timepoint//32)
        x = x.view(-1, self.F2 * (self.tp // 32))


        print(x.shape)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out


class EEGNetBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EEGNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size=(1, 64), padding=(0, 32), bias=False)
        self.batchnorm = nn.BatchNorm2d(hidden_size)
        self.activation = nn.ELU()
        self.pooling = nn.AvgPool2d(kernel_size=(1, 4))

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.pooling(x)
        return x
class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ConvLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.block1 = EEGNetBlock(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, 10, num_layers, batch_first=True)
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x):
        x = self.block1(x)
        # Prepare input shape for LSTM
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(x.size(0), x.size(2), -1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class CNNLSTM(nn.Module):
    def __init__(self, num_classes, num_channels=19, num_samples=512, dropout_rate=0.5, weight_decay=1):
        super(CNNLSTM, self).__init__()


        # Define the Conv2d and DepthwiseConv2d blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 64), bias=False),
            nn.BatchNorm2d(64,False),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(64, 1, kernel_size=(num_channels, 1), bias=False, padding=(0, 0), padding_mode='zeros'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout2d(p=dropout_rate)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=(1, 16), padding=(0, 7), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout2d(p=dropout_rate)
        )

        # Define the LSTM layers
        self.lstm1 = nn.LSTM(input_size=16, hidden_size=10, batch_first=True, bidirectional=False, dropout=dropout_rate)
        self.lstm2 = nn.LSTM(input_size=10, hidden_size=10, batch_first=True, bidirectional=False, dropout=dropout_rate)

        # Define the output layer
        self.dense = nn.Linear(10, 1)

    def forward(self, x):
        # Apply the convolutional blocks
        x = self.block1(x)
        x = self.block2(x)

        # Reshape the tensor
        x = x.view(x.size(0), x.size(1), -1)

        # Apply the LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Apply the output layer
        x = self.dense(x[:, -1, :])  # Use only the last time step's output for classification
        #x = torch.sigmoid(x)

        return x

# Create an instance of the model
cnn_lstm_model = CNNLSTM(num_classes=1, num_channels=19, num_samples=512).double()
print(cnn_lstm_model)

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def load_model(model, model_path, use_cuda=False):
    map_location = 'cpu'
    if use_cuda and torch.cuda.is_available():
        map_location = 'cuda:0'
    model.load_state_dict(torch.load(model_path, map_location))
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Define hyperparameters
input_size = 19  # Number of EEG channels
hidden_size = 64
num_layers = 2
num_classes = 2  # Binary classification for ADHD detection

# Create the ConvLSTM model
model = ConvLSTM(input_size, hidden_size, num_layers, num_classes)

# Print the model architecture
print(model)

lstm= EEGNet()
print(lstm)

total_params = count_parameters(model)
print(f"Total number of parameters: {total_params}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from adhd_classification import data_load
from sklearn.model_selection import train_test_split
import numpy as np

# Define your model, input_size, output_size, and other hyperparameters here

# Initialize the model, loss function, and optimizer
#model = YourModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.BCELoss()  # Use an appropriate loss function for your task
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Use an optimizer of your choice



x_data, y_data = data_load(PATH_DATASET_MAT)
x_data = np.swapaxes(x_data, 2, 0)
y_data = np.swapaxes(y_data, 1, 0)
print(y_data[0:600, 1:4])
print('x_data.shape: ', x_data.shape)
print('y_data.shape: ', y_data.shape)

X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(x_data, y_data, test_size=0.2, shuffle=True,random_state=42)
X_train = torch.Tensor(X_train_org)  # Your training data tensor
print(X_train)
print(X_train.shape)
y_train = torch.Tensor(y_train_org)  # Your training labels tensor

print("shaape input",X_test_org.shape[0])
#x_train = X_train_org.reshape(X_train_org.shape[0],1,19, 512)
# Create a TensorDataset from your training data and labels
#train_dataset = TensorDataset(X_train, y_train)
#train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train_org), torch.from_numpy(y_train_org))


x_train = X_train_org.reshape(X_train_org.shape[0], 1, 512, 19)
x_test = X_test_org.reshape(X_test_org.shape[0], 1, 19, 512)

train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_org))
print(train.__len__())
train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
print(train_loader.__len__())

# Define your data loader
batch_size = 32  # Adjust this according to your needs
num_epochs =300
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print(train_loader)
print(train_loader.dataset)


# Define your training loop
def train_model(model, train_loader, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            print(inputs.shape)
            print(labels.shape)
            #inputs = inputs.reshape( 32, 1, 19,512)
            #inputs = np.float64(inputs)
            # Zero the gradients
            # Convert input tensor to double
            inputs= inputs.double()
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update the weights
            optimizer.step()

            running_loss += loss.item()

        # Print the average loss for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_loader)}")


# Train the model
train_model(cnn_lstm_model, train_loader, num_epochs)

# Save the trained model to a file
torch.save(cnn_lstm_model.state_dict(), 'trained_model.pth')
