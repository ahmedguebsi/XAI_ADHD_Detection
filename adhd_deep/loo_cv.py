import torch
import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score
import torch.optim as optim
from xai_all import CompactCNN
from adhd_classification import data_load
from sklearn.model_selection import train_test_split

#torch.cuda.empty_cache()
#torch.manual_seed(0)

def run():
    PATH_DATASET_MAT = r"C:\Users\Ahmed Guebsi\Downloads\ADHD_part1"

    x_data, y_data, subIdx = data_load(PATH_DATASET_MAT)
    print("suuuuuuub shape",subIdx.shape)
    print(max(subIdx))
    x_data = np.swapaxes(x_data, 2, 0)
    y_data = np.swapaxes(y_data, 1, 0)
    subIdx = np.swapaxes(subIdx, 1, 0)
    print(y_data[0:600, 1:4])
    print('x_data.shape: ', x_data.shape)
    print('y_data.shape: ', y_data.shape)
    #label.astype(int)
    subIdx.astype(int)

    X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(x_data, y_data, test_size=0.2, shuffle=True,random_state=42)
    samplenum = y_data.shape[0]
    label = y_data[:, 0]
    print("laaaaaaaaaaabel",label.shape)

    channelnum = 19
    subjnum = 120
    samplelength = 4
    sf = 128

    lr = 1e-2
    batch_size = 50
    n_epoch = 6

    # ydata contains the label of samples
    ydata = np.zeros(samplenum, dtype=np.longlong)

    for i in range(samplenum):
        ydata[i] = label[i]

    # only channel 5 is used, which corresponds to the Fz channel
    selectedchan = [5]

    #   update the xdata and channel number
    xdata = x_data[:, selectedchan, :]
    channelnum = len(selectedchan)

    #   the result stores accuracies of every subject
    results = np.zeros(subjnum)

    #   it performs leave-one-subject-out training and classfication
    for i in range(1, subjnum + 1):

        # form the training data all subjects except i
        trainindx = np.where(subIdx != i)[0]
        xtrain = xdata[trainindx]
        x_train = xtrain.reshape(xtrain.shape[0], 1, channelnum, samplelength * sf)
        y_train = ydata[trainindx]

        # form the testing data subject i
        testindx = np.where(subIdx == i)[0]
        xtest = xdata[testindx]
        x_test = xtest.reshape(xtest.shape[0], 1, channelnum, samplelength * sf)
        y_test = ydata[testindx]

        train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        # load the CNN model to deal with 1D EEG signals
        my_net = CompactCNN().double()

        optimizer = optim.Adam(my_net.parameters(), lr=lr)
        loss_class = torch.nn.NLLLoss()

        for p in my_net.parameters():
            p.requires_grad = True

        # train the classifier
        for epoch in range(n_epoch):
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data

                input_data = inputs
                class_label = labels

                my_net.zero_grad()
                my_net.train()

                class_output = my_net(input_data)
                err_s_label = loss_class(class_output, class_label)
                err = err_s_label

                err.backward()
                optimizer.step()

        # test the results
        my_net.train(False)
        with torch.no_grad():
            x_test = torch.DoubleTensor(x_test)
            answer = my_net(x_test)
            probs = answer.cpu().numpy()
            preds = probs.argmax(axis=-1)
            acc = accuracy_score(y_test, preds)

            print(acc)
            results[i - 1] = acc

    print('mean accuracy:', np.mean(results))


if __name__ == '__main__':
    run()
