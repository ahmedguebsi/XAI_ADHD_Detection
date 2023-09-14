
import torch
import scipy.io as sio
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from xai_all import CompactCNN, EEGNet
from adhd_classification import data_load
from sklearn.model_selection import train_test_split
plt.rcParams.update({'font.size': 12})

#torch.cuda.empty_cache()
#torch.manual_seed(0)
PATH_DATASET_MAT= r"C:\Users\Ahmed Guebsi\Downloads\ADHD_part1"

class FeatureVis():
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_heatmap(self, allsignals, sampleidx, subid, samplelabel, multichannelsignal, likelihood):
        """
        input:
           allsignals:          all the signals in the batch
           sampleidx:           the index of the sample
           subid:               the ID of the subject
           samplelabel:         the ground truth label of the sample
           multichannelsignal:  the signals from all channels for the sample
           likelihood:          the likelihood of the sample to be classified into alert and drowsy state
        """

        if likelihood[0] > likelihood[1]:
            state = 0
        else:
            state = 1

        if samplelabel == 0:
            labelstr = 'normal'
        else:
            labelstr = 'adhd'

        fig = plt.figure(figsize=(14, 6))
        fig.suptitle('Subject:' + str(int(subid)) + '   ' + 'Label:' + labelstr + '   ' + '$P_{normal}=$' + str(
            round(likelihood[0], 2)) + '   $P_{adhd}=$' + str(round(likelihood[1], 2)))  # , fontsize=12)

        # devide the figure layout
        gridlayout = gridspec.GridSpec(ncols=2, nrows=3, figure=fig, wspace=0.2, hspace=0.5)
        axis0 = fig.add_subplot(gridlayout[0:2, 0])
        axis1 = fig.add_subplot(gridlayout[2, 0])
        axis2 = fig.add_subplot(gridlayout[0:3, 1])

        # do some preparations

        rawsignal = allsignals[sampleidx].cpu().detach().numpy().squeeze()
        print("tyyyyyyyyyyyyyyyyyyyyyyype",type(rawsignal))
        print(rawsignal)
        print("rawsignal",rawsignal.shape)
        channelnum = multichannelsignal.shape[0]
        print(channelnum)
        samplelength = multichannelsignal.shape[1]
        print(samplelength)
        maxvalue = np.max(np.abs(rawsignal))

        convkernelLength = self.model.kernelLength

        # calculate the heatmap for the sample
        source = self.model.conv(allsignals)
        source = self.model.batch(source)
        source = torch.nn.ELU()(source)

        activations = source[sampleidx].cpu().detach().numpy().squeeze()

        weights = self.model.fc.weight[state].cpu().detach().numpy().squeeze()
        cam = np.matmul(weights, activations)

        heatmap = np.zeros(samplelength)
        halfkerlength = int(convkernelLength / 2)

        heatmap[halfkerlength:(samplelength - halfkerlength + 1)] = cam

        for i in range(halfkerlength - 1):
            heatmap[i] = heatmap[halfkerlength] * i / (halfkerlength - 1)
        for i in range((samplelength - halfkerlength), samplelength):
            heatmap[i] = heatmap[halfkerlength] * (samplelength - halfkerlength + 1 - i) / (halfkerlength)

        heatmap = (heatmap - np.mean(heatmap)) / np.sqrt(np.sum(heatmap ** 2) / (samplelength))

        # calculate the band power components

        psd, freqs = psd_array_multitaper(rawsignal, 128, adaptive=True, normalization='full', verbose=0)
        freq_res = freqs[1] - freqs[0]
        bandpowers = np.zeros(4)

        idx_band = np.logical_and(freqs >= 1, freqs <= 4)
        bandpowers[0] = simps(psd[idx_band], dx=freq_res)
        idx_band = np.logical_and(freqs >= 4, freqs <= 8)
        bandpowers[1] = simps(psd[idx_band], dx=freq_res)
        idx_band = np.logical_and(freqs >= 8, freqs <= 12)
        bandpowers[2] = simps(psd[idx_band], dx=freq_res)
        idx_band = np.logical_and(freqs >= 12, freqs <= 30)
        bandpowers[3] = simps(psd[idx_band], dx=freq_res)

        totalpower = simps(psd, dx=freq_res)
        if totalpower < 0.00000001:
            bandpowers = np.zeros(4)
        else:
            bandpowers /= totalpower

        barx = np.arange(1, 5)
        axis1.bar(barx, bandpowers)
        axis1.set_xlim([0, 5])
        axis1.set_ylim([0, 0.8])

        axis1.set_ylabel("Relative power")

        axis1.set_xticks([1, 2, 3, 4])
        axis1.set_xticklabels(['Delta', 'Theta', 'Alpha', 'Beta'])

        # draw the heatmap
        xx = np.arange(1, (samplelength + 1))
        axis0.set_xticks([])
        axis0.set_ylim([-maxvalue - 10, maxvalue + 10])
        axis0.set_xlim([0, (samplelength + 1)])
        axis0.set_ylabel("mV")

        print("heatmap",heatmap.shape)
        print(heatmap)

        points = np.array([xx, rawsignal]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        print("points shape",points.shape)
        print("segments shape",segments.shape)
        norm = plt.Normalize(-1, 1)
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(heatmap)
        lc.set_linewidth(2)
        axis0.add_collection(lc)
        fig.colorbar(lc, ax=axis0, orientation="horizontal", ticks=[-1, -0.5, 0, 0.5, 1])

        # draw all the signals

        thespan = np.percentile(multichannelsignal, 98)
        print(thespan)
        yttics = np.zeros(channelnum)
        for i in range(channelnum):
            yttics[i] = i * thespan

        axis2.set_ylim([-thespan, thespan * channelnum])
        axis2.set_xlim([0, samplelength + 1])


        labels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8',
               '01', '02']

        plt.sca(axis2)
        plt.yticks(yttics, labels)
        #plt.show()

        heatmap1 = np.zeros((channelnum, samplelength)) - 1
        heatmap1[5, :] = heatmap
        print("heatmap 1",heatmap1)
        xx = np.arange(1, samplelength + 1)
        print("teeeeeeeeeeeeeeest",multichannelsignal)

        for i in range(0, channelnum):
            y = multichannelsignal[i, :] + thespan * (i)
            print("y",y.shape)
            print(y)
            dydx = heatmap1[i, :]
            #dydx=heatmap[i , :]
            print("heatmap i",dydx)
            print("dydx",dydx.shape)

            points = np.array([xx, y]).T.reshape(-1, 1, 2)
            print("points",points.shape)
            print(points)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            print(segments.shape)

            norm = plt.Normalize(-1, 1)
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(dydx)
            lc.set_linewidth(2)
            axis2.add_collection(lc)
            axis2.plot(xx, y, color='black', linewidth=0.5)


        plt.show()



def run():

    x_data, y_data, subIdx = data_load(PATH_DATASET_MAT)
    print("suuuuuuub shape",subIdx.shape)
    print(subIdx)
    x_data = np.swapaxes(x_data, 2, 0)
    y_data = np.swapaxes(y_data, 1, 0)
    subIdx = np.swapaxes(subIdx, 1, 0)
    print(y_data[0:600, 1:4])
    print('x_data.shape: ', x_data.shape)
    print('y_data.shape: ', y_data.shape)
    #label.astype(int)
    subIdx.astype(int)

    X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(x_data, y_data, test_size=0.2, shuffle=True,
                                                                        random_state=42)

    samplenum = y_data.shape[0]
    label = y_data[:, 0]
    print("laaaaaaaaaaabel",label.shape)

    #   there are 11 subjects in the dataset. Each sample is 3-seconds data from 30 channels with sampling rate of 128Hz.

    channelnum = 19
    subjnum = 120
    samplelength = 4
    sf = 128

    #   define the learning rate, batch size and epoches
    lr = 1e-2
    batch_size = 50
    n_epoch = 6

    #   ydata contains the label of samples
    ydata = np.zeros(samplenum, dtype=np.longlong)

    for i in range(samplenum):
        ydata[i] = label[i]

    #   only channel 28 is used, which corresponds to the Oz channel
    selectedchan = [5]
    rawx =x_data
    print("rawx shape",rawx.shape)
    print(type(rawx))

    #   update the xdata and channel number
    xdata = x_data[:, selectedchan, :]
    #xdata = x_data
    print("xdata shape",xdata.shape)
    channelnum = len(selectedchan)
    #  you can set the subject id here


    for i in range(2, 9):

        trainindx = np.where(subIdx != i)[0]
        print("train index",trainindx)
        xtrain = xdata[trainindx]
        x_train = xtrain.reshape(xtrain.shape[0], 1, channelnum, samplelength * sf)
        y_train = ydata[trainindx]

        testindx = np.where(subIdx == i)[0]
        print("test index",testindx)

        xtest = xdata[testindx]
        print("xtest shape",xtest.shape)
        rawxdata = rawx[testindx]
        print("rawxdata shape",rawxdata.shape)
        x_test = xtest.reshape(xtest.shape[0], 1, channelnum, samplelength * sf)
        print("x_test shape",x_test.shape)
        y_test = ydata[testindx]

        train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        my_net = CompactCNN().double()

        optimizer = optim.Adam(my_net.parameters(), lr=lr)
        loss_class = torch.nn.NLLLoss()

        for p in my_net.parameters():
            p.requires_grad = True

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

        my_net.train(False)
        with torch.no_grad():
            x_test = torch.DoubleTensor(x_test)
            answer = my_net(x_test)
            probs = np.exp(answer.cpu().numpy())
            sampleVis = FeatureVis(my_net)

            #   you can set the sample index here
            sampleidx = 1
            print("raw x data shape",rawxdata.shape)
            print(rawxdata[sampleidx].shape)
            print(rawxdata[sampleidx])
            sampleVis.generate_heatmap(allsignals=x_test, sampleidx=sampleidx, subid=i, samplelabel=y_test[sampleidx],
                                       multichannelsignal=rawxdata[sampleidx], likelihood=probs[sampleidx])


if __name__ == '__main__':
    run()