import torch
import scipy.io as sio
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import mne

from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
from adhd_classification import data_load
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score,confusion_matrix,roc_curve,f1_score

from electrodes_positions import get_electrodes_coordinates, set_electrodes_montage, get_electrodes_positions
from early_stopping import EarlyStopping
PATH_DATASET_MAT= r"C:\Users\Ahmed Guebsi\Downloads\ADHD_part1"


torch.cuda.empty_cache()
torch.manual_seed(0)
np.random.seed(0)
plt.rcParams.update({'font.size': 14})


class EEGNet(torch.nn.Module):
    def __init__(self, channelnum=19):
        super(EEGNet, self).__init__()

        # model parameters
        self.eps = 1e-05

        self.f1 = 8
        self.d = 2
        self.conv1 = torch.nn.Conv2d(1, self.f1, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = torch.nn.BatchNorm2d(self.f1, track_running_stats=False)
        self.batchnorm2 = torch.nn.BatchNorm2d(self.f1 * self.d, track_running_stats=False)
        self.batchnorm3 = torch.nn.BatchNorm2d(self.f1 * self.d, track_running_stats=False)
        self.activ1 = torch.nn.ELU()
        self.activ2 = torch.nn.ELU()
        self.depthconv = torch.nn.Conv2d(self.f1, self.f1 * self.d, (19, 1), groups=self.f1, bias=False)
        self.avgpool = torch.nn.AvgPool2d((1, 4))
        self.separable = torch.nn.Conv2d(self.f1 * self.d, self.f1 * self.d, (1, 16), padding=(0, 8),
                                         groups=self.f1 * self.d, bias=False)
        self.fc1 = torch.nn.Linear(256, 2)  # 128
        self.softmax = nn.LogSoftmax(dim=1)
        self.softmax1 = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)

        # parameters for the interpretation techniques
        self.batch_mean1 = 0
        self.batch_std1 = 0
        self.gamma1 = 0
        self.beta1 = 0
        self.batch_mean2 = 0
        self.batch_std2 = 0
        self.gamma2 = 0
        self.beta2 = 0
        self.batch_mean3 = 0
        self.batch_std3 = 0
        self.gamma3 = 0
        self.beta3 = 0
        self.activ_in1 = 0
        self.activ_out1 = 0
        self.activ_baseline_in1 = 0
        self.activ_baseline_out1 = 0
        self.activ_in2 = 0
        self.activ_out2 = 0
        self.activ_baseline_in2 = 0
        self.activ_baseline_out2 = 0

    def forward(self, inputdata):
        intermediate = self.conv1(inputdata)

        intermediate = self.batchnorm1(intermediate)

        intermediate = self.depthconv(intermediate)

        intermediate = self.batchnorm2(intermediate)

        intermediate = self.activ1(intermediate)

        intermediate = F.avg_pool2d(intermediate, (1, 4))

        intermediate = self.dropout(intermediate)

        intermediate = self.separable(intermediate)

        intermediate = self.batchnorm3(intermediate)

        intermediate = self.activ2(intermediate)

        intermediate = F.avg_pool2d(intermediate, (1, 8))

        intermediate = self.dropout(intermediate)

        intermediate = intermediate.view(intermediate.size()[0], -1)

        intermediate = self.fc1(intermediate)

        output = self.softmax(intermediate)
        print(output.shape)
        print(output)

        return output

    def update_softmax_forward(self):
        def softmax_forward_hook_function(module, ten_in, ten_out):
            return ten_in[0]

        handle = self.softmax.register_forward_hook(softmax_forward_hook_function)

        return handle

    # make the batch normalization layer a linear operation before applying backpropagation to remove the effects of other samples in the batch

    def update_batch_forward(self):
        def batch_forward_hook_function1(module, ten_in, ten_out):
            data = ten_in[0]
            batchmean1 = self.batch_mean1.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),
                                                 int(data.size(3)))
            batchstd1 = self.batch_std1.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),
                                               int(data.size(3)))

            data = torch.div((ten_in[0] - batchmean1), batchstd1)
            gammamatrix = (self.gamma1).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),
                                                      int(data.size(3)))
            betamatrix = (self.beta1).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),
                                                    int(data.size(3)))

            output = data * gammamatrix + betamatrix

            return output

        def batch_forward_hook_function2(module, ten_in, ten_out):
            data = ten_in[0]
            batchmean2 = self.batch_mean2.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),
                                                 int(data.size(3)))
            batchstd2 = self.batch_std2.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),
                                               int(data.size(3)))

            data = torch.div((ten_in[0] - batchmean2), batchstd2)
            gammamatrix = (self.gamma2).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),
                                                      int(data.size(3)))
            betamatrix = (self.beta2).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),
                                                    int(data.size(3)))

            output = data * gammamatrix + betamatrix

            return output

        def batch_forward_hook_function3(module, ten_in, ten_out):
            data = ten_in[0]
            batchmean3 = self.batch_mean3.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),
                                                 int(data.size(3)))
            batchstd3 = self.batch_std3.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),
                                               int(data.size(3)))

            data = torch.div((ten_in[0] - batchmean3), batchstd3)
            gammamatrix = (self.gamma3).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),
                                                      int(data.size(3)))
            betamatrix = (self.beta3).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),
                                                    int(data.size(3)))

            output = data * gammamatrix + betamatrix

            return output

        handle1 = self.batchnorm1.register_forward_hook(batch_forward_hook_function1)
        handle2 = self.batchnorm2.register_forward_hook(batch_forward_hook_function2)
        handle3 = self.batchnorm3.register_forward_hook(batch_forward_hook_function3)

        return [handle1, handle2, handle3]

    # Save the batch mean and std

    def update_batch_forward_meanstd(self):
        def batch_forward_hook_function1(module, ten_in, ten_out):
            data = ten_in[0].clone().detach().requires_grad_(False).cpu().double()

            self.batch_mean1 = torch.mean(data, [0, 2, 3], True)
            self.batch_std1 = torch.sqrt(torch.mean((data - self.batch_mean1) ** 2, [0, 2, 3], True) + self.eps)

            self.gamma1 = torch.DoubleTensor(1, ten_in[0].size(1), 1, 1)
            self.beta1 = torch.DoubleTensor(1, ten_in[0].size(1), 1, 1)

            self.gamma1[0, :, 0, 0] = self.batchnorm1.weight.clone().detach().requires_grad_(False).cpu()
            self.beta1[0, :, 0, 0] = self.batchnorm1.bias.clone().detach().requires_grad_(False).cpu()

        def batch_forward_hook_function2(module, ten_in, ten_out):
            data = ten_in[0].clone().detach().requires_grad_(False).cpu().double()

            self.batch_mean2 = torch.mean(data, [0, 2, 3], True)
            self.batch_std2 = torch.sqrt(torch.mean((data - self.batch_mean2) ** 2, [0, 2, 3], True) + self.eps)

            self.gamma2 = torch.DoubleTensor(1, ten_in[0].size(1), 1, 1)
            self.beta2 = torch.DoubleTensor(1, ten_in[0].size(1), 1, 1)

            self.gamma2[0, :, 0, 0] = self.batchnorm2.weight.clone().detach().requires_grad_(False).cpu()
            self.beta2[0, :, 0, 0] = self.batchnorm2.bias.clone().detach().requires_grad_(False).cpu()

        def batch_forward_hook_function3(module, ten_in, ten_out):
            data = ten_in[0].clone().detach().requires_grad_(False).cpu().double()

            self.batch_mean3 = torch.mean(data, [0, 2, 3], True)
            self.batch_std3 = torch.sqrt(torch.mean((data - self.batch_mean3) ** 2, [0, 2, 3], True) + self.eps)

            self.gamma3 = torch.DoubleTensor(1, ten_in[0].size(1), 1, 1)
            self.beta3 = torch.DoubleTensor(1, ten_in[0].size(1), 1, 1)

            self.gamma3[0, :, 0, 0] = self.batchnorm3.weight.clone().detach().requires_grad_(False).cpu()
            self.beta3[0, :, 0, 0] = self.batchnorm3.bias.clone().detach().requires_grad_(False).cpu()

        handle1 = self.batchnorm1.register_forward_hook(batch_forward_hook_function1)
        handle2 = self.batchnorm2.register_forward_hook(batch_forward_hook_function2)
        handle3 = self.batchnorm3.register_forward_hook(batch_forward_hook_function3)

        return [handle1, handle2, handle3]

    def update_activ_forward(self):
        def activ_forward_hook_function1(module, ten_in, ten_out):
            self.activ_in1 = ten_in[0].clone().detach().requires_grad_(False).cpu()
            self.activ_out1 = ten_out.clone().detach().requires_grad_(False).cpu()

        def activ_forward_hook_function2(module, ten_in, ten_out):
            self.activ_in2 = ten_in[0].clone().detach().requires_grad_(False).cpu()
            self.activ_out2 = ten_out.clone().detach().requires_grad_(False).cpu()

        handle1 = self.activ1.register_forward_hook(activ_forward_hook_function1)
        handle2 = self.activ2.register_forward_hook(activ_forward_hook_function2)
        #
        return [handle1, handle2]


    def update_activ_deconvolution(self):
        def activ_backward_hook_function(mmodule, grad_in, grad_out):
            modified_grad = torch.clamp(grad_out[0], min=0.0)

            return (modified_grad,)

        handle1 = self.activ1.register_backward_hook(activ_backward_hook_function)
        handle2 = self.activ2.register_backward_hook(activ_backward_hook_function)
        return [handle1, handle2]

    def update_activ_guidedbackpropogation(self):
        def activ_backward_hook_function1(mmodule, grad_in, grad_out):
            forwardpass = torch.where(self.activ_out1 > 0, torch.ones_like(self.activ_out1),torch.zeros_like(self.activ_out1))
            modified_grad = forwardpass * torch.clamp(grad_out[0], min=0.0)

            return (modified_grad,)

        def activ_backward_hook_function2(mmodule, grad_in, grad_out):
            forwardpass = torch.where(self.activ_out2 > 0, torch.ones_like(self.activ_out2),torch.zeros_like(self.activ_out2))
            modified_grad = forwardpass * torch.clamp(grad_out[0], min=0.0)

            return (modified_grad,)

        handle1 = self.activ1.register_backward_hook(activ_backward_hook_function1)
        handle2 = self.activ2.register_backward_hook(activ_backward_hook_function2)
        return [handle1, handle2]


class VisTech():
    def __init__(self, model):
        self.model = model
        self.model.eval()

        self.eps = 0.000001
        self.method = None

    def enhanceheatmap(self, heatmap, r=5):

        sampleChannel = heatmap.shape[0]
        sampleLength = heatmap.shape[1]

        newmap = np.zeros((sampleChannel, sampleLength))
        for i in range(sampleChannel):
            for j in range(sampleLength):
                if j < r:
                    newmap[i, j] = np.mean(heatmap[i, :j + r])
                elif j + r > sampleLength:
                    newmap[i, j] = np.mean(heatmap[i, j - r:])
                else:
                    newmap[i, j] = np.mean(heatmap[i, j - r:j + r])

        return newmap

    def convert_batchlayer_to_linear(self, batchInput):

        handles = self.model.update_batch_forward_meanstd()
        self.model(batchInput)
        self.remove_registered_functions(handles)
        handles = self.model.update_batch_forward()

        return handles

    def remove_registered_functions(self, handles):
        for handle in handles:
            handle.remove()

    def heatmap_calculation_backpropogation(self, batchInput, sampleidx, method='EpsilonLRP'):
        # This function output the heatmaps generate with different interpretation techniques.
        # Most of the techques can be achieved by modifying the nonlinear activation layers

        def calculate_one_hot_out_put(output):
            result = output.cpu().detach().numpy()
            preds = result.argmax(axis=-1)
            one_hot_output = np.zeros(result.shape)

            for i in range(preds.shape[0]):
                one_hot_output[i, preds[i]] = 1

            one_hot_output = torch.DoubleTensor(one_hot_output)

            return one_hot_output

        sampleInput = batchInput
        sampleInput.requires_grad = True

        handles0 = self.convert_batchlayer_to_linear(batchInput)

        if method == "guidedbackpropogation":
            handles1 = self.model.update_activ_forward()
            handles2 = self.model.update_activ_guidedbackpropogation()

            output = self.model(sampleInput)
            one_hot_output = calculate_one_hot_out_put(output)
            output.backward(gradient=one_hot_output)
            grad = sampleInput.grad
            heatmap = grad.cpu().detach().numpy().squeeze()

            self.remove_registered_functions(handles1 + handles2)


        elif method == "Saliencymap":
            output = self.model(sampleInput)

            one_hot_output = calculate_one_hot_out_put(output)
            output.backward(gradient=one_hot_output)
            grad = sampleInput.grad
            heatmap = grad.cpu().detach().numpy().squeeze()


        self.remove_registered_functions(handles0)
        # the methods will generate heatmaps for a batch, otherwise return the heatmap for a sample
        if sampleidx != None:
            heatmap = heatmap[sampleidx]

        return heatmap


    def generate_interpretation(self, batchInput, sampleidx, subid, samplelabel, likelihood, method):

        if likelihood[0] > likelihood[1]: #likelihood of the sample to be classified into normal and adhd state
            state = 0
        else:
            state = 1

        if samplelabel == 0:
            labelstr = 'normal'
        else:
            labelstr = 'adhd'

        sampleInput = batchInput[sampleidx].cpu().detach().numpy().squeeze()
        sampleChannel = sampleInput.shape[0]
        sampleLength = sampleInput.shape[1]

        channelnames =['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8','P7', 'P3', 'Pz', 'P4', 'P8', '01', '02']


        heatmap_sample_thres = 2
        heatmap_channel_thres = 1

        # generate the original sample and channel contribution maps
        heatmap = self.heatmap_calculation_backpropogation(batchInput=batchInput, sampleidx=sampleidx, method=method)
        heatmap_channel = np.mean(heatmap, axis=1)


        # Step 1: normalization
        heatmap = (heatmap - np.mean(heatmap)) / (np.std(heatmap))
        heatmap_channel = (heatmap_channel - np.mean(heatmap_channel)) / (np.std(heatmap_channel))

        # Step 2: thresholding
        heatmap_channel = heatmap_channel - heatmap_channel_thres
        heatmap = heatmap - heatmap_sample_thres

        # set values below lower bound of color map -1 to -1
        for u in range(sampleChannel):
            for l in range(sampleLength):
                if heatmap[u, l] < -1:
                    heatmap[u, l] = -1
                # Step 3: smoothing
        smooth_factor = 5
        heatmap = self.enhanceheatmap(heatmap, smooth_factor)



        # draw the figure
        rowdivide = 4
        fig = plt.figure(figsize=(15, 9))
        gridlayout = gridspec.GridSpec(ncols=2, nrows=rowdivide, figure=fig, wspace=0.05, hspace=0.3)
        axs0 = fig.add_subplot(gridlayout[0:rowdivide - 1, 0])
        axs1 = fig.add_subplot(gridlayout[0:rowdivide - 1, 1])
        axs2 = fig.add_subplot(gridlayout[rowdivide - 1, :])

        axs2.xaxis.set_ticks([])
        axs2.yaxis.set_ticks([])

        # display the  results
        axs2.text(0.01, 0.8, 'Model: EEGNET   Interpretation: ' + method ,horizontalalignment='left', fontsize=15)
        fig.suptitle('Subject:' + str(int(subid)) + '   ' + 'Label:' + labelstr + '   ' + '$P_{normal}=$' + str(
            round(likelihood[0], 2)) + '   $P_{adhd}=$' + str(round(likelihood[1], 2)), y=0.985, fontsize=17)


        thespan = np.percentile(sampleInput, 98)
        xx = np.arange(1, sampleLength + 1)

        for i in range(0, sampleChannel):
            y = sampleInput[i, :] + thespan * (sampleChannel - 1 - i)
            dydx = heatmap[i, :]

            points = np.array([xx, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(-1, 1)
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(dydx)
            lc.set_linewidth(2)
            axs0.add_collection(lc)

        yttics = np.zeros(sampleChannel)
        for gi in range(sampleChannel):
            yttics[gi] = gi * thespan

        axs0.set_ylim([-thespan, thespan * sampleChannel])
        axs0.set_xlim([0, sampleLength + 1])
        axs0.set_xticks([1, 128, 256, 384,512])
        axs0.set_xticklabels(['0', '1', '2','3', '4(s)'])

        inversechannelnames = []
        for i in range(sampleChannel):
            inversechannelnames.append(channelnames[sampleChannel - 1 - i])

        plt.sca(axs0)
        plt.yticks(yttics, inversechannelnames)

        montage = 'standard_1020'
        sfreq = 128

        info = mne.create_info(
            channelnames,
            ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', \
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg', \
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg', \
                      'eeg', 'eeg', 'eeg', 'eeg'],
            sfreq=sfreq
        )

        electrodes_coordinates = get_electrodes_coordinates(channelnames)
        # print(electrodes_coordinates)
        dig_points = get_electrodes_positions(channelnames, electrodes_coordinates)
        _,info = set_electrodes_montage(channelnames, electrodes_coordinates,sampleInput)

        im, cn = mne.viz.plot_topomap(data=heatmap_channel, pos=info, vmin=-1, vmax=1, axes=axs1, names=channelnames,
                                      show_names=True, outlines='head', cmap='viridis', show=False)
        fig.colorbar(im, ax=axs1)
        plt.show()

def plot_roc(fpr, tpr):
    plt.plot(fpr, tpr, label = 'ROC curve', linewidth = 2)
    plt.plot([0,1],[0,1], 'k--', linewidth = 2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve ')
    plt.show()
def plot_cm(cm):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.tight_layout()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

def specificity(y_true, y_pred):
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp)
def sensitivity(y_true, y_pred):
    tp = sum((y_true == 1) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn)


def run():


    channelnum = 19
    subjnum =120
    samplelength = 4
    sf = 128

    #   define the learning rate, batch size and epoches
    lr = 1e-3
    batch_size = 32
    n_epoch = 2

    x_data, y_data, subIdx = data_load(PATH_DATASET_MAT)
    x_data = np.swapaxes(x_data, 2, 0)
    y_data = np.swapaxes(y_data, 1, 0)
    subIdx = np.swapaxes(subIdx, 1, 0)
    print(y_data[0:600, 1:4])
    print('x_data.shape: ', x_data.shape)
    print('y_data.shape: ', y_data.shape)
    print(subIdx)
    subIdx.astype(int)


    samplenum = y_data.shape[0]
    label = y_data[:, 0]
    print("laaaaaaaaaaabel",label.shape)
    print(np.unique(subIdx))

    #   ydata contains the label of samples
    ydata = np.zeros(samplenum, dtype=np.longlong)

    #   the result stores accuracies of every subject
    results = []

    for i in range(samplenum):
        ydata[i] = label[i]

    X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(x_data, y_data, test_size=0.2, shuffle=True,
                                                                        random_state=42)

    # select the subject index here
    for i in range(1, subjnum + 1):
        #       form the training data
        trainindx = np.where(subIdx != i)[0]
        xtrain = x_data[trainindx]
        x_train = xtrain.reshape(xtrain.shape[0], 1, channelnum, samplelength * sf)
        y_train = ydata[trainindx]

        #       form the testing data
        testindx = np.where(subIdx == i)[0]
        xtest = x_data[testindx]
        x_test = xtest.reshape(xtest.shape[0], 1, channelnum, samplelength * sf)
        y_test = ydata[testindx]

        train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        # select the deep learning model to be used
        #my_net = InterpretableCNN().double()
        my_net = EEGNet().double()

        for p in my_net.parameters():
            p.requires_grad = True

        optimizer = optim.Adam(my_net.parameters(), lr=lr)
        loss_class = torch.nn.NLLLoss()
        # Define ReduceLROnPlateau scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, verbose=True)

        # Define early stopping parameters
        early_stopping = {
            'patience': 20,  # Number of epochs with no improvement after which training will be stopped
            'min_delta': 0.001,  # Minimum change in validation loss to be considered as an improvement
            'best_loss': float('inf'),  # Initialize with a large value
            'counter': 0  # Counter for the number of epochs with no improvement
        }

        # Tensorboard writer for logging


        train_accuracies = []
        val_accuracies = []

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        patience = 20
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        # train the classifier
        for epoch in range(1,n_epoch+1):
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data

                input_data = inputs
                class_label =labels
                #class_label = labels.view(-1, 1 ).double() for BCELoss

                train_loss =0.0

                my_net.zero_grad()
                my_net.train()

                class_output = my_net(input_data)
                err_s_label = loss_class(class_output, class_label)
                err = err_s_label

                err.backward()
                optimizer.step()
                train_loss += err.item()
                # record training loss
                train_losses.append(err.item())

            # Calculate average training loss for the epoch
            avg_train_loss = train_loss / len(train_loader)
            print("train loss avg",avg_train_loss)

        my_net.eval()
        val_loss =0.0
        with torch.no_grad():

            x_test = torch.DoubleTensor(x_test)
            answer = my_net(x_test)
            print("y_test",y_test)
            y_test = torch.from_numpy(y_test)
            #y_test=y_test.view(-1,1).double() for BCELoss
            print(type(y_test))
            loss = loss_class(answer, y_test)
            val_loss += loss.item()
            valid_losses.append(loss.item())
            probs = np.exp(answer.cpu().numpy())
            print("probs",probs)

            preds = probs.argmax(axis=-1)
            print("preds",preds)
            acc = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds)
            recall = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            print(acc)
            print(precision)
            print(recall)
            print(f1)
            print("val loss",val_loss)
            print(valid_losses)
            print(specificity(y_test, preds))
            print(sensitivity(y_test, preds))
            results.append(acc)
            fpr, tpr, t = roc_curve(y_test, preds)
            cm = confusion_matrix(y_test, preds,labels=[0,1])
            print("conv matrix",cm)
            #plot_roc(fpr, tpr)
            plot_cm(cm)

            # print training/validation statistics
            # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epoch))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epoch:>{epoch_len}}] ' +f'train_loss: {train_loss:.5f} ' +f'valid_loss: {valid_loss:.5f}')
        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # load the last checkpoint with the best model
        #my_net.load_state_dict(torch.load('checkpoint.pt'))

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, my_net)

        if early_stopping.early_stop:
            print("Early stopping")
            break


        print('mean accuracy:', np.mean(results))



        # Save the trained model to a file
        torch.save(my_net.state_dict(), 'trained_cnn_model.pth')
        sampleVis = VisTech(my_net)

        # select the interpretation method to be used
        method="guidedbackpropogation"
        # method="Saliencymap"
        ########################################

        sampleidx = 8
        sampleVis.generate_interpretation(batchInput=x_test, sampleidx=sampleidx, subid=i,
                                          samplelabel=y_test[sampleidx], likelihood=probs[sampleidx], method=method)


torch.cuda.empty_cache()



if __name__ == '__main__':
    run()
