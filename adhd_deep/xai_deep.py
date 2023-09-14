import tensorflow as tf
import torch

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()

from adhd_classification import data_load
from eeg_visualize import test_visualize
from sklearn.model_selection import train_test_split
import numpy as np

import deeplift
from deeplift.layers import NonlinearMxtsMode
from deeplift.conversion import kerasapi_conversion as kc
import shap
from lime import lime_tabular

print("TensorFlow version:", tf.__version__)


PATH_DATASET_MAT= r"C:\Users\Ahmed Guebsi\Downloads\ADHD_part1"





if __name__ == "__main__":
    params = {
        'subject_num': 144, 'HC_num': 44, 'ADD_num': 52, 'ADHD_num': 48,
        'HC_trials': 10129, 'ADD_trials': 13031, 'ADHD_trials': 10742,
        'output_dir': r'C:\Users\Ahmed Guebsi\Desktop\ahmed_files',
        'strategy': tf.distribute.MultiWorkerMirroredStrategy(),
        # tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        'num_epochs': 300, 'batch_size': 32,
        'k': 5, 'lr': 1e-3,
        'kernels': 1, 'chans': 19, 'samples': 512,
        'mode_': 'k-fold', 'acc': [],
        'model_weights_path': '', 'model_structure_path': ''
    }
    # ----------------------------------------------data load------------------------

    saved_model_file = r"C:\Users\Ahmed Guebsi\Desktop\ahmed_files\test\saved_models\model-weights-1.hdf5"
    keras_model = tf.keras.models.load_model(saved_model_file)
    summary = keras_model.summary()
    print(summary)

    x_data, y_data = data_load(PATH_DATASET_MAT)
    x_data = np.swapaxes(x_data, 2, 0)
    y_data = np.swapaxes(y_data, 1, 0)
    print(y_data[0:600, 1:4])

    print('x_data.shape: ', x_data.shape)
    print('y_data.shape: ', y_data.shape)

    # train_main(params, x_data, y_data)

    # print('y_test subject index : ', y_data[11000:19000, 0])
    #X_test, y_test, y_idx = x_data[11150:11400, :], y_data[11150:11400, 1:4], y_data[11150:11400, 0]

    X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(x_data, y_data, test_size=0.2, shuffle=True,
                                                                        random_state=42)
    print(X_test_org.shape)
    print(y_test_org.shape)
    print(X_test_org)
    print(y_test_org)

    from xai_all import InterpretableCNN
    from torchsummary import summary
    import numpy as np
    # Initialize an instance of your model
    model = InterpretableCNN()
    inputs =(1,19,512)
    input_tensor =torch.tensor(np.array(inputs))
    input_data = input_tensor.double()
    print(summary(model, inputs))
    # Load the pre-trained weights
    checkpoint = torch.load('trained_model.pth')

    print(checkpoint.keys())
    model.load_state_dict(checkpoint)
    model.eval()  # Set the model to evaluation mode

    x_data, y_data = data_load(PATH_DATASET_MAT)
    x_data = np.swapaxes(x_data, 2, 0)
    y_data = np.swapaxes(y_data, 1, 0)
    print(y_data[0:600, 1:4])

    print('x_data.shape: ', x_data.shape)
    print('y_data.shape: ', y_data.shape)

    # train_main(params, x_data, y_data)

    # print('y_test subject index : ', y_data[11000:19000, 0])
    #X_test, y_test, y_idx = x_data[11150:11400, :], y_data[11150:11400, 1:4], y_data[11150:11400, 0]

    X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(x_data, y_data, test_size=0.2, shuffle=True,
                                                                        random_state=42)
    print(X_test_org.shape)
    print(y_test_org.shape)
    print(X_test_org)
    print(y_test_org)

    background = X_train_org[np.random.choice(X_train_org.shape[0], 100, replace=False)]
    print(background.shape)
    x_train = X_train_org.reshape(X_train_org.shape[0], 1, 19, 512)
    print("exp xtrain",x_train.shape)

    # explain predictions of the model on four images
    e = shap.DeepExplainer(model, x_train)
    # ...or pass tensors directly
    # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    shap_values = e.shap_values(X_test_org[1:5])
    print(shap_values)

    # plot the feature attributions
    shap.image_plot(shap_values, -x_test[1:5])

    from tf_explain.callbacks.smoothgrad import SmoothGradCallback


    callbacks = [
        SmoothGradCallback(
            validation_data=(x_val, y_val),
            class_index=0,
            num_samples=20,
            noise=1.,
            output_dir=output_dir,
        )
    ]

    keras_model.fit(X_train_org, y_train_org, batch_size=32, epochs=300, callbacks=callbacks)

    from alibi.explainers import IntegratedGradients

    #model = tf.keras.models.load_model("path_to_your_model")

    ig = IntegratedGradients(keras_model,
                             layer=None,
                             taget_fn=None,
                             method="gausslegendre",
                             n_steps=50,
                             internal_batch_size=100)

    # we use the first 100 training examples as our background dataset to integrate over
    explainer = shap.DeepExplainer(keras_model, X_train_org[:100])

    # explain the first 10 predictions
    # explaining each prediction requires 2 * background dataset size runs
    shap_values = explainer.shap_values(X_test_org[:10])
    print(shap_values)



