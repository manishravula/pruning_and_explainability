import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras import backend as k
from keras.utils import np_utils
from keras.datasets import cifar10
import keras.metrics

from math import ceil
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

import cifar10vgg

import innvestigate
import innvestigate.utils

import gc
import glob

def top_3_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

keras.metrics.top_3_accuracy = top_3_accuracy

label_to_str = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}


def preprocess_image(x):
    #this function is used to normalize instances in production according to saved ta = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))raining set statistics
    # Input: X - a training set
    # Output X - a normalized training set according to normalization constants.
    #these values produced during first training and are general for the standard cifar10 training set normalization
    mean = 120.707
    std = 64.15
    return (x-mean)/(std+1e-7)


def get_lrp_analyzer(weightsFile_name):
    model = keras.models.load_model(weightsFile_name)
    model = innvestigate.utils.model_wo_softmax(model)
    analyzer = innvestigate.create_analyzer("deep_taylor", model)
    return analyzer


def analyze_images(images_batch,analyzer):
    analysis_results = analyzer.analyze(images_batch)
    res = analysis_results
    #Normalizing.
    res = res.sum(axis=np.argmax(np.asarray(res.shape) == 3))
    return res

def get_predictions(name,imgs):
    model = keras.models.load_model(name)
    predictions = model.predict(imgs)
    del model
    gc.collect()
    return predictions

if __name__ == '__main__':

    pruned_model_name = 'first_attempt/model_latest_pruned_0.13669841077690123_0.9630681818181818.h5'
    pruned_model_list = glob.glob('first_attempt/model_latest_pruned_0*')
    pruned_model_list.sorted()
    pruned_model_list.pop()
    original_model_name = 'first_attempt/model_latest_pruned_0.9125416010673124_0.9985795454545454.h5'

    (x_train , y_train), (x_test, y_test) = cifar10.load_data()
    n_ele = 100
    imset_x = x_train[0:0+n_ele]
    imset_y = y_train[0:0+n_ele]

    imset_x = preprocess_image(imset_x)

    #for idx,im in enumerate(imset_x):
    #plt.imsave('result_images/orig_{}.png'.format(idx),im)
    #print("Saved original image {}".format(idx))

    # First get predictions of original model for a set of images.
    orig_pred = get_predictions(original_model_name,imset_x)

    pruned_pred_list = []
    for name in pruned_model_list:
        pruned_pred = get_predictions(name,imset_x)
        pruned_pred_list.append(pruned_pred)

    #Now analyze the pruned model to see what happens.
    alr_original = get_lrp_analyzer(original_model_name)
    ares_original = analyze_images(imset_x, alr_original)

    ares_pruned_list = []
    for name in pruned_model_list:
        alr_pruned = get_lrp_analyzer(name)
        ares_pruned = analyze_images(imset_x, alr_pruned)
        ares_pruned_list.append(ares_pruned)

    hits_orig = 0
    hits_pruned = 0
    n_pruned_models = len(pruned_model_list)

    for idx in range(len(imset_x)):
        pruned_pred = pruned_pred_list[0]
        print(pruned_pred[idx])
        print(orig_pred[idx])
        curr_pruned_pred = np.argmax(pruned_pred[idx])
        curr_orig_pred = np.argmax(orig_pred[idx])

        if curr_pruned_pred == imset_y[idx]:
            hits_pruned += 1
        if curr_orig_pred == imset_y[idx]:
            hits_orig += 1 
        
        print(curr_orig_pred , curr_orig_pred , imset_y[idx])

        #Image from the dataset
        fig,axes = plt.subplots(1,n_pruned_models+2)
       
        axes[0].imshow(imset_x[idx])
        axes[0].set_xlabel(label_to_str[imset_y[idx][0]])
        axes[0].set_title("Original Image")

        #Image from analysis of original model
        a = ares_original[idx]
        a /= np.max(np.abs(a))
        axes[1].imshow(a, cmap="seismic")
        axes[1].set_xlabel(label_to_str[curr_orig_pred])
        axes[1].set_title("Unpruned Model LRP")


        for i in range(n_pruned_models):
            #Image from analysis of pruned model
            a = ares_pruned_list[i][idx]
            a /= np.max(np.abs(a))
            axes[i].imshow(a, cmap="seismic")
            axes[i].set_xlabel(label_to_str[pruned_pred_list[i][idx]])
            axes[i].set_title("Pruned Model {} LRP".format(pruned_model_list[i].split('_')[-2]))

        if (curr_pruned_pred == curr_orig_pred):
            if (curr_pruned_pred == imset_y[idx]):
                img_name = 'result_images/same/correct/res_{}.png'.format(idx)
            else:
                img_name = 'result_images/same/wrong/res_{}.png'.format(idx)
        else:
            img_name = 'result_images/diff/res_{}.png'.format(idx)

        fig.savefig(img_name)
        plt.close(fig)
        print("Saved image to {}".format(img_name))
    print("Accuracy of original model is {}".format(hits_orig/len(imset_x)))
    print("Accuracy of pruned model is {}".format(hits_pruned/len(imset_x)))
