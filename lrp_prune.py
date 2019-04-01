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
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cifar10vgg

import innvestigate
import innvestigate.utils

import gc
import glob
import ipdb
import pickle, os



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


(x_train , y_train), (x_test, y_test) = cifar10.load_data()
n_ele = 50
imset_x = x_train[0:0+n_ele]
imset_y = y_train[0:0+n_ele]

print("Preprocessing Images \n ")
imset_x = preprocess_image(imset_x)
print("Preprocessing done \n ")

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
    return np.argmax(predictions,axis=1)

def make_result(pruned_model_list,ares_list,pred_list,imset_x):
    hits_orig = 0
    hits_pruned = 0
    n_pruned_models = len(pruned_model_list)

    for idx in range(len(imset_x)):

        most_pruned_pred = pred_list[-1][idx]
        curr_orig_pred = pred_list[0][idx]

        if most_pruned_pred == imset_y[idx]:
            hits_pruned += 1
        if curr_orig_pred == imset_y[idx]:
            hits_orig += 1 
        
        if (most_pruned_pred == curr_orig_pred):
            if (most_pruned_pred == imset_y[idx]):
                img_name = 'result_images/same/correct/res_{}.png'.format(idx)
            else:
                img_name = 'result_images/same/wrong/res_{}.png'.format(idx)
        else:
            img_name = 'result_images/diff/res_{}.png'.format(idx)


        print(most_pruned_pred , curr_orig_pred , imset_y[idx])

        #Image from the dataset
        fig,axes = plt.subplots(1,n_pruned_models+4)
       
        axes[0].imshow(imset_x[idx])
        axes[0].set_xlabel(label_to_str[imset_y[idx][0]])
        axes[0].set_title("Original Image")

        ares_original = ares_list[0]
        #Image from analysis of original model
        a = ares_original[idx]
        a /= np.max(np.abs(a))
        axes[1].imshow(a, cmap="seismic")
        axes[1].set_xlabel(label_to_str[curr_orig_pred])
        axes[1].set_title("Original")
        orig_lrp = a

        ares_pruned_list = ares_list[1:]
        pruned_pred_list = pred_list[1:]

        for i in range(n_pruned_models):
            #Image from analysis of pruned model
            a = ares_pruned_list[i][idx]
            a /= np.max(np.abs(a))
            im = axes[i+2].imshow(a, cmap="seismic")
            axes[i+2].set_xlabel(label_to_str[pruned_pred_list[i][idx]])
            axes[i+2].set_title("{}% Pruned".format(100-int(float(pruned_model_list[i].split('_')[-2]) * 100)))
            pruned_lrp = a

        a = (pruned_lrp - orig_lrp)
        a = a*(a>0)
        im2 = axes[-2].imshow(a, cmap="seismic")
        axes[-2].set_title("P-O Diff")
        #cbar = fig.colorbar(im,ax=axes[-1],fraction=0.05,pad=0.04,drawedges=False)

        a = -(pruned_lrp - orig_lrp)
        a = a*(a>0)
        axes[-1].imshow(a, cmap="seismic")
        axes[-1].set_title("O-P Diff")
        #cbar = fig.colorbar(im2,ax=axes[-1],fraction=0.05,pad=0.04,drawedges=False)


        fig.savefig(img_name)
        plt.close(fig)
        print("Saved image to {}".format(img_name))
    print("Accuracy of original model is {}".format(hits_orig/len(imset_x)))
    print("Accuracy of pruned model is {}".format(hits_pruned/len(imset_x)))

if __name__ == '__main__':


    if not os.path.isfile('first'):
        pruned_model_name = 'first_attempt/model_latest_pruned_0.13669841077690123_0.9630681818181818.h5'
        pruned_model_list = glob.glob('first_attempt/model_latest_pruned_0*')
        pruned_model_list.sort()
        pruned_model_list.pop()
        pruned_model_list = pruned_model_list[:1]
        original_model_name = 'first_attempt/model_latest_pruned_0.9125416010673124_0.9985795454545454.h5'

                #for idx,im in enumerate(imset_x):
        #plt.imsave('result_images/orig_{}.png'.format(idx),im)
        #print("Saved original image {}".format(idx))

        # First get predictions of original model for a set of images.

        pred_list = []
        pred_list.append(get_predictions(original_model_name,imset_x))

        for name in pruned_model_list:
            print("Getting predictions of {} \n".format(name))
            pred = get_predictions(name,imset_x)
            pred_list.append(pred)

        #Now analyze the pruned model to see what happens.
        ares_list = [] 
        alr_original = get_lrp_analyzer(original_model_name)
        ares_original = analyze_images(imset_x, alr_original)
        ares_list.append(ares_original)

        for name in pruned_model_list:
            print("Deriving analyzer of {} \n".format(name))
            alr_pruned = get_lrp_analyzer(name)
            ares_pruned = analyze_images(imset_x, alr_pruned)
            ares_list.append(ares_pruned)
        pickle.dump((pruned_model_list,ares_list,pred_list),open('first','wb'))

    else:
        pruned_model_list,ares_list,pred_list = pickle.load(open('first','rb'))

    ipdb.set_trace()
    make_result(pruned_model_list,ares_list,pred_list,imset_x)


