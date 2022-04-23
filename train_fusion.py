# Feature fusion

import os
import time
import utils
import random
import joblib
import argparse
import cv2 as cv
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import NearestNeighbors as KNN
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import io, filters
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from skimage.transform import resize
from sklearn.metrics import accuracy_score
from skimage.measure import moments_hu, moments_central, moments_normalized
from skimage.feature import hog
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='CURE-TSR Training and Evaluation')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--test-path', '-tp', default='ChallengeFree/test', type=str,
                    metavar='S', help='Path to test dataset (default: ChallengeFree)')

def main():
    global args 
    args = parser.parse_args()

    traindir = os.path.join(args.data, 'ChallengeFree/train')
    eval = False
    if eval:
        print('e')
        evaluate(args.data, model_file='svm_fusion_gabor_hu.pkl')
        return

    imgs = make_dataset_svm(traindir, transforms.Resize([28, 28]))

    train_imgs = []
    train_labels = []
    lengthV = []


    # Gabor filters
    kernels = create_kernels()
    cnt=0
    for img, label in imgs:
        train_labels.append(label)
        
        feature_vector, hog_img = hog(np.array(img).astype('uint8'), orientations=9, 
                            pixels_per_cell=(14,14),cells_per_block=(1,1), visualize=True)
        # feature_vector, hog_img = hog(np.array(img).astype('uint8'), orientations=9, 
        #                     pixels_per_cell=(8,8),cells_per_block=(1,1), visualize=True)
    
        cnt+=1
        # Gabor features
        gabor_feats = compute_feats(np.array(img), kernels)
        if cnt>2000:
             gabor_feats = compute_feats(np.array(img), kernels)
        feature_vector = np.append(feature_vector, gabor_feats)

        # HU moments
        gray = rgb2gray(np.array(img))
        mu = moments_central(gray)
        nu = moments_normalized(mu)
        hu_mmts = moments_hu(nu)
        feature_vector = np.append(feature_vector, hu_mmts)

        # LBPs
        radius = 3
        n_points = 8*radius

        lbp = local_binary_pattern(gray, n_points, radius, 'uniform')

        # lengthV.append(len(feature_vector))
        # #add featureVector list to dataset that is fed into svm
        train_imgs.append(feature_vector)


    best_params = gridSearch(train_imgs, train_labels)
    print(best_params)
    # # best_params = {'C': 100.0, 'gamma': 0.1}
    # best_params = {'estimator__gamma': 0.01, 'estimator__C': 10000.0}
    # # Train SVM on the best parameters
    print('Training...')
    clf = OneVsRestClassifier(SVC()).set_params(**best_params)
    clf.fit(train_imgs, train_labels)
    joblib.dump(clf, 'svm_fusion_hu.pkl')

    # scores = cross_val_score(clf, train_imgs, train_labels, cv=10)
    # print(scores)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return    

# Perform randomized search
def gridSearch(train_imgs, train_labels, num_fold = 5):
    # C from 1e-2 to 1e10
    C_range = np.logspace(-2, 10, 13)
    # gamma from 1e-9 to 1e3
    gamma_range = np.logspace(-9, 0, 10)
    param_grid = dict(estimator__gamma=gamma_range, estimator__C=C_range)

    kf = KFold(n_splits = num_fold, shuffle = False)
    grid = RandomizedSearchCV(OneVsRestClassifier(SVC()), cv=kf, n_jobs=int(os.cpu_count()-2), param_distributions=param_grid, scoring='accuracy', verbose=True)
    # grid = GridSearchCV(OneVsRestClassifier(SVC()), cv=kf, n_jobs=int(os.cpu_count()-2), param_grid=param_grid, scoring='accuracy', verbose=True)
    grid.fit(train_imgs, train_labels)

    mean_scores = np.array(grid.cv_results_['mean_test_score'])
    acc = np.max(mean_scores)
#     print(acc)
    best_params = grid.best_params_
    return best_params

def make_dataset_svm(traindir, transform = None):
    imgs = []
    # cnt = 0
    for fname in sorted(os.listdir(traindir)):
        target = int(float(fname[3:5])) - 1
        path = os.path.join(traindir, fname)

        # Load and transform image
        img = utils.pil_loader(path)
        if transform:
            img = transform(img)

        item = (img, target)
        imgs.append(item)
        # cnt += 1
    return imgs

def evaluate(data_path, model_file='svm_hog.pkl'):
    # testdir = os.path.join(data_path, 'ChallengeFree/test')
    testdir = os.path.join(data_path, args.test_path)
    print('Testing..')

    imgs = make_dataset_svm(testdir, transforms.Resize([28, 28]))

    test_imgs = []
    test_labels = []
    cnt = 0 
    kernels = create_kernels()
    for img, label in imgs:
        test_labels.append(label)
        
        feature_vector, hog_img = hog(np.array(img).astype('uint8'), orientations=9, 
                            pixels_per_cell=(14,14),cells_per_block=(1,1), visualize=True)
        # feature_vector, hog_img = hog(np.array(img).astype('uint8'), orientations=9, 
        #                     pixels_per_cell=(8,8),cells_per_block=(1,1), visualize=True)
        
        cnt+=1
        gabor_feats = compute_feats(np.array(img), kernels)
        feature_vector = np.append(feature_vector, gabor_feats)

        gray = rgb2gray(np.array(img))
        mu = moments_central(gray)
        nu = moments_normalized(mu)
        hu_mmts = moments_hu(nu)
        feature_vector = np.append(feature_vector, hu_mmts)

        test_imgs.append(feature_vector)

    clf = joblib.load(model_file)
    acc = accuracy_score(test_labels, clf.predict(test_imgs))
    print(args.test_path, acc*100)

# Compute Gabor features - obatined after filtering the image using Gabor features
def compute_feats(image, kernels, plotp= False):
    feats = np.zeros((len(kernels)*2), dtype=np.double)
    cnt = 0 
    image = rgb2gray(image)
    if plotp:
        plt.imshow(image)
        plt.show()
    for kernel in kernels:
        # print (np.array(kernel).shape)
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[cnt] = filtered.mean()
        feats[cnt+1] = filtered.var()
        if plotp:
            plt.imshow(kernel)
            plt.show()
            plt.imshow(filtered)
            plt.show()
        cnt += 2
    return feats

def create_kernels():
    # prepare filter bank kernels
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                            sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    return kernels



if __name__=='__main__':
    main()
