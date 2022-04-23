# https://github.com/tsunghao-huang/SVM-Fruit-Image-Classifier/blob/master/main.py4
# https://github.com/mayankvik2/Handwritten-Digits-Classification/blob/master/Handwritten%20Digit%20Classification.ipynb
# https://medium.com/@basu369victor/handwritten-digits-recognition-d3d383431845


import os
import time
import utils
import random
import joblib
import argparse
import cv2 as cv
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from skimage import io, filters
from skimage.transform import resize
from sklearn.metrics import accuracy_score
from skimage.feature import hog, ORB, CENSURE, corner_peaks, corner_harris, BRIEF
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
                    metavar='S', help='Path to test dataset (default: 10)')

def main():
    global args 
    args = parser.parse_args()

    traindir = os.path.join(args.data, 'ChallengeFree/train')
    eval = True
    if eval:
        # print('e')
        evaluate(args.data, model_file='svm_hog_base_good.pkl')
        return

    imgs = make_dataset_svm(traindir, transforms.Resize([28, 28]))

    train_imgs = []
    train_labels = []
    lengthV = []

    # features = ['ColorHist', 'HoG', 'Censure', 'ORB','BRIEF']
    features = ['HoG']

    for img, label in imgs:
        train_labels.append(label)
        
        feature_vector, hog_img = hog(np.array(img).astype('uint8'), orientations=9, 
                            pixels_per_cell=(14,14),cells_per_block=(1,1), visualize=True)
        # feature_vector, hog_img = hog(np.array(img).astype('uint8'), orientations=9, 
        #                     pixels_per_cell=(8,8),cells_per_block=(1,1), visualize=True)
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(hog_img)
        # plt.show()
        # exit()

        # if 'ColorHist' in features:
        #     # b = cv.calcHist([img], [0], None, [256], (0, 256), accumulate=False) #image, channel, mask
        #     # g = cv.calcHist([img], [1], None, [256], (0, 256), accumulate=False)
        #     # r = cv.calcHist([img], [2], None, [256], (0, 256), accumulate=False)
        #     hist0 = np.histogram(img[0], bins=256)
        #     hist1 = np.histogram(img[1], bins=256)
        #     hist2 = np.histogram(img[2], bins=256)
        #     for h in hist0:
        #         feature_vector = np.append(feature_vector, h)
        #     for h in hist1:
        #         feature_vector = np.append(feature_vector, h)
        #     for h in hist2:
        #         feature_vector = np.append(feature_vector, h)
        # if 'Censure' in features:
        #     #Censure extractor
        #     detector = CENSURE()
        #     detector.detect(img)
        #     feature_vector = np.append(feature_vector, [np.array(detector.keypoints).flatten()])
        # if 'ORB' in features:
        #     #ORB extractor
        #     orb_detector = ORB(n_keypoints=50)
        #     orb_detector.detect_and_extract(img)
        #     feature_vector = np.append(feature_vector, orb_detector.keypoints)
        # lengthV.append(len(feature_vector))
        # #add featureVector list to dataset that is fed into svm
        train_imgs.append(feature_vector)

    # max = np.amax(lengthV)
    # lengthV = []
    # train_imgs2 = []
    # #pad dataset with zeroes so that all featurevectors have the same length --> important for svm
    # for d in train_imgs:
    #     d = np.pad(d, (0, max - len(d)), 'constant')
    #     train_imgs2.append(d)
    #     lengthV.append(len(d))
    # train_imgs = train_imgs2

    best_params = gridSearch(train_imgs, train_labels)
    print(best_params)
    # best_params = {'C': 100.0, 'gamma': 0.1}
    # best_params = {'estimator__gamma': 0.01, 'estimator__C': 10000.0}
    # Train SVM on the best parameters
    print('Training...')
    clf = OneVsRestClassifier(SVC()).set_params(**best_params)
    clf.fit(train_imgs, train_labels)
    joblib.dump(clf, 'svm_hog_64.pkl')

    scores = cross_val_score(clf, train_imgs, train_labels, cv=10)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    

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

def evaluate(data_path, model_file='svm_hog_base.pkl'):
    # testdir = os.path.join(data_path, 'ChallengeFree/test')
    testdir = os.path.join(args.data, args.test_path)
    print('Testing..')

    imgs = make_dataset_svm(testdir, transforms.Resize([28, 28]))

    test_imgs = []
    test_labels = []

    for img, label in imgs:
        test_labels.append(label)
        
        feature_vector, hog_img = hog(np.array(img).astype('uint8'), orientations=9, 
                            pixels_per_cell=(14,14),cells_per_block=(1,1), visualize=True)
        # feature_vector, hog_img = hog(np.array(img).astype('uint8'), orientations=9, 
        #                     pixels_per_cell=(8,8),cells_per_block=(1,1), visualize=True)
        test_imgs.append(feature_vector)

    clf = joblib.load(model_file)
    acc = accuracy_score(test_labels, clf.predict(test_imgs))
    print(args.test_path, acc*100)


if __name__=='__main__':
    main()
