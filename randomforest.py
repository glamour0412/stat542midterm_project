



import os
import pandas as pd
import numpy as np
import time
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tic = time.time()

def get_data(path, mode='train', n_cats=10000, n_dogs=10000, n_test=200):
    images = []
    labels = []
    ctr_cats = 0
    ctr_dogs = 0
    
    for filename in os.listdir(path):
        image = imread(path + '/' + filename)
        image = resize(image, (128, 64))
#         dog = 1, cat = 0
        if mode == 'train':
            if 'cat' in filename and ctr_cats < n_cats:
                images.append(image)
                labels.append(0)
                ctr_cats += 1
            elif 'dog' in filename and ctr_dogs < n_dogs:
                images.append(image)
                labels.append(1)
                ctr_dogs += 1
                
            if ctr_cats == n_cats and ctr_dogs == n_dogs: break
        else:
            images.append(image)
            if len(images) == n_test: break
    
    if mode == 'train':
        return images, labels  
    else:
        return images


TRAIN_DIR = '/Users/ruanyufei/Desktop/542mid/dogs-vs-cats/train'
TEST_DIR = '/Users/ruanyufei/Desktop/542mid/dogs-vs-cats/test1'
xtrain, ytrain = get_data(TRAIN_DIR)
xtest = get_data(TEST_DIR, mode='test')

def HOG(image,orientations,pixelsPerCell,cellsPerBlock,block_norm):
    hist = hog(image, orientations = orientations,
                       pixels_per_cell = pixelsPerCell,
                       cells_per_block = cellsPerBlock,
                       block_norm = block_norm)
    return hist

lst=[]
for x in xtrain:
    h = HOG(x,orientations = 9, pixelsPerCell = (8, 8),cellsPerBlock = (2, 2), block_norm = 'L2-Hys')
    lst.append(h)
X = np.array(lst)
y = ytrain

"""
lsttest=[]
for x in xtest:
    h = HOG(x,orientations = 9, pixelsPerCell = (8, 8),cellsPerBlock = (2, 2), block_norm = 'L2-Hys')
    lsttest.append(h)
X_test = np.array(lsttest)
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
rf_model = RandomForestClassifier(n_estimators=100, max_features="auto", random_state=124)
rf_model.fit(X_train, y_train)
predictions = rf_model.predict(X_test)
accuracy = 1 - sum(abs(predictions - y_test))/len(y_test)
toc = time.time()

print("accuracy is ", accuracy, ", training time is ", toc - tic)


