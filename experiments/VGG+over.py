import sys
import os
import pandas as pd
sys.path.append('../')
from prediction.predict import get_prediction
from training.train import get_classifier, train_model
from data_IO.data_reader import get_data
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold,cross_val_score
from scipy.stats import sem
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

from collections import Counter
LABEL_PATH = r'/home/xujingning/ocean/ocean_data/label.csv'
DATA_PATH = '/home/xujingning/ocean/ocean_data/data_img/'
MODEL_PATH = '/home/xujingning/ocean/ocean_data/VGG+over/'
EPOCH = 30
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def add_noise(_img):
    #n_img = _img.copy()
    noise = np.random.randint(5, size=(224, 224, 3), dtype='uint8')
    return noise+_img


labels, imgs, _ = get_data(DATA_PATH, LABEL_PATH)
print(len(imgs), len(labels))
#imgs = imgs / 255.0
clf = get_classifier(MODEL_PATH, 'VGG_net')
total_imgs_train, total_imgs_test, total_labels_train, total_labels_test = train_test_split(imgs, labels, test_size=0.3, random_state=42)
#total_imgs_train = total_imgs_train
#total_imgs_test = total_imgs_test
# print(len(total_imgs_train), len(total_labels_train), len(total_imgs_test), len(total_labels_test))
print("gcy begin")
#print(total_imgs_train[0],total_labels_train[0])
#print(total_imgs_train[1],total_labels_train[1])
nsamples, nx, ny, nz = total_imgs_train.shape
d2_train_dataset = total_imgs_train.reshape((nsamples, nx*ny*nz))
# #nsamples, nx, ny, nz = total_imgs_test.shape
# #d2_test_dataset = total_imgs_test.reshape((nsamples, nx*ny*nz))
#
ros = RandomOverSampler(random_state=42)
#X_resampled, y_resampled = ros.fit_sample(X, y)

xx, yy = ros.fit_sample(d2_train_dataset, total_labels_train)
# #xxx, yyy = ADASYN().fit_sample(d2_test_dataset, total_labels_test)
#
print('gcy end')
xx = xx.reshape((len(xx), nx, ny, nz))
# #xxx = xxx.reshape((len(xx), nx, ny, nz))
# print(len(xx), len(yy))
# print(xx.shape)
total_imgs_train = np.array(xx).astype(np.float32)
total_labels_train = np.array(yy)

#print(total_imgs_train[0],total_labels_train[0])
#print(total_imgs_train[1],total_labels_train[1])
# total_imgs_train=total_imgs_train
# print(total_labels_train)
# print(total_imgs_train)
print(total_imgs_train.shape)
print(total_labels_train.shape)
print("train zhuan")
# copy = list(total_imgs_train.copy())
# for img in total_imgs_train:
#     copy.append(list(np.fliplr(img)))
#
# print(len(copy))
# total_imgs_train = np.array(copy).copy()
#
# print("train noise")
# copy = list(total_imgs_train.copy())
# for img in total_imgs_train:
#     copy.append(add_noise(img))
#
# print(len(copy))
# total_imgs_train = np.array(copy).copy()
#
# print("test zhuan")
# copy = list(total_imgs_test.copy())
# for img in total_imgs_test:
#     copy.append(list(np.fliplr(img)))
# print(len(copy))
# total_imgs_test = np.array(copy).copy()
#
# print("test noise")
# copy = list(total_imgs_test.copy())
# for img in total_imgs_test:
#     copy.append(add_noise(img))
#
# print(len(copy))
# total_imgs_test = np.array(copy).copy()
#
#
#
#
# copy = list(total_labels_train.copy())
#
# print("train label")
# for i in range(0,3):
#     for k in range(len(total_labels_train)):
#         copy.append(total_labels_train[k])
#
#
# print(len(copy))
#
# total_labels_train = np.array(copy).copy()
#
#
# print("test label")
# copy = list(total_labels_test.copy())
#
# for i in range(0,3):
#     for k in range(len(total_labels_test)):
#
#         copy.append(total_labels_test[k])
#
# print(len(copy))
# total_labels_test = np.array(copy).copy()
#
#
# print(len(total_imgs_train), len(total_labels_train), len(total_imgs_test), len(total_labels_test))



# ss = StratifiedShuffleSplit(n_splits=6,test_size=0.3,train_size=0.7,random_state=0)
# for train_index, test_index in ss.split(total_imgs_train, total_labels_train):
#     print("TRAIN:", len(train_index), "TEST:", len(test_index))
#     imgs_train, imgs_test = total_imgs_train[train_index], total_imgs_train[test_index]
#     labels_train, labels_test = total_labels_train[train_index], total_labels_train[test_index]
for i in range(EPOCH):
        print('round:' + str(i))
        print('train')
        train_model(clf, {'X1': total_imgs_train}, total_labels_train)

        print('test')
        mypred = get_prediction(clf, {'X1': total_imgs_test}, total_labels_test)
        pred = []
        for j in mypred:
            pred.append(j)
        final_pred = []
        for j in pred:
            final_pred.append(j['classes'])
        print('precision micro: ', metrics.precision_score(total_labels_test, final_pred, average='micro'))
        print('precision macro: ', metrics.precision_score(total_labels_test, final_pred, average='macro'))
        print('recall micro: ', metrics.recall_score(total_labels_test, final_pred, average='micro'))
        print('recall macro: ', metrics.recall_score(total_labels_test, final_pred, average='macro'))
        print('f1 micro: ', metrics.f1_score(total_labels_test, final_pred, average='micro'))
        print('f1 macro: ', metrics.f1_score(total_labels_test, final_pred, average='macro'))


print('test')
mypred = get_prediction(clf, {'X1': total_imgs_test}, total_labels_test)
pred = []
for j in mypred:
    pred.append(j)
final_pred = []
for j in pred:
    final_pred.append(j['classes'])
print('precision micro: ', metrics.precision_score(total_labels_test, final_pred, average='micro'))
print('precision macro: ', metrics.precision_score(total_labels_test, final_pred, average='macro'))
print('recall micro: ', metrics.recall_score(total_labels_test, final_pred, average='micro'))
print('recall macro: ', metrics.recall_score(total_labels_test, final_pred, average='macro'))
print('f1 micro: ', metrics.f1_score(total_labels_test, final_pred, average='micro'))
print('f1 macro: ', metrics.f1_score(total_labels_test, final_pred, average='macro'))


#evaluate_cross_validation(clf, total_imgs_test, total_labels_test, 6)
result = confusion_matrix(total_labels_test, final_pred)
df = pd.DataFrame(result)
df.to_csv("/home/xujingning/ocean/ocean_data/result.csv")




del clf
