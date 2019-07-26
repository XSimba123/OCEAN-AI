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
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
LABEL_PATH = r'/home/xujingning/ocean/ocean_data/test_label.csv'
DATA_PATH = '/home/xujingning/ocean/ocean_data/test_data_img/'
MODEL_PATH = '/home/xujingning/ocean/ocean_data/alex+over/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


total_labels_test, total_imgs_test, _ = get_data(DATA_PATH, LABEL_PATH)
clf = get_classifier(MODEL_PATH, 'alex_net')

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
