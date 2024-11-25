from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import svm
import pandas as pd
import numpy as np
import pickle

data = pd.read_csv("Train_speaker.csv")
data = data.drop(['Unnamed: 0'], axis = 1)

# Standardize Data
scaler = StandardScaler()
scaler.fit(np.array(data.drop(['label'], axis=1), dtype = float))
with open('scaler.pkl','wb') as f:
    pickle.dump(scaler,f)

X = scaler.transform(np.array(data.drop(['label'], axis=1), dtype = float))

y = np.array(data['label'],dtype=str)

X_10min, dummy, y_10min, dummy = train_test_split(X, y, test_size=0.5)
X_5min, dummy, y_5min, dummy = train_test_split(X, y, train_size=0.25)
X_1min, dummy, y_1min, dummy = train_test_split(X, y, train_size=0.05) 

svm_clf = svm.SVC(kernel='linear', probability=True)
svm_clf.fit(X, y)
with open('model_20.pkl','wb') as f:
    pickle.dump(svm_clf,f)

svm_clf_10min = svm.SVC(kernel='linear', probability=True)
svm_clf_10min.fit(X_10min, y_10min)
with open('model_10.pkl','wb') as f:
    pickle.dump(svm_clf_10min,f)


svm_clf_5min = svm.SVC(kernel='linear', probability=True)
svm_clf_5min.fit(X_5min, y_5min)
with open('model_5.pkl','wb') as f:
    pickle.dump(svm_clf_5min,f)

svm_clf_1min = svm.SVC(kernel='linear', probability=True)
svm_clf_1min.fit(X_1min, y_1min)
with open('model_1.pkl','wb') as f:
    pickle.dump(svm_clf_1min,f)