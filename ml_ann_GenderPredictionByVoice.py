import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def readCSV(filename):
    df = pd.read_csv(filename)

    for i, row in df.iterrows():
        _label = row['label']
        if _label == 'male':
            df.set_value(i, 'label', np.array([1, 0]))
        else:
            df.set_value(i, 'label', np.array([0, 1]))
    
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values

    # _targets = df['label'].values
    # print(_targets)
    return (features, labels)

def generateDatasets(_features, _labels):
    x_train, x_test, y_train, y_test = train_test_split(_features, _labels, test_size=0.2)
    print x_train.shape, y_train.shape
    print x_train.shape, y_test.shape

    return (x_train, x_test, y_train, y_test)

def visualizeData(_features):
    _scaler = StandardScaler()
    _features = _scaler.fit_transform(_features)

    return

_features, _labels = readCSV("./Data/data.csv")
generateDatasets(_features, _labels)
