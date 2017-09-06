import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz as grv

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import utils
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn import tree

# different learning rate schedules and momentum parameters
params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'adam', 'learning_rate_init': 0.01}]

labels = ["constant learning-rate", "constant with momentum",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate", "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum", "adam"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'}]


def readCSV(filename):
    df = pd.read_csv(filename)

    # for i, row in df.iterrows():
    #     _label = row['label']
    #     if _label == 'male':
    #         df.set_value(i, 'label', 1)
    #     else:
    #         df.set_value(i, 'label', 0)
    
    # features = df.iloc[:, :-1].values
    # labels = df.iloc[:, -1].values

    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    labels = LabelEncoder().fit_transform(labels)

    print(utils.multiclass.type_of_target(features))
    print(utils.multiclass.type_of_target(labels))
    
    return (features, labels)

def generateDatasets(_features, _labels):
    _scaler = StandardScaler()
    _features = _scaler.fit_transform(_features)

    x_train, x_test, y_train, y_test = train_test_split(_features, _labels, test_size=0.2)
    # x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, test_size=0.2)
    x_cv = x_test
    y_cv = y_test
    print x_train.shape, y_train.shape
    print x_test.shape, y_test.shape

    return (x_train, y_train, x_test, y_test, x_cv, y_cv)

def visualizeData(_features, _labels):
    
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf() # clear figure
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    pca = PCA(n_components=3)
    pca.fit(_features)
    X = pca.transform(_features)

    for name, label in [('Male', 1), ("Female", 0)]:
        ax.text3D(X[_labels == label, 0].mean(),
              X[_labels == label, 1].mean() + 1.5,
              X[_labels == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

    # Reorder the labels to have colors matching the cluster results
    y = np.choose(_labels, [1, 0]).astype(np.float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral,
            edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()

    return

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.clf()
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def multiLayerPerceptronModel(x_train, y_train, x_test, y_test):
    
    print x_train.shape[0], y_train.shape[0]

    title = "Learning Curves (ANN)"
    
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500) # 3 hiddens layers with 13 units each
    mlp.fit(x_train, y_train)
    pred = mlp.predict(x_test)
    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test,pred))

    print("Training set score: %f" % mlp.score(x_train, y_train))
    print("Training set loss: %f" % mlp.loss_)
    print("Testing set score: %f" % mlp.score(x_test, y_test))

    plot_learning_curve(mlp, title, x_train, y_train)

    plt.show()

    return

def CARTModel(x_train, y_train, x_test, y_test):

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)

    pred = clf.predict(x_test)
    print(classification_report(y_test,pred))

    print("DecisionTree score: %f" % clf.score(x_train, y_train))

    # dot_data = tree.export_graphviz(clf, out_file=None)
    # graph = grv.Source(dot_data)
    # graph.render("voice_recognition_by_gender")

    return


def plot_on_dataset(X, y, ax, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)
    # X = MinMaxScaler().fit_transform(X)
    mlps = []

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(hidden_layer_sizes=(13,13,13), verbose=0, random_state=0,
                            max_iter=500, **param)
        mlp.fit(X, y)
        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
            ax.plot(mlp.loss_curve_, label=label, **args)


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
_features, _labels = readCSV("./Data/data.csv")
x_train, y_train, x_test, y_test, x_cv, y_cv = generateDatasets(_features, _labels)
#visualizeData(_features, _labels)
multiLayerPerceptronModel(x_train, y_train, x_test, y_test)
CARTModel(x_train, y_train, x_test, y_test)

# for ax, data, name in zip(axes.ravel(), [(x_train, y_train)], ['voice_recognition_by_gender']):
#     plot_on_dataset(*data, ax=ax, name=name)

# fig.legend(ax.get_lines(), labels=labels, ncol=3, loc="upper center")
# plt.show()

