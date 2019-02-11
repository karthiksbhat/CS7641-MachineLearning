import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve

# Read datasets
dataset_train = pd.read_csv("~/Documents/Datasets/ML/fashion-mnist_train.csv")
dataset_test = pd.read_csv("~/Documents/Datasets/ML/fashion-mnist_test.csv")

# Sample smaller amount of entire set, and set global* variables
dt_train_sample = dataset_train.sample(n=10000, random_state=1)
dt_test_sample = dataset_test.sample(n=2500, random_state=1)
X_train = dt_train_sample.drop('label', axis=1)
y_train = dt_train_sample['label']
X_test = dt_test_sample.drop('label', axis=1)
y_test = dt_test_sample['label']

# Function taken from SKLearn documentation to plot a learning curve
def plot_learning_curve(estimator, title, X, y, cv, train_sizes):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    print "plotting learning curve"
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print "after plot block"
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


# DT Classifier with Pruning
#------------------------------------------------------
# Function to identify and plot best hyperparameters
def dt_hyperparameter_choice(X_train, y_train):
	dt_crossVal = {}
	dt_crossVal_1 = {}

	for i in xrange(3,30):
		print "in iteration: ", i
		dt_classifier = tree.DecisionTreeClassifier(max_depth= i)
		dt_classifier.fit(X_train, y_train)
		dt_cv = cross_val_score(dt_classifier, X_train, y_train)
		dt_crossVal[i] = sum(dt_cv)/len(dt_cv)

	for i in xrange(3,30):
		print "in iteration: ", i
		dt_classifier = tree.DecisionTreeClassifier(max_leaf_nodes = i)
		dt_classifier.fit(X_train, y_train)
		dt_cv = cross_val_score(dt_classifier, X_train, y_train)
		dt_crossVal_1[i] = sum(dt_cv)/len(dt_cv)

	best_depth = max(dt_crossVal, key=dt_crossVal.get)
	best_nodes = max(dt_crossVal_1, key=dt_crossVal_1.get)
	print "max depth for best CV: ", best_depth
	print "max leaf nodes for best CV: ", best_nodes

	plt.plot(dt_crossVal.keys(), dt_crossVal.values(), color="blue", label="CV score - depth")
	plt.plot(dt_crossVal_1.keys(), dt_crossVal_1.values(), color="green", label="CV score - nodes")
	plt.xlabel("Value of hyperparameters")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.show()
	return best_depth, best_nodes

best_depth, best_nodes = dt_hyperparameter_choice(X_train, y_train)
# ************ best_depth turned out to be 9! ************
# best_depth = 9
dt_classifier = tree.DecisionTreeClassifier(max_depth=best_depth)
dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict(X_test)


plotted = plot_learning_curve(dt_classifier, "Decision Tree Fashion", X_train, y_train, 10, np.linspace(.1, 1.0, 5))
plotted.show()
# results analysis
print confusion_matrix(y_test, y_pred) 
print classification_report(y_test, y_pred)



#ADA Boosting
#------------------------------------------------------
def ada_hyperparameter_choice(X_train, y_train):
	ada_crossVal = {}
	for i in xrange(3,10):
		print "in iteration: ", i
		dt_classifier = tree.DecisionTreeClassifier(max_depth=i)
		ada_classifier = AdaBoostClassifier(base_estimator = dt_classifier, n_estimators = 50)
		ada_classifier.fit(X_train, y_train)
		ada_cv = cross_val_score(ada_classifier, X_train, y_train)
		ada_crossVal[i] = sum(ada_cv)/len(ada_cv)

	best_depth = max(ada_crossVal, key=ada_crossVal.get)
	print "max depth for best CV: ", best_depth

	plt.plot(ada_crossVal.keys(), ada_crossVal.values(), color="blue", label="CV score - depth")
	plt.xlabel("Value of hyperparameters")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.show()
	return best_depth

# best_depth graph was inconclusive, so running for 9, and then for something smaller
best_depth = ada_hyperparameter_choice(X_train, y_train)

# ************ best_depth is 6 ************

dt_classifier = tree.DecisionTreeClassifier(max_depth=best_depth)
ada_classifier = AdaBoostClassifier(base_estimator = dt_classifier, n_estimators = 50)
print "initializing classifier"
ada_classifier.fit(X_train, y_train)
print "finished fitting classifier"
y_pred = ada_classifier.predict(X_test)

plotted = plot_learning_curve(ada_classifier, "Adaptive Boosted Fashion", X_train, y_train, 10, np.linspace(.1, 1.0, 5))
# results analysis
print confusion_matrix(y_test, y_pred) 
print classification_report(y_test, y_pred)
plotted.show()



#kNN Algorithm
#------------------------------------------------------
def knn_hyperparameter_choice(X_train, y_train):
	knn_crossVal = {}
	for i in xrange(2, 20):
		print "in iteration: ", i
		knn_classifier = KNeighborsClassifier(n_neighbors = i)
		knn_classifier.fit(X_train, y_train)
		knn_cv = cross_val_score(knn_classifier, X_train, y_train)
		knn_crossVal[i] = sum(knn_cv)/len(knn_cv)
	best_n = max(knn_crossVal, key=knn_crossVal.get)
	print "n for best CV is: ", best_n
	plt.plot(knn_crossVal.keys(), knn_crossVal.values(), color="blue", label="CV score - n")
	plt.xlabel("k-value")
	plt.ylabel("Accuracy")
	plt.legend()
	# plt.show()
	return best_n

best_n = knn_hyperparameter_choice(X_train, y_train)
knn_classifier = KNeighborsClassifier(n_neighbors = best_n)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

plotted = plot_learning_curve(knn_classifier, "kNN Classifier Fashion", X_train, y_train, 4, np.linspace(.1, 1.0, 5))
# results analysis
print confusion_matrix(y_test, y_pred) 
print classification_report(y_test, y_pred)
plotted.show()



# Artificial Neural Networks
#------------------------------------------------------
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# This code is for gridSearching the best hyperparameters
parameters = {
    'hidden_layer_sizes': [(10,10,10), (50,50,50), (10,), (50,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
mlp = MLPClassifier(max_iter=1000)
print "initializing"
clf = GridSearchCV(mlp, parameters, cv=5)
print "fitting"
clf.fit(X_train, y_train)
print "best parameters are: ", clf.best_params_
# Grid search code ends

# best parameters are:  {'alpha': 0.0001, 'activation': 'tanh', 'solver': 'sgd', 'learning_rate': 'constant', 'hidden_layer_sizes': (50,)}
mlp = MLPClassifier(alpha=0.0001, activation="tanh", solver="sgd", learning_rate="constant", hidden_layer_sizes=(50,), max_iter=1000)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
plotted = plot_learning_curve(mlp, "MLP Classifier Fashion", X_train, y_train, 10, np.linspace(.1, 1.0, 5))
print confusion_matrix(y_test, y_pred) 
print classification_report(y_test, y_pred)
plotted.show()


#SVM with different kernels
#------------------------------------------------------

def svc_param_selection(X, y, nfolds, kernel):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel=kernel), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

# print "best params are for sigmoid: ", svc_param_selection(X_train, y_train, 5, "sigmoid")
# print "best params are for linear: ", svc_param_selection(X_train, y_train, 5, "linear")
# best params are for sigmoid:  {'C': 0.001, 'gamma': 0.001}
# best params are for linear:  {'C': 0.001, 'gamma': 0.001}

rbfSVC = svm.SVC(C=0.001, gamma=0.001)
rbfSVC.fit(X_train, y_train)
y_pred = rbfSVC.predict(X_test)
plotted = plot_learning_curve(rbfSVC, "RBF SVC Classifier Fashion", X_train, y_train, 5, np.linspace(.1, 1.0, 5))
print "RBF"
print confusion_matrix(y_test, y_pred) 
print classification_report(y_test, y_pred)
# plotted.show()

sigmoidSVC = svm.SVC(kernel="sigmoid", C=0.001, gamma=0.001)
sigmoidSVC.fit(X_train, y_train)
y_pred = sigmoidSVC.predict(X_test)
plotted = plot_learning_curve(sigmoidSVC, "Sigmoid SVC Classifier Fashion", X_train, y_train, 5, np.linspace(.1, 1.0, 5))
print "Sigmoid"
print confusion_matrix(y_test, y_pred) 
print classification_report(y_test, y_pred)
# plotted.show()

linearSVC = svm.SVC(kernel='linear', C=0.001, gamma=0.001)
linearSVC.fit(X_train, y_train)
y_pred = linearSVC.predict(X_test)
print cross_val_score(linearSVC, X_train, y_train)
plotted = plot_learning_curve(linearSVC, "Linear SVC Classifier Fashion", X_train, y_train, 5, np.linspace(.1, 1.0, 5))
print "Linear"
print confusion_matrix(y_test, y_pred) 
print classification_report(y_test, y_pred)
plotted.show()