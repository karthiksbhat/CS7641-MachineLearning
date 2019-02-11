import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

dataset_train = pd.read_csv("~/Documents/Datasets/ML/fifa-processed-train.csv")
dataset_test = pd.read_csv("~/Documents/Datasets/ML/fifa-processed-test.csv")

# set global* variables
X_train = dataset_train[['Weak Foot', 'Skill Moves', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing','Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']]
y_train = dataset_train['position_label']
X_test = dataset_test[['Weak Foot', 'Skill Moves', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing','Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']]
y_test = dataset_test['position_label']

fig = plt.figure()
ax = plt.axes()

# Function taken from SKLearn documentation to plot a learning curve
def plot_learning_curve(estimator, title, X, y, cv, train_sizes):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes)
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

# best depth comes out as 7!
best_depth, best_nodes = dt_hyperparameter_choice(X_train, y_train)

dt_classifier = tree.DecisionTreeClassifier(max_depth=best_depth)
dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict(X_test)

plotted = plot_learning_curve(dt_classifier, "Decision Tree FIFA", X_train, y_train, 10, np.linspace(.1, 1.0, 5))
plotted.show()
print confusion_matrix(y_test, y_pred) 
print classification_report(y_test, y_pred)



# ADA Boosted DT classifier
# --------------------------------------------------------------
# Function to identify and plot best hyperparameters
# This function currently only produces the "best_n"
# it previously was used to iterate on depth, but is no longer the case
def ada_hyperparameter_choice(X_train, y_train):
	ada_crossVal = {}
	for i in xrange(1,100):
		print "in iteration: ", i
		dt_classifier = tree.DecisionTreeClassifier(max_depth=5)
		ada_classifier = AdaBoostClassifier(base_estimator = dt_classifier, n_estimators = i)
		ada_classifier.fit(X_train, y_train)
		ada_cv = cross_val_score(ada_classifier, X_train, y_train)
		ada_crossVal[i] = sum(ada_cv)/len(ada_cv)

	best_n = max(ada_crossVal, key=ada_crossVal.get)
	print "Best n for best CV: ", best_n

	plt.plot(ada_crossVal.keys(), ada_crossVal.values(), color="blue", label="CV score - n")
	plt.xlabel("Value of hyperparameters")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.show()
	return best_n

# best max depth keeping 50 estimators turns out to be 20
#best_depth = ada_hyperparameter_choice(X_train, y_train)
best_n = ada_hyperparameter_choice(X_train, y_train)
# using 5 because since it's boosted, I can afford to be more aggressive in pruning
best_depth = 5 # -- used for one curve
dt_classifier = tree.DecisionTreeClassifier(max_depth=best_depth)
ada_classifier = AdaBoostClassifier(base_estimator = dt_classifier, n_estimators = best_n)
ada_classifier.fit(X_train, y_train)
y_pred = ada_classifier.predict(X_test)

plotted = plot_learning_curve(ada_classifier, "Adaptive Boosted FIFA", X_train, y_train, 5, np.linspace(.1, 1.0, 5))
plotted.show()
print confusion_matrix(y_test, y_pred) 
print classification_report(y_test, y_pred)


#KNN algorithm
# --------------------------------------------------------------
def knn_hyperparameter_choice(X_train, y_train):
	knn_crossVal = {}
	for i in xrange(2, 40):
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

plotted = plot_learning_curve(knn_classifier, "kNN Classifier FIFA", X_train, y_train, 10, np.linspace(.1, 1.0, 5))
print confusion_matrix(y_test, y_pred) 
print classification_report(y_test, y_pred)
plotted.show()


# Artificial Neural Networks
# --------------------------------------------------------------
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

# best parameters are:  {'alpha': 0.0001, 'activation': 'relu', 'solver': 'adam', 'learning_rate': 'adaptive', 'hidden_layer_sizes': (10, 10, 10)}
mlp = MLPClassifier(alpha=0.0001, activation="relu", solver="adam", learning_rate="adaptive", hidden_layer_sizes=(10,10,10), max_iter=1000)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
plotted = plot_learning_curve(mlp, "MLP Classifier FIFA", X_train, y_train, 10, np.linspace(.1, 1.0, 5))
print confusion_matrix(y_test, y_pred) 
print classification_report(y_test, y_pred)
plotted.show()


#SVM with different kernels
# --------------------------------------------------------------
# This code is for gridSearching the best hyperparameters
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

# Below comments were used to run gridSearch, and have been commented out.
# print "best params are for Sigmoid: ", svc_param_selection(X_train, y_train, 5)
# best params are for RBF:  {'C': 1, 'gamma': 0.001}
# best params are for Sigmoid:  {'C': 0.1, 'gamma': 0.001}
# best params are for Linear:  {'C': 0.001, 'gamma': 0.001}


rbfSVC = svm.SVC(C=1, gamma=0.001)
rbfSVC.fit(X_train, y_train)
y_pred = rbfSVC.predict(X_test)
plotted = plot_learning_curve(rbfSVC, "RBF SVC Classifier FIFA", X_train, y_train, 10, np.linspace(.1, 1.0, 5))
print confusion_matrix(y_test, y_pred) 
print classification_report(y_test, y_pred)
# plotted.show()

sigmoidSVC = svm.SVC(kernel="sigmoid", C=0.1, gamma=0.001)
sigmoidSVC.fit(X_train, y_train)
y_pred = sigmoidSVC.predict(X_test)
plotted = plot_learning_curve(sigmoidSVC, "Sigmoid SVC Classifier FIFA", X_train, y_train, 10, np.linspace(.1, 1.0, 5))
print confusion_matrix(y_test, y_pred) 
print classification_report(y_test, y_pred)
# plotted.show()

linearSVC = svm.SVC(kernel='linear', C=0.001, gamma=0.001)
linearSVC.fit(X_train, y_train)
y_pred = linearSVC.predict(X_test)
print cross_val_score(linearSVC, X_train, y_train)
plotted = plot_learning_curve(linearSVC, "Linear SVC Classifier FIFA", X_train, y_train, 10, np.linspace(.1, 1.0, 5))
print confusion_matrix(y_test, y_pred) 
print classification_report(y_test, y_pred)
plotted.show()