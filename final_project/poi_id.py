#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import tester
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import exploring_dataset as expl
import dataset_features as df
from sklearn.model_selection import GridSearchCV
from time import time

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances',
                       'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                       'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                       'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi']
poi_feature = ['poi']

features_list = poi_feature + financial_features + email_features  # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Exploring dataset
expl.explore_dataset(data_dict)
expl.counting_NaNs(data_dict, features_list)
expl.find_empty_rows(data_dict)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)

## Uncomment for data visualization
##expl.draw_scatter_plot(data_dict, "salary", "bonus")
##expl.draw_scatter_plot(data_dict, "total_payments", "total_stock_value")

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
df.fillFraction(my_dataset)
features_list = features_list + ['fraction_from_poi', 'fraction_to_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

## SelectKBest to find best features to use
kBest_features = poi_feature + df.perform_SelectKBest(21, features, labels,  features_list)

# Scale features
from sklearn import preprocessing
data = featureFormat(my_dataset, kBest_features, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

features_train, features_test, labels_train, labels_test = \
    df.peform_StratifiedShuffleSplit(features, labels)

## GaussianNB
print 'Performing GaussianNB'
from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()
''' Finding best number of features
df.perform_plot_evaluation_metrics(clf_NB, my_dataset, kBest_features, 'GaussianNB')
'''
tester.dump_classifier_and_data(clf_NB, my_dataset, kBest_features[:7])
t0 = time()
tester.main()
print "training time GaussianNB: ", round(time()-t0, 3), "s"

## Decision Tree
print 'Performing Decision Tree'
from sklearn import tree
clf_DT = tree.DecisionTreeClassifier()
params = {'criterion':('gini', 'entropy'), 'splitter':('best','random')}
clf = GridSearchCV(clf_DT, params)
clf.fit(features_train, labels_train)
best_params = clf.best_params_
print 'Best parameters for Decision Tree: '
print best_params
clf_DT = tree.DecisionTreeClassifier(splitter = 'random', criterion = 'entropy')
''' Finding best number of features
df.perform_plot_evaluation_metrics(clf_DT, my_dataset, kBest_features, 'Decision Tree')
'''
tester.dump_classifier_and_data(clf_DT, my_dataset, kBest_features[:4])
t0 = time()
tester.main();
print "training time Decision Tree: ", round(time()-t0, 3), "s"

## K-means with 2 clusters
print 'Performing K-means with 2 clusters'
from sklearn.cluster import KMeans
clf_K = KMeans(n_clusters=2)
''' Finding best number of features
df.perform_plot_evaluation_metrics(clf_K, my_dataset, kBest_features, 'K-Means')
'''
tester.dump_classifier_and_data(clf_K, my_dataset, kBest_features[:4])
t0 = time()
tester.main()
print "training time K-means: ", round(time()-t0, 3), "s"

## Logistic Regression
print 'Performing Logistic Regression'
from sklearn.linear_model import LogisticRegression
clf_LR = LogisticRegression()
params = {'tol': [1, 0.1, 0.01, 0.001, 0.0001],
'C': [0.1, 0.01, 0.001, 0.0001]}
clf = GridSearchCV(clf_LR, params)
clf.fit(features_train, labels_train)
best_params = clf.best_params_
print 'Best parameters for Logistic Regression: '
print best_params
clf_LR = LogisticRegression(C = 0.1, tol = 1)
''' Finding best number of features
df.perform_plot_evaluation_metrics(clf_LR, my_dataset, kBest_features, 'Logistic Regression')
'''
tester.dump_classifier_and_data(clf_LR, my_dataset, kBest_features[:18])
t0 = time()
tester.main()
print "training time Logistic Regression: ", round(time()-t0, 3), "s"

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

clf = clf_NB
features_list = kBest_features[:7]

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
