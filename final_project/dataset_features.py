from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np

def computeFraction( poi_messages, all_messages ):
    if (float(all_messages) == 0) or (float(all_messages) != float(all_messages)):
        fraction = 0    
    else : fraction = float(poi_messages) / float(all_messages)
    return fraction

def fillFraction(data_dict):
    for name in data_dict:
        data_point = data_dict[name]
        from_poi_to_this_person = data_point["from_poi_to_this_person"]
        to_messages = data_point["to_messages"]
        fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
        data_point["fraction_from_poi"] = fraction_from_poi
        from_this_person_to_poi = data_point["from_this_person_to_poi"]
        from_messages = data_point["from_messages"]
        fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
        data_point["fraction_to_poi"] = fraction_to_poi

def perform_SelectKBest(k_num, features, labels, features_list):   
    clf = SelectKBest(f_classif, k = k_num)
    clf.fit(features, labels)
    scores = zip(features_list[1:], clf.scores_)
    scores.sort(key = lambda t: t[1], reverse = True)
    print 'SelectKBest scores: ', scores   
    kBest_features = [(score[0]) for score in scores[0:k_num]]
    print 'KBest features', kBest_features
    return kBest_features

def perform_plot_evaluation_metrics(clf, dataset, feature_list, plot_title):
    accuracy_arr = []
    precision_arr = []
    recall_arr = []
    f1_arr = []
    x_arr = []
    for x in range(3, 21):
        fl = feature_list[0:x]
        accuracy, precision, recall, f1 = my_test_classifier(clf, dataset, fl)
        x_arr.append(x)
        accuracy_arr.append(accuracy)
        precision_arr.append(precision)
        recall_arr.append(recall)
        f1_arr.append(f1)
        
    plt.plot(x_arr, accuracy_arr, label='accuracy')
    plt.plot(x_arr, precision_arr, label='precision')
    plt.plot(x_arr, recall_arr, label='recall')
    plt.plot(x_arr, f1_arr, label='f1')
    plt.legend(loc='best')
    plt.xlabel('Number of features')
    plt.ylabel('Evaluation metrics value')
    plt.xticks(np.arange(1, 21, 1))
    plt.title(plot_title)
    plt.show()

def peform_StratifiedShuffleSplit(features,labels, folds = 1000):
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
    return features_train, features_test, labels_train, labels_test
    
def my_test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        return accuracy, precision, recall, f1
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."
