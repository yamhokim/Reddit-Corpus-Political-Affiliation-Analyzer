import argparse
import os
from scipy.stats import ttest_ind
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  
from sklearn.preprocessing import StandardScaler # MIGHT NEED TO REMOVE THIS FOR ACTUAL SUBMISSION
import random
# set the random state for reproducibility 
import numpy as np
np.random.seed(401)

def accuracy(C):
    """ Compute accuracy given NumPy array confusion matrix C. Returns a floating point value. """
    diagonal_sum = C.trace()
    total_sum = C.sum()

    accuracy = diagonal_sum / total_sum
    return accuracy


def recall(C):
    """ Compute recall given NumPy array confusion matrix C. Returns a list of floating point values. """
    num_classes = C.shape[0]
    recall_vals = []

    for i in range(num_classes):
        true_positives = C[i, i]
        false_negatives = np.sum(C[i:]) - true_positives
        if (true_positives + false_negatives) == 0:
            recall_i = 0.0
        else:
            recall_i = true_positives / (true_positives + false_negatives)
        
        recall_vals.append(recall_i)
    
    return recall_vals

def precision(C):
    """ Compute precision given NumPy array confusion matrix C. Returns a list of floating point values. """
    num_classes = C.shape[0]
    precision_vals = []

    for i in range(num_classes):
        true_positives = C[i, i]
        false_positives = np.sum(C[:, i]) - true_positives
        if (true_positives + false_positives) == 0:
            precision_i = 0.0
        else:
            precision_i = true_positives / (true_positives + false_positives)
        
        precision_vals.append(precision_i)

    return precision_vals
    

def class31(output_dir, X_train, y_train, X_test, y_test):
    """ 
    This function performs experiment 3.1.
    
    Parameters:
    - output_dir: path of directory to write output to.
    - X_train: NumPy array, with the selected training features.
    - y_train: NumPy array, with the selected training classes.
    - X_test:  NumPy array, with the selected testing features.
    - y_test:  NumPy array, with the selected testing classes.

    Returns:      
    - best_index: int, the index of the supposed best classifier.
    """

    best_index, optimal_accuracy, optimal_F1 = 0, 0, 0
    scaler = StandardScaler()
    classifiers = [('SGDClassifier', SGDClassifier(loss='hinge')), ('GaussianNB', GaussianNB()), ('RandomForestClassifier', RandomForestClassifier(max_depth=5, n_estimators=10)), ('MLPClassifier', MLPClassifier(alpha=0.05)), ('AdaBoostClassifier', AdaBoostClassifier())]
    for i in range(len(classifiers)):
        print(f'class 31 test for classifier {i}')
        classifier_name = classifiers[i][0]
        classifier = classifiers[i][1]
        if (classifier_name == 'SGDClassifier') or (classifier_name == 'MLPClassifier') or (classifier_name == 'AdaBoostClassifier'):
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        print('start fit')
        classifier.fit(X_train, y_train)
        print('done fit')
        print('start prediction')
        y_predictions = classifier.predict(X_test)
        print('done prediction')

        conf_matrix = confusion_matrix(y_test, y_predictions)
        classifier_accuracy = accuracy(conf_matrix)
        classifier_recall = recall(conf_matrix)
        classifier_precision = precision(conf_matrix)

        if classifier_accuracy > optimal_accuracy:
            best_index = i
            optimal_accuracy = classifier_accuracy
        
        # Didn't end up using F1_score at all, but it is a potential metric that can be used for analysis
        # if classifier_accuracy > optimal_accuracy and F1_score > optimal_F1:
        #     optimal_index = i
        #     optimal_accuracy = classifier_accuracy
        #     optimal_F1 = F1_score    

        with open(f"{output_dir}/a1_3.1.txt", "a") as outf:
            # For each classifier, compute results and write the following output:
            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {classifier_accuracy:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in classifier_recall]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in classifier_precision]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

    return best_index

# Helper function for class32() function
def best_classifier(best_index):
    classifiers = [('SGDClassifier', SGDClassifier(loss='hinge')), ('GaussianNB', GaussianNB()), ('RandomForestClassifier', RandomForestClassifier(max_depth=5, n_estimators=10)), ('MLPClassifier', MLPClassifier(alpha=0.05)), ('AdaBoostClassifier', AdaBoostClassifier())]
    optimal_classifier = classifiers[best_index]

    return optimal_classifier[0], optimal_classifier[1]

def class32(output_dir, X_train, y_train, X_test, y_test, best_index):
    """ 
    This function performs experiment 3.2.
    
    Parameters:
    - output_dir: path of directory to write output to.
    - X_train: NumPy array, with the selected training features.
    - y_train: NumPy array, with the selected training classes.
    - X_test:  NumPy array, with the selected testing features.
    - y_test:  NumPy array, with the selected testing classes.
    - best_index:   int, the index of the supposed best classifier (from task 3.1).

    Returns:
       X_1k: NumPy array, just 1K rows of X_train.
       y_1k: NumPy array, just 1K rows of y_train.
    """
    num_rows = [1000, 5000, 10000, 15000, 20000]
    X_1k, y_1k = 0, 0
    scaler = StandardScaler()

    for i in range(len(num_rows)):
        print(f'class 32 test for {num_rows[i]} rows')
        classifier_name, classifier = best_classifier(best_index)
        num_train = num_rows[i]
        if (classifier_name == 'SGDClassifier') or (classifier_name == 'MLPClassifier') or (classifier_name == 'AdaBoostClassifier'):
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            np.random.seed(1)
            random_indexes = random.sample(range(len(X_train)), num_train)
            X_train_subset = X_train_scaled[random_indexes]
            y_train_subset = y_train[random_indexes]

            if num_train == 1000:
                X_1k = X_train_subset
                y_1k = y_train_subset
            
            classifier.fit(X_train_subset, y_train_subset)
            y_predictions = classifier.predict(X_test_scaled)
            conf_matrix = confusion_matrix(y_test, y_predictions)
            acc = accuracy(conf_matrix)

        else:
            np.random.seed(1)
            random_indexes = random.sample(range(len(X_train)), num_train)
            X_train_subset = X_train[random_indexes]
            y_train_subset = y_train[random_indexes]

            if num_train == 1000:
                X_1k = X_train_subset
                y_1k = y_train_subset
            
            classifier.fit(X_train_subset, y_train_subset)
            y_predictions = classifier.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_predictions)
            acc = accuracy(conf_matrix)


        with open(f"{output_dir}/a1_3.2.txt", "a") as outf:
            # For each number of training examples, compute results and write
            # the following output:
            outf.write(f'{num_train}: {acc:.4f}\n')

    return (X_1k, y_1k)


def class33(output_dir, X_train, y_train, X_test, y_test, best_index, X_1k, y_1k):
    """ 
    This function performs experiment 3.3.
    
    Parameters:
    - output_dir: path of directory to write output to.
    - X_train: NumPy array, with the selected training features.
    - y_train: NumPy array, with the selected training classes.
    - X_test:  NumPy array, with the selected testing features.
    - y_test:  NumPy array, with the selected testing classes.
    - best_index: int, the index of the supposed best classifier (from task 3.1).
    - X_1k:    NumPy array, just 1K rows of X_train (from task 3.2).
    - y_1k:    NumPy array, just 1K rows of y_train (from task 3.2).
    """

    # Get the best classifier
    scaler = StandardScaler()
    classifier_name, classifier = best_classifier(best_index)
    if (classifier_name == 'SGDClassifier') or (classifier_name == 'MLPClassifier') or (classifier_name == 'AdaBoostClassifier'):   # Used the scaled versions of the datasets
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Part 1: Find the best k features according to the 32K training data
    # Initialize the different selectors
    selector_5_feats_32k = SelectKBest(f_classif, k=5) 
    selector_50_feats_32k = SelectKBest(f_classif, k=50)
    selector_5_feats_1k = SelectKBest(f_classif, k=5)

    # Start by selecting the best 5 features
    print('Getting 5 feats for 32k')
    X_new_5_feats_32k = selector_5_feats_32k.fit_transform(X_train, y_train)
    selector_5_feats_32k.fit(X_train, y_train)
    pp_5_feats_32k = selector_5_feats_32k.pvalues_
    indices_5_feats_32k = selector_5_feats_32k.get_support(indices=True)    # List of indices for the features
    print('Finished getting 5 feats for 32k')

    # Now select the best 50 features   
    selector_50_feats_32k.fit(X_train, y_train)
    pp_50_feats_32k = selector_50_feats_32k.pvalues_

    # Part 2: 
    # a) Train the best classifier from section 3.1 on the 1K training set using the best k=5 features
    print('Getting 5 feats for 1k')
    X_new_5_feats_1k = selector_5_feats_1k.fit_transform(X_1k, y_1k)
    indices_5_feats_1k = selector_5_feats_1k.get_support(indices=True)      # List of indices for the features
    print('Finished getting 5 feats for 1k')

    classifier.fit(X_new_5_feats_1k, y_1k)
    y_predictions_5_feat_1k = classifier.predict(X_test[:, indices_5_feats_1k])
    conf_matrix_5_feats_1k = confusion_matrix(y_test, y_predictions_5_feat_1k)
    accuracy_1k = accuracy(conf_matrix_5_feats_1k)
    print(f'Accuracy for 5 feats, 1K: {accuracy_1k}')

    # b) Train the best classifier from section 3.1 on the 32K training set using the best k=5 features
    classifier.fit(X_new_5_feats_32k, y_train)
    y_predictions_5_feat_32k = classifier.predict(X_test[:, indices_5_feats_32k])
    conf_matrix_5_feats_32k = confusion_matrix(y_test, y_predictions_5_feat_32k)
    accuracy_full = accuracy(conf_matrix_5_feats_32k)
    print(f'Accuracy for 5 feats, 32K: {accuracy_full}')

    # Part 3: 
    # a) Extract indices of the top k=5 features using the 1k training set and take the intersection wuth the k=5 features using the 32K training set
    feature_intersection = []
    for feature in indices_5_feats_1k:
        if feature in indices_5_feats_32k:
            feature_intersection.append(feature)

    # Part 4: Format the top k=5 feature indices extracted from the 32K training set to the file  
    # This is done in the open() block below

    # Initialize two arrays for writing the pp_vals into the results text files
                 
    k_feats = [5, 50]
    p_vals = [pp_5_feats_32k, pp_50_feats_32k]

    with open(f"{output_dir}/a1_3.3.txt", "a") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        
        # for each number of features k_feat, write the p-values for that number of features:
        for i in range(len(k_feats)):
            outf.write(f'{k_feats[i]} p-values: {[format(pval) for pval in p_vals[i]]}\n')
        
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {indices_5_feats_32k}\n')


def class34(output_dir, X_train, y_train, X_test, y_test, best_index):
    """ 
    This function performs experiment 3.4.
    
    Parameters:
    - output_dir: path of directory to write output to.
    - X_train: NumPy array, with the selected training features.
    - y_train: NumPy array, with the selected training classes.
    - X_test:  NumPy array, with the selected testing features.
    - y_test:  NumPy array, with the selected testing classes.
    - best_index:       int, the index of the supposed best classifier (from task 3.1).
    """
    kfold_accuracies = []   # Consists of arrays where each subarray represents each k fold, and each entry in the subarray is for each classifier
                            # Example: [fold1[class1, class2, class3, class4, class5], fold2[class1, class2, class3, class4, class5], ...]
    classifier_accuracies = []  # Consists of subarrays, where each subarray represents each classifier, and each entry in the subarray represents the accuracy of the k folds for each classifier
                                # Example: [classifier_1[fold1, fold2, fold3, fold4, fold5], classifier_2[fold1, fold2, fold3, fold4, fold5], ...]
    p_values = []
    print(f'length X_train: {len(X_train)}')
    print(f'length X_test: {len(X_test)}')
    print(f'length y_train: {len(y_train)}')
    print(f'length y_test: {len(y_test)}')
    X_total = np.concatenate((X_train, X_test), axis=0)
    y_total = np.concatenate((y_train, y_test), axis=0)
    print(f'length X_total: {len(X_total)}')
    print(f'length y_total: {len(y_total)}')
    scaler = StandardScaler()

    k_fold_cross_validator = KFold(n_splits=5, shuffle=True, random_state=1)
    # Order of classifiers = SGDClassifier -> GaussianNB -> RandomForestClassifier -> MLPClassifier -> AdaBoostClassifier 
    classifiers = [('SGDClassifier', SGDClassifier(loss='hinge')), 
                   ('GaussianNB', GaussianNB()), 
                   ('RandomForestClassifier', RandomForestClassifier(max_depth=5, n_estimators=10)), 
                   ('MLPClassifier', MLPClassifier(alpha=0.05)), 
                   ('AdaBoostClassifier', AdaBoostClassifier())]
    
    print(f'Testing print, check for indexing here: {k_fold_cross_validator.split(X_total)}')

    for train_index, test_index in k_fold_cross_validator.split(X_total):
        
        X_train_fold = X_total[train_index] 
        X_test_fold = X_total[test_index]
        y_train_fold = y_total[train_index]
        y_test_fold = y_total[test_index]
        temp_accuracies = []    # Array to house all the accuracies for each k fold

        for i in range(5):
            classifier_name, classifier = classifiers[i]
            print(f'Running for classifier: {classifier_name}')
            if (classifier_name == 'SGDClassifier') or (classifier_name == 'MLPClassifier') or (classifier_name == 'AdaBoostClassifier'):
                X_train_fold = scaler.fit_transform(X_train_fold)
                X_test_fold = scaler.transform(X_test_fold)
            
            print('starting to fit')
            classifier.fit(X_train_fold, y_train_fold)
            print('finished fitting')
            y_prediction_fold = classifier.predict(X_test_fold)
            conf_matrix = confusion_matrix(y_test_fold, y_prediction_fold)
            acc = accuracy(conf_matrix)
            temp_accuracies.append(acc)
            print('accuracy calculated')

        kfold_accuracies.append(temp_accuracies)   
        print(temp_accuracies)
    print(kfold_accuracies)
    
    # Now populate the array where each subarray represents the k fold accuracies for each classifier
    for j in range(5):
        temp_classifier_accuracies = [] # This is a subarray representing the k fold accuracices for a single classifier
        for n in range(5):
            temp_classifier_accuracies.append(kfold_accuracies[n][j])
        classifier_accuracies.append(temp_classifier_accuracies)

    # Now compare the accuracies of the classifiers
    for k in range(5):
        # If the index we're comparing against is the index of the best classifier, skip to the next iteration
        if k == best_index:
            continue
        else:
            S = ttest_ind(classifier_accuracies[best_index], classifier_accuracies[k])
            pval = S.pvalue
            p_values.append(pval)
    
    # MAKE SURE THE ABOVE IS CORRECT
            
    with open(f"{output_dir}/a1_3.4.txt", "a") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # for each fold:
        for fold, accuracies in enumerate(kfold_accuracies):
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in accuracies]}\n')
        outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="The input npz file from Task 2.", required=True)
    parser.add_argument(
        "-o", "--output-dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    # TODO: load data and split into train and test.
    features = np.load('feats.npz')['arr_0']
    X = features[:, :173]
    y = features[:, 173]
    X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

    # TODO: complete each classification experiment, in sequence.
    # Class 31 experiment
    print('Starting class 31 test')
    best_index = class31(args.output_dir, X_train_set, y_train_set, X_test_set, y_test_set)
    print('Class 31 test run finished')

    # Class 32 experiment
    print('Starting class 32 test')
    X_1k, y_1k = class32(args.output_dir, X_train_set, y_train_set, X_test_set, y_test_set, best_index)
    print('Class 32 test run finished')

    # Class 33 experiment
    print('Starting class 33 test')
    class33(args.output_dir, X_train_set, y_train_set, X_test_set, y_test_set, best_index, X_1k, y_1k)
    print('Class 33 test run finished')

    # Class 34 experiment
    print('Starting class 34 test')
    class34(args.output_dir, X_train_set, y_train_set, X_test_set, y_test_set, best_index)
    print('Class 34 test run finished')
