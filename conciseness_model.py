import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_predict, train_test_split, KFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
import lightgbm as lgb
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from sklearn import svm
import matplotlib.pyplot as plt


df = pd.read_csv('training\data_train.csv')

df_list = df.values.tolist()

product_titles = [lst[2] for lst in df_list[:1000]]

df = pd.read_csv('training/conciseness_train.labels')
conciseness_labels = []
for labels in df.values.tolist()[:1000]:
    conciseness_labels.append(labels[0])
Y = conciseness_labels

def NWordFeatures(X):
    # Get N word features (1 to 3)
    features = CountVectorizer(ngram_range=(1,3), max_features=200, stop_words='english').fit(X)
    return features.get_feature_names_out().tolist()

def NWordTransform(features, X):
    # Features are the N-word features
    # X is the dataset to transform into the N-word features
    return features.transform(X).toarray()

def NGramFeatures(X):
    # Get Ngram features for n = [3,4,5,6]
    features = dict()
    for title in product_titles:
        for i in [3,4,5,6]:
            ngrams = [title[j:j+i] for j in range(len(title) - i)]
            for ngram in ngrams:
                if ngram not in features:
                    features[ngram] = 0
                else:
                    features[ngram] += 1
    features = dict(sorted(features.items(), key = lambda feature : -feature[1]))
    features = list(features)[:200]
    return features

# This function transforms the initial array X of features [f1, f2, f3, ..., fk] into new features [g1, g2, ..., gl]
def Transform(X, features):
    ret = []
    for i in range(len(X)):
        ret.append([])
        for j in range(len(features)):
            cnt = 0
            for k in range(len(X[i])):
                l = 0
                while k+l < len(X[i]) and l < len(features[j]) and features[j][l] == X[i][k+l]:
                    l += 1
                if l == len(features[j]):
                    cnt += 1
            ret[i].append(cnt)
    return ret

# Combine the features of N-words and N-grams
features = NWordFeatures(product_titles) + NGramFeatures(product_titles)

# Transform the original dataset into a 2d array (n_samples, n_features)
X = Transform(product_titles, features)
X = np.array(X)
Y = np.array(Y)

# This function will compute the error obtained from the estimators
def compute_error(predictions, actual):
    err = 0
    for i in range(0, len(predictions)):
        if predictions[i] != actual[i]:
            err += 1
    return (err / len(predictions))

final_predictions = [0] * len(Y)
# Fold Set Generation
for Fold_set in range(4):

    errors = []
    # LGB Base Model (Light Gradient Boosting)
    LGB = lgb.LGBMClassifier(learning_rate=0.4)

    # 'cross_val_predict' converts the training data into k-folds where k = cv
    # and then performs testing on each of the folds to generate the predictions
    pred = cross_val_predict(LGB, X, Y, cv = 10)
    errors.append(compute_error(pred, Y))

    # SVM Base Model (Support Vector Machines)
    svm_clf = svm.SVC()
    predictions = cross_val_predict(svm_clf, X, Y, cv = 10)
    errors.append(compute_error(predictions, Y))
    
    # XGB Base Model (Extreme Gradient Boosting)
    xgb_cl = xgb.XGBClassifier()
    predictions = cross_val_predict(xgb_cl, X, Y, cv = 10)
    errors.append(compute_error(predictions, Y))
    
    # Multinomial NBC (Naive Bayes Classifier)
    nbc = MultinomialNB()
    predictions = cross_val_predict(nbc, X, Y, cv = 10)
    errors.append(compute_error(predictions, Y))

    # SGD Base Model (Stochastic Gradient Descent)
    sgdc = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    predictions = cross_val_predict(sgdc, X, Y, cv = 10)
    errors.append(compute_error(predictions, Y))

    # Stacked Ensemble Model
    base = []
    base.append(('lgb', LGB))
    base.append(('xgb', xgb_cl))
    base.append(('svm', svm_clf))
    base.append(('multi_nbc', nbc))
    base.append(('sgdc', sgdc))

    ensemble_model = StackingClassifier(estimators=base, final_estimator=xgb_cl, cv = 10)
    predictions = cross_val_predict(ensemble_model, X, Y, cv = 10)
    
    for idx in range(len(predictions)):
        final_predictions[idx] += predictions[idx]

    errors.append(compute_error(predictions, Y))
    if Fold_set == 0:
        fig = plt.figure(figsize = (8,6))
        Models = ['LGB', 'XGB', 'SVM', 'Multinomial NBC', 'SGD', 'Ensemble']
        Errors = errors
        plt.bar(Models,Errors)
        plt.ylabel('Errors')
        plt.title('Error analysis of machine learning algorithms')
        plt.show()


tp = 0
fn = 0
tn = 0
fp = 0
# Final Prediction averaged over all 4 fold-sets
for i in range(len(final_predictions)):
    p = final_predictions[i]
    p = p / 4
    if p < 0.5:
        p=0
    else:
        p=1
    final_predictions[i] = p
    if final_predictions[i] == 1:
        if Y[i] == 0:
            fp += 1
        else:
            tp += 1
    else:
        if Y[i] == 0:
            tn += 1
        else:
            fn += 1

print('Final Predictions Error =', compute_error(final_predictions, Y))
precision = tp / (tp + fp)
print('Precision = ', precision)
recall = tp / (tp + fn)
print('Recall = ', recall)
f_measure = (2 * precision * recall) / (precision + recall)
print('F-measure =', f_measure)
