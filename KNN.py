import pandas as pd
import numpy as np
from BayesianUtils import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from imblearn.under_sampling import RandomUnderSampler

np.random.seed(8)
patterns_df = pd.read_csv("./data/pd_speech_features.csv")
# hyper-parameters
test_size = 0.20
sampling_ratio = 0.5625
k_values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

# safer to make all of them float, I saw app_entropy_shannon_10_coef which is int while app_entropy_shannon_9_coef is float
patterns_df = patterns_df.astype(float)
patterns_df[["gender", "class"]] = patterns_df[["gender", "class"]].astype("category")
patterns_df[["numPulses", "numPeriodsPulses", "id"]] = patterns_df[["numPulses", "numPeriodsPulses", "id"]].astype(int)

n = len(patterns_df)
test_sample_total = np.ceil(test_size * n)
# this step is to get closest number of individuals to selected sample size, for example 31 is not divisible by 3,
# so we find closest number that is divisible by 3 because we want to keep each individual in 1 set.
test_sample_total = 3 * np.ceil(test_sample_total / 3)
# print("total number of test samples required {0}".format(test_sample_total))

# data split, includes set sampling and id grouping
healthy_individuals = patterns_df[patterns_df["class"] == 0]
patients = patterns_df[patterns_df["class"] == 1]
healthy_train_individuals, healthy_test_individuals = get_half_test_samples(test_sample_total, healthy_individuals)
patients_train_individuals, patients_test_individuals = get_half_test_samples(test_sample_total, patients)

X_train = pd.concat([healthy_train_individuals, patients_train_individuals])
X_test = pd.concat([healthy_test_individuals, patients_test_individuals]) # 50% are healty, 50% are patients
X_train = X_train.sample(frac=1) # shuffle rows, because currently they are first half healthy, second half patients
X_test = X_test.sample(frac= 1)

Y_train = X_train[['class']]
Y_test = X_test[['class']]



X_train = X_train.drop(columns=["class","id"])
X_test = X_test.drop(columns=["class","id"])

# paper for this dataset suggests to remove all the tqwt features
X_train = X_train[X_train.columns[0: -432]]
X_test = X_test[X_test.columns[0: -432]]

# feature engineering
X_engineered = get_selected_features(X_train)
X_test_engineered = get_selected_features(X_test)


normalized_X=(X_engineered-X_engineered.mean())/X_engineered.std()
normalized_X_test = (X_test_engineered-X_engineered.mean())/X_engineered.std()

undersample = RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=8)
undersampled_X, Y_undersampled= undersample.fit_resample(normalized_X, Y_train)

F1_scores =[]
best_k =None
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(undersampled_X, Y_undersampled.squeeze())
    y_pred = knn.predict(normalized_X_test)
    cm = confusion_matrix(Y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    NPV = tn/(tn+fn)
    specificity = tn/(tn+fp)
    # PPV is precision, sensitivity is recall
    PPV, sensitivity, f1, support = precision_recall_fscore_support(y_pred, Y_test , average="weighted")
    if all(i < f1 for i in F1_scores):
        best_k=k
        print(f1)
    F1_scores.append(f1)
plt.xlabel("k")
plt.ylabel("F1 score")
plt.plot(k_values, F1_scores, label="f1 against k")
print("best k {0}".format(best_k))
print("Positive Predictive Values: {0}".format(PPV))
print("Negative Predictive Values: {0}".format(NPV))
print("Sensitivity: {0}".format(sensitivity))
print("Specificity: {0}".format(specificity))
print("F1 score: {0}".format(f1))
print("Confusion matrix \n{0}".format(cm))
plt.show()