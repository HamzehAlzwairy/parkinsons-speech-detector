import pandas as pd
import numpy as np
from BayesianUtils import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from imblearn.under_sampling import RandomUnderSampler

np.random.seed(8)
patterns_df = pd.read_csv("./data/pd_speech_features.csv")
# hyper-parameters
test_size = 0.20
lambda_0_1 = 0.5
lambda_1_0 = 5
sampling_ratio = 0.5625

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

# normalize features

normalized_X=(X_engineered-X_engineered.mean())/X_engineered.std()
normalized_X_test = (X_test_engineered-X_engineered.mean())/X_engineered.std()

plot_pca_scatter(normalized_X, Y_train)
# perform normalized_X
undersample = RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=8)
undersampled_X, Y_undersampled= undersample.fit_resample(normalized_X, Y_train)
# visualize distributions of normalized data
plot_distributions_single_fig(pd.concat([undersampled_X.assign(label=Y_undersampled.values),
                                       normalized_X_test.assign(label=Y_test.values)], ignore_index=True))




undersampled_X = undersampled_X.assign(label=Y_undersampled.values)
train_patients_data = undersampled_X.loc[undersampled_X["label"]==1].drop(columns='label')

train_healthy_data = undersampled_X.loc[undersampled_X["label"]==0].drop(columns='label')

# estimate parameters of gaussian from the data
ω1_mue, ω1_covariance = get_mean_and_covariance(train_patients_data)
ω0_mue, ω0_covariance = get_mean_and_covariance(train_healthy_data)
ω1_pdf = multivariate_normal(mean=ω1_mue, cov=ω1_covariance)
ω0_pdf = multivariate_normal(mean=ω0_mue, cov=ω0_covariance)

classifier = BayesClassifier()
classifier.fit(PDFs=[ω1_pdf, ω0_pdf],priors=None,losses=[lambda_0_1, lambda_1_0],classes=[0,1])

#
y_pred = normalized_X_test.apply(classifier.classify, axis=1)
cm = confusion_matrix(Y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
NPV = tn/(tn+fn)
specificity = tn/(tn+fp)
# PPV is precision, sensitivity is recall
PPV, sensitivity, f1, support = precision_recall_fscore_support(y_pred, Y_test , average="weighted")
print("Positive Predictive Values: {0}".format(PPV))
print("Negative Predictive Values: {0}".format(NPV))
print("Sensitivity: {0}".format(sensitivity))
print("Specificity: {0}".format(specificity))
print("F1 score: {0}".format(f1))
print("Confusion matrix \n{0}".format(cm))
