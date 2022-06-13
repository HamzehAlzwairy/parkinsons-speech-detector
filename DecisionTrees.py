from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import numpy as np
from DecisionTreeUtils import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

np.random.seed(8)
patterns_df = pd.read_csv("./data/pd_speech_features.csv")
# hyper-parameters
test_size = 0.20
min_samples_split = 5 # decrease for overfit, increase for underfit
n_features = 30 # features to select from FE
correlation_threshold = 0.6 # threshold of correlation to remove

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

# ------------ uncommenct to obtain uncorrelated plots along with the distributions of features ------------------------
# corr = plot_corr(X_train)
#
# # remove features with correlation >0.6
# columns_to_drop = get_uncorr_features(corr, correlation_threshold)
#
#
# # Drop features
# X_train_uncorrelated= X_train.drop(columns_to_drop, axis=1)
# X_test_uncorrelated = X_test.drop(columns_to_drop, axis=1)
#
# # for the sake of plotting the histograms, add y again then remove it.
# X_train_uncorrelated = X_train_uncorrelated.assign(label=Y_train.values)
# X_test_uncorrelated = X_test_uncorrelated.assign(label=Y_test.values)
# plotting_df = pd.concat([X_train_uncorrelated, X_test_uncorrelated], ignore_index=True)
# X_train_uncorrelated = X_train_uncorrelated.drop(columns=["label"])
# X_test_uncorrelated = X_test_uncorrelated.drop(columns=["label"])
#
#
# plot_distributions(plotting_df)
# plot_corr(X_train_uncorrelated)
# ----------------------------------------------------------------------------------------------------------------------

# apply sequential floating feature selection
# balanced option in tree weights every class by its distribution's inverse, it means if we have in a group 5 '1'
# and 2 '0', and ratio is 3:1, it x3 the 0's so they classifier will decide based on 5 and 6 for 1 and 0
tree_classifier = DecisionTreeClassifier(class_weight='balanced', min_samples_split=min_samples_split,
                                         random_state=8)
# less features means less computational complex, less overfitting

X_engineered, X_test_engineered, selected_features_names = apply_sffs(X_train,Y_train,
                                                                X_test, n_features, tree_classifier)
#plot_distributions_single_fig(plotting_df)
#  adding important features
X_engineered = X_engineered.assign(PPE= X_train[['PPE']], RPDE= X_train[['RPDE']],
                                   DFA= X_train[['DFA']], numPulses= X_train[['numPulses']],
                                   meanHarmToNoiseHarmonicity=X_train[["meanHarmToNoiseHarmonicity"]],
                                   locShimmer=X_train[["locShimmer"]]
                                   )

X_test_engineered = X_test_engineered.assign(PPE= X_test[['PPE']], RPDE= X_test[['RPDE']],
                                             DFA= X_test[['DFA']], numPulses= X_test[['numPulses']],
                                             meanHarmToNoiseHarmonicity=X_test[["meanHarmToNoiseHarmonicity"]],
                                             locShimmer = X_test[["locShimmer"]])

tree_classifier.fit(X_engineered, Y_train)
y_pred = tree_classifier.predict(X_test_engineered)

cm = confusion_matrix(Y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
NPV = tn/(tn+fn)
specificity = tn/(tn+fp)
# PPV is precision, sensitivity is recall
PPV, sensitivity, f1, support = precision_recall_fscore_support(y_pred, Y_test , average="weighted")


print("Final {0} selected features are: \n{1}".format(len(X_engineered.columns.values), X_engineered.columns.values))
print("Positive Predictive Values: {0}".format(PPV))
print("Negative Predictive Values: {0}".format(NPV))
print("Sensitivity: {0}".format(sensitivity))
print("Specificity: {0}".format(specificity))
print("F1 score: {0}".format(f1))

# ----------uncomment to plot features importance ------------
# importances = tree_classifier.feature_importances_
# indices = np.argsort(importances) # top 10 features
# plt.figure(figsize = (15, 8))
# plt.title('Feature Importances')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [X_engineered.columns[i] for i in indices])
# plt.xlabel('Relative Importance')
# -------------------------------------------------------------------------------------

# ----------uncomment to plot boxplots for features in relation to classes------------
# plotting_df = patterns_df[[X_engineered.columns[i] for i in indices[-6:]]]
# plotting_df = plotting_df.assign(label=patterns_df[["class"]])
# plot_features_relation_to_y(plotting_df)
# -------------------------------------------------------------------------------------

plt.show()

