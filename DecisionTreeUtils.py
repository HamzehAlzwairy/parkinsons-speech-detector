from sklearn.model_selection import GroupShuffleSplit
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
def get_half_test_samples(test_sample_total, df):
    # ex, total test is 40, so 20 is the number of samples needed from this group, we convert it to percentage.
    group_precentage = (test_sample_total / 2) / len(df)

    # split data to train and test by patient ID to avoid patient overlap ( data leakage to test )
    splitter = GroupShuffleSplit(test_size=group_precentage, n_splits=1, random_state=8)
    split = splitter.split(df, groups=df['id'])

    train_individuals, test_individuals = next(split)
    train_individuals = df.iloc[train_individuals]  # 486 x 755 for patients, 116 x 755 for healthy
    test_individuals = df.iloc[test_individuals]  # 78 x 755 for patients, 76 x 755 for healthy

    return train_individuals, test_individuals

def plot_distributions(df):
    n_cols = 5
    n_rows = int((len(df.columns.values)/2/n_cols))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    fig.tight_layout()

    axes = axes.ravel()
    # fig 2
    fig2, axes2 = plt.subplots(nrows=n_rows, ncols=n_cols)
    fig2.tight_layout()
    axes2 = axes2.ravel()
    if "gender" in df.columns.values:
        df[["gender"]] = df[["gender"]].astype("int")

    first_half = df.columns.values[:int(len(df.columns.values)/ 2)]
    second_half =df.columns.values[int(len(df.columns.values)/ 2):]
    for i, column in enumerate(first_half):
        sns.kdeplot(data=df, x=column, hue="label", fill=True,
                        palette=["blue",'red'], legend=False, ax=axes[i])

    for i, column in enumerate(second_half):
        sns.kdeplot(data=df, x=column, hue="label", fill=True,
                        palette=["blue",'red'], legend=False,  ax=axes2[i])
    fig.legend(["patient","healthy"],loc='upper left',)
    fig2.legend(["patient", "healthy"], loc='upper left', )
    #plt.show()

def plot_distributions_single_fig(df):
    n_cols = 3
    n_rows = int(math.ceil((len(df.columns.values)/n_cols)))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    fig.tight_layout()
    axes = axes.ravel()
    for i, column in enumerate(df.columns.values):
        print(column)
        if column == "class":
            break
        sns.kdeplot(data=df, x=column, hue="class", fill=True,
                        palette=["blue",'red'], legend=False, ax=axes[i])
    fig.legend(["patient", "healthy"], loc='upper left', )
    #plt.show()

def apply_sffs(X,Y, X_test,n_features, classifier):
    sffs = SFS(classifier,
               k_features=n_features,
               forward=True,
               floating=True,
               scoring='f1',
               cv=3,
               n_jobs=-1)
    sffs = sffs.fit(X, Y, custom_feature_names=X.columns.values)
    X_engineered= X[list(sffs.k_feature_names_)]
    X_test_engineered= X_test[list(sffs.k_feature_names_)]
    return X_engineered, X_test_engineered, sffs.k_feature_names_

def plot_corr(X):
    corr = X.corr().abs()
    sns.set(font_scale=0.7)
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    #plt.show()
    return corr

def get_uncorr_features(corr, thresh):
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > thresh)]
    return to_drop

def plot_features_relation_to_y(df):
    n_cols = 2
    n_rows = int(math.floor((len(df.columns.values)/n_cols)))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    fig.tight_layout()
    axes = axes.ravel()
    for i, column in enumerate(df.columns.values):
        if column == "label":
            break
        sns.boxplot(x=df['label'], y=df[column], data=df, ax=axes[i])
    #plt.show()