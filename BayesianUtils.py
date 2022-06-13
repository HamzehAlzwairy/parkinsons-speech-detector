import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
def get_selected_features(X):
    # based on analysis done in DecisionTrees.py
    return X[[ 'rapJitter', 'meanAutoCorrHarmonicity'
        , 'meanNoiseToHarmHarmonicity', 'GQ_prc5_95', 'mean_1st_delta'
        , 'mean_7th_delta_delta', 'mean_9th_delta_delta', 'std_1st_delta'
        , 'std_delta_delta_log_energy', 'std_12th_delta_delta'
        , 'det_entropy_log_10_coef', 'det_TKEO_mean_2_coef', 'det_TKEO_std_9_coef'
        , 'app_entropy_shannon_5_coef', 'app_entropy_log_3_coef'
        , 'app_det_TKEO_mean_6_coef', 'app_det_TKEO_mean_7_coef'
        , 'app_TKEO_std_5_coef', 'app_TKEO_std_6_coef', 'app_TKEO_std_7_coef'
        , 'app_TKEO_std_9_coef', 'det_LT_entropy_shannon_1_coef'
        , 'det_LT_entropy_shannon_10_coef', 'det_LT_entropy_log_5_coef'
        , 'app_LT_entropy_shannon_3_coef', 'app_LT_entropy_log_3_coef'
        , 'app_LT_entropy_log_7_coef', 'app_LT_TKEO_mean_8_coef'
        , 'app_LT_TKEO_std_8_coef', 'PPE', 'RPDE', 'DFA', 'numPulses'
        , 'meanHarmToNoiseHarmonicity', 'locShimmer']]
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


def plot_distributions_single_fig(df):
    n_cols = 6
    n_rows = int(math.floor((len(df.columns.values)/n_cols)))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    fig.tight_layout()
    axes = axes.ravel()
    for i, column in enumerate(df.columns.values):
        if column == "label":
            break
        sns.kdeplot(data=df, x=column, hue="label", fill=True,
                        palette=["blue",'red'], legend=False, ax=axes[i])
    fig.legend(["patient", "healthy"], loc='upper left', )
    plt.show()

# parameters of gaussian
def get_mean_and_covariance(data):
    mue_vector = data.mean(axis=0)
    covariance_matrix = data.cov(ddof=1)
    return mue_vector, covariance_matrix

def plot_pca_scatter(X, Y):
    pca = PCA(n_components=2, svd_solver='full')
    # apply PCA to normalized training data
    transformed_data = pca.fit_transform(X)
    transformed_data = pd.DataFrame(transformed_data)
    transformed_data = transformed_data.assign(label=Y.values)
    groups = [data for i, data in transformed_data.groupby('label')]
    color = ['blue', 'red']
    marker = ["o", "x"]
    classes = ["0", "1"]
    for i, group in enumerate(groups):
        plt.scatter(group[[0]], group[[1]], c=color[i], marker=marker[i], label=classes[i])
    plt.legend(loc="upper left")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

class BayesClassifier(object):
    def __init__(self):
        self.class_conditional_probabilities = None
        self.prior_probabilities = None
        self.losses = None

    def fit(self, PDFs, priors, losses, classes):
        self.class_conditional_probabilities = PDFs
        self.prior_probabilities = priors
        self.losses = losses
        self.classes = classes

    def classify(self, x):
        g_x_risks = []
        for PDF,loss in zip(self.class_conditional_probabilities,self.losses):
            p_x = PDF.pdf(x)
            posterior_x = p_x * 0.5 # assume equal priors, because we have class imbalance, if we provide p(w0) it will increase its risk
            risk_x = loss * posterior_x
            g_x_risks.append(risk_x)
        class_index = np.argmin(g_x_risks)
        return self.classes[class_index]
