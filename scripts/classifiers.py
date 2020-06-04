# LIBRAIRIES

import os
import argparse
import pandas               as pd
import numpy                as np
import matplotlib.pyplot    as plt
import seaborn              as sns
from argparse                   import ArgumentParser
from tabulate                   import tabulate
from sklearn                    import preprocessing
from sklearn                    import metrics
from sklearn                    import tree
from sklearn                    import svm
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.linear_model       import LogisticRegression
from sklearn.model_selection    import train_test_split
from sklearn.model_selection    import cross_val_score
from sklearn.metrics            import log_loss
from sklearn.metrics            import f1_score
from sklearn.metrics            import jaccard_similarity_score

parser = ArgumentParser()
parser.add_argument(
    "--export", 
    metavar = "export",
    default = True
)
args = parser.parse_args()
home_dir = '/Users/kjarr/Documents/Ressources/Data Science/Portofolio/Loan'

# DATA

df = pd.read_csv(os.path.join(home_dir, 'data', 'train_set.csv'))

# RESHAPE DATES

df['due_date']       = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df['dayofweek']      = df['effective_date'].dt.dayofweek
df['weekend']        = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

# CAT TO NUM

df['Gender'].replace(
    to_replace = ['male','female'], 
    value      = [0, 1],
    inplace    = True
)

# 1HOT

ft = df[['Principal', 'terms', 'age', 'Gender', 'weekend', 'dayofweek']]
ft = pd.concat(
    [ft, pd.get_dummies(df['education'])], 
    axis = 1
)
ft.drop(
    ['Master or Above'], 
    axis    = 1,
    inplace = True
)

#=========#
# * VIZ * #
#=========#

if args.export:
    plot = sns.FacetGrid(
        df, 
        col      = "Gender", 
        hue      = "loan_status", 
        palette  = "Set1", 
        col_wrap = 2
    )
    plot.map(
        plt.hist, 
        'Principal', 
        bins = np.linspace(df.Principal.min(), df.Principal.max(), 10), 
        ec   = "k"
    )
    plot.axes[-1].legend()
    plt.savefig(
        'reports/plt_principal.png',
        dpi = 300
    )
    del plot

    plot = sns.FacetGrid(
        df, 
        col      = "Gender", 
        hue      = "loan_status", 
        palette  = "Set1", 
        col_wrap =2
    )
    plot.map(
        plt.hist, 
        'age', 
        bins = np.linspace(df.age.min(), df.age.max(), 10), 
        ec   = "k"
    )
    plot.axes[-1].legend()
    plt.savefig(
        'reports/plt_age.png',
        dpi = 300
    )
    del plot

    plot = sns.FacetGrid(
        df, 
        col      = "Gender", 
        hue      = "loan_status", 
        palette  = "Set1", 
        col_wrap = 2
    )
    plot.map(
        plt.hist, 
        'dayofweek', 
        bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10), 
        ec   = "k"
    )
    plot.axes[-1].legend()
    plt.savefig(
        'reports/plt_dayofweek.png',
        dpi = 300
    )
    del plot

#====================#
# * CLASSIFICATION * #
#====================#

train_scores = {}

# IN / OUT

X = ft
X = preprocessing.StandardScaler().fit(X).transform(X)

y = df['loan_status'].replace(
    to_replace = ['PAIDOFF','COLLECTION'], 
    value      = [0,1]
).values.astype(float)

""" X.shape
    y.shape """

# CLASS_KNN

score_best = 0

for k in range(5, 15):
    knn = KNeighborsClassifier(
        n_neighbors = k,
        algorithm   = 'auto'
    )
    scores = cross_val_score(
        knn, 
        X, y, 
        cv = 10
    )
    score = scores.mean()
    if score > score_best:
        knn_best   = knn
        score_best = score
        k_best     = k

class_knn = knn_best
class_knn.fit(X, y)
yhat = class_knn.predict(X)

train_scores['KNN - jaccard'] = jaccard_similarity_score(y, yhat)
train_scores['KNN - f1']      = f1_score(y, yhat, average = 'weighted')

del knn, knn_best, yhat, score, score_best, k, k_best

# CLASS_TREE

class_tree = tree.DecisionTreeClassifier()
class_tree = class_tree.fit(X, y)
yhat = class_tree.predict(X)

train_scores['TREE - jaccard'] = jaccard_similarity_score(y, yhat)
train_scores['TREE - f1']      = f1_score(y, yhat, average = 'weighted')

del yhat

# CLASS_SVM

class_svm = svm.LinearSVC(random_state = 7)
class_svm.fit(X, y)

yhat = class_svm.predict(X)

train_scores['SVM - jaccard'] = jaccard_similarity_score(y, yhat)
train_scores['SVM - f1']      = f1_score(y, yhat, average = 'weighted')

del yhat

# CLASS_LOGREG

class_log = LogisticRegression(
    random_state = 0, 
    solver       = 'lbfgs',
    multi_class  = 'multinomial'
)
class_log.fit(X, y)

yhat  = class_log.predict(X)

train_scores['LOGREG - jaccard']  = jaccard_similarity_score(y, yhat)
train_scores['LOGREG - f1-score'] = f1_score(y, yhat, average = 'weighted')  
train_scores['LOGREG - logLoss']  = log_loss(y, class_log.predict_proba(X))

del yhat

#================#
# * EVALUATION * #
#================#

# TEST DATA

test_df = pd.read_csv(os.path.join(home_dir, 'data', 'test_set.csv'))

# RESHAPE DATES

test_df['due_date']       = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek']      = test_df['effective_date'].dt.dayofweek
test_df['weekend']        = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

# CAT TO NUM

test_df['Gender'].replace(
    to_replace = ['male','female'], 
    value      = [0,1],
    inplace    = True
)

# FEAT SEL

ft = test_df[['Principal', 'terms', 'age', 'Gender', 'weekend', 'dayofweek']]
ft = pd.concat(
    [ft, pd.get_dummies(test_df['education'])], 
    axis = 1
)
ft.drop(
    ['Master or Above'], 
    axis    = 1,
    inplace = True
)

# IN/OUT

X = ft
X = preprocessing.StandardScaler().fit(X).transform(X)

y = test_df['loan_status'].replace(
    to_replace = ['PAIDOFF','COLLECTION'], 
    value      = [0,1]
).values

X.shape[0] == y.shape[0]

# SCORES

test_scores = {}

pred_knn  = class_knn.predict(X)
pred_tree = class_tree.predict(X)
pred_svm  = class_svm.predict(X)
pred_log  = class_log.predict(X)

test_scores['KNN - jaccard']    = jaccard_similarity_score(y, pred_knn)
test_scores['KNN - f1']         = f1_score(y, pred_knn, average = 'weighted')

test_scores['TREE - jaccard']   = jaccard_similarity_score(y, pred_tree)
test_scores['TREE - f1']        = f1_score(y, pred_tree, average = 'weighted')

test_scores['SVM - jaccard']    = jaccard_similarity_score(y, pred_svm)
test_scores['SVM - f1']         = f1_score(y, pred_svm, average = 'weighted')

test_scores['LOGREG - jaccard'] = jaccard_similarity_score(y, pred_log)
test_scores['LOGREG - f1']      = f1_score(y, pred_log, average = 'weighted')  
test_scores['LOGREG - logloss'] = log_loss(y, class_log.predict_proba(X))

train_scores
test_scores

report = {
    'Model' : [
        'KNN',
        'DECISION TREE',
        'SVM',
        'LOGISTIC REGRESSION'
    ],
    'Jaccard' : [
        test_scores.get('KNN - jaccard'),
        test_scores.get('TREE - jaccard'),
        test_scores.get('SVM - jaccard'),
        test_scores.get('LOGREG - jaccard')
    ],
    'F1-score' : [
        test_scores.get('KNN - f1'),
        test_scores.get('TREE - f1'),
        test_scores.get('SVM - f1'),
        test_scores.get('LOGREG - f1')
    ],
    'LogLoss' : [
        None,
        None,
        None,
        test_scores.get('LOGREG - logloss')
    ]
}

report = pd.DataFrame(
    report, 
    columns = ['Model', 'Jaccard', 'F1-score', 'LogLoss']
)

print(report)

if args.export:
    with open(os.path.join(home_dir, 'reports', 'report.txt'), 'a') as f:
        f.write(report.to_string())