
# coding: utf-8

# # Setup
# ## Imports

# In[1]:

from itertools import chain
import html
import ujson as json
import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../../data/merged/data_to_use.csv')


# In[2]:

print(df.columns)


# ## Induced features, relabeling, etc.
# We could (and probably should) do this in the other notebook. But this is a fine  place for now

# In[3]:

# df['class'] = df['class'].isin(['positive'])
df['class'] = df['class'].apply(lambda x: 1 if x == 'positive' else 0)
df['class'] = df['class'].astype('bool')
df['lt_23'] = df.age.apply(lambda x: True if x < 23 else False)


# ## Messing around
# cross-tabs, etc. Can junk this section

# In[4]:

ct = pd.crosstab(df['class'], df.flag)
sns.heatmap(ct.T, annot=True, fmt='d', cmap='bone_r')


# In[5]:

sns.heatmap(ct.T/ct.sum(axis=1), annot=True, cmap='bone_r')


# In[6]:

ct = pd.crosstab(df['class'], df.flag)
sns.heatmap(ct.T, annot=True, fmt='d', cmap='bone_r')


# ## Generate Splits
# We can share these splits (or whatever the final splits end up being) across classifiers

# In[7]:

splitter = ShuffleSplit(df.shape[0], 10)
splits = [x for x in splitter]


# # Classifiers
# ## Linreg
# 
# We're using **dumb** folds. These should be smartened up.

# ### Slice DF

# In[8]:

df_X = df.ix[:, ['age',
                 'lt_23',
                 'price',
                 'duration_in_mins',
                 'price_per_min',
                 'flag', 'ethnicity']].copy()

df_X.age = df_X.age.fillna(5000000)
df_X.price = df_X.price.fillna(5000000)
df_X.duration_in_mins = df_X.duration_in_mins.fillna(5000000)
df_X.price_per_min = df_X.price_per_min.fillna(5000000)
df_X = pd.get_dummies(df_X)


# ### Run model

# In[33]:

def train_tester(df_X_train, y_train, df_X_test, y_test):
    lr = LinearRegression()
    lr.fit(df_X_train, y_train)
    y_pred = lr.predict(df_X_test)
    fpr, tpr, thresholds = roc_curve(y_test.values, y_pred)
    return {'model':lr,
            'y_pred': y_pred,
            'y_test': y_test.values,
            'lr_score': lr.score(df_X_test, y_test),
            'roc': (fpr, tpr, thresholds),
            'auc': auc(fpr, tpr)
            }


#p = mp.Pool(10)
#lrs = p.starmap(train_tester,
                #[(df_X.iloc[train_ix, :],
                  #df['class'].iloc[train_ix],
                  #df_X.iloc[test_ix, :],
                  #df['class'].iloc[test_ix])
                 #for train_ix, test_ix in splits])
#lrs = train_tester(
        #df_X.iloc[train_ix, :],
        #df['class'].iloc[train_ix],
        #df_X.iloc[test_ix, :],
        #df['class'].iloc[test_ix]
        #)
def score_metrics(y_test, y_pred):
    true_positives = (y_test & y_pred).sum()
    true_negatives = ((~y_test) & (~y_pred)).sum()
    false_positives = ((~y_test) & y_pred).sum()
    false_negatives = (y_test & (~y_pred)).sum()
    f1=(2*true_positives)/float(2*true_positives + false_negatives + false_positives)
    true_positive_rate= true_positives/float(true_positives + false_negatives)
    true_negative_rate = (true_negatives/float(true_negatives + false_positives))
    accuracy = (true_positives + true_negatives)/float(true_positives + true_negatives+false_positives + false_negatives)
    ipdb.set_trace()
    return(
            {
                'true_positive_rate':true_positive_rate,
                'true_negative_rate':true_negative_rate,
                'f1':f1,
                'accuracy':accuracy
                }
            )

def all_scoring_metrics(clf, X, y, stratified_kfold):
    out = []
    for i, (train, test) in enumerate(stratified_kfold):
        clf.fit(X.loc[train], y.loc[train])
        y_pred = clf.predict(X.loc[test])
        y_test=y.loc[test]

    output_features = score_metrics(y_test, y_pred)
    #output_features.update({i[0]:i[1] for i in zip(X.columns, clf.feature_importances_)})
    output_features['roc_auc'] = roc_auc_score(y_test, clf.predict_proba(X.loc[test])[:,1])
    out.append(output_features)
    return pd.DataFrame(out)

clf = RandomForestClassifier(oob_score=True, random_state=2, n_estimators=100, n_jobs=-1, class_weight="balanced")
del df_X['ethnicity']
metrics = all_scoring_metrics(clf, df_X, df['class'], StratifiedKFold(df['class'], 10))
print("Results (averaged from 10 fold cross validation and computed out of sample)")
print(metrics)
#p.close()
#p.join()


## In[34]:

#pd.Series([lr['lr_score'] for lr in lrs]).describe()


## In[36]:

#pd.Series([lr['auc'] for lr in lrs]).describe()


## With these splits and features, our linear model _juuuuust_ a little better than the mean. Thanks, skewed data :(

## In[29]:

#roc_df = pd.DataFrame({'fpr': pd.DataFrame.from_records([lr['roc'][0] for lr in lrs]).apply(np.mean),
                       #'tpr': pd.DataFrame.from_records([lr['roc'][1] for lr in lrs]).apply(np.mean)})
#roc_df.set_index('fpr', inplace=True)


## In[32]:

#roc_df.plot()


## In[ ]:

#df_slice = df.ix[:, ['label','age', 'lt_23', 'price', 'duration_in_mins', 'price_per_min']].copy()

#for x in ['age', 'price', 'duration_in_mins', 'price_per_min']:
    #df_slice[x] = df_slice[x].fillna(5000000)

#df_X = df_slice.ix[:, ['age', 'price', 'lt_23', 'duration_in_mins', 'price_per_min']]
#y = df_slice.label

#lr = LinearRegression()
#lr.fit(df_X, y)
#print(lr.coef_.shape)
#print(lr.intercept_)
#lr.score(df_X, y)


## In[ ]:




## ## Simple, unfolded, dumb random forest

## In[ ]:

#etc = ExtraTreesClassifier()


