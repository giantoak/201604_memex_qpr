
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
#import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import ipdb

#df = pd.read_csv('../../data/merged/data_to_use_by_ad_v2.csv')
df = pd.read_csv('../../data/merged/data_to_use_by_ad_v3_with_exp_imgs.csv')

steve = pd.read_csv('../../data/merged/phones_by_month.csv')
steve_phone=steve[['phone','n_ads','n_distinct_locations','location_tree_length','n_outcall','n_incall','n_incall_and_outcall','n_cooccurring_phones']].drop_duplicates()
df = df.merge(steve_phone, how='left')
df = df.loc[df[['dd_id','phone']].drop_duplicates().index]

df['has_images'] = df['images_count'].notnull()
# In[2]:



# ## Induced features, relabeling, etc.
# We could (and probably should) do this in the other notebook. But this is a fine  place for now

# In[3]:

# df['class'] = df['class'].isin(['positive'])
df['class'] = df['class'].apply(lambda x: 1 if x == 'positive' else 0)
df['class'] = df['class'].astype('bool')


# ## Messing around
# cross-tabs, etc. Can junk this section

# In[4]:

ct = pd.crosstab(df['class'], df.flag)
#sns.heatmap(ct.T, annot=True, fmt='d', cmap='bone_r')


# In[5]:

#sns.heatmap(ct.T/ct.sum(axis=1), annot=True, cmap='bone_r')


# In[6]:

ct = pd.crosstab(df['class'], df.flag)
#sns.heatmap(ct.T, annot=True, fmt='d', cmap='bone_r')


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
numerical_vars = ['age', 'price', 'duration_in_mins', 'price_per_min' , 'images_count','exp_ads_from_simimages_count','similar_images_count']
phone_level_vars = ['n_ads', 'n_distinct_locations', 'location_tree_length', 'n_outcall', 'n_incall', 'n_incall_and_outcall', 'n_cooccurring_phones']
#continuous_means = df.groupby('phone')[continuous_vars].mean()
#continuous_means = continuous_means.rename(columns = {k+'_mean':k for k in continuous_means.columns})
#continuous_stds = df.groupby('phone')[continuous_vars].std()
#continuous_stds = continuous_stds.rename(columns = {k+'_std':k for k in continuous_stds.columns})
#continuous_sums = df.groupby('phone')[continuous_vars].sum()
#continuous_sums = continuous_sums.rename(columns = {k+'_sum':k for k in continuous_sums.columns})
#continuous = pd.concat([continuous_means, continuous_stds], axis=1)
numerical = df.groupby('phone')[numerical_vars].describe().unstack() # add distributions of these continuous variables
phone_level_vars = df.groupby('phone')[phone_level_vars].max()
#continuous = continuous_means
flag_dummies = pd.get_dummies(df['flag'])
flag_dummies = pd.concat([df['phone'], flag_dummies], axis=1)
#discrete = flag_dummies.groupby('phone').aggregate([np.mean, np.sum])
discrete = flag_dummies.groupby('phone').mean()
phone_level = pd.concat([numerical, discrete, phone_level_vars], axis=1)
phone_level['has_images'] = df.groupby('phone')['has_images'].max()
phone_level_label = df.groupby('phone')['class'].max()
phone_level.index = range(len(phone_level))
phone_level = phone_level.reindex()
phone_level_label.index = range(len(phone_level_label))
phone_level_label = phone_level_label.reindex()
phone_level = phone_level.fillna(0)

df_X = df.ix[:, ['age',
                 'price',
                 'duration_in_mins',
                 'n_ads',
                 'n_distinct_locations',
                 'location_tree_length',
                 'n_outcall',
                 'n_incall',
                 'n_incall_and_outcall',
                 'n_cooccurring_phones',
                 'price_per_min',
                 'flag', 'ethnicity']].copy()

del df_X['ethnicity']
df_X['age_missing'] = df_X['age'].isnull()
df_X['price_missing'] = df_X['price'].isnull()
df_X['n_ads'] = df_X['n_ads'].isnull()
df_X['n_distinct_locations'] = df_X['n_distinct_locations'].fillna(0)
df_X['location_tree_length'] = df_X['location_tree_length'].fillna(0)
df_X['n_outcall'] = df_X['n_outcall'].fillna(0)
df_X['n_incall'] = df_X['n_incall'].fillna(0)
df_X['n_incall_and_outcall'] = df_X['n_incall_and_outcall'].fillna(0)
df_X['n_cooccurring_phones'] = df_X['n_cooccurring_phones'].fillna(0)
df_X.age = df_X.age.fillna(0)
df_X.price = df_X.price.fillna(0)
df_X.duration_in_mins = df_X.duration_in_mins.fillna(0)
df_X.price_per_min = df_X.price_per_min.fillna(0)
df_X = pd.get_dummies(df_X)


# ### Run model

# In[33]:

#def train_tester(df_X_train, y_train, df_X_test, y_test):
    #lr = LinearRegression()
    #lr.fit(df_X_train, y_train)
    #y_pred = lr.predict(df_X_test)
    #fpr, tpr, thresholds = roc_curve(y_test.values, y_pred)
    #return {'model':lr,
            #'y_pred': y_pred,
            #'y_test': y_test.values,
            #'lr_score': lr.score(df_X_test, y_test),
            #'roc': (fpr, tpr, thresholds),
            #'auc': auc(fpr, tpr)
            #}


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
        output_features.update({i[0]:i[1] for i in zip(X.columns, clf.feature_importances_)})
        output_features['roc_auc'] = roc_auc_score(y_test, clf.predict_proba(X.loc[test])[:,1])
        out.append(output_features)
    return pd.DataFrame(out)

#def phone_number_stratification(phone_numbers, X, y, num_folds, X_phone_column='phone'):
    #'''
    #phone_numbers is a list of the phone numbers that will be drawn from. X will have a column with these phone numbers.
    #'''
    #out = []
    #me=StratifiedKFold(phone_numbers, 3)
    #for i, (train, test) in enumerate(me):
        #ipdb.set_trace()

#me = phone_number_stratification(df['phone'], df, df['class'], 2)

eval_columns = ['f1','accuracy','true_negative_rate','true_positive_rate','roc_auc']
price_cols = ['duration_in_mins','price','price_per_min']

# work at phone level
print("_____")
print("Work at phone level...")
num_folds = 10
phone_clf = RandomForestClassifier(oob_score=True, random_state=2, n_estimators=100, n_jobs=-1, class_weight="balanced")
metrics = all_scoring_metrics(phone_clf, phone_level, phone_level_label, StratifiedKFold(phone_level_label, num_folds))
print("Results (averaged from %s fold cross validation and computed out of sample)" % num_folds)
means = metrics.mean()[[i for i in metrics.columns if i in eval_columns]]
stds = metrics.std()[[i for i in metrics.columns if i in eval_columns]]
print(pd.concat([means,stds], axis=1).rename(columns={0:'mean',1:'std'}))
#importances = metrics.mean()[[i for i in metrics.columns if i not in eval_columns]]
#print('Price importances: %s' % importances[[i for i in importances.columns if 'price' in i or 'dur' in i]].sum(axis=1).iloc[0])
#print('Age importances: %s' % importances[[i for i in importances.columns if 'age' in i]].sum(axis=1).iloc[0])
#print('Age importances: %s' % importances['age'].iloc[0])
#p.close()
#p.join()


## work at ad level
#print("_____")
#print("Work at ad level...")
#clf = RandomForestClassifier(oob_score=True, random_state=2, n_estimators=100, n_jobs=-1, class_weight="balanced")
#metrics = all_scoring_metrics(clf, df_X, df['class'], StratifiedKFold(df['class'], 2))
#print("Results (averaged from 10 fold cross validation and computed out of sample)")
#print(metrics[[i for i in metrics.columns if i in eval_columns]])
#importances = metrics[[i for i in metrics.columns if i not in eval_columns]]
#print('Price importances: %s' % importances[[i for i in importances.columns if i in price_cols]].sum(axis=1).iloc[0])
#print('Age importances: %s' % importances['age'].iloc[0])

