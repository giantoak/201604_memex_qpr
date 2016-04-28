
# coding: utf-8

# # Setup
# ## Imports

# In[1]:

from itertools import chain
import html
import ujson as json
import multiprocessing as mp
import cPickle
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
from sklearn.ensemble import ExtraTreesClassifier
import ipdb

fit_cf = cPickle.load(open('giant_oak_RF.pkl'))
phone_level_old = pd.read_csv('giant_oak_features.csv', index_col='phone')
old_cols = set(phone_level_old.columns)
def dummify_df(df, cols_to_dummy, sep, vals_to_drop='nan'):
    """
    get_dummy() on a df has some issues with dataframe-level operations
    when the column has co-occuring values.
    :param pandas.DataFrame df:
    :param list|str cols_to_dummy:
    :param str sep:
    :param list|str vals_to_drop:
    :returns: `pandas.DataFrame` --
    """
    if isinstance(cols_to_dummy, str):
        cols_to_dummy = [cols_to_dummy]

    if isinstance(vals_to_drop, str):
        vals_to_drop = [vals_to_drop]

    for col_to_dummy in cols_to_dummy:
        dummy_df = df[col_to_dummy].str.get_dummies(sep=sep)
        for col in vals_to_drop:
            if col in dummy_df.columns:
                del dummy_df[col]

        dummy_df.columns = ['{}_{}'.format(col_to_dummy, x) for x in dummy_df.columns]
        df = df.join(dummy_df)
        del df[col_to_dummy]

    return df

def disaggregated_df(df, aggregate_col, sep):
    """
    DOES NOT save the original index
    You could definitely do this faster outside of python, but sometimes that isn't possible
    Takes a column of strings with spearators, and splits them s.t. each row gets a new entity per row.
    :param pandas.DataFrame df:
    :param str aggregate_col:
    :param str sep:
    :returns: `pandas.DataFrame` --
    """
    from itertools import chain
    import pandas as pd

    good_slice = df.ix[df[aggregate_col].apply(lambda x: x.find(sep) == -1)]
    bad_slice = df.ix[df[aggregate_col].apply(lambda x: x.find(sep) > -1)]

    def row_breaker(x):
        broken_row = []
        for new_val in x[aggregate_col].split(sep):
            broken_row.append([x[key]
                               if key != aggregate_col else new_val
                               for key in df.columns])
        return broken_row

    rows = list(chain(*bad_slice.apply(row_breaker, axis=1).values))
    new_df = pd.concat([good_slice, pd.DataFrame.from_records(rows, columns=df.columns)]).drop_duplicates()
    new_df.reset_index(inplace=True, drop=True)
    return new_df
#df = pd.read_csv('../../data/merged/data_to_use_by_ad_v2.csv')
#df = pd.read_csv('../../data/merged/data_to_use_by_ad_v3_with_exp_imgs.csv')
phones_in_dd = set(['(214) 643-0854', '(214) 905-1457', '(225) 572-9627',
                    '(281) 384-0032', '(281) 818-8756', '(325) 267-2364',
                    '(401) 212-7204', '(401) 523-2205', '(469) 478-6538',
                    '(469) 671-6959', '(501) 952-3516', '(504) 613-5108',
                    '(520) 312-2202', '(561) 727-7483', '(585) 284-9482',
                    '(619) 419-9933', '(702) 208-8444', '(702) 706-4860',
                    '(713) 598-4991', '(806) 239-8076', '(806) 544-8003',
                    '(806) 730-5586', '(817) 727-8494', '(832) 247-9222',
                    '(832) 757-8556', '(832) 918-7312', '(832) 994-9119',
                    '(912) 318-2015', '(912) 318-2157'])

df = pd.read_csv('../../data/merged/partial_test_data_to_use_by_ad_with_exp_imgs.csv')
df['phone'] = df['phone'].fillna('')
df=df[df['phone'].isin(phones_in_dd)]
df = disaggregated_df(df, 'phone', '|')
df_old = df.copy()

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


# ## Messing around
# cross-tabs, etc. Can junk this section

# In[4]:



# In[5]:

#sns.heatmap(ct.T/ct.sum(axis=1), annot=True, cmap='bone_r')


# In[6]:

#sns.heatmap(ct.T, annot=True, fmt='d', cmap='bone_r')


# ## Generate Splits
# We can share these splits (or whatever the final splits end up being) across classifiers

# In[7]:

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
agg_category_cols = ['flag', 'ethnicity']

df_old = df.copy()
for a_c_c in agg_category_cols:
    df[a_c_c].fillna('', inplace=True)
    #df[a_c_c] = df[a_c_c].apply(lambda x: x.replace('|',';'))
    df[a_c_c] = df[a_c_c].apply(lambda x: x.replace(' ','|'))
    df = dummify_df(df, a_c_c, '|')
    print('post-{}: {}'.format(a_c_c, df.shape))
old_colums_ad_level = set(df.columns)
#flag_dummies = pd.get_dummies(df['flag'])
#flag_dummies = pd.concat([df['phone'], flag_dummies], axis=1)
#discrete = flag_dummies.groupby('phone').aggregate([np.mean, np.sum])
discrete = df.groupby('phone')[[i for i in df.columns if 'flag' in i or 'ethnicity' in i]].mean()
#discrete = flag_dummies.groupby('phone').mean()
phone_level = pd.concat([numerical, discrete, phone_level_vars], axis=1)
phone_level_index = phone_level.index
phone_level['has_images'] = df.groupby('phone')['has_images'].max()
phone_level.index = range(len(phone_level))
phone_level = phone_level.reindex()
phone_level = phone_level.fillna(0)
phone_level.to_csv('temp.csv', index=False)
phone_level_again=pd.read_csv('temp.csv')
phone_level_again = phone_level_again[[i for i in phone_level_again.columns if i in phone_level.columns]]
new_cols = set(phone_level_again.columns)
print(new_cols-old_cols)
print(old_cols-new_cols)
for i in old_cols:
    if i not in new_cols:
        phone_level_again[i]=0
new_cols = set(phone_level_again.columns)
print(new_cols-old_cols)
print(old_cols-new_cols)
phone_level_again = phone_level_again[list(old_cols)]

results = fit_cf.predict_proba(phone_level_again)[:,1]
output=[{'score':i[0],'phone':i[1]} for i in zip(results,phone_level_index)]

open('giant_oak_phone_results.json','w').write(json.dumps(output))

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
                 'price_per_min'] + [i for i in df.columns if 'ethnicity' in i or 'flag' in i]].copy()

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
new_cols_ad_level = set(df_X.columns)

old_df_X = pd.read_csv('giant_oak_features_ad_level.csv')
old_colums_ad_level = set(old_df_X.columns)
ad_level_clf = cPickle.load(open('giant_oak_RF_ad_level.pkl'))
print(new_cols_ad_level-old_colums_ad_level)
print(old_colums_ad_level-new_cols_ad_level)
for i in old_colums_ad_level:
    if i not in new_cols_ad_level:
        df_X[i]=0
cdr_id = df['cdr_id']
del df_X['cdr_id']
ad_level_results = ad_level_clf.predict_proba(df_X)[:,1]
ad_level_output=[{'score':i[0],'cdr_id':i[1]} for i in zip(ad_level_results,cdr_id)]

open('giant_oak_ad_results.json','w').write(json.dumps(ad_level_output))
