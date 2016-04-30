
# coding: utf-8

# # This Notebook
# 
# In here, everything on which we're planning to train gets merged together and dumped to a CSV and a pickle for quick reference.
# 
# ------
# 
# # Setup
# ## Imports

# In[1]:

import ujson as json
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from itertools import chain
from tqdm import tqdm, tqdm_pandas
from sqlalchemy import create_engine
get_ipython().magic(u'matplotlib inline')


# ## Helper Functions

# In[2]:

def phone_str_to_dd_format(phone_str):
    """
    :param str phone_str:
    :returns: `str` --
    """
    if len(phone_str) !=  10:
        return phone_str
    return '({}) {}-{}'.format(phone_str[:3], phone_str[3:6], phone_str[6:])


def disaggregated_df(df, aggregate_col, sep):
    """
    DOES NOT save the original index
    You could definitely do this faster outside of python, but sometimes that isn't possible
    :param pandas.DataFrame df:
    :param str aggregate_col:
    :param str sep:
    :returns: `pandas.DataFrame` -- 
    """
    from itertools import chain

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


# In[3]:

def df_of_tables_for_dd_ids(dd_ids, sqlite_tables, sql_con):
    """
    :param list dd_ids: list of Deep Dive IDs to retrieve
    :param list sqlite_tables: list of SQLite tables to join
    :param sqlalchemy.create_engine sql_con: Connection to SQLite (can be \
    omitted)
    :returns: `pandas.DataFrame` -- dataframe of tables, joined using the Deep \
    Dive IDs.
    """
    import pandas as pd

    dd_ids_str = ','.join(['"{}"'.format(x) for x in dd_ids])
    query_fmt = 'select * from {} where dd_id in ({})'.format

    df = pd.read_sql(query_fmt(sqlite_tables[0], dd_ids_str), sql_con).drop_duplicates()
    df['dd_id'] = df.dd_id.astype(int)

    for s_t in sqlite_tables[1:]:
        df_2 = pd.read_sql(query_fmt(s_t, dd_ids_str), sql_con)
        df_2['dd_id'] = df_2.dd_id.astype(int)
        
        # We use outer joins because dd_ids in one table may be missing from the other.
        df = df.merge(df_2, on=['dd_id'], how='outer')

    if 'post_date' in df:
        df['post_date'] = df.post_date.apply(pd.to_datetime)
        
    if 'duration_in_mins' in df:
        df['duration_in_mins'] = df.duration_in_mins.apply(lambda x: float(x) if x != '' else np.nan)

    return df


# # Read Training Data

# In[4]:

jsns = [json.loads(x) for x in open('../../data/orig_ht_data/ht_training.json', 'r')]
train_df = pd.DataFrame.from_records(jsns)
train_df = train_df.ix[:, ['class', 'phone', 'url']]


# In[5]:

print(train_df.shape)
train_df['phone'] = train_df.phone.apply(lambda x:','.join(x))
train_df = disaggregated_df(train_df, 'phone', ',')
print(train_df.shape)
train_df = train_df.ix[train_df.phone != 'nan', :]
print(train_df.shape)

train_df['phone'] = train_df.phone.apply(phone_str_to_dd_format)


# We drop phone numbers that have both positive and negative values. This is an oversimplification, and we could also try something like calculating a real score. This is okay for now.

# In[6]:

pos_neg_set = set(train_df.ix[(train_df['class'] == 'positive'), 'phone']) &set(train_df.ix[(train_df['class'] == 'negative'), 'phone'])
train_df = train_df.ix[~train_df.phone.isin(pos_neg_set), :]
print(train_df.shape)


# We're not using urls, so we drop them too. We can put these back if we want them.

# In[7]:

train_df = train_df.ix[:, ['class', 'phone']].drop_duplicates()
print(train_df.shape)


# # Read DeepDive Data w/ Training Phones

# In[8]:

sql_con = create_engine('sqlite:////Users/pmlandwehr/wkbnch/memex/memex_queries/dd_dump.db')
query_str_fmt = 'select {} from {} where {} in ({})'.format


# In[9]:

phone_str_list = ','.join(['"{}"'.format(x) for x in train_df.phone.unique()])
query_str = query_str_fmt('*', 'dd_id_to_phone', 'phone', phone_str_list)
df = pd.read_sql(query_str, sql_con)
print(df.shape)
df = df.drop_duplicates()
print(df.shape)


# In[10]:

df_2 = df_of_tables_for_dd_ids(list(df.dd_id.unique()),
                                ['dd_id_to_price_duration',
                                'dd_id_to_flag',
                                'dd_id_to_age',
                                'dd_id_to_cbsa'],
                               sql_con)
print(df_2.shape)


# For those ads with prices and durations, let's add a price per minute value

# In[11]:

df_2['price_per_min'] = df_2.price / df_2.duration_in_mins


# In[12]:

## DD IDs in the first list are definitive and complete, so left join
df_3 = df.merge(df_2, on='dd_id', how='left')
print(df_3.shape)


# **Clean up**

# In[13]:

del df
del df_2


# ## Join Deep Dive Data with Training Data

# In[14]:

df_4 = train_df.merge(df_3, on='phone', how='left')


# **Clean up**

# In[15]:

del df_3


# # Join Deep Dive Data with Greg's HT Data
# ## STD Data
# We rename the "Name" column to "area"

# In[16]:

std_df = pd.read_excel('../../data/greg_correlates/std.xlsx')


# In[17]:

std_df.head()


# In[18]:

df_4 = df_4.merge(std_df.ix[:, ['Name', 'Disease', 'Year', 'Cases', 'Rate', 'MSA']],
                  left_on='area',
                  right_on='Name',
                  how='left')
del df_4['Name']
print(df_4.shape)


# **Clean up**

# In[19]:

del std_df


# ## MSA Characteristics
# Note that we're using the **yearly** version of the file. We could also use the **monthly** version. I'm primarily choosing yearly because monthly had some import issues that it doesn't seem worth wrangling at this second.

# In[20]:

msa_df = pd.read_csv('../../data/greg_correlates/msa_characteristics.csv')


# In[21]:

df_4 = df_4.merge(msa_df,
                  left_on='MSA',
                  right_on='census_msa_code',
                 how='left')
del df_4['census_msa_code']


# **Clean up**

# In[22]:

del msa_df


# # Save the Results

# In[23]:

df_4.to_csv('../../data/merged/data_to_use.csv', index=False)
df_4.to_pickle('../../data/merged/data_to_use.pkl')

