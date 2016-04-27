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
    Takes a column of strings with spearators, and splits them s.t. each row gets a new entity per row.
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

def aggregated_df(df, disaggregated_col, key_cols, sep):
    """
    Takes a colomn that has been disaggregated, and fuses the contents back together
    :param pandas.DataFrame df:
    :param str disaggregated_col:
    :param str|list key_cols:
    :param str sep:
    :returns: `pandas.DataFrame` -- 
    """
    if isinstance(key_cols, str):
        key_cols = [key_cols]
    
    col_subset = key_cols+[disaggregated_col]
    grpr = df.ix[:, col_subset].drop_duplicates().groupby(key_cols)
    df_2 = grpr[disaggregated_col].apply(lambda x: sep.join([str(y)
                                                             for y in sorted(set(x))]))
    df_2 = df_2.reset_index()
    
    df_2['temp'] = df_2[disaggregated_col]
    del df_2[disaggregated_col]
    df = df.merge(df_2, on=key_cols)
    df[disaggregated_col] = df['temp']
    del df['temp']
    del df_2
    return df.drop_duplicates()


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