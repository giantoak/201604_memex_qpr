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
    import numpy as np

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

    # I melted some rows when making this, and it's proven a mistake. Let's unmelt
    melted_cols = ['ethnicity', 'flag']
    for m_c in melted_cols:
        if m_c in df.columns:
            df = aggregated_df(df, m_c, 'dd_id', '|')

    return df


def phone_str_to_dd_format(phone_str):
    """
    :param str phone_str:
    :returns: `str` --
    """
    if len(phone_str) != 10:
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


def aggregated_df(df, disaggregated_col, key_cols, sep):
    """
    Takes a column that has been disaggregated, and fuses the contents back together.

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


def lr_train_tester(df_X_train, y_train, df_X_test, y_test):
    """
    Take some training and test data, fit a lin. reg, produce scores.

    :param pandas.DataFrame df_X_train:
    :param pandas.Series y_train:
    :param pandas.DataFrame df_X_tes:
    :param pandas.Series y_test:
    :returns: `dict` --
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    # from sklearn.metrics import precision_recall_fscore_support

    lr = LinearRegression()
    lr.fit(df_X_train, y_train)
    y_pred = lr.predict(df_X_test)
    fpr, tpr, thresholds = roc_curve(y_test.values, y_pred)
    # precision, recall, f_score, support = precision_recall_fscore_support(y_test.values, y_pred)
    return {'model':lr,
            'y_pred': y_pred,
            'y_test': y_test.values,
            'lr_score': lr.score(df_X_test, y_test),
            'roc': {'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': thresholds,
                    'auc': auc(fpr, tpr)},
            # 'precision_recall': {'p': precision,
            #                     'r': recall,
            #                     'f1': f_score,
            #                     'support': support}
            }


def score_metrics(y_test, y_pred):
    """
    :param y_test:
    :param y_pred:
    :returns: `dict` --
    """
    true_positives = (y_test & y_pred).sum()
    true_negatives = ((~y_test) & (~y_pred)).sum()
    false_positives = ((~y_test) & y_pred).sum()
    false_negatives = (y_test & (~y_pred)).sum()
    f1 = (2 * true_positives) / float(2 * true_positives +
                                      false_negatives + false_positives)
    true_positive_rate = true_positives / \
        float(true_positives + false_negatives)
    true_negative_rate = (
        true_negatives / float(true_negatives + false_positives))
    accuracy = (true_positives + true_negatives) / float(true_positives +
                                                         true_negatives + false_positives + false_negatives)
    return(
        {
            'true_positive_rate': true_positive_rate,
            'true_negative_rate': true_negative_rate,
            'f1': f1,
            'accuracy': accuracy
        }
    )


def all_scoring_metrics(clf, X, y, stratified_kfold):
    """
    :param clf:
    :param X:
    :param y:
    :param stratified_kfold:
    :returns: `pandas.DataFrame` --
    """
    from sklearn.metrics import roc_auc_score
    import pandas as pd
    
    out = []
    for i, (train, test) in enumerate(stratified_kfold):
        clf.fit(X.loc[train], y.loc[train])
        y_pred = clf.predict(X.loc[test])
        y_test = y.loc[test]

        output_features = score_metrics(y_test, y_pred)
        output_features.update({i[0]: i[1]
                            for i in zip(X.columns, clf.feature_importances_)})
        output_features['roc_auc'] = roc_auc_score(
        y_test, clf.predict_proba(X.loc[test])[:, 1])
        out.append(output_features)
    return pd.DataFrame(out)
