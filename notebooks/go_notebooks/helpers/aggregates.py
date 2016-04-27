import pandas as pd
from scipy import stats

"""
Adds aggregates of ads to a data frame. Aggregates are grouped by phone.
This will modify the input data frame!
"""
def add_aggregates(df):

    features = [
        'n_distinct_locations',
        'location_tree_length',
        'n_cooccurring_phones'
        ]

    phone_data = pd.read_csv('../data/phone_aggregates/phones.csv')
    phones = df['phone'].unique()

    new_features = []
    for i in range(len(phones)):
        n_records = len(phone_data[phone_data['phone'] == phones[i]])
        if n_records > 1:
            raise Exception("More than one record for a phone in phone data.")
        elif n_records == 1:
            index = len(new_features)
            new_features.append({})
            new_features[index]['phone'] = phones[i]
            for feature in features:
                score = stats.percentileofscore(phone_data[feature], phone_data[phone_data['phone'] == phones[i]][feature].iloc[0])
                new_features[index][feature] = score

    df.merge(pd.DataFrame(new_features), on='phone', how='left')


"""
Convenience method for calling add_aggregates(DataFrame) that will load data frame
from disk and save the result data frame to disk.
"""
def add_aggregates(input_df_path, output_df_path):
    df = pd.read_csv(input_df_path)
    add_aggregates(df)
    df.to_csv(output_df_path, index=False)
