import pandas as pd
import numpy as np
import os
import re
from ico_tools import evaluate_model
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

base_filepath = r'C:\PythonPrjFiles\forecasting_winner_ico'

def ico_ml_app_preparation(input_csv, output_csv, verbose=False):
    df = pd.read_csv(input_csv, index_col=0)
    print(df.info()) if verbose else False
    print(df.head(5)) if verbose else False

    model_prefix = 'ml_model_'

    df[model_prefix + 'goal'] = df['goal'].apply(
        lambda element: re.sub(r'[^0-9]', '', element))
    df[model_prefix + 'goal'] = df[model_prefix + 'goal'].apply(
        lambda element: np.nan if element == '' else element)
    df[model_prefix + 'goal'] = df[model_prefix + 'goal'].astype(np.float64)
    # df[model_prefix + 'goal_pct'] = df[model_prefix + 'goal_received'] / df[model_prefix + 'goal']

    df[model_prefix + 'on_exchanges_dummy'] = df['on_exchanges'].apply(
        lambda element: 1 if element != 'Not traded on exchanges' else 0)
    df[model_prefix + 'facebook_dummy'] = df['facebook'].apply(
        lambda element: 1 if element != 'None' else 0)
    df[model_prefix + 'reddit_dummy'] = df['reddit'].apply(
        lambda element: 1 if element != 'None' else 0)
    df[model_prefix + 'github_dummy'] = df['github'].apply(
        lambda element: 1 if element != 'None' else 0)
    df[model_prefix + 'github_dummy'] = df['github'].apply(
        lambda element: 1 if element != 'None' else 0)
    df[model_prefix + 'twitter_dummy'] = df['twitter'].apply(
        lambda element: 1 if element != 'None' else 0)
    df[model_prefix + 'telegram_dummy'] = df['telegram'].apply(
        lambda element: 1 if element != 'None' else 0)
    df[model_prefix + 'linkedin_dummy'] = df['linkedin'].apply(
        lambda element: 1 if element != 'None' else 0)
    df[model_prefix + 'medium_dummy'] = df['medium'].apply(
        lambda element: 1 if element != 'None' else 0)
    df[model_prefix + 'slack_dummy'] = df['slack'].apply(
        lambda element: 1 if element != 'None' else 0)
    df[model_prefix + 'btc_dummy'] = df['btc'].apply(
        lambda element: 1 if element != 'None' else 0)
    df[model_prefix + 'youtube_dummy'] = df['youtube'].apply(
        lambda element: 1 if element != 'None' else 0)
    df[model_prefix + 'presale_dummy'] = df['presale'].apply(
        lambda element: 1 if element != 'None' else 0)
    df[model_prefix + 'bonus_first_dummy'] = df['bonus_first'].apply(
        lambda element: 1 if element != 'None' else 0)

    original_fileds = ['ico_category_name',
                       'token_type',
                       'sentences',
                       'coherence_values',
                       'tech_sen_pct',
                       'location_num',
                       'organization_num',
                       'person_num',
                       'duration_days',
                       'open_start',
                       'af_cluster_labels',
                       'af_cluster_centers_indices',
                       'centre_cluster_euclidean_distance',
                       'tokens_in_sentiment_analysis',
                       'Avg_obj_score',
                       'Avg_pos_score',
                       'Avg_neg_score'
    ]

    for col in df:
        if col.startswith('w2v_avg_'):
            original_fileds.append(col)

    for field in original_fileds:
        df[model_prefix + str(field)] = df[str(field)]

    df.to_csv(output_csv, index_label='index')
    print('ML Model completed')


def ico_ml_app_model(train_df, test_df, threshold, trees):
    model_prefix = 'ml_model_'
    y_col = 'goal_pct'

    x_train = train_df.loc[:, train_df.columns.str.startswith(model_prefix)]
    x_test = test_df.loc[:, test_df.columns.str.startswith(model_prefix)]
    y_train = train_df[y_col].apply(lambda element: 1 if element >= threshold else 0)
    y_test = test_df[y_col]
    # print(y_test)
    # print(x_train.columns)
    # print(y_train.columns)

    numeric_col = x_train.select_dtypes(include='number')
    categorical_col = x_train.select_dtypes(include='object')
    numeric_features = list(numeric_col)
    categorical_features = list(categorical_col)
    # print(numeric_features)
    # print(categorical_features)

    numeric_transformer = Pipeline([
        ('fill_na', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ('fill_na', SimpleImputer(strategy='constant', fill_value='missing')),
        ("onehot", OneHotEncoder()),
    ])
    preprocessor = ColumnTransformer([
        ('numeric', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features),
    ])

    classifier = Pipeline([
        ('pre-process', preprocessor),
        ("random forest", RandomForestClassifier(n_estimators=trees, criterion='entropy', n_jobs=-1)),
    ])

    clf = classifier.fit(x_train, y_train)
    feature_importances = clf[1].feature_importances_
    names = x_train.columns
    # for item1, item2 in zip(names, feature_importances):
    #     print('Importance %s: %.4f' % (item1, item2))

    predictions = pd.DataFrame(clf.predict(x_test), columns=[model_prefix + 'y_predictions'], index=test_df.index)

    df_final = pd.concat([y_test, predictions], axis='columns', join='outer', ignore_index=False, sort=False)

    # Reference to evaluate model predictions
    df_final[model_prefix + y_col + '_trasformed'] = df_final[y_col].apply(lambda element: 1 if element >= threshold else 0)
    df_final = evaluate_model(df_final, model_prefix)


    # df_final.to_csv(os.sep.join([base_filepath, '01_' + model_prefix + 'Results.csv']))
    return df_final
