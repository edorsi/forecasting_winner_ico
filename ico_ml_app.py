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

    cleanup_nums = {
        'ico_category_name': {
            'Advertising': 1,
            'Artificial Intelligence': 2,
            'Banking': 3,
            'Bets': 4,
            'Blockchain': 5,
            'Blockchain Platform': 6,
            'Blockchain Service': 7,
            'Business': 8,
            'Card': 9,
            'Cloud Computing': 10,
            'Cloud Storage': 11,
            'Collaboration': 12,
            'Crowdfunding': 13,
            'CryptoFund': 14,
            'Currency': 15,
            'Dapp': 16,
            'Data Service': 17,
            'E-commerce': 18,
            'Education': 19,
            'Energy': 20,
            'Exchange': 21,
            'Finance': 22,
            'Gambling': 23,
            'Gaming': 24,
            'Healthcare': 25,
            'Hi-Tech': 26,
            'Hybrid Intellingence': 27,
            'Insurance': 28,
            'IOT': 29,
            'Market': 30,
            'Marketing': 31,
            'Marketplace': 32,
            'Masternode': 33,
            'Media': 34,
            'Mining': 35,
            'Mobile': 36,
            'Network': 37,
            'Payments': 38,
            'Protocol': 39,
            'Real Assets': 40,
            'Real Business': 41,
            'Real Estate': 42,
            'Security': 43,
            'Smart Contract': 44,
            'Social': 45,
            'Social Network': 46,
            'Ticketing': 47,
            'Token Discounter': 48,
            'Trading': 49,
            'Verification': 50,
            'VR': 51
        },
        'token_type': {
            'BEP2': 1,
            'EIP20': 2,
            'EOS': 3,
            'ERC20': 4,
            'ERC223': 5,
            'ERC23': 6,
            'ICON': 7,
            'IRC-20': 8,
            'NEM Blockchain': 9,
            'NEO Blockchain': 10,
            'NEP-5': 11,
            'None': 12,
            'NRC20': 13,
            'OEP-4': 14,
            'Own wallet': 15,
            'QRC': 16,
            'ST-20': 17,
            'Stellar': 18,
            'TRC-10': 19,
            'Waves blockchain': 20,
            'WRC-20': 21
        }
    }
    df.replace(cleanup_nums, inplace=True)

    original_fileds = [
        'ico_category_name',
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

    # df = pd.concat([df, pd.get_dummies(df['ico_category_name'], prefix=model_prefix + 'ico_category_name')], axis=1)
    # df = pd.concat([df, pd.get_dummies(df['token_type'], prefix=model_prefix + 'token_type')], axis=1)

    df.to_csv(output_csv, index_label='index')
    print('ML Model completed')


def ico_ml_app_model(train_df, test_df, threshold, trees, iter):
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
        ('onehot', OneHotEncoder()),
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
    with open(base_filepath + r'\feature_importance.csv', 'a+') as f:
        for item1, item2 in zip(names, feature_importances):
            print('Iteration %d - feature - %s - importance %.4f' % (iter, item1, item2), file=f)

    predictions = pd.DataFrame(clf.predict(x_test), columns=[model_prefix + 'y_predictions'], index=test_df.index)

    df_final = pd.concat([y_test, predictions], axis='columns', join='outer', ignore_index=False, sort=False)
    # df_ml_features = pd.DataFrame(data=feature_importances, columns=names)

    # Reference to evaluate model predictions
    df_final[model_prefix + y_col + '_trasformed'] = df_final[y_col].apply(lambda element: 1 if element >= threshold else 0)
    df_final = evaluate_model(df_final, model_prefix)


    # df_final.to_csv(os.sep.join([base_filepath, '01_' + model_prefix + 'Results.csv']))
    return df_final
