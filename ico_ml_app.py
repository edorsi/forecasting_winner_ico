import pandas as pd
import numpy as np
import os
import re
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
    # df[model_prefix + 'goal_received'] = df['goal_received'].apply(
    #     lambda element: re.sub(r'[^0-9]', '', element))
    # df[model_prefix + 'goal_received'] = df[model_prefix + 'goal_received'].apply(
    #     lambda element: np.nan if element == '' else element)
    # df[model_prefix + 'goal_received'] = df[model_prefix + 'goal_received'].astype(np.float64)

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


def testing():
    class NumberSelector(BaseEstimator, TransformerMixin):
        '''
        Transformer to select a single column from the data frame to perform additional transformations on
        Use on numeric columns in the data
        '''

        def __init__(self, column):
            self.column = column

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X_new = X[[self.column]]
            # print(X_new)
            return X_new

    feats = FeatureUnion([
        ('ico_category_name', Pipeline([
            ('selector', NumberSelector(column='ico_category_name')),
            ('scaler', MinMaxScaler()),
        ])),
        ('goal', Pipeline([
            ('selector', NumberSelector(column='goal')),
            ('scaler', MinMaxScaler()),
        ])),
        ('on_exchanges_dummy', Pipeline([
            ('selector', NumberSelector(column='on_exchanges_dummy')),
            ('scaler', MinMaxScaler()),
        ])),
        ('facebook_dummy', Pipeline([
            ('selector', NumberSelector(column='facebook_dummy')),
            ('scaler', MinMaxScaler()),
        ])),
        ('reddit_dummy', Pipeline([
            ('selector', NumberSelector(column='reddit_dummy')),
            ('scaler', MinMaxScaler()),
        ])),
        ('github_dummy', Pipeline([
            ('selector', NumberSelector(column='github_dummy')),
            ('scaler', MinMaxScaler()),
        ])),
        ('twitter_dummy', Pipeline([
            ('selector', NumberSelector(column='twitter_dummy')),
            ('scaler', MinMaxScaler()),
        ])),
        ('telegram_dummy', Pipeline([
            ('selector', NumberSelector(column='telegram_dummy')),
            ('scaler', MinMaxScaler()),
        ])),
        ('linkedin_dummy', Pipeline([
            ('selector', NumberSelector(column='linkedin_dummy')),
            ('scaler', MinMaxScaler()),
        ])),
        ('medium_dummy', Pipeline([
            ('selector', NumberSelector(column='medium_dummy')),
            ('scaler', MinMaxScaler()),
        ])),
        ('slack_dummy', Pipeline([
            ('selector', NumberSelector(column='slack_dummy')),
            ('scaler', MinMaxScaler()),
        ])),
        ('btc_dummy', Pipeline([
            ('selector', NumberSelector(column='btc_dummy')),
            ('scaler', MinMaxScaler()),
        ])),
        ('youtube_dummy', Pipeline([
            ('selector', NumberSelector(column='youtube_dummy')),
            ('scaler', MinMaxScaler()),
        ])),
        ('token_type', Pipeline([
            ('selector', NumberSelector(column='token_type')),
            ('scaler', MinMaxScaler()),
        ])),
        ('sentences', Pipeline([
            ('selector', NumberSelector(column='sentences')),
            ('scaler', MinMaxScaler()),
        ])),
        ('presale_dummy', Pipeline([
            ('selector', NumberSelector(column='presale_dummy')),
            ('scaler', MinMaxScaler()),
        ])),
        ('bonus_first_dummy', Pipeline([
            ('selector', NumberSelector(column='bonus_first_dummy')),
            ('scaler', MinMaxScaler()),
        ])),
        ('coherence_values', Pipeline([
            ('selector', NumberSelector(column='coherence_values')),
            ('scaler', MinMaxScaler()),
        ])),
        ('tech_sen_pct', Pipeline([
            ('selector', NumberSelector(column='tech_sen_pct')),
            ('scaler', MinMaxScaler()),
        ])),
        ('location_num', Pipeline([
            ('selector', NumberSelector(column='location_num')),
            ('scaler', MinMaxScaler()),
        ])),
        ('organization_num', Pipeline([
            ('selector', NumberSelector(column='organization_num')),
            ('scaler', MinMaxScaler()),
        ])),
        ('person_num', Pipeline([
            ('selector', NumberSelector(column='person_num')),
            ('scaler', MinMaxScaler()),
        ])),
        ('duration_days', Pipeline([
            ('selector', NumberSelector(column='duration_days')),
            ('scaler', MinMaxScaler()),
        ])),
        ('open_start', Pipeline([
            ('selector', NumberSelector(column='open_start')),
            ('scaler', MinMaxScaler()),
        ])),
        ('w2v_avg_0', Pipeline([
            ('selector', NumberSelector(column='w2v_avg_0')),
            ('scaler', MinMaxScaler()),
        ])),
        ('w2v_avg_1', Pipeline([
            ('selector', NumberSelector(column='w2v_avg_1')),
            ('scaler', MinMaxScaler()),
        ])),
        ('w2v_avg_2', Pipeline([
            ('selector', NumberSelector(column='w2v_avg_2')),
            ('scaler', MinMaxScaler()),
        ])),
        ('w2v_avg_3', Pipeline([
            ('selector', NumberSelector(column='w2v_avg_3')),
            ('scaler', MinMaxScaler()),
        ])),
        ('w2v_avg_4', Pipeline([
            ('selector', NumberSelector(column='w2v_avg_4')),
            ('scaler', MinMaxScaler()),
        ])),
        # ('af_cluster_labels', Pipeline([
        #     ('selector', NumberSelector(column='af_cluster_labels')),
        #     ('scaler', MinMaxScaler()),
        # ])),
        # ('af_cluster_centers_indices', Pipeline([
        #     ('selector', NumberSelector(column='af_cluster_centers_indices')),
        #     ('scaler', MinMaxScaler()),
        # ])),
        # ('tsne_0', Pipeline([
        #     ('selector', NumberSelector(column='tsne_0')),
        #     ('scaler', MinMaxScaler()),
        # ])),
        # ('tsne_1', Pipeline([
        #     ('selector', NumberSelector(column='tsne_1')),
        #     ('scaler', MinMaxScaler()),
        # ])),
        ('centre_cluster_euclidean_distance', Pipeline([
            ('selector', NumberSelector(column='centre_cluster_euclidean_distance')),
            ('scaler', MinMaxScaler()),
        ])),
        # ('tokens_in_sentiment_analysis', Pipeline([
        #     ('selector', NumberSelector(column='tokens_in_sentiment_analysis')),
        #     ('scaler', MinMaxScaler()),
        # ])),
        ('Avg_obj_score', Pipeline([
            ('selector', NumberSelector(column='Avg_obj_score')),
            ('scaler', MinMaxScaler()),
        ])),
        ('Avg_pos_score', Pipeline([
            ('selector', NumberSelector(column='Avg_pos_score')),
            ('scaler', MinMaxScaler()),
        ])),
        ('Avg_neg_score', Pipeline([
            ('selector', NumberSelector(column='Avg_neg_score')),
            ('scaler', MinMaxScaler()),
        ])),
    ])

    all_models = [
        ('Multinomial classifier Naive Bayes', Pipeline([
            ('features', feats),
            ('clf', MultinomialNB())
        ])),
        ("Bernoulli classifier Naive Bayes", Pipeline([
            ('features', feats),
            ('clf', BernoulliNB())
        ])),
        ("Support Vector Classification", Pipeline([
            ('features', feats),
            ('clf', SVC(gamma='scale'))
        ])),
        ("Extra Trees Classifier", Pipeline([
            ('features', feats),
            ("extra trees", ExtraTreesClassifier(n_estimators=200))
        ])),
        ("Random Forest Classifier", Pipeline([
            ('features', feats),
            ("random forest", RandomForestClassifier(n_estimators=200))
        ])),
    ]

    X = df[[
        'ico_category_name',
        'goal',
        'on_exchanges_dummy',
        'facebook_dummy',
        'reddit_dummy',
        'github_dummy',
        'twitter_dummy',
        'telegram_dummy',
        'linkedin_dummy',
        'medium_dummy',
        'slack_dummy',
        'btc_dummy',
        'youtube_dummy',
        'token_type',
        'sentences',
        'presale_dummy',
        'bonus_first_dummy',
        'coherence_values',
        'tech_sen_pct',
        'location_num',
        'organization_num',
        'person_num',
        'duration_days',
        'open_start',
        'w2v_avg_0',
        'w2v_avg_1',
        'w2v_avg_2',
        'w2v_avg_3',
        'w2v_avg_4',
        # 'af_cluster_labels',
        # 'af_cluster_centers_indices',
        # 'tsne_0',
        # 'tsne_1',
        'centre_cluster_euclidean_distance',
        # 'tokens_in_sentiment_analysis',
        'Avg_obj_score',
        'Avg_pos_score',
        'Avg_neg_score'
    ]]

    print(X) if verbose else False

    print("ML dataset final: %d" % len(df))
    print('-----------------')
    y = df['goal_pct'].apply(lambda element: 1 if element >= 1 else 0)

    unsorted_scores = []
    for name, model in all_models:
        unsorted_scores.append((name,
                                cross_val_score(model, X, y, cv=5, scoring='f1').mean(),
                                cross_val_score(model, X, y, cv=5, scoring='accuracy').mean(),
                                cross_val_score(model, X, y, cv=5, scoring='precision').mean(),
                                cross_val_score(model, X, y, cv=5, scoring='recall').mean()))
        print(datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - evaluated model ' + str(name))
    print('-----------------')

    scores = sorted(unsorted_scores, key=lambda x: -x[2])
    print(datetime.datetime.utcnow().strftime('%H:%M:%S') + ' - Scores:')
    print(tabulate(scores, floatfmt='.4f',
                   headers=('model', 'f1 micro', 'accuracy', 'precision micro', 'recall micro')))
    print('-----------------')


def ico_ml_app_model(train_df, test_df, threshold):
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
        ("random forest", RandomForestClassifier(n_estimators=200, criterion='entropy', n_jobs=-1)),
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


    # df_final.to_csv(os.sep.join([base_filepath, '01_' + model_prefix + 'Results.csv']))
    return df_final
