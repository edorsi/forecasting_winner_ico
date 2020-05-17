from ico_econ_app import ico_eco_app_model
from ico_ml_app import ico_ml_app_model
import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

base_filepath = r'C:\PythonPrjFiles\forecasting_winner_ico'


def model_compare(input_csv, output_csv, debug=True):
    df = pd.read_csv(input_csv, index_col=0)

    thresholds = np.arange(0, 1.05, 0.05).tolist() if not debug else [1]
    iters = 1 if debug else 15

    df_csv = pd.DataFrame()
    for k, threshold in enumerate(thresholds):
        y = df['goal_pct'].apply(lambda element: 1 if element >= threshold else 0)
        for i in range(iters):
            skf = StratifiedKFold(n_splits=10, shuffle=True)
            j = 0
            for train_index, test_index in skf.split(df, y):
                print('Starting threshold %d iteration %d fold %d' % (threshold, i + 1, j + 1))
                df_train = df.iloc[train_index]
                df_test = df.iloc[test_index]
                eco_app_model = ico_eco_app_model(df_train, df_test, threshold)
                ml_app_model = ico_ml_app_model(df_train, df_test, threshold, trees=200)
                df_final = pd.concat([eco_app_model, ml_app_model], axis='columns', join='outer', ignore_index=False, sort=False)
                df_final['threshold'] = threshold
                df_final['iteration'] = i + 1
                df_final['fold_number'] = j + 1
                j += 1
                df_csv = df_csv.append(df_final)
    df_csv.to_csv(output_csv, index_label='index')


def model_evolution(input_csv, output_csv, debug=True):
    df = pd.read_csv(input_csv, index_col=0)
    threshold = 1
    test_index = [2, 23, 58, 67, 73, 76, 87, 89, 110, 140, 141, 144, 156, 187, 208, 210, 211, 214, 223, 232, 234, 245,
                  261, 263, 275, 276, 284, 286, 293, 297, 303, 305, 347, 358, 367, 368, 371, 376, 377]
    train_index = [t for t in df.index]
    train_index = list(set(train_index) - set(test_index))

    df_csv = pd.DataFrame()
    for t in [10,50,100,150,200]:
        for i in range(1,11):
            for j in range(1,len(train_index)):
                random.shuffle(train_index)
                df_train = df.iloc[train_index[0:j]]
                df_test = df.iloc[test_index]
                eco_app_model = ico_eco_app_model(df_train, df_test, threshold)
                ml_app_model = ico_ml_app_model(df_train, df_test, threshold, t)
                df_final = pd.concat([eco_app_model, ml_app_model], axis='columns', join='outer', ignore_index=False, sort=False)
                df_final['threshold'] = threshold
                df_final['training'] = j
                df_final['iteration'] = i
                df_final['trees'] = t
                df_csv = df_csv.append(df_final)
                print('Trees %d - Iteration %d - Processed %d / %d - %.2f' % (t, i, j, len(train_index), j/len(train_index)*100))

    df_csv.to_csv(output_csv, index_label='index')



if __name__ == '__main__':
    model_compare(
        input_csv=os.sep.join([base_filepath, '39_icodrops_feature_ml.csv']),
        output_csv=os.sep.join([base_filepath, '00_Results.csv'])
    )

    # model_evolution(
    #     input_csv=os.sep.join([base_filepath, '39_icodrops_feature_ml.csv']),
    #     output_csv=os.sep.join([base_filepath, '00_Evo_Results_200.csv'])
    # )
