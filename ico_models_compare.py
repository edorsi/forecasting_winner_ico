from ico_econ_app import ico_eco_app_model
from ico_ml_app import ico_ml_app_model
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

base_filepath = r'C:\PythonPrjFiles\forecasting_winner_ico'


def model_compare(input_csv, debug=True):
    df = pd.read_csv(input_csv, index_col=0)

    thresholds = np.arange(0, 1.05, 0.05).tolist() if not debug else [1]
    t = len(thresholds)
    iters = 1 if debug else 15

    for k, threshold in enumerate(thresholds):
        y = df['goal_pct'].apply(lambda element: 1 if element >= threshold else 0)
        for i in range(iters):
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            j = 1
            for train_index, test_index in skf.split(df, y):
                print('Starting threshold %d iteration %d fold %d' % (threshold, i + 1, j))
                df_train = df.iloc[train_index]
                df_test = df.iloc[test_index]
                eco_app_model = ico_eco_app_model(df_train, df_test)
                ml_app_model = ico_ml_app_model(df_train, df_test)
                # df_final = pd.concat([eco_app_model, ml_app_model], axis='columns', join='outer', ignore_index=False, sort=False)
                # df_final['fold_number'] = j
                j += 1


if __name__ == '__main__':
    model_compare(
        input_csv=os.sep.join([base_filepath, '39_icodrops_feature_ml.csv'])
    )
