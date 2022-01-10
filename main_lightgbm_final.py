import argparse
import os
import time
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix
import lightgbm as lgb
from lightgbm import LGBMClassifier
import nsml
from nsml import DATASET_PATH
pd.set_option('display.max_columns', None)  # Look over all columns
pd.set_option('display.max_rows', None)     # Look over all rows

def bind_model(model, optimizer=None):
    def save(path, *args, **kwargs):
        with open(os.path.join(path, 'model_lgb.pkl'), 'wb') as fp:
            joblib.dump(model, fp)
        print('Model saved')
    def load(path, *args, **kwargs):
        with open(os.path.join(path, 'model_lgb.pkl'), 'rb') as fp:
            temp_class = joblib.load(fp)
        nsml.copy(temp_class, model)
        print('Model loaded')
    # Inference
    def infer(path, **kwargs):
        return inference(path, model)
    nsml.bind(save=save, load=load, infer=infer)  # 'nsml.bind' function must be called at the end.

# Inference
def inference(path, model, **kwargs):
    data = preproc_data(pd.read_csv(path), train=False)
    pred_proba = model.predict_proba(data)[:, 1]
    pred_labels = np.where(pred_proba >= .5, 1, 0).reshape(-1)
    # output format
    # [(proba, label), (proba, label), ..., (proba, label)]
    results = [(proba, label) for proba, label in zip(pred_proba, pred_labels)]
    return results

def elipsoid(x):
    result = np.square(x['FBG']) / np.square(125) + np.square(x['HbA1c']) / np.square(6.4)
    if (result >= 1):
        value = 1
    else:
        value = 0
    return value

# preprocessing our data
def preproc_data(data, label=None, train=True, val_ratio=0.2, seed=1234):
    if train:
        dataset = dict()
        data['age'] = data['age'].apply(lambda x: np.NaN if x > 102 else (np.NaN if x < 20 else x))
        data['Ht'] = data['Ht'].apply(lambda x: np.NaN if x < 0 else x)
        data['Wt'] = data['Wt'].apply(lambda x: np.NaN if x < 0 else x)
        data['BMI'] = data['BMI'].apply(lambda x: np.NaN if x > 50 else (np.NaN if x < 10 else x))
        data['SBP'] = data['SBP'].apply(lambda x: np.NaN if x > 250 else (np.NaN if x < 0 else x))
        data['DBP'] = data['DBP'].apply(lambda x: np.NaN if x > 175 else (np.NaN if x < 4 else x))
        data['PR'] = data['PR'].apply(lambda x: np.NaN if x > 200 else (np.NaN if x < 20 else x))
        data['Cr'] = data['Cr'].apply(lambda x: np.NaN if x < 0 else x)
        data['AST'] = data['AST'].apply(lambda x: np.NaN if x > 300 else (np.NaN if x < 0 else x))
        data['ALT'] = data['ALT'].apply(lambda x: np.NaN if x > 300 else (np.NaN if x < 0 else x))
        data['GGT'] = data['GGT'].apply(lambda x: np.NaN if x < 0 else x)
        data['ALP'] = data['ALP'].apply(lambda x: np.NaN if x < 0 else x)
        data['BUN'] = data['BUN'].apply(lambda x: np.NaN if x < 0 else x)
        data['Alb'] = data['Alb'].apply(lambda x: np.NaN if x < 0 else x)
        data['TG'] = data['TG'].apply(lambda x: np.NaN if x < 0 else x)
        data['CrCl'] = data['CrCl'].apply(lambda x: np.NaN if x < 0 else x)
        data['FBG'] = data['FBG'].apply(lambda x: np.NaN if x < 0 else x)
        data['HbA1c'] = data['HbA1c'].apply(lambda x: np.NaN if x > 15 else (np.NaN if x < 0 else x))
        data['LDL'] = data['LDL'].apply(lambda x: np.NaN if x < 0 else x)
        data['HDL'] = data['HDL'].apply(lambda x: np.NaN if x < 0 else x)
        # basic vairables
        data['BMI_DIA_YES'] = data['BMI'].apply(lambda x: 0 if x > 28 else (1 if x >= 24 else 0))
        data['BMI_DIA_NO'] = data['BMI'].apply(lambda x: 0 if x > 26 else (1 if x >= 22 else 0))
        data['LDL_DIA_YES'] = data['LDL'].apply(lambda x: 2 if x > 180 else (1 if x > 154 else (0 if x > 105 else( 1 if x > 70 else 2))))
        data['LDL_DIA_NO'] = data['LDL'].apply(lambda x: 1 if x > 150 else (1 if x < 100 else 0))
        data['HDL_DIA_YES'] = data['HDL'].apply(lambda x: 0 if x > 55 else (1 if x >= 38 else 0))
        data['HDL_DIA_NO'] = data['HDL'].apply(lambda x: 0 if x > 62 else (1 if x >= 42 else 0))
        data['HbA1c_DIA_YES'] = data['HbA1c'].apply(lambda x: 3 if x > 6.0 else (2 if x > 5.7 else (1 if x > 5.5 else 0)))
        data['HbA1c_DIA_NO'] = data['HbA1c'].apply(lambda x: 1 if x > 5.8 else (2 if x >= 5.3 else (1 if x < 5.3 else 0)))
        data['FBG_DIA_YES'] = data['FBG'].apply(lambda x: 3 if x > 110 else (2 if x > 95 else (1 if x > 85 else 0)))
        data['FBG_DIA_NO'] = data['FBG'].apply(lambda x: 2 if x > 105 else (1 if x > 95 else (1 if x < 85 else 0)))
        data['TG_DIA_YES'] = data['TG'].apply(lambda x: 0 if x > 400 else (1 if x > 200 else (2 if x > 100 else 3)))
        data['TG_DIA_NO'] = data['TG'].apply(lambda x: 2 if x > 200 else (1 if x > 100 else 0))
        data['CrCl_DIA_YES'] = data['CrCl'].apply(lambda x: 0 if x > 115 else (1 if x > 75 else 0))
        data['CrCl_DIA_NO'] = data['CrCl'].apply(lambda x: 0 if x > 110 else (1 if x > 60 else 0))
        #data['DBP_DIA_YES'] = data['DBP'].apply(lambda x: 2 if x > 91 else (1 if x > 85 else (0 if x > 67 else (1 if x >62 else 2))))
        #data['DBP_DIA_NO'] = data['DBP'].apply(lambda x: 0 if x > 90 else (1 if x > 81 else (2 if x > 65 else (1 if x > 57 else 0))))
        #data['SBP_DIA_YES'] = data['SBP'].apply(lambda x: 2 if x > 150 else (1 if x > 131 else (0 if x > 119 else (2 if x < 104 else 1))))
        #data['SBP_DIA_NO'] = data['SBP'].apply(lambda x: 0 if x > 138 else (1 if x > 102 else 0))
        #data['AST_DIA_YES'] = data['AST'].apply(lambda x: 0 if x > 40 else 1)
        #data['AST_DIA_NO'] = data['AST'].apply(lambda x: 0 if x > 27 else 1) 
        #data['GGT_DIA_YES'] = data['GGT'].apply(lambda x: 0 if x > 100 else (1 if x > 50 else 2))
        #data['GGT_DIA_NO'] = data['GGT'].apply(lambda x: 0 if x > 60 else 1)
        #data['ALP_DIA_YES'] = data['ALP'].apply(lambda x: 0 if x > 110 else (1 if x > 55 else 2))
        #data['ALP_DIA_NO'] = data['ALP'].apply(lambda x: 0 if x > 60 else 1)
        #Add level variable
        #data['LDL_LEVEL']= data['LDL'].apply(lambda x: 1 if x > 125 else 0)
        #data['HDL_LEVEL']= data['HDL'].apply(lambda x: 1 if x > 60 else 0)
        #data['TG']= data['TG'].apply(lambda x: 1 if x > 200 else 0)
        #data['TC']= data['TC'].apply(lambda x: 1 if x > 200 else 0)
        #data['FBG_level'] = data['FBG'].apply(lambda x: 2 if x >= 110 else (1 if x >= 100 else 0))
        #data['HbA1c_level'] = data['HbA1c'].apply(lambda x: 2 if x >= 6.1 else (1 if x >= 5.7 else 0)
        #Add risk factors
        data['RISK_1'] = data['HbA1c'] * data['FBG'] + data['BMI']
        data['RISK_2']=(data['TC']+data['TG']+data['LDL'])-100*data['HDL'] 
        data['RISK_3']=data['DBP'].apply(lambda x: 1 if x > 90 else 0)& data['SBP'].apply(lambda x: 1 if x > 140 else 0)
        #data['dangnyo'] = data.apply(elipsoid, axis=1)
        ##40-60에 해당하는 index
        data = data.dropna(subset=['FBG', 'HbA1c'], how='any')
        # NaN 값 평균으로 채우기
        NUMERIC_COLS = ['HbA1c', 'FBG', 'TG', 'LDL', 'HDL', 'Alb', 'BUN', 'Cr', 'CrCl', 'AST', 'ALT', 'GGT', 'ALP',
                        'TC', 'PR', 'DBP', 'SBP', 'BMI', 'Wt', 'Ht', 'age']
        # 성별 ['M', 'F'] -> [0, 1]로 변환
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)
        # 컬럼명에 특수 JSON 문자 포함시 발생하는 오류 방지
        data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        # 날짜 datetime으로 변환
        data.loc[:, 'date'] = pd.to_datetime(data['date'], format='%Y.%m.%d')
        data.loc[:, 'date_E'] = pd.to_datetime(data['date_E'], format='%Y.%m.%d')
        data['delta_date'] = data['date_E'] - data['date']
        data['delta_date'] = data['delta_date'].astype(str)
        data['delta_date'] = data['delta_date'].apply(lambda x: int(re.sub('[a-z]+', '', x)))
        #data['delta_date'] = 100*data['delta_date']/data['delta_date'].max()
        data['delta_date'] = data['delta_date'].apply(lambda x: 4 if x > 4000 else (4 if x > 3000 else (2 if x > 2000 else (2 if x > 1000 else 0))))
        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E','BMI','HbA1c','TG','CrCl','Ht','Wt','LDL','HDL','TC','FBG']
        X = data.drop(columns=DROP_COLS).copy()
        y = label
        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          stratify=y,
                                                          test_size=val_ratio,
                                                          random_state=seed,
                                                        )
        X_cols = X.columns
        X_train = pd.DataFrame(X_train, columns=X_cols)
        X_train['gender_enc'] = X_train['gender_enc'].astype('category')
        X_val = pd.DataFrame(X_val, columns=X_cols)
        X_val['gender_enc'] = X_val['gender_enc'].astype('category')
        dataset['X_train'] = X_train
        dataset['y_train'] = y_train
        dataset['X_val'] = X_val
        dataset['y_val'] = y_val
        return dataset
    else:
        data['age'] = data['age'].apply(lambda x: np.NaN if x > 102 else (np.NaN if x < 20 else x))
        data['Ht'] = data['Ht'].apply(lambda x: np.NaN if x < 0 else x)
        data['Wt'] = data['Wt'].apply(lambda x: np.NaN if x < 0 else x)
        data['BMI'] = data['BMI'].apply(lambda x: np.NaN if x > 50 else (np.NaN if x < 10 else x))
        data['SBP'] = data['SBP'].apply(lambda x: np.NaN if x > 250 else (np.NaN if x < 0 else x))
        data['DBP'] = data['DBP'].apply(lambda x: np.NaN if x > 175 else (np.NaN if x < 4 else x))
        data['PR'] = data['PR'].apply(lambda x: np.NaN if x > 200 else (np.NaN if x < 20 else x))
        data['Cr'] = data['Cr'].apply(lambda x: np.NaN if x < 0 else x)
        data['AST'] = data['AST'].apply(lambda x: np.NaN if x > 300 else (np.NaN if x < 0 else x))
        data['ALT'] = data['ALT'].apply(lambda x: np.NaN if x > 300 else (np.NaN if x < 0 else x))
        data['GGT'] = data['GGT'].apply(lambda x: np.NaN if x < 0 else x)
        data['ALP'] = data['ALP'].apply(lambda x: np.NaN if x < 0 else x)
        data['BUN'] = data['BUN'].apply(lambda x: np.NaN if x < 0 else x)
        data['Alb'] = data['Alb'].apply(lambda x: np.NaN if x < 0 else x)
        data['TG'] = data['TG'].apply(lambda x: np.NaN if x < 0 else x)
        data['CrCl'] = data['CrCl'].apply(lambda x: np.NaN if x < 0 else x)
        data['FBG'] = data['FBG'].apply(lambda x: np.NaN if x < 0 else x)
        data['HbA1c'] = data['HbA1c'].apply(lambda x: np.NaN if x > 15 else (np.NaN if x < 0 else x))
        data['LDL'] = data['LDL'].apply(lambda x: np.NaN if x < 0 else x)
        data['HDL'] = data['HDL'].apply(lambda x: np.NaN if x < 0 else x)
        data['BMI_DIA_YES'] = data['BMI'].apply(lambda x: 0 if x > 28 else (1 if x >= 24 else 0))
        data['BMI_DIA_NO'] = data['BMI'].apply(lambda x: 0 if x > 26 else (1 if x >= 22 else 0))
        data['LDL_DIA_YES'] = data['LDL'].apply(lambda x: 2 if x > 180 else (1 if x > 154 else (0 if x > 105 else (1 if x > 70 else 2))))
        data['LDL_DIA_NO'] = data['LDL'].apply(lambda x: 1 if x > 150 else (1 if x < 100 else 0))
        data['HDL_DIA_YES'] = data['HDL'].apply(lambda x: 0 if x > 55 else (1 if x >= 38 else 0))
        data['HDL_DIA_NO'] = data['HDL'].apply(lambda x: 0 if x > 62 else (1 if x >= 42 else 0))
        data['HbA1c_DIA_YES'] = data['HbA1c'].apply(lambda x: 3 if x > 6.0 else (2 if x > 5.7 else (1 if x > 5.5 else 0)))
        data['HbA1c_DIA_NO'] = data['HbA1c'].apply(lambda x: 1 if x > 5.8 else (2 if x >= 5.3 else (1 if x < 5.3 else 0)))
        data['FBG_DIA_YES'] = data['FBG'].apply(lambda x: 3 if x > 110 else (2 if x > 95 else (1 if x > 85 else 0)))
        data['FBG_DIA_NO'] = data['FBG'].apply(lambda x: 2 if x > 105 else (1 if x > 95 else (1 if x < 85 else 0)))
        data['TG_DIA_YES'] = data['TG'].apply(lambda x: 0 if x > 400 else (1 if x > 200 else (2 if x > 100 else 3)))
        data['TG_DIA_NO'] = data['TG'].apply(lambda x: 2 if x > 200 else (1 if x > 100 else 0))
        data['CrCl_DIA_YES'] = data['CrCl'].apply(lambda x: 0 if x > 115 else (1 if x > 75 else 0))
        data['CrCl_DIA_NO'] = data['CrCl'].apply(lambda x: 0 if x > 110 else (1 if x > 60 else 0))
        #data['DBP_DIA_YES'] = data['DBP'].apply(lambda x: 2 if x > 91 else (1 if x > 85 else (0 if x > 67 else (1 if x > 62 else 2))))
        #data['DBP_DIA_NO'] = data['DBP'].apply(lambda x: 0 if x > 90 else (1 if x > 81 else (2 if x > 65 else (1 if x > 57 else 0))))
        #data['SBP_DIA_YES'] = data['SBP'].apply(lambda x: 2 if x > 150 else (1 if x > 131 else (0 if x > 119 else (2 if x < 104 else 1))))
        #data['SBP_DIA_NO'] = data['SBP'].apply(lambda x: 0 if x > 138 else (1 if x > 102 else 0))
        #data['AST_DIA_YES'] = data['AST'].apply(lambda x: 0 if x > 40 else 1)
        #data['AST_DIA_NO'] = data['AST'].apply(lambda x: 0 if x > 27 else 1)
        #data['GGT_DIA_YES'] = data['GGT'].apply(lambda x: 0 if x > 100 else (1 if x > 50 else 2))
        #data['GGT_DIA_NO'] = data['GGT'].apply(lambda x: 0 if x > 60 else 1)
        #data['ALP_DIA_YES'] = data['ALP'].apply(lambda x: 0 if x > 110 else (1 if x > 55 else 2))
        #data['ALP_DIA_NO'] = data['ALP'].apply(lambda x: 0 if x > 60 else 1)
        # data['FBG_level'] = data['FBG'].apply(lambda x: 2 if x >= 110 else (1 if x >= 100 else 0))
        # data['HbA1c_level'] = data['HbA1c'].apply(lambda x: 2 if x >= 6.1 else (1 if x >= 5.7 else 0))
        data['RISK_1'] = data['HbA1c'] * data['FBG'] + data['BMI']
        data['RISK_2']=(data['TC']+data['TG']+data['LDL'])-100*data['HDL'] 
        data['RISK_3']=data['DBP'].apply(lambda x: 1 if x > 90 else 0)& data['SBP'].apply(lambda x: 1 if x > 140 else 0)
        #data['dangnyo'] = data.apply(elipsoid, axis=1)
        ##40-60에 해당하는 index
        # NaN 값 평균으로 채우기
        NUMERIC_COLS = ['HbA1c', 'FBG', 'TG', 'LDL', 'HDL', 'Alb', 'BUN', 'Cr', 'CrCl', 'AST', 'ALT', 'GGT', 'ALP',
                        'TC', 'PR', 'DBP', 'SBP', 'BMI', 'Wt', 'Ht', 'age']
        # 성별 ['M', 'F'] -> [0, 1]로 변환
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)
        # 컬럼명에 특수 JSON 문자 포함시 발생하는 오류 방지
        data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        # 날짜 datetime으로 변환
        data.loc[:, 'date'] = pd.to_datetime(data['date'], format='%Y.%m.%d')
        data.loc[:, 'date_E'] = pd.to_datetime(data['date_E'], format='%Y.%m.%d')
        data['delta_date'] = data['date_E'] - data['date']
        data['delta_date'] = data['delta_date'].astype(str)
        data['delta_date'] = data['delta_date'].apply(lambda x: int(re.sub('[a-z]+', '', x)))
        #data['delta_date'] = 100 * data['delta_date'] / data['delta_date'].max()
        data['delta_date'] = data['delta_date'].apply(lambda x: 4 if x > 4000 else (4 if x > 3000 else (2 if x > 2000 else (2 if x > 1000 else 0))))
        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E','BMI','HbA1c','TG','CrCl','Ht','Wt','LDL','HDL','TC','FBG']
        data = data.drop(columns=DROP_COLS).copy()
        scaler = MinMaxScaler()
        X_cols = data.columns
        X_test = pd.DataFrame(data, columns=X_cols)
        X_test['gender_enc'] = X_test['gender_enc'].astype('category')
        return X_test

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    args.add_argument('--seed', type=int, default=42)
    config = args.parse_args()
    time_init = time.time()
    np.random.seed(config.seed)
    params = {
        'boosting_type': 'rf',
        'objective': 'binary',
        'metric': 'auc',
        'bagging_freq': 1,
        'bagging_fraction': 0.356,
        # 'metric': 'binary_logloss',
        'is_unbalance': 'true',
        'n_estimators': 700
        # 'early_stopping': 15,
    }
    model = LGBMClassifier(**params)
    # nsml.bind() should be called before nsml.paused()
    bind_model(model)
    # DONOTCHANGE: They are reserved for nsml
    # Warning: Do not load data before the following code!
    # test mode

    if config.pause:
        nsml.paused(scope=locals())

    # training mode
    if config.mode == 'train':
        data_path = DATASET_PATH + '/train/train_data'
        label_path = DATASET_PATH + '/train/train_label'
        raw_data = pd.read_csv(data_path)
        raw_labels = np.loadtxt(label_path, dtype=np.int16)
        dataset = preproc_data(raw_data, raw_labels, train=True, val_ratio=0.023, seed=1234)
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_val = dataset['X_val']
        y_val = dataset['y_val']
        time_dl_init = time.time()
        print('Time to dataset initialization: ', time_dl_init - time_init)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=15)
        nsml.save(0)  # name of checkpoint; 'model_lgb.pkl' will be saved
        final_time = time.time()
        print("Time to dataset initialization: ", time_dl_init - time_init)
        print("Time spent on training :", final_time - time_dl_init)
        print("Total time: ", final_time - time_init)
        print("Done")