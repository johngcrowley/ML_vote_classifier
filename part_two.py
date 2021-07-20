import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

#set our views for pandas
pd.set_option('display.max_columns',None)

#set for plots --- not needed for script
#sns.set(style='darkgrid',palette='deep',font_scale=1.1,rc={'figure.figsize':[20,10]})

#training files
survey = pd.read_csv('csvs/survey_responses_training.csv')
voter_info = pd.read_csv('csvs/voter_information_training.csv')
voter_hist = pd.read_csv('csvs/voting_history_training.csv')

#testing files
survey_test = pd.read_csv('csvs/survey_responses_testing.csv')
voter_info_test = pd.read_csv('csvs/voter_information_testing.csv')
voter_hist_test = pd.read_csv('csvs/voting_history_testing.csv')

# -------------------------------- Functions --------------------------------------------
def merge_survey_info(survey,voter_info):
    df = pd.merge(survey,voter_info,how='outer',on='myv_van_id')
    return df

def clean_voter_history_training(voter_hist):
    voter_hist['voted'] = np.where(voter_hist['election_date'] == '2018-11-06',1,0)
    voted = voter_hist.loc[voter_hist['voted'] == 1]
    voter_hist = voter_hist.drop_duplicates(subset='myv_van_id')
    notvoted = voter_hist.loc[voter_hist['voted'] == 0]
    notvoted = notvoted[notvoted['myv_van_id'].isin(voted['myv_van_id']) == False]
    df3 = pd.concat([voted,notvoted])
    df3 = df3.drop(columns=['Unnamed: 0','party_id']).set_index('myv_van_id')
    return df3

def clean_voter_history_testing(voter_hist_test):
    voter_hist_test = voter_hist_test.drop_duplicates(subset='myv_van_id')
    df3 = voter_hist_test.drop(columns=['party_id']).set_index('myv_van_id')
    return df3

def feature_engineer_voterhistory(cleaned_voter_hist,voter_hist):
    
    elecs = len(voter_hist['election_date'].unique())
    gb = voter_hist.groupby('myv_van_id')['election_date']
    voter_gb_van_id = (gb.count()).apply(lambda x: round(100* x/elecs,2))
    
    history_gb_elections = pd.DataFrame(voter_gb_van_id).reset_index().rename(columns={'election_date':'% election'})
    df2 = pd.merge(history_gb_elections,cleaned_voter_hist,how='inner',on='myv_van_id')
    df2 = df2.drop(columns=['election_date'])
    df3 = pd.pivot_table(data=df2,index='myv_van_id')
    return df3

def feature_engineering(df):
    df['eday'] = '2018-11-06'
    df['registration_date'] = pd.to_datetime(df['registration_date']).dt.date
    df['eday'] = pd.to_datetime(df['eday']).dt.date


    # get days since, normalize by years. note: same day registration = 0
    df['vote_since_register'] = (df['eday'] - df['registration_date'])
    df['vote_since_register'] = df['vote_since_register'].apply(lambda x: round(int(x.days) / 365,1))
    return df

def merge(df,df3):
    col_renames = {'age_combined':'age',
             'gender_combined':'gender',
             'ethnicity_combined':'ethnicity',
             'state_house_district_latest':'state_house',
             'state_senate_district_latest':'state_senate',
             'vote_since_register':'yrs_reg',
             'us_cong_district_latest':'CD'}
    df = df.rename(columns=col_renames)
    
    col_to_drop = ['datetime_canvassed','master_survey_question_name',
                   'contact_type_name','registration_date','van_precinct_name','eday',
                   'master_survey_response_name','zip','state_house','state_senate']
    df = df.drop(col_to_drop,axis=1)
    
    training = pd.merge(df3,df,on='myv_van_id',how='inner')
    training = training.drop_duplicates(subset='myv_van_id')
    training = training[training['gender'].isin(['M','F'])]
    return training

def encode_X_y(data):
    data_encoded = pd.get_dummies(data,columns=['CD','gender','ethnicity']).set_index('myv_van_id')
    data_encoded = data_encoded.drop(columns=['CD_26'])
    y = data_encoded.voted.dropna(how='all')
    X = data_encoded.drop(columns='voted').dropna(how='all')
    return X,y

def encode_test(data):
    XT = pd.get_dummies(data,columns=['CD','gender','ethnicity']).set_index('myv_van_id')
    return XT

def train_test_clf(X,y,XT):
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=47)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_scaled = min_max_scaler.fit_transform(X_train)
    X_test_scaled = min_max_scaler.transform(X_test)
    
    clf = GradientBoostingClassifier(learning_rate=.01,max_depth=2, n_estimators=1000, random_state=47)
    clf.fit(X_train_scaled,y_train)
    y_pred_prob = clf.predict_proba(X_test_scaled)[:,1]
    auc = roc_auc_score(y_test,y_pred_prob)
    print(f'training auc score is {auc}')
    
    XT_scaled = min_max_scaler.transform(XT)
    y_pred = clf.predict(XT_scaled)
    output = pd.DataFrame(XT.index,y_pred).reset_index()
    output = output.rename(columns={'index':'Voted'})
    return output

def run_it():
    df = merge_survey_info(survey,voter_info)
    cleaned_voter_hist = clean_voter_history_training(voter_hist)
    df3 = feature_engineer_voterhistory(cleaned_voter_hist,voter_hist)
    df = feature_engineering(df)
    training = merge(df,df3)
    X = encode_X_y(training)[0]
    y = encode_X_y(training)[1]

    # 2.) Run Test Files thru functions
    df_test = merge_survey_info(survey_test,voter_info_test)
    cleaned_voter_hist_test = clean_voter_history_testing(voter_hist_test)
    df3_test = feature_engineer_voterhistory(cleaned_voter_hist_test,voter_hist_test)
    df_test = feature_engineering(df_test)
    testing = merge(df_test,df3_test)
    XT = encode_test(testing)

    # 3.) Run Through Classifier
    output = train_test_clf(X,y,XT)
    return output.to_csv('part_two_output.csv')

# -------------------- Run It ------------------------

# 1.) Run Training Files thru functions
run_it()