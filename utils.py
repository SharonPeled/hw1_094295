import pickle
import pandas as pd
import numpy as np 
import json
import ast
from textblob import TextBlob
import datetime
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer

def adjust_prediction(pred):
    pred = np.e ** pred - 1
    pred[pred<0] = 1
    return pred


def predict(model,X_test):
    y_pred = model.predict(X_test.values)
    y_pred = adjust_prediction(y_pred)
    return y_pred

def get_values(elem_str,field):
    elem_list = [] if pd.isna(elem_str) else ast.literal_eval(elem_str)
    if isinstance(elem_list, dict):
        elem_list = [elem_list]
    values = []
    for elem in elem_list:
        values.append(elem[field])
    return values


def json_to_onehot(df,col,field,one_hot_columns):
    column_format = lambda val : col+"_"+str(val)
    possible_values = sorted(one_hot_columns[col])
    new_columns = [column_format(val) for val in possible_values]
    for new_col in new_columns:
        df[new_col] = 0
    df[new_columns].replace(np.nan, 0, inplace=True)
#     print(col," added ",len(new_columns), " columns")
#     print(col,field,possible_values)
    for i,row in df.iterrows():
        for val in [elem for elem in get_values(row[col],field) if elem in possible_values]:
            df.loc[i,column_format(val)] = 1
    


def reformat_data(df_input):
    df = df_input.copy()
    with open("one_hot_columns", 'rb') as file:
        one_hot_columns = pickle.load(file)
    # adding one hot columns only if the number of samples reaches the threshold
    json_to_onehot(df,"belongs_to_collection","name",one_hot_columns)
    json_to_onehot(df,"genres","name",one_hot_columns)
    json_to_onehot(df,"production_companies","id",one_hot_columns)
    json_to_onehot(df,"production_countries","name",one_hot_columns)
    json_to_onehot(df,"spoken_languages","iso_639_1",one_hot_columns)
    json_to_onehot(df,"cast","name",one_hot_columns)
    json_to_onehot(df,"crew","name",one_hot_columns)
    return df 


def add_numeric_features(X,df):
    # finds all the columns who are numeric and withour nulls - "ready for being features"
    numeric_columns = []
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            numeric_columns.append(col)
    for num_col in numeric_columns:
        X[num_col] = df[num_col]
    return numeric_columns


def add_features(X,df,columns,column_format,func):
    for col in columns:
        X[column_format(col)] = df[col].apply(lambda elem : func(elem))    
    

def get_json_list_values(elem_str,field):
    elem_list = [] if pd.isna(elem_str) else ast.literal_eval(elem_str)
    if isinstance(elem_list, dict):
        elem_list = [elem_list]
    values = []
    for elem in elem_list:
        values.append(elem[field])
    return values


def add_time_features(X,df):
    dates = df["release_date"].apply(lambda date_str:datetime.datetime.strptime(date_str, "%Y-%m-%d") if not pd.isna(date_str) else np.nan)
    add_time_range_features(X,dates.apply(lambda date : date.year if not pd.isna(date) else np.nan),1980,2020,5,lambda t : "year_"+str(t))
    add_time_range_features(X,dates.apply(lambda date : date.month if not pd.isna(date) else np.nan),3,12,3,lambda t : "month_"+str(t))
    X["month_1st"] = dates.apply(lambda date : int(date.day==1) if not pd.isna(date) else np.nan)
    X["is_friday"] = dates.apply(lambda date : int(date.today().weekday()==5) if not pd.isna(date) else np.nan)
    
    
def add_time_range_features(X,time_args,low_bound,up_bound,interval,column_format):
    intervals = np.arange(low_bound,up_bound,interval)
    lower_time_range = -np.inf
    for time_range in intervals:
        X[column_format(time_range)] = time_args.apply(lambda time_arg:int(lower_time_range<time_arg<=time_range) if not pd.isna(time_arg) else np.nan)    
        lower_time_range = time_range
        
        
def generate_feature_matrix(df_input):
    df = df_input.copy()
    df.drop(["id"],axis=1,inplace=True) # non relevant numeric column
    X = pd.DataFrame(index=df.index)
    # Make sure to leave the NaNs and not change to 0 when needed.
    add_numeric_features(X,df)
    # no missing values features (if there are missing values we make sure to fill them)
    add_features(X,df,["homepage"],lambda col : "has_"+col,lambda elem : int(not pd.isna(elem)))
    add_features(X,df,["video"],lambda col : "has_"+col,lambda elem : int(elem) if not pd.isna(elem) else 0)
    add_features(X,df,["belongs_to_collection"],lambda col : "is_"+col,lambda elem: len(get_json_list_values(elem,"id")) if not pd.isna(elem) else 0) # binary
    # potentially has missing values (+ numerical features)
    add_features(X,df,["title","overview","tagline"],lambda col : col + "_length",lambda elem: len(elem.split(" ")) if not pd.isna(elem) else np.nan)
    add_features(X,df,["genres","production_companies","Keywords"],lambda col : "num_"+col,lambda elem: len(get_json_list_values(elem,"id")) if not pd.isna(elem) else np.nan)
    add_features(X,df,["production_countries"],lambda col : "num_"+col,lambda elem: len(get_json_list_values(elem,"iso_3166_1")) if not pd.isna(elem) else np.nan)
    add_features(X,df,["spoken_languages"],lambda col : "num_"+col,lambda elem: len(get_json_list_values(elem,"iso_639_1")) if not pd.isna(elem) else np.nan)
    add_features(X,df,["cast"],lambda col : "num_"+col,lambda elem: len(get_json_list_values(elem,"cast_id")) if not pd.isna(elem) else np.nan)
    add_features(X,df,["crew"],lambda col : "num_"+col,lambda elem: len(get_json_list_values(elem,"credit_id")) if not pd.isna(elem) else np.nan)
    add_features(X,df,["overview"],lambda col : "polarity",lambda elem: TextBlob(elem).sentiment.polarity if not pd.isna(elem) else np.nan)
    add_features(X,df,["overview"],lambda col : "subjectivity",lambda elem: TextBlob(elem).sentiment.subjectivity if not pd.isna(elem) else np.nan)
    add_time_features(X,df)
    return X


def feature_tranform(X,col,func,new_name):
    X[col] = X[col].apply(lambda elem : func(elem))
    X.rename(columns={col:new_name},inplace=True)

    
# Feature transformations
def transform(X):
    budget_threshold = 30000
    X["budget"] = X["budget"].apply(lambda elem: elem if elem>=budget_threshold else np.nan)
    log_transform = lambda elem:np.log(elem+1) if not pd.isna(elem) else np.nan
    feature_tranform(X,"budget",log_transform,"log_budget")
    feature_tranform(X,"popularity",log_transform,"log_popularity")
    feature_tranform(X,"vote_average",log_transform,"log_vote_average")
    feature_tranform(X,"vote_count",log_transform,"log_vote_count")
    feature_tranform(X,"runtime",log_transform,"log_runtime")
    return X


def impute(X):
    missing_data_columns = X.columns[X.isnull().sum()>0]
    print(missing_data_columns)
    model_imp = IterativeImputer(max_iter=30,imputation_order='ascending',verbose=2,tol=0.01)
    X = pd.DataFrame(data=model_imp.fit_transform(X), columns=X.columns)
    missing_data_columns = X.columns[X.isnull().sum()>0]
    print(missing_data_columns)
    return X


def final_feature_matrix(df_input):
    df = reformat_data(df_input)
    if "revenue" in df.columns:
        df.drop(columns=["revenue"],inplace=True) # so we won't use it as feature
    X_test = generate_feature_matrix(df)
    X_test = transform(X_test)
    X_test = impute(X_test)
    with open("columns.pkl", 'rb') as file:
        columns = pickle.load(file)
    X_test = X_test[columns]
    return X_test