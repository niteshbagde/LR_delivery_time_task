import os,sys
import numpy as np
import pandas as pd
import pandas_profiling

import pickle

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from model.models.regression_model1.exception import CustomeException
from model.models.regression_model1.logger import lg, project_name




# generate pandas profile


def generate_pandas_data_report(data_path:str):
# read data from a CSV file

    try:

        if data_path == "":
            
            data = "deliverytime_ML_model\data\processed\Delivery_data_proc.csv"
            return "data path is invalid, undable to prcess"
        else:
            data = pd.read_csv(data_path)
            

        # generate a profile report using pandas profiling
        profile = pandas_profiling.ProfileReport(data)

        # save the report as an HTML file
        profile.to_file("train_data_report.html")
        
    except Exception as e:
        raise CustomeException(e,sys)


# save pickle object

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        lg.info("saving pickel object ...")
        lg.info("this is the pickel file path : "+ str(file_path) )
        with open(file_path,"wb") as f:
            pickle.dump(obj,f)
    except Exception as e:
        lg.error("error occured in save object function")
        raise CustomeException(e,sys)
    

def load_obj(file_path):
    try:
        lg.info("reading object file from : "+file_path)
        with open(file_path,"rb") as f:
            return pickle.load(f)
        
        lg.info("file object loaded successfully")
        
    except Exception as e:
        lg.error("error occured in loading pickel object : "+file_path)
        raise CustomeException(e,sys)


def data_pathfor_csv():
    csv_data_path = os.getcwd()+"\\"+project_name
    df = pd.read_csv(csv_data_path+"\data\processed\Delivery_data_proc.csv")
    lg.info("reading data from file : "+csv_data_path+"\data\processed\Delivery_data_proc.csv")
    return df


def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(x_train,y_train)

            # Predict Testing data
            y_test_pred =model.predict(x_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report
    except Exception as e:
        lg.error("error occured while evaluating model")
        raise CustomeException(e,sys)




# ---------------------------------------------------------------------------------------------------------- #



# custom function 


# drop nun values and duplicate if any
def drop_na_dup(df_test:pd.Series)-> pd.DataFrame:
    df = df_test.copy()
    if max([i  for i in df.isna().sum()])>0:
        lg.info("dropping null or empty values")
        df = df.dropna(axis=0)
    elif max([i  for i in df.duplicated()]):
        lg.info("dropping duplicate values")
        df = df.drop_duplicates(axis=0)
        
    try:
        lg.info("checking time format from the data")
        # regex expression 
        df = df[df['Time_Orderd'].str.contains(r'^\d{2}:\d{2}$')] 
        df = df[df['Time_Order_picked'].str.contains(r'^\d{2}:\d{2}$')]
    except Exception as e:
        raise CustomeException(e,sys)

    if df.shape[0]>500:
        print("values are more than 500")
    else:
        print("values are less than 500")
        
    return df
    # df = drop_na_dup(df)

# process time data 

def convert_to_daytime_cat(time_str:str):
    """
    Converts a string representing a time value in 24-hour format
    to a categorical label for the corresponding day-time.
    """
    hour = int(time_str.split(':')[0])
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    elif 21 <= hour < 24 or 0 <= hour < 2:
        return "night"
    else:
        return 'midnight'

    # Assuming your DataFrame is called df and the column with time values is 'Time_Order_picked':
    # df['Daytime'] = df['Time_Order_picked'].apply(convert_to_daytime_cat)


# timr differene between ordered and picked

def order_time_difference(df_test:pd.DataFrame, order_col:str='Time_Orderd', picked_col:str='Time_Order_picked',
                          output_col:str='Time_Difference_Minutes'):
    """
    Calculates the time difference in minutes between two columns in a pandas DataFrame.
    
    Args:
        df (pandas DataFrame): The DataFrame containing the time columns.
        order_col (str): The name of the column containing the order time.
        picked_col (str): The name of the column containing the picked-up time.
        output_col (str): The name of the output column to create.
        
    Returns:
        pandas DataFrame: The modified DataFrame with the time difference column added.
    """
    df =df_test.copy()
    # Convert time strings to datetime objects
    df[order_col] = pd.to_datetime(df[order_col], format='%H:%M')
    df[picked_col] = pd.to_datetime(df[picked_col], format='%H:%M')

    # Calculate time difference in minutes
    df[output_col] = ((df[picked_col] - df[order_col]).dt.total_seconds() / 60).astype(int)
    
    # Create columns for hour and minute values
    df['Order_Hour'] = df[order_col].dt.hour
    df['Order_Minute'] = df[order_col].dt.minute
    df['Picked_Hour'] = df[picked_col].dt.hour
    df['Picked_Minute'] = df[picked_col].dt.minute
    df = df.drop([order_col,picked_col],axis=1)
    
    return df
    # df = order_time_difference(df, order_col="Time_Orderd", picked_col="Time_Order_picked", output_col="Time_Difference_Minutes" )


def process_coordinates(df_test:pd.DataFrame, lat_col:str="Restaurant_latitude" or "Delivery_location_latitude", lon_col:str="Restaurant_longitude" or "Delivery_location_longitude"):
    df = df_test.copy()
    # Convert coordinates to strings and split on dot separator
    lat_str = df[lat_col].astype(str)
    lat_parts = lat_str.str.split('.')
    lon_str = df[lon_col].astype(str)
    lon_parts = lon_str.str.split('.')

    # Count the length of the parts before and after the dot separator
    lat_before = lat_parts.str[0].str.len()
    lat_after = lat_parts.str[1].str.len()
    lon_before = lon_parts.str[0].str.len()
    lon_after = lon_parts.str[1].str.len()

    # Combine conditions to select invalid coordinates
    invalid_coords = (lat_before != 2) | (lat_after != 6) | (lon_before != 2) | (lon_after != 6)

    # Drop rows with invalid coordinates
    df = df[~invalid_coords]

    return df

    # df = process_coordinates(df, 'Restaurant_latitude', 'Restaurant_longitude')
    # df = process_coordinates(df, 'Delivery_location_latitude', 'Delivery_location_longitude')


























