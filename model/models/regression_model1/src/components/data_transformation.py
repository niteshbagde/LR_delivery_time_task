import pandas as pd
import os,sys
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split

from model.models.regression_model1.exception import CustomeException
from model.models.regression_model1.logger import lg , project_name
from model.models.regression_model1.utils import save_obj , data_pathfor_csv

@dataclass
class DataTransformationConfig:
    pickel_data_path = os.getcwd()+"\\"+project_name
    preprocessor_obj_file_path = pickel_data_path+"\\model\\artifacts\\"+"processor.pkl"

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            lg.info("Data transformation initialized")

            df = data_pathfor_csv()

            for i in df.columns:
                if i == "Unnamed: 0":
                    df = df.drop("Unnamed: 0", axis=1)
                    lg.info("found and dropping column Unnamed: 0 ")
                else:
                    pass
            lg.info(df.columns)
            # Define which columns should be ordinal-encoded and which should be scaled

            categorical_cols = df.select_dtypes(include='object').columns
            numerical_cols = df.select_dtypes(exclude='object').drop('Time_taken (min)',axis=1).columns

            cat_for_ord_columns = df[["Weather_conditions","Road_traffic_density", "Festival"]].columns

            cat_for_ohe_columns = df[['Type_of_order', 'Type_of_vehicle', 'City','Daytime']].columns


            #-------------------------------------------------------------------------#
            # ordinal cat
            wather_cat = ["Fog", "Stormy","Sandstorms","Windy", "Cloudy", "Sunny"]
            traffic_cat = ['Low','Medium','High','Jam']
            festival_cat = ['No', 'Yes']
            #-------------------------------------------------------------------------#
            # one Hot encoding cat
            typeoforder = ['Snack' ,'Meal', 'Buffet', 'Drinks']
            typeofvehicle = ['motorcycle', 'scooter', 'electric_scooter']
            city = ['Metropolitian' ,'Urban' ,'Semi-Urban']
            day_time = ['afternoon','night','morning', 'evening' ]
            #-------------------------------------------------------------------------#

            lg.info("pipeline initiated")

            num_pipeline=Pipeline(
                    steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False))

                    ]

                )

            # Categorigal Pipeline
            cat_pipeline1=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[wather_cat,traffic_cat,festival_cat])),
                ('scaler',StandardScaler(with_mean=False))
                ]

            )

            cat_pipeline2 = Pipeline(
                steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                    ("ohe", OneHotEncoder(categories=[typeoforder,typeofvehicle,city,day_time],handle_unknown="ignore")),
                    ('scaler',StandardScaler(with_mean=False))]
            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline1',cat_pipeline1,cat_for_ord_columns),
            ("cat_pipeline2", cat_pipeline2,cat_for_ohe_columns)
            ])

            return preprocessor

        except Exception as e:
            lg.error("error in data transformation stage")
            raise CustomeException(e,sys)

    def initate_data_transfomration(self, train_path, test_path):
        try:
            lg.info("reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            lg.info("read train and test data completed")
            preprocessor_obj = self.get_data_transformation_obj()
            lg.info("obtained preprocessor object")

            target_column_name = "Time_taken (min)"
            drop_columns = [target_column_name]

            input_features_train_df = train_df.drop(drop_columns,axis=1)
            lg.info(train_df.head())
            target_fetaures_train_df = train_df[target_column_name]
            lg.info(target_fetaures_train_df.head())
            input_feature_test_df = test_df.drop(drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]
            lg.info("applying the preprocessing object on train and test dataset")


            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_features_train_arr, np.array(target_fetaures_train_df)]
            test_arr = np.c_[input_features_test_arr,np.array(target_feature_test_df)]

            save_obj(
                file_path= self.data_tranformation_config.preprocessor_obj_file_path,
                obj= preprocessor_obj
            )
            lg.info("pickel file saved from initat data transformation ")

            lg.info("initat data transformation completed")

            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            lg.error("error occured in initated data transfomration stage  ")
            raise CustomeException(e,sys)






