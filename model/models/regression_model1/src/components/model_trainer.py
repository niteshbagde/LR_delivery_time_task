import numpy as np
import pandas as pd

import os,sys

from sklearn.linear_model import LinearRegression,Ridge, Lasso, ElasticNet


from model.models.regression_model1.exception import CustomeException
from model.models.regression_model1.logger import lg,project_name
from model.models.regression_model1.utils import save_obj

from model.models.regression_model1.utils import evaluate_model

from dataclasses import dataclass


@dataclass
class ModelTrainierConfig:

    pickel_data_path = os.getcwd()+"\\"+project_name
    trained_model_file_path = pickel_data_path+"\\model\\artifacts\\"+"Model.pkl"
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainierConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            lg.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
        }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            lg.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score*100}')
            print('\n====================================================================================\n')
            lg.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score*100}')

            save_obj(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            lg.info('Exception occured at Model Training')
            raise CustomeException(e,sys)













