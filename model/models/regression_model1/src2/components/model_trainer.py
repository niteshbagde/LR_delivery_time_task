import numpy as np
import pandas as pd

import os,sys

from sklearn.linear_model import LinearRegression,Ridge, Lasso, ElasticNet


from model.models.regression_model1.exception import CustomeException
from model.models.regression_model1.M2logger import lg,project_name
from model.models.regression_model1.M2utils import save_obj 

from model.models.regression_model1.M2utils import evaluate_model

from dataclasses import dataclass


@dataclass
class ModelTrainierConfig:

    pickel_data_path = os.getcwd()+"\\"+project_name
    trained_model_file_path = pickel_data_path+"\\model\\artifacts\\"+"M2Model.pkl"
    text_model_data_file_path = pickel_data_path+"\\model\\artifacts\\"+"M2model_details.txt"
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainierConfig()


    
    def initate_model_training(self,train_array,test_array):

        text_model_data = list()
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


            for i in model_report:
                daat = i + "=" + str(model_report[i]) + "\n"
                text_model_data.append(daat)


            text_model_data.append("\n"+f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score*100}') , 


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
        
    # exporting model details to text file
        lg.info("exportingmodel details to text file : " + self.model_trainer_config.text_model_data_file_path)
        with open(self.model_trainer_config.text_model_data_file_path,"w") as f:
            for i in text_model_data:
                f.writelines(i)










