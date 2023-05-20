import os, sys
from model.models.regression_model1.M2logger import lg
from model.models.regression_model1.M2utils import CustomeException
from model.models.regression_model1.src2.components.data_ingestion import DataIngestion
import pandas as pd
from model.models.regression_model1.src2.components.data_transformation import DataTransformation
from model.models.regression_model1.src2.components.model_trainer import ModelTrainer


def train_model2()->str:
    try:
        lg.info("training pipeline started...")
        obj = DataIngestion()
        train_data_p,test_data_p = obj.initiate_data_ingestion()

        data_transform = DataTransformation()
        tra_arr,tes_arr,_ = data_transform.initate_data_transfomration(train_data_p,test_data_p)

        model_trainer = ModelTrainer()
        model_trainer.initate_model_training(train_array=tra_arr,test_array=tes_arr)
        lg.info("traing pipeline ended ...")
        return "completed"
    except Exception as e:
        lg.info("erro occured in traing pipeline ...")
        raise CustomeException(e,sys)


# test 
if __name__ == '__main__':
    train_model2()


# tested and working fine