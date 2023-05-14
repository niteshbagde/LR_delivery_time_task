import os, sys
from model.models.regression_model1.logger import lg
from model.models.regression_model1.utils import CustomeException
from model.models.regression_model1.src.components.data_ingestion import DataIngestion
import pandas as pd
from model.models.regression_model1.src.components.data_transformation import DataTransformation
from model.models.regression_model1.src.components.model_trainer import ModelTrainer

# test 
if __name__ == '__main__':
    lg.info("training pipeline started...")
    obj = DataIngestion()
    train_data_p,test_data_p = obj.initiate_data_ingestion()

    data_transform = DataTransformation()
    tra_arr,tes_arr,_ = data_transform.initate_data_transfomration(train_data_p,test_data_p)

    model_trainer = ModelTrainer()
    model_trainer.initate_model_training(train_array=tra_arr,test_array=tes_arr)
    lg.info("traing pipeline ended ...")


# tested and working fine