import os , sys
import pandas as pd 
# import pandas_profiling

from model.models.regression_model1.exception import CustomeException
from model.models.regression_model1.M2logger import lg , project_name

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from model.models.regression_model1.src2.components.data_transformation import DataTransformation
from model.models.regression_model1.M2utils import data_pathfor_csv


@dataclass
class DataIngestionConfig():

    path__ = os.getcwd()+"/"+project_name

    train_data_path = os.path.join(path__+"/data/interim", "model2_train.csv")
    test_data_path = os.path.join(path__+"/data/interim", "model2_test.csv")
    raw_data_path = os.path.join(path__+"/data/interim", "model2_raw_copy.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()   #tupel of 3 path


    def initiate_data_ingestion(self):
        lg.info("Data ingestion method starts")
        try:
            
            df = data_pathfor_csv()

            for i in df.columns:
                if i == "Unnamed: 0":
                    df = df.drop("Unnamed: 0", axis=1)
                    lg.info("found and dropping column Unnamed: 0 ")
                else:
                    pass
            lg.info("data read complete to pandas dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            
            lg.info("train test split path initialized")
            train_set , test_set = train_test_split(df, test_size=0.3, random_state=44)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            lg.info("data ingestion completed for defining path")

            lg.info("returning train and test data path for further pipeline")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        

        except Exception as e:
            lg.error("exception occured at data ingestion stage")
            raise CustomeException(e,sys)

# test 
# if __name__ == '__main__':
#     obj = DataIngestion()
#     train_data_p,test_data_p = obj.initiate_data_ingestion()

#     data_transform = DataTransformation()
#     tra_arr,tes_arr,_ = data_transform.initate_data_transfomration(train_data_p,test_data_p)

# tested and working fine






