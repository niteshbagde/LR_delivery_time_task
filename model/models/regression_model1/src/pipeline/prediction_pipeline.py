import sys, os, pickle 
from model.models.regression_model1.exception import CustomeException
from model.models.regression_model1.logger import lg, project_name
from model.models.regression_model1.utils import load_obj, convert_to_daytime_cat

import pandas as pd


class PredictionPipleine:
    def __init__(self):
        pass


    def predict(self,features):

        try:
            pickel_data_path = os.getcwd()
            preprocessor_path = os.path.join(pickel_data_path+"\\model\\artifacts\\"+"processor.pk1")
            model_path = os.path.join(pickel_data_path+"\\model\\artifacts\\"+"Model.pkl") # note extenison was not properly maintained

            processor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scale = processor.transform(features)
            pred = model.predict(data_scale)

            return pred


        except Exception as e:
            lg.info("error occured in prediction pipleine")
            raise CustomeException(e,sys)
        


class CustomData:
        def __init__(
                        self, 
                        Delivery_person_Age : int, 
                        Delivery_person_Ratings : float,
                        Restaurant_latitude : float,
                        Restaurant_longitude : float, 
                        Delivery_location_latitude : float,
                        Delivery_location_longitude : float,
                        Time_Orderd:str,
                        Time_Order_picked:str, 
                        Weather_conditions : float,
                        Road_traffic_density  : str, 
                        Vehicle_condition : int, 
                        Type_of_order  : str,
                        Type_of_vehicle  : str, 
                        multiple_deliveries : int, 
                        Festival  : str, 
                        City  : str,
        ):
                        self.Delivery_person_Age = Delivery_person_Age
                        self.Delivery_person_Ratings = Delivery_person_Ratings
                        self.Restaurant_latitude = Restaurant_latitude                  
                        self.Restaurant_longitude = Restaurant_longitude
                        self.Delivery_location_latitude = Delivery_location_latitude   
                        self.Delivery_location_longitude = Delivery_location_longitude
                        self.Time_Orderd = Time_Orderd
                        self.Time_Order_picked = Time_Order_picked
                        self.Weather_conditions = Weather_conditions
                        self.Road_traffic_density = Road_traffic_density
                        self.Vehicle_condition = Vehicle_condition
                        self.Type_of_order = Type_of_order
                        self.Type_of_vehicle = Type_of_vehicle
                        self.multiple_deliveries = multiple_deliveries
                        self.Festival = Festival
                        self.City  = City



                        # self.Time_Difference_Minutes = Time_Difference_Minutes 
                        # self.Order_Hour = Order_Hour
                        # self.Order_Minute  = Order_Minute
                        # self.Picked_Hour = Picked_Hour
                        # self.Picked_Minute = Picked_Minute
                        # self.Daytime = Daytime 
        
        def get_data_as_dataframe(self):
            
            try:
                    
                    self.Time_Order_picked = self.Time_Order_picked.replace(".", ":")
                    self.Time_Orderd = self.Time_Orderd.replace(".", ":")

                    custom_data_input_dict = {
                                                'Delivery_person_Age' : [self.Delivery_person_Age], 
                                                'Delivery_person_Ratings' : [self.Delivery_person_Ratings],
                                                'Restaurant_latitude' : [self.Restaurant_latitude],
                                                'Restaurant_longitude' : [self.Restaurant_longitude], 
                                                'Delivery_location_latitude' : [self.Delivery_location_latitude],
                                                'Delivery_location_longitude' : [self.Delivery_location_longitude],
                                                'Time_Orderd':[self.Time_Orderd],
                                                'Time_Order_picked':[self.Time_Order_picked], 
                                                'Weather_conditions' : [self.Weather_conditions],
                                                'Road_traffic_density'  : [self.Road_traffic_density], 
                                                'Vehicle_condition' : [self.Vehicle_condition], 
                                                'Type_of_order'  : [self.Type_of_order],
                                                'Type_of_vehicle'  : [self.Type_of_vehicle], 
                                                'multiple_deliveries' : [self.multiple_deliveries], 
                                                'Festival'  : [self.Festival], 
                                                'City'  : [self.City],
                                                }
                    
                    df = pd.DataFrame(custom_data_input_dict)
                    lg.info("data gathered. Inputs converted to data frame")

                    return df

            except Exception as e:
                    raise CustomeException(e,sys)
        










