import sys, os, pickle 
from model.models.regression_model1.exception import CustomeException
from model.models.regression_model1.M2logger import lg, project_name
from model.models.regression_model1.M2utils import load_obj

import pandas as pd


class PredictionPipleine:
    def __init__(self):
        pass


    def predict(self,features):

        try:
            pickel_data_path = os.getcwd()
            preprocessor_path = os.path.join(pickel_data_path+"\\deliverytime_ML_model\\model\\artifacts\\"+"M2processor.pkl")
            model_path = os.path.join(pickel_data_path+"\\deliverytime_ML_model\\model\\artifacts\\"+"M2Model.pkl") # note extenison was not properly maintained

            processor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scale = processor.transform(features)
            pred = model.predict(data_scale)

            return pred


        except Exception as e:
            lg.info("error occured in prediction pipleine")
            raise CustomeException(e,sys)
        
# temp = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Weather_conditions',
#        'Road_traffic_density', 'Vehicle_condition', 'Type_of_order',
#        'Type_of_vehicle', 'multiple_deliveries', 'Festival', 'City',
#        'Time_taken (min)', 'Daytime', 'order_pick_Time_Difference_Minutes',
#        'onground_dist(KM)']

class CustomData:
        def __init__(
                        self, 
                        Delivery_person_Age : int, 
                        Delivery_person_Ratings : float,
                        Weather_conditions : float,
                        Road_traffic_density  : str, 
                        Vehicle_condition : int, 
                        Type_of_order  : str,
                        Type_of_vehicle  : str, 
                        multiple_deliveries : int, 
                        Festival  : str, 
                        City  : str,
                        Daytime : str,
                        order_pick_Time_Difference_Minutes : int,
                        onground_dist_in_KM : float
                        ):
                        self.Delivery_person_Age = Delivery_person_Age
                        self.Delivery_person_Ratings = Delivery_person_Ratings
                        self.Weather_conditions = Weather_conditions
                        self.Road_traffic_density = Road_traffic_density
                        self.Vehicle_condition = Vehicle_condition
                        self.Type_of_order = Type_of_order
                        self.Type_of_vehicle = Type_of_vehicle
                        self.multiple_deliveries = multiple_deliveries
                        self.Festival = Festival
                        self.City  = City
                        self.Daytime = Daytime
                        self.order_pick_Time_Difference_Minutes = order_pick_Time_Difference_Minutes
                        self.onground_dist_in_KM = onground_dist_in_KM

        
        def get_data_as_dataframe(self):
            
            try:
                    

                    custom_data_input_dict = {
                                                'Delivery_person_Age' : [self.Delivery_person_Age], 
                                                'Delivery_person_Ratings' : [self.Delivery_person_Ratings],
                                                'Weather_conditions' : [self.Weather_conditions],
                                                'Road_traffic_density'  : [self.Road_traffic_density], 
                                                'Vehicle_condition' : [self.Vehicle_condition], 
                                                'Type_of_order'  : [self.Type_of_order],
                                                'Type_of_vehicle'  : [self.Type_of_vehicle], 
                                                'multiple_deliveries' : [self.multiple_deliveries], 
                                                'Festival'  : [self.Festival], 
                                                'City'  : [self.City],
                                                'Daytime' : [self.Daytime],
                                                'order_pick_Time_Difference_Minutes' : [self.order_pick_Time_Difference_Minutes],
                                                'onground_dist_in_KM' : [self.onground_dist_in_KM]

                                                }
                    
                    df = pd.DataFrame(custom_data_input_dict)
                    lg.info("data gathered. Inputs converted to data frame")

                    return df

            except Exception as e:
                    raise CustomeException(e,sys)
        










