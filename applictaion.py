from flask import  Flask , request ,render_template , json,jsonify , flash

from model.models.regression_model1.src.pipeline.prediction_pipeline import CustomData, PredictionPipleine


from model.models.regression_model1.src2.pipeline.prediction_pipeline import CustomData as CustomData2
from model.models.regression_model1.src2.pipeline.prediction_pipeline import PredictionPipleine as PredictionPipleine2

from model.models.regression_model1.utils import order_time_difference, convert_to_daytime_cat
from model.models.regression_model1.logger import lg, project_name
from model.models.regression_model1.M2logger import lg as lg2


from model.models.regression_model1.src.pipeline.training_pipeline import train_model1
from model.models.regression_model1.src2.pipeline.training_pipeline import train_model2

import os
from pandas_profiling import ProfileReport
import pandas as pd

model_data_path = os.getcwd()
model1_path = os.path.join(model_data_path+"\\deliverytime_ML_model\\model\\artifacts\\"+"model_details.txt")
model2_path = os.path.join(model_data_path+"\\deliverytime_ML_model\\model\\artifacts\\"+"M2model_details.txt") # note extenison was not properly maintained


application = Flask(__name__)


@application.route('/')
@application.route('/home')
def home_page():
    return render_template('index.html')

@application.route('/model_details')
def model_details():
    # model 1
    with open(model1_path,"r") as f1:
        model1 = f1.readlines()
    # model 2
    with open(model2_path,"r") as f2:
        model2 = f2.readlines()
    return render_template('model_details.html', len=6, model1 = model1, model2 = model2)







@application.route('/report1')
def custom_pandas_profile1():

    data1 = pd.read_csv(os.path.join(os.getcwd()+"\\deliverytime_ML_model\\data\\processed\\")+"Delivery_data_proc.csv")
    for i in data1.columns:
        if i == "Unnamed: 0":
            data1=data1.drop("Unnamed: 0",axis=1)
    else:
        pass
    profile1 = ProfileReport(data1, title='Model1_Data_Profiling_Report', explorative=True)
    profile1.to_widgets()
    profile1.to_file(os.path.join(os.getcwd()+"\\deliverytime_ML_model\\templates\\")+"Data1_Profiling_Report.html")
    return render_template('Data1_Profiling_Report.html')

@application.route('/train_data_report1')
def train_data_report1():
    # report from pandas profiling
    return render_template("Data1_Profiling_Report.html")





@application.route('/report2')
def custom_pandas_profile2():
    
    data2 = pd.read_csv(os.path.join(os.getcwd()+"\\deliverytime_ML_model\\data\\processed\\")+"Processed_Delivery_data_WithDist.csv")
    for i in data2.columns:
        if i == "Unnamed: 0":
            data2=data2.drop("Unnamed: 0",axis=1)
    else:
        pass
    profile2 = ProfileReport(data2, title='Model2_Data_Profiling_Report', explorative=True)
    profile2.to_widgets()
    profile2.to_file(os.path.join(os.getcwd()+"\\deliverytime_ML_model\\templates\\")+"Data2_Profiling_Report.html")
    return render_template('Data2_Profiling_Report.html')



@application.route('/train_data_report2')
def train_data_report2():
    # report from pandas profiling
    return render_template("Data2_Profiling_Report.html")





@application.route('/data_details')
def data_details():
    return render_template('data_details.html')



@application.route('/predictm1',methods=['GET','POST'])
def run_model1():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data=CustomData(
                        Delivery_person_Age = int(request.form.get("Delivery_person_Age")), 
                        Delivery_person_Ratings = float(request.form.get("Delivery_person_Ratings")),
                        Restaurant_latitude = float(request.form.get("Restaurant_latitude")),
                        Restaurant_longitude = float(request.form.get("Restaurant_longitude")), 
                        Delivery_location_latitude = float(request.form.get("Delivery_location_latitude")),
                        Delivery_location_longitude = float(request.form.get("Delivery_location_longitude")),
                        Time_Orderd=str(request.form.get("Time_Orderd")),
                        Time_Order_picked=str(request.form.get("Time_Order_picked")), 
                        Weather_conditions = str(request.form.get("Weather_conditions")),
                        Road_traffic_density  = str(request.form.get("Road_traffic_density")), 
                        Vehicle_condition = int(request.form.get("Vehicle_condition")), 
                        Type_of_order  = str(request.form.get("Type_of_order")),
                        Type_of_vehicle  = str(request.form.get("Type_of_vehicle")), 
                        multiple_deliveries = int(request.form.get("Multiple_deliveries")), 
                        Festival  = str(request.form.get("Festival")), 
                        City  = str(request.form.get("City")),

        )

        lg.info("collected data for : "+ project_name)
        df = data.get_data_as_dataframe()
        lg.info(df.to_string())
        # self.Daytime = Daytime 
        df['Daytime'] = df['Time_Order_picked'].apply(convert_to_daytime_cat)
        # self.Order_Hour = Order_Hour
        # self.Order_Minute  = Order_Minute
        # self.Picked_Hour = Picked_Hour
        # self.Picked_Minute = Picked_Minute
        # self.Time_Difference_Minutes = Time_Difference_Minutes
        df = order_time_difference(df, order_col="Time_Orderd", picked_col="Time_Order_picked",output_col="Time_Difference_Minutes" )
        PrediPipleine = PredictionPipleine()
        lg.info("processing prediction for data :")
        lg.info("\n"+df.to_string())
        pred = PrediPipleine.predict(df)
        # result = pred
        return render_template("result.html", result = round(pred[0],2))

@application.route('/predictm2',methods=['GET','POST'])
def run_model2():
    if request.method=='GET':
        return render_template('index.html')
    else:
        
        data2=CustomData2(
                        Delivery_person_Age = int(request.form.get("Delivery_person_Age")), 
                        Delivery_person_Ratings = float(request.form.get("Delivery_person_Ratings")), 
                        Weather_conditions = str(request.form.get("Weather_conditions")),
                        Road_traffic_density  = str(request.form.get("Road_traffic_density")), 
                        Vehicle_condition = int(request.form.get("Vehicle_condition")), 
                        Type_of_order  = str(request.form.get("Type_of_order")),
                        Type_of_vehicle  = str(request.form.get("Type_of_vehicle")), 
                        multiple_deliveries = int(request.form.get("Multiple_deliveries")), 
                        Festival  = str(request.form.get("Festival")), 
                        City  = str(request.form.get("City")),
                        Daytime = str(request.form.get("Daytime")),
                        order_pick_Time_Difference_Minutes = str(request.form.get("order_pick_Time_Difference_Minutes")),
                        onground_dist_in_KM = float(request.form.get("onground_dist_in_KM"))
                        )

        lg2.info("collected data for : "+ project_name)
        df2 = data2.get_data_as_dataframe()
        PrediPipleine2 = PredictionPipleine2()
        lg2.info("processing prediction for data :")
        lg2.info("\n"+df2.to_string())
        pred2 = PrediPipleine2.predict(df2)
        # result = pred
        return render_template("result.html", result = round(pred2[0],2))







# Define the endpoint to receive requests
@application.route('/predict', methods=['GET','POST'])
def predict():
    input_params = str(request.form.get("model"))
    # Check which model to use based on input parameter
    if input_params == "model1":
        return render_template('model1.html')
    elif input_params == "model2":
        return render_template('model2.html')
    else:
        return render_template('index.html', msg="invalid")





@application.route('/train1', methods=['GET','POST'])
def re_train_model1():
    train1 = train_model1()
    if train1 == "completed":
        message1=["Model Training completed"]
        model_result = os.getcwd()+"\\"+project_name+"\\model\\artifacts\\"+"model_details.txt"
        with open(model_result,"r") as f:
            message_read = f.read()
        for i in message_read.splitlines():
            message1.append(i)
        return message1
    else:
        message1 = "Model Training Failed"
        return message1




@application.route('/train2', methods=['GET','POST'])
def re_train_model2():
    train2 = train_model2()
    if train2 == "completed":
        message2=["Model Training completed"]
        model_result = os.getcwd()+"\\"+project_name+"\\model\\artifacts\\"+"M2model_details.txt"
        with open(model_result,"r") as f:
            message_read = f.read()
        for i in message_read.splitlines():
            message2.append(i)
        return message2
    else:
        message2 = "Model Training Failed"
        return message2






if __name__=="__main__":
    application.run(host='0.0.0.0',debug=True)









