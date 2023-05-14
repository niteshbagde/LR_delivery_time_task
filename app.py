from flask import  Flask , request ,render_template , json,jsonify

from model.models.regression_model1.src.pipeline.prediction_pipeline import CustomData, PredictionPipleine

from model.models.regression_model1.utils import order_time_difference, convert_to_daytime_cat
from model.models.regression_model1.logger import lg, project_name

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home_page():
    return render_template('index.html')

@app.route('/model_details')
def model_details():
    return render_template('model_details.html')

@app.route('/train_data_report')
def train_data_report():
    # report from pandas profiling
    return render_template("train_data_report.html")


@app.route('/data_details')
def data_details():
    return render_template('data_details.html')


@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
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




if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)









