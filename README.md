# Project README

## Delivery Time Estimation

This project aims to estimate delivery time based on various factors such as delivery person details, restaurant and delivery location coordinates, weather conditions, road traffic density, vehicle condition, type of order and vehicle, multiple deliveries, festival, city, and time taken.

### Data

The raw dataset had 20 columns, out of which unnecessary columns were removed, and exploratory data analysis (EDA) and feature engineering (FE) were applied. The processed dataset has 15 columns including the time taken in minutes, daytime, time difference in minutes, and order and pick-up times in hours and minutes.

Note that the coordinate columns can be replaced with on-ground distance for testing real-world accuracy, and some columns like order time and pick-up times may not be necessary due to their high correlation with other features.

### Model

A linear regression model was built using the processed dataset and deployed as a Flask app. The model's accuracy can be further improved by considering additional parameters, such as route network analysis or on-ground distance.

### Flask App

The Flask app consists of three nav tabs: prediction, model details, and data. Users can input their data via an HTML form and get the estimated delivery time. The model details tab provides information about the backend details of the model like accuracy and name. The data tab presents a Pandas profiling report.

## Quick Walkthrough

The raw data contains 20 columns, including ID, delivery person details, restaurant and delivery location coordinates, order and pickup times, weather conditions, road traffic density, vehicle condition, type of order and vehicle, multiple deliveries, festival, city, and time taken.

Unnecessary columns were removed via EDA & FE, and the following 15 columns represent the processed data:

1. Delivery_person_Age
2. Delivery_person_Ratings
3. Restaurant_latitude
4. Restaurant_longitude
5. Delivery_location_latitude
6. Delivery_location_longitude
7. Weather_conditions
8. Road_traffic_density
9. Vehicle_condition
10. Type_of_order
11. Type_of_vehicle
12. Multiple_deliveries
13. Festival
14. City
15. Time_taken (min)
16. Daytime - This column was added based on the order time. We can also add days as a category to see its impact.
17. Time_Difference_Minutes - This was added to remove order time and order pick time since their correlation was high with the dependent feature.
18. Order_Hour
19. Order_Minute
20. Picked_Hour
21. Picked_Minute

Note:

a. Columns for coordinate data can be removed, and instead, on-ground distance can be applied to give the model the actual scenario of the real world. More parameters can be applied concerning it, and the model will keep on growing. For this example, I have kept it as it is. Although I have considered 1000 data points and applied on-ground distance from Google Maps, there was not much difference in prediction accuracy observed via R2 score. Also, the coordinate values contained some negative values that caused data imbalance and outliers. The best approach would be to feed actual on-ground distance and maybe add some new data using route network analysis since the coordinates are available to plot them on a map. Another reason was that the correlation observed was high among themselves, which might cause model overfitting.

        3. Restaurant_latitude
        4. Restaurant_longitude
        5. Delivery_location_latitude
        6. Delivery_location_longitude

b. The following columns can be removed since their correlation might be affecting the prediction:

        18. Order_Hour
        19. Order_Minute
        20. Picked_Hour
        21. Picked_Minute

#

The pipeline was built for the linear regression model considering the above columns. The columns mentioned in the notes section can be applied, and we can observe the changes in accuracy, but they are not applied in this model. We will update the repo once done.

The Flask app was created for the model to test and apply the prediction on user input data via an HTML form.

The Flask app contains three nav tabs. Apart from prediction, it will give details on the model (backend details like model accuracy, name of the model, etc.) and data (Pandas profiling). The model details tab is not updated for now...

#

> The project is completed and is working fine on testing.



























