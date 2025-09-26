AQI Forecasting for Indian Cities
<img width="2537" height="1295" alt="image" src="https://github.com/user-attachments/assets/795852f4-47e6-4425-bc82-84c20ae62b5b" />

View the Live Deployed Application Here [https://aqiforcastingapp.streamlit.app/]
1. Project Overview
This project is a comprehensive data science application designed to forecast the Air Quality Index (AQI) for major cities across India. It leverages historical air quality data to train a time-series model (Facebook's Prophet) capable of predicting future AQI values.

The primary goal is to transform a large, complex dataset into an interactive and user-friendly web application that provides actionable environmental insights. This end-to-end project demonstrates skills in data cleaning, time-series analysis, predictive modeling, and web application development with Streamlit.

2. Tech Stack & Libraries
Language: Python

Core Libraries:

Pandas: For data manipulation and cleaning.

Prophet (by Facebook): For time-series forecasting.

Streamlit: For building the interactive web dashboard.

Statsmodels: (Used in initial analysis phases).

3. Key Features
Interactive Dashboard: A clean, user-friendly interface built with Streamlit.

City Selection: Users can select from a dropdown list of dozens of Indian cities with sufficient historical data.

Dynamic Forecasting: The ability to generate AQI forecasts for a user-defined period (7 to 90 days).

Data Visualization: An interactive line chart that clearly displays the historical trends and future predictions.

Cached Backend: Utilizes Streamlit's caching to store the prediction engine, ensuring a fast and smooth user experience after the initial load.

4. Dataset
This project uses the "Air Quality Data in India (2015 - 2020)" dataset, which is publicly available on Kaggle. It contains daily and hourly AQI measurements from numerous stations across India.

Source: Kaggle Dataset Link

Primary File Used: city_day.csv

5. Setup & How to Run Locally
To run this project on your local machine, please follow these steps:

Clone the Repository:

git clone [https://github.com/your-username/AQI-Forecasting-App.git](https://github.com/your-username/AQI-Forecasting-App.git)
cd AQI-Forecasting-App

Install Dependencies:
Make sure you have Python 3.8+ installed. Then, install the required libraries from the requirements.txt file.

pip install -r requirements.txt

Place the Dataset:
Download the city_day.csv file from the Kaggle link above and place it in the root directory of the project.

Run the Streamlit App:
Execute the following command in your terminal:

streamlit run app.py

The application will open in a new tab in your web browser.
