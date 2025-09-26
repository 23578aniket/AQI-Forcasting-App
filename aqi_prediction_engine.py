import pandas as pd
from prophet import Prophet
import logging
from typing import Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AQIPredictionEngine:
    """
    Handles loading historical AQI data and training a time-series forecasting
    model using Facebook's Prophet library. This version is optimized to use
    the pre-aggregated city_day.csv dataset.
    """

    def __init__(self, data_filepath: str):
        """
        Initializes the engine and loads the historical data.

        Args:
            data_filepath (str): The path to the historical AQI data CSV file (city_day.csv).
        """
        self.data_filepath = data_filepath
        self.data = self._load_and_prepare_data()
        self.models: Dict[str, Prophet] = {}  # Cache for trained models

    def _load_and_prepare_data(self) -> Optional[pd.DataFrame]:
        """
        Loads the historical AQI data from the city_day.csv and performs
        essential cleaning and preparation.
        """
        try:
            logger.info(f"Loading historical data from {self.data_filepath}...")
            # Use only the essential columns to save memory
            df = pd.read_csv(self.data_filepath, usecols=['City', 'Date', 'AQI'])

            # --- Data Cleaning and Preparation ---
            df['Date'] = pd.to_datetime(df['Date'])
            df['AQI'] = df['AQI'].ffill()
            df.dropna(subset=['AQI', 'City'], inplace=True)
            df['AQI'] = df['AQI'].astype(int)

            logger.info("Historical data loaded and prepared successfully.")
            return df
        except FileNotFoundError:
            logger.error(f"Error: The data file was not found at {self.data_filepath}")
            return None
        except Exception as e:
            logger.error(f"An error occurred while loading or preparing the data: {e}")
            return None

    def get_available_cities(self) -> list:
        """Returns a sorted list of unique cities available in the dataset."""
        if self.data is not None:
            return sorted(self.data['City'].unique().tolist())
        return []

    def train_model_for_city(self, city: str) -> Optional[Prophet]:
        """
        Trains a Prophet forecasting model for a specific city.
        Caches the model for future use.
        """
        if city in self.models:
            logger.info(f"Using cached model for {city}.")
            return self.models[city]

        if self.data is None:
            logger.error("Data not loaded. Cannot train model.")
            return None

        logger.info(f"Training new forecasting model for {city}...")

        city_data = self.data[self.data['City'] == city].copy()

        if len(city_data) < 730:  # Prophet works best with at least 2 years of data
            logger.warning(f"Warning: Insufficient data for {city} (less than 2 years). Forecast may be less reliable.")

        prophet_df = city_data[['Date', 'AQI']].rename(columns={'Date': 'ds', 'AQI': 'y'})

        model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
        model.fit(prophet_df)

        self.models[city] = model

        logger.info(f"Model for {city} trained successfully.")
        return model

    def get_forecast(self, city: str, periods: int = 30) -> Optional[pd.DataFrame]:
        """
        Generates a future AQI forecast for a given city.
        """
        model = self.train_model_for_city(city)
        if model is None:
            return None

        logger.info(f"Generating a {periods}-day forecast for {city}...")
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        logger.info("Forecast generated successfully.")
        # Return only essential columns and round the predicted values
        forecast['yhat'] = forecast['yhat'].round().astype(int)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


if __name__ == "__main__":
    # --- This block allows us to test the prediction engine directly ---

    # IMPORTANT: Ensure 'city_day.csv' is in the same directory as this script.
    engine = AQIPredictionEngine(data_filepath='city_day.csv')

    if engine.data is not None:
        available_cities = engine.get_available_cities()
        if available_cities:
            test_city = "Delhi"
            if test_city not in available_cities:
                test_city = available_cities[0]

            print(f"\n--- Testing AQI Prediction Engine for {test_city} ---")

            future_forecast = engine.get_forecast(test_city, periods=7)

            if future_forecast is not None:
                print("\n--- 7-Day AQI Forecast ---")
                print(future_forecast[['ds', 'yhat']].tail(7))
            else:
                print(f"Could not generate a forecast for {test_city}.")
        else:
            print("No cities available in the dataset.")

