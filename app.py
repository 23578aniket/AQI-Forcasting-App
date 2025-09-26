import streamlit as st
import pandas as pd
from aqi_prediction_engine import AQIPredictionEngine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="AQI Forecasting India",
    page_icon="💨",
    layout="wide"
)


# --- Caching the Prediction Engine ---
# This is a key Streamlit optimization. It prevents the app from reloading
# the large dataset and retraining models every time the user interacts with a widget.
@st.cache_resource
def load_engine():
    """Loads the AQIPredictionEngine and caches it."""
    logger.info("Initializing AQIPredictionEngine for the first time...")
    try:
        engine = AQIPredictionEngine(data_filepath='city_day.csv')
        # Pre-train a model for a common city to make initial load feel faster
        if engine.data is not None and "Delhi" in engine.get_available_cities():
            engine.train_model_for_city("Delhi")
        return engine
    except Exception as e:
        st.error(f"Fatal error during engine initialization: {e}")
        return None


# --- Main App UI ---
st.title("💨 Air Quality Index (AQI) Forecasting for Indian Cities")
st.markdown("Select a city and a forecast period to see the predicted AQI for the upcoming days.")

# Load the engine from the cache
engine = load_engine()

if engine and engine.data is not None:
    # --- Sidebar for User Inputs ---
    st.sidebar.header("Forecast Controls")

    # 1. City Selection
    available_cities = engine.get_available_cities()
    selected_city = st.sidebar.selectbox(
        "Select a City:",
        options=available_cities,
        index=available_cities.index("Delhi") if "Delhi" in available_cities else 0
    )

    # 2. Forecast Period Selection
    forecast_days = st.sidebar.slider(
        "Select Forecast Period (days):",
        min_value=7,
        max_value=90,
        value=30,
        step=7
    )

    # --- Main Panel for Displaying Results ---
    if st.sidebar.button("Generate Forecast", type="primary"):
        with st.spinner(f"Training model and forecasting for {selected_city}... This may take a moment."):
            try:
                # Generate the forecast
                forecast = engine.get_forecast(selected_city, periods=forecast_days)

                if forecast is not None:
                    st.success(f"Forecast for {selected_city} generated successfully!")

                    # --- Display Metrics ---
                    latest_actual_aqi = int(engine.data[engine.data['City'] == selected_city]['AQI'].iloc[-1])
                    predicted_tomorrow_aqi = int(forecast['yhat'].iloc[-forecast_days])

                    col1, col2 = st.columns(2)
                    col1.metric("Latest Recorded AQI", f"{latest_actual_aqi}")
                    col2.metric("Predicted AQI for Tomorrow", f"{predicted_tomorrow_aqi}")

                    # --- Display Chart ---
                    st.subheader(f"Predicted AQI for the next {forecast_days} days in {selected_city}")

                    # Prepare data for charting
                    chart_data = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted AQI'})
                    chart_data.set_index('Date', inplace=True)

                    st.line_chart(chart_data)

                    # --- Display Data Table ---
                    with st.expander("View Forecast Data"):
                        display_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted AQI'})
                        display_df['Date'] = display_df['Date'].dt.date
                        st.dataframe(display_df.tail(forecast_days))

                else:
                    st.error(
                        f"Could not generate a forecast for {selected_city}. The city may not have enough historical data for a reliable prediction.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

else:
    st.error(
        "Failed to load the AQI Prediction Engine. Please ensure the 'city_day.csv' file is in the correct directory and is not corrupted.")

st.sidebar.markdown("---")
st.sidebar.info(
    "This application uses a time-series model (Prophet) to forecast future AQI values based on historical data.")
