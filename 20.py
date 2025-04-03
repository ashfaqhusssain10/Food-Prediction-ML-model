import streamlit as st
import pandas as pd
import numpy as np
import dill
import logging
import re
import math
from typing import Dict, Tuple
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from xgboost import XGBRegressor
from dataclasses import dataclass

# Define ItemMetadata class to match ramp.py
@dataclass
class ItemMetadata:
    category: str
    unit: str
    is_veg: bool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FoodPrediction")

# Load the model from the .dill file
@st.cache_resource
def load_model(filename: str = "adouble.dill") -> object:
    try:
        with open(filename, 'rb') as f:
            predictor = dill.load(f)
        logger.info(f"Model loaded successfully from {filename}")
        st.success(f"Model loaded successfully from {filename}")
        predictor.ItemMetadata = ItemMetadata
        return predictor
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error(f"Failed to load model: {e}")
        return None

# Function to extract quantity and unit
def extract_quantity_and_unit(qty_str: str) -> Tuple[float, str]:
    try:
        qty_val = float(''.join(filter(lambda x: x.isdigit() or x == '.', qty_str)))
        unit = ''.join(filter(str.isalpha, qty_str))
        return qty_val, unit
    except Exception as e:
        logger.warning(f"Failed to extract quantity and unit from '{qty_str}': {e}")
        return 0.0, "g"

# Main Streamlit app
def main():
    st.title("Food Quantity and Price Prediction System")

    # Load the predictor model
    predictor = load_model()
    if predictor is None:
        st.error("Cannot proceed without a loaded model.")
        return

    # Sidebar for inputs
    with st.sidebar:
        st.header("Event Details")
        event_time = st.selectbox("Event Time", ["Morning", "Afternoon", "Evening", "Night"])
        meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner", "Hi-Tea"])
        event_type = st.selectbox("Event Type", ["Wedding", "Birthday Party", "Corporate Event", "Other"])
        guest_count = st.number_input("Number of Guests", min_value=1, value=10, step=1)

        st.header("Menu Items")
        menu_input = st.text_area("Enter menu items (one per line)", height=200)
        predict_button = st.button("Predict Quantities and Prices")

    # Main content area for outputs
    if predict_button and menu_input:
        menu_items = [item.strip() for item in menu_input.split("\n") if item.strip()]
        if not menu_items:
            st.warning("Please enter at least one menu item.")
            logger.warning("No menu items provided by user")
            return

        # Get predictions
        with st.spinner("Calculating predictions..."):
            try:
                logger.info(f"Starting prediction for {len(menu_items)} items")
                predictions = predictor.predict(event_time, meal_type, event_type, guest_count, menu_items)
                logger.info("Prediction completed successfully")
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                st.error(f"Error during prediction: {e}")
                return

        # Prepare data for table
        table_data = []
        total_event_cost = 0

        for item, qty_str in predictions.items():
            total_qty_val, unit = extract_quantity_and_unit(qty_str)
            std_item = predictor.standardize_Item_name(item)
            mapped_item = getattr(predictor, 'Item_name_mapping', {}).get(std_item, item)

            # Determine category (needed for internal logic, but not displayed)
            if mapped_item in predictor.item_metadata:
                category = predictor.item_metadata[mapped_item].category
            else:
                category = predictor.guess_item_category(item)
                unit_inferred = predictor.food_rules.infer_default_unit(category)
                predictor.item_metadata[mapped_item] = predictor.ItemMetadata(category=category, unit=unit_inferred, is_veg=True)
                predictor.item_metadata[std_item] = predictor.ItemMetadata(category=category, unit=unit_inferred, is_veg=True)
                logger.info(f"Assigned category '{category}' to unmapped item '{item}'")

            # Calculate prices
            try:
                total_price, base_price_per_unit, price_per_person = predictor.calculate_price(
                    total_qty_val, category, guest_count, item, predicted_unit=unit
                )
                logger.info(f"Price calculated for {item}: Total=₹{total_price:.2f}, Per Person=₹{price_per_person:.2f}")
            except Exception as e:
                logger.warning(f"Price calculation failed for {item}: {e}. Using default values.")
                st.warning(f"Price calculation failed for {item}: {e}")
                total_price, base_price_per_unit, price_per_person = 0, 0, 0

            total_event_cost += total_price

            # Per person quantity
            per_person_qty = total_qty_val / guest_count if guest_count > 0 else 0
            per_person_weight = f"{per_person_qty:.2f}{unit}"

            # Add to table data (exclude Category)
            table_data.append({
                "Item": item,
                "Quantity": qty_str,
                "Per Person Weight": per_person_weight,
                #"Per Person Price": f"₹{price_per_person:.2f}",
                "Per Person Price": f"₹{(total_price / guest_count):.2f}",
                "Total Price": f"₹{total_price:.2f}",
            })

        # Display results in the middle
        st.header("Prediction Results")

        # Display a single table with all items
        df = pd.DataFrame(table_data)
        st.table(df)  # Show all items in one table, with "Total" as the last column

        # Overall event summary
        st.header("Event Summary")
        price_per_guest = total_event_cost / guest_count if guest_count > 0 else 0
        st.write(f"*Total Event Cost:* ₹{total_event_cost:.2f}")
        st.write(f"*Price per Guest:* ₹{price_per_guest:.2f}")

        logger.info(f"Event summary: Total Cost=₹{total_event_cost:.2f}, Price per Guest=₹{price_per_guest:.2f}")

    elif predict_button and not menu_input:
        st.warning("Please enter menu items before predicting.")
        logger.warning("Prediction button clicked without menu items")

if _name_ == "_main_":
    main()

