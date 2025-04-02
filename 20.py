import streamlit as st
import pandas as pd
import numpy as np
import dill
import logging
import re
import math
import os
import gdown  # NEW for Google Drive download
from typing import Dict, Tuple
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from xgboost import XGBRegressor
from dataclasses import dataclass

# 🧠 Define your ItemMetadata class
@dataclass
class ItemMetadata:
    category: str
    unit: str
    is_veg: bool

# 🔧 Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FoodPrediction")

# ✅ Load model from Google Drive
@st.cache_resource
def load_model():
    file_id = ""  # 🔁 Replace with your actual Google Drive file ID
    output_path = "adouble.dill"

    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, output_path, quiet=False)
            st.success("✅ Model downloaded from Google Drive.")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            st.error(f"❌ Could not download model from Google Drive: {e}")
            return None

    try:
        with open(output_path, 'rb') as f:
            predictor = dill.load(f)
        logger.info("✅ Model loaded successfully")
        predictor.ItemMetadata = ItemMetadata
        return predictor
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error(f"❌ Failed to load model: {e}")
        return None

# 📦 Extract quantity and unit from string
def extract_quantity_and_unit(qty_str: str) -> Tuple[float, str]:
    try:
        qty_val = float(''.join(filter(lambda x: x.isdigit() or x == '.', qty_str)))
        unit = ''.join(filter(str.isalpha, qty_str))
        return qty_val, unit
    except Exception as e:
        logger.warning(f"Failed to extract quantity and unit from '{qty_str}': {e}")
        return 0.0, "g"

# 🚀 Main App
def main():
    st.title("🍽️ Food Quantity & Price Prediction System")

    predictor = load_model()
    if predictor is None:
        st.stop()

    # 🎛️ Sidebar Inputs
    with st.sidebar:
        st.header("📋 Event Details")
        event_time = st.selectbox("Event Time", ["Morning", "Afternoon", "Evening", "Night"])
        meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner", "Hi-Tea"])
        event_type = st.selectbox("Event Type", ["Wedding", "Birthday Party", "Corporate Event", "Other"])
        guest_count = st.number_input("Number of Guests", min_value=1, value=10, step=1)

        st.header("🍱 Menu Items")
        menu_input = st.text_area("Enter menu items (one per line)", height=200)
        predict_button = st.button("Predict Quantities and Prices")

    if predict_button and menu_input:
        menu_items = [item.strip() for item in menu_input.split("\n") if item.strip()]
        if not menu_items:
            st.warning("Please enter at least one menu item.")
            return

        with st.spinner("⏳ Calculating predictions..."):
            try:
                predictions = predictor.predict(event_time, meal_type, event_type, guest_count, menu_items)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                return

        # 🧮 Build results
        table_data = []
        total_event_cost = 0

        for item, qty_str in predictions.items():
            total_qty_val, unit = extract_quantity_and_unit(qty_str)
            std_item = predictor.standardize_Item_name(item)
            mapped_item = getattr(predictor, 'Item_name_mapping', {}).get(std_item, item)

            if mapped_item in predictor.item_metadata:
                category = predictor.item_metadata[mapped_item].category
            else:
                category = predictor.guess_item_category(item)
                unit_inferred = predictor.food_rules.infer_default_unit(category)
                predictor.item_metadata[mapped_item] = predictor.ItemMetadata(category=category, unit=unit_inferred, is_veg=True)
                predictor.item_metadata[std_item] = predictor.ItemMetadata(category=category, unit=unit_inferred, is_veg=True)

            try:
                total_price, _, price_per_person = predictor.calculate_price(
                    total_qty_val, category, guest_count, item, predicted_unit=unit
                )
            except Exception as e:
                st.warning(f"Price calculation failed for {item}: {e}")
                total_price, price_per_person = 0, 0

            total_event_cost += total_price
            per_person_qty = total_qty_val / guest_count if guest_count > 0 else 0
            per_person_weight = f"{per_person_qty:.2f}{unit}"

            table_data.append({
                "Item": item,
                "Quantity": qty_str,
                "Per Person Weight": per_person_weight,
                "Per Person Price": f"₹{price_per_person:.2f}",
                "Total Price": f"₹{total_price:.2f}",
            })

        # 📊 Display results
        st.header("📊 Prediction Results")
        df = pd.DataFrame(table_data)
        st.table(df)

        # 📋 Event Summary
        st.header("📋 Event Summary")
        price_per_guest = total_event_cost / guest_count if guest_count > 0 else 0
        st.write(f"**Total Event Cost:** ₹{total_event_cost:.2f}")
        st.write(f"**Price per Guest:** ₹{price_per_guest:.2f}")

    elif predict_button:
        st.warning("Please enter menu items before predicting.")

if __name__ == "__main__":
    main()
