import math
import pandas as pd
import numpy as np
import logging
import re
import warnings
#import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from tempfile import NamedTemporaryFile
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from xgboost import XGBRegressor
import dill



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FoodPrediction")


#####################################
# PART 1: MODEL IMPLEMENTATION CODE #
#####################################
class FoodItemMatcher:
    """Two-tier hash table system for efficient food item matching."""

    def __init__(self, item_metadata=None):
        self.direct_lookup = {}  # Tier 1: Direct lookup hash table
        self.token_to_items = {}  # Tier 2: Token-based lookup
        self.category_items = {}  # Items organized by category

        if item_metadata:
            self.build_hash_tables(item_metadata)

    def build_hash_tables(self, item_metadata):
        """Build the two-tier hash table system from item metadata."""
        # Reset tables
        self.direct_lookup = {}
        self.token_to_items = {}
        self.category_items = {}

        # Process each item
        for item_name, metadata in item_metadata.items():
            if not item_name or pd.isna(item_name):
                continue

            # Standardize name
            std_name = self._standardize_name(item_name)

            # Add to direct lookup (Tier 1)
            self.direct_lookup[std_name] = item_name

            # Add common variants
            variants = self._generate_variants(std_name)
            for variant in variants:
                if variant not in self.direct_lookup:
                    self.direct_lookup[variant] = item_name

            # Add to token lookup (Tier 2)
            tokens = self._tokenize(std_name)
            for token in tokens:
                if token not in self.token_to_items:
                    self.token_to_items[token] = []
                if item_name not in self.token_to_items[token]:
                    self.token_to_items[token].append(item_name)

            # Add to category items
            category = metadata.category
            if category not in self.category_items:
                self.category_items[category] = []
            self.category_items[category].append(item_name)

    def find_item(self, query_item, item_metadata):
        """Find a food item using the two-tier hash table approach."""
        if not query_item or pd.isna(query_item):
            return None, None

        # Standardize query - handle '>' prefix automatically
        std_query = self._standardize_name(query_item)
        std_query = std_query.replace('> ', '').replace('>', '')  # Remove '>' prefixes

        # Tier 1: Try direct lookup first (fast path)
        if std_query in self.direct_lookup:
            item_name = self.direct_lookup[std_query]
            if item_name in item_metadata:
                return item_name, item_metadata[item_name]

        # Additional direct lookups for common variations
        # Try with spaces removed
        compact_query = std_query.replace(' ', '')
        for key in self.direct_lookup:
            compact_key = key.replace(' ', '')
            if compact_query == compact_key:
                item_name = self.direct_lookup[key]
                if item_name in item_metadata:
                    return item_name, item_metadata[item_name]

        # Tier 2: Enhanced token-based lookup
        tokens = self._tokenize(std_query)
        if tokens:
            # Find candidates with improved scoring
            candidates = {}
            for token in tokens:
                # Handle token variations (plurals, singulars)
                token_variants = [token]
                if token.endswith('s'):
                    token_variants.append(token[:-1])  # Remove 's' for plurals
                elif len(token) > 3:
                    token_variants.append(token + 's')  # Add 's' for singulars

                for variant in token_variants:
                    if variant in self.token_to_items:
                        for item_name in self.token_to_items[variant]:
                            if item_name not in candidates:
                                candidates[item_name] = 0
                            candidates[item_name] += 1

            # Enhance scoring with additional contextual factors
            scored_candidates = []
            for item_name, token_matches in candidates.items():
                if item_name in item_metadata:
                    item_tokens = self._tokenize(self._standardize_name(item_name))
                    if not item_tokens:
                        continue

                    # Basic token match score
                    token_score = token_matches / max(len(tokens), len(item_tokens))

                    # Enhanced substring matching
                    contains_score = 0
                    if std_query in self._standardize_name(item_name):
                        contains_score = 0.8
                    elif self._standardize_name(item_name) in std_query:
                        contains_score = 0.6

                    # Word overlap score (considering word position)
                    word_overlap = 0
                    std_query_words = std_query.split()
                    item_words = self._standardize_name(item_name).split()
                    for i, qword in enumerate(std_query_words):
                        for j, iword in enumerate(item_words):
                            if qword == iword:
                                # Words in same position get higher score
                                pos_factor = 1.0 - 0.1 * abs(i - j)
                                word_overlap += pos_factor

                    if len(std_query_words) > 0:
                        word_overlap_score = word_overlap / len(std_query_words)
                    else:
                        word_overlap_score = 0

                    # Combined score with weights
                    final_score = max(token_score * 0.4 + contains_score * 0.4 + word_overlap_score * 0.2,
                                      contains_score)

                    scored_candidates.append((item_name, final_score))

            # Sort by score and get best match
            if scored_candidates:
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                best_match = scored_candidates[0]

                # Lower threshold for matching (0.4 instead of 0.5)
                if best_match[1] >= 0.4:
                    return best_match[0], item_metadata[best_match[0]]

        return None, None

    def _standardize_name(self, name):
        """Standardize item name for matching."""
        if pd.isna(name):
            return ""
        name = str(name).strip().lower()
        name = " ".join(name.split())  # Normalize whitespace
        return name

    def _tokenize(self, text):
        """Split text into tokens for matching."""
        if not text:
            return []

        # Simple tokenization by whitespace
        tokens = text.split()

        # Remove very common words and short tokens
        stop_words = {"and", "with", "the", "in", "of", "a", "an"}
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

        return tokens

    def _generate_variants(self, name):
        """Generate common variants of a food item name."""
        variants = []

        # Common misspellings and variations
        replacements = {
            "biryani": ["biriyani", "briyani", "biryani"],
            "chicken": ["chiken", "chikken", "checken"],
            "paneer": ["panner", "panir", "pnr"],
            "masala": ["masala", "masaala", "masalla"]
        }

        # Generate simple word order variations
        words = name.split()
        if len(words) > 1:
            # Add reversed order for two-word items
            variants.append(" ".join(reversed(words)))

        # Apply common spelling variations
        for word, alternatives in replacements.items():
            if word in name:
                for alt in alternatives:
                    variant = name.replace(word, alt)
                    variants.append(variant)

        return variants






class FoodCategoryRules:
    """
    Class to manage all food category rules, dependencies, and modifiers
    """

    def __init__(self):
        logger.info("Initializing FoodCategoryRules")
        self.category_rules = self._initialize_category_rules()
        self.category_dependencies = self._initialize_category_dependencies()
        self.meal_type_modifiers = self._initialize_meal_type_modifiers()

    def _initialize_category_rules(self) -> Dict[str, Dict[str, Any]]:
        # Added missing categories like Breakfast, Sandwich, Cup Cakes
        return {
            "Welcome_Drinks": {
                "min_quantity": "100ml",
                "max_quantity": "120ml",
                "default_quantity": "120ml",
                "vc_price": 180,
                "adjustments": {
                    "large_event": lambda guest_count: "120ml" if guest_count > 50 else "120ml"
                }
            },
            "Biryani": {
                "min_quantity": "250g",
                "max_quantity": "450g",
                "default_quantity": "300g",
                "vc_price": 300,
                "adjustments": {
                    "per_person": lambda guest_count: "250g" if guest_count > 100 else "300g",
                    "multiple_varieties": lambda count: "250g" if count > 2 else "300g",
                    "total_items": lambda total_items: "450g" if total_items <= 3 else (
                        "320g" if total_items >= 3 else "300g")
                }
            },
            "Salad": {
                "min_quantity": "50g",
                "max_quantity": "50g",
                "default_quantity": "50g",
                "vc_price": 70,
            },
            "Podi": {
                "min_quantity": "10g",
                "max_quantity": "10g",
                "default_quantity": "10g",
                "vc_price": 100,
            },
            "Fried_Items": {
                "min_quantity": "40g",
                "max_quantity": "60g",
                "default_quantity": "50g",
                "vc_price": 100,
            },
            "Ghee": {
                "min_quantity": "3g",
                "max_quantity": "5g",
                "default_quantity": "3g",
                "vc_price": 150,
            },
            "Pickle": {
                "min_quantity": "10g",
                "max_quantity": "10g",
                "default_quantity": "10g",
                "vc_price": 250,
            },
            "Flavored_Rice": {
                "min_quantity": "80g",
                "max_quantity": "100g",
                "default_quantity": "100g",
                "vc_price": 150,
                "adjustments": {
                    "variety_count": lambda count: {
                        1: "100g",
                        2: "80g",
                        3: "80g",
                        4: "80g",
                        5: "80g",
                    }.get(count, "80g")
                }
            },
            "Soups": {
                "min_quantity": "120ml",
                "max_quantity": "120ml",
                "default_quantity": "120ml",
                "vc_price": 150,
            },
            "Crispers": {
                "min_quantity": "5g",
                "max_quantity": "5g",
                "default_quantity": "5g",
                "vc_price": 40,
            },
            "Fried_Rice": {
                "min_quantity": "80g",
                "max_quantity": "120g",
                "default_quantity": "80g",
                "vc_price": 70,
                "adjustments": {
                    "per_person": lambda guest_count: "100g" if guest_count > 100 else "100g",
                    "multiple_varieties": lambda count: "80g" if count > 3 else "100g"
                }
            },
            "Fry": {
                "min_quantity": "30g",
                "max_quantity": "40g",
                "default_quantity": "30g",
                "vc_price": 60,
            },
            "Fryums": {
                "min_quantity": "5g",
                "max_quantity": "10g",
                "default_quantity": "5g",
                "vc_price": 10,
            },
            "Salan": {
                "min_quantity": "40g",
                "max_quantity": "40g",
                "default_quantity": "40g",
                "vc_price": 80,
            },
            "Cakes": {
                "min_quantity": "500g",
                "max_quantity": "1000g",
                "default_quantity": "500g",
                "vc_price": 400,
                "adjustments": {
                    "per_person": lambda guest_count: "500g" if guest_count > 30 else "1000g"
                }
            },
            "Cup_Cakes": {
                "min_quantity": "50g",
                "max_quantity": "100g",
                "default_quantity": "50g",
                "vc_price": 200,
            },
            "Hot_Beverages": {
                "min_quantity": "100ml",
                "max_quantity": "120ml",
                "default_quantity": "120ml",
                "vc_price": 150,
            },
            "Pulav": {
                "min_quantity": "250g",
                "max_quantity": "450g",
                "default_quantity": "300g",
                "vc_price": 300,
                "adjustments": {
                    "per_person": lambda guest_count: "250g" if guest_count > 100 else "300g",
                    "multiple_varieties": lambda count: "250g" if count > 2 else "300g"
                }
            },
            "Appetizers": {
                "min_quantity": "80g",
                "max_quantity": "120g",
                "default_quantity": "100g",
                "vc_price": 270,
                "adjustments": {
                    "variety_count": lambda count: {
                        1: "120g",
                        2: "100g",
                        3: "80g",
                        4: "60g",
                        5: "50g",
                        6: "50g",
                        7: "50g",
                        8: "50g",
                        9: "50g",
                        10: "50g"
                    }.get(count, "80g")
                },
                "by_weight": {
                    "default_quantity": "100g",
                    "adjustments": {
                        "variety_count": lambda count: {
                            1: "120g", 2: "100g", 3: "80g", 4: "60g", 5: "50g"
                        }.get(count, "80g")
                    }
                },
                "by_pieces": {
                    "default_quantity": "2pcs",
                    "adjustments": {
                        "variety_count": lambda count: {
                            1: "3pcs", 2: "2pcs", 3: "2pcs", 4: "1pcs", 5: "1pcs"
                        }.get(count, "1pcs")
                    }
                }

            },
            "Roti_Pachadi": {
                "min_quantity": "15g",
                "max_quantity": "20g",
                "default_quantity": "15g",
                "vc_price": 80,
            },
            "Curries": {
                "min_quantity": "120g",
                "max_quantity": "180g",
                "default_quantity": "120g",
                "vc_price": 270,
                "adjustments": {
                    "variety_count": lambda count: {
                        1: "120g",
                        2: "100g",
                        3: "80g",
                        4: "80g",
                        5: "80g",
                        6: "80g",
                    }.get(count, "100g")
                }
            },
            "Rice": {
                "min_quantity": "150g",
                "max_quantity": "250g",
                "default_quantity": "200g",
                "vc_price": 80,
                "adjustments": {
                    "with_curry": lambda: "200g",
                    "standalone": lambda: "250g"
                }
            },
            "Liquids(Less_Dense)": {
                "min_quantity": "60ml",
                "max_quantity": "100ml",
                "default_quantity": "70ml",
                "vc_price": 100,
                "adjustments": {
                    "with_dal": lambda: "60ml",
                    "standalone": lambda: "100ml"
                }
            },
            "Liquids(High_Dense)": {
                "min_quantity": "30ml",
                "max_quantity": "30ml",
                "default_quantity": "30ml",
                "vc_price": 160,
            },
            "Dal": {
                "min_quantity": "60g",
                "max_quantity": "80g",
                "default_quantity": "70g",
                "vc_price": 120,
            },
            "Desserts": {
                "min_quantity": "80g",
                "max_quantity": "120g",
                "default_quantity": "100g",
                "vc_price": 170,
                "adjustments": {
                    "variety_count": lambda count: "80g" if count > 2 else "100g"
                },
                # Add special handling structures
                "by_weight": {
                    "default_quantity": "100g",
                    "adjustments": {
                        "variety_count": lambda count: "80g" if count > 2 else "100g"
                    }
                },
                "by_pieces": {
                    "default_quantity": "1pcs",
                    #"adjustments": {
                     #   "variety_count": lambda count: "1pcs" if count > 2 else "2pcs"
                    #}
                }
            },
            "Curd": {
                "min_quantity": "50g",
                "max_quantity": "60g",
                "default_quantity": "50g",
                "vc_price": 50,
            },
            "Fruits": {
                "min_quantity": "100g",
                "max_quantity": "120g",
                "default_quantity": "100g",
                "vc_price": 150,
            },
            "Paan": {
                "min_quantity": "1pcs",
                "max_quantity": "1pcs",
                "default_quantity": "1pcs",
                "vc_price": 50,
            },
            "Omlette": {
                "min_quantity": "1pcs",
                "max_quantity": "1pcs",
                "default_quantity": "1pcs",
                "vc_price": 150,
            },
            "Bread": {
                "min_quantity": "1pcs",
                "max_quantity": "2pcs",
                "default_quantity": "1pcs",
                "vc_price": 40,
                "adjustments": {
                    "with_curry": lambda: "1pcs"
                }
            },
            "Italian": {
                "min_quantity": "100g",
                "max_quantity": "120g",
                "default_quantity": "100g",
                "vc_price": 150,
                "adjustments": {
                    "with_pasta": lambda: "120g"
                }
            },
            "Pizza": {
                "min_quantity": "100g",
                "max_quantity": "120g",
                "default_quantity": "100g",
                "vc_price": 200,
            },
            "Raitha": {
                "min_quantity": "50g",
                "max_quantity": "60g",
                "default_quantity": "60g",
                "vc_price": 500,
                "adjustments": {
                    "with_biryani": lambda: "70g",
                    "standalone": lambda: "50g"
                }
            },
            "Breakfast": {
                "min_quantity": "100g",
                "max_quantity": "150g",
                "default_quantity": "100g",
                "vc_price": 150,
                "by_weight": {
                    "default_quantity": "120g"
                },
                "by_pieces": {
                    "default_quantity": "4pcs",
                    "adjustments": {
                        "variety_count": lambda count: "2pcs" if count > 2 else "4pcs"
                    }
                }
            },
            "Sandwich": {
                "min_quantity": "100g",
                "max_quantity": "150g",
                "default_quantity": "100g",
                "vc_price": 150,
            }
        }

    def _initialize_category_dependencies(self) -> Dict[str, List[str]]:
        return {
            "Pulav": ["Raita", "Salad"],
            "Rice": ["Curries", "Liquids(Less_Dense)", "Curd"],
            "Breads": ["Curries"],
            "Curries": ["Rice", "Breads"],
            "Dal": ["Rice", "Breads"],
            "Salan": ["Biryani", "Pulav"],
            "Curd": ["Rice"],
            "Rasam": ["Rice", "Dal"],
            "Chutneys": ["Rice"],
            "Roti_Pachadi": ["Rice"],
            "Pickles": ["Rice", "Flavoured_Rice"],
            "Podi": ["Rice", "Flavoured_Rice"]
        }

    def _initialize_meal_type_modifiers(self) -> Dict[str, float]:
        return {
            "Breakfast": 0.8,
            "Lunch": 1.0,
            "Hi-Tea": 0.9,
            "Dinner": 1.0
        }

    def extract_quantity_value(self, quantity_str: str) -> float:
        if pd.isna(quantity_str):
            return 0.0
        match = re.search(r'(\d+\.?\d*)', str(quantity_str))
        if match:
            return float(match.group(1))
        return 0.0

    def extract_unit(self, quantity_str: str) -> str:
        if pd.isna(quantity_str):
            return ''
        match = re.search(r'[a-zA-Z]+', str(quantity_str))
        if match:
            return match.group(0)
        return ''

    def infer_default_unit(self, category: str) -> str:
        """Infer the default unit for a category if not specified in the data."""
        liquid_categories = ["Liquids(Less_Dense)", "Liquids(High_Dense)", "Soups", "Welcome_Drinks", "Hot_Beverages"]
        piece_categories = ["Breads", "Paan", "Omlette"]
        if category in liquid_categories:
            return "ml"
        elif category in piece_categories:
            return "pcs"
        else:
            return "g"

    def validate_unit(self, category: str, unit: str) -> str:
        """Validate and correct the unit based on the category."""
        liquid_categories = ["Liquids(Less_Dense)", "Liquids(High_Dense)", "Soups", "Welcome_Drinks", "Hot_Beverages"]
        piece_categories = ["Breads", "Paan", "Omlette"]
        if not unit:
            inferred_unit = self.infer_default_unit(category)
            return inferred_unit
        if category in liquid_categories and unit not in ["ml", "l"]:
            logger.warning(f"Invalid unit '{unit}' for liquid category '{category}', defaulting to 'ml'")
            return "ml"
        elif category in piece_categories and unit != "pcs":
            logger.warning(f"Invalid unit '{unit}' for piece category '{category}', defaulting to 'pcs'")
            return "pcs"
        elif unit not in ["g", "kg", "ml", "l", "pcs"]:
            logger.warning(f"Invalid unit '{unit}' for category '{category}', defaulting to 'g'")
            return "g"
        return unit

    def normalize_category_name(self, category: str) -> str:
        return category.replace(" ", "_").strip()

    def get_default_quantity(self, category: str) -> Tuple[float, str]:
        normalized_category = self.normalize_category_name(category)
        if normalized_category in self.category_rules:
            rule = self.category_rules[normalized_category]
            default_qty_str = rule["default_quantity"]
            default_qty = self.extract_quantity_value(default_qty_str)
            unit = self.extract_unit(default_qty_str)
            unit = self.validate_unit(normalized_category, unit)
            return default_qty, unit
        inferred_unit = self.infer_default_unit(normalized_category)
        return 0.0, inferred_unit


    def apply_category_rules(self, category: str, guest_count: int, item_count: int, **kwargs) -> Tuple[float, str]:
        """
        Apply category rules to determine per-person quantity with proper unit-type handling

        Parameters:
        category: The food category
        guest_count: Number of guests
        item_count: Number of items in this category
        **kwargs: Additional parameters including:
            - unit_type: Unit type of the item ('pcs', 'g', 'ml')
            - total_items: Total number of menu items (optional)
            - meal_type: Type of meal (optional)
            - has_biryani, has_rice, has_curry, has_dal: Dependency flags

        Returns:
        Tuple of (quantity_value, unit)
        """
        normalized_category = self.normalize_category_name(category)
        if normalized_category not in self.category_rules:
            qty, unit = self.get_default_quantity(normalized_category)
            return qty, unit

        rule = self.category_rules[normalized_category]

        # Special handling for dual-structure categories with explicit unit type
        if "unit_type" in kwargs:
            unit_type = kwargs.get("unit_type")
            item_name = kwargs.get("item_name", "unknown")
            special_categories = ["Appetizers", "Desserts", "Breakfast"]

            if normalized_category in special_categories and unit_type in ["g", "ml", "pcs"]:
                # Select appropriate sub-structure
                sub_key = "by_weight" if unit_type in ["g", "ml"] else "by_pieces"

                if sub_key in rule:
                    sub_rule = rule[sub_key]
                    # Get default quantity
                    default_qty_str = sub_rule.get("default_quantity", rule["default_quantity"])
                    default_qty = self.extract_quantity_value(default_qty_str)
                    unit = self.extract_unit(default_qty_str) or unit_type

                    # Apply adjustments if available
                    adjusted_qty = default_qty
                    if "adjustments" in sub_rule and "variety_count" in sub_rule["adjustments"]:
                        try:
                            adj_result = sub_rule["adjustments"]["variety_count"](item_count)
                            if isinstance(adj_result, dict):
                                adj_qty_str = adj_result.get(item_count, default_qty_str)
                            else:
                                adj_qty_str = adj_result
                            adjusted_qty = self.extract_quantity_value(adj_qty_str)
                        except Exception as e:
                            logger.error(f"Error applying special category adjustment: {e}")

                    logger.info(
                        f"Special category {category} with unit {unit_type}: adjusted qty = {adjusted_qty}{unit}")
                    # Return early with special handling result
                    return adjusted_qty, unit

        # Standard rule processing
        default_qty_str = rule["default_quantity"]
        default_qty = self.extract_quantity_value(default_qty_str)
        unit = self.extract_unit(default_qty_str)
        unit = self.validate_unit(normalized_category, unit)
        adjusted_qty = default_qty

        # Apply standard adjustments if available
        if "adjustments" in rule:
            # Large event adjustment
            if "large_event" in rule["adjustments"]:
                adjusted_qty_str = rule["adjustments"]["large_event"](guest_count)
                adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
                logger.info(f"Category {category}: Large event adjustment = {adjusted_qty_str}")

            # Per-person adjustment
            if "per_person" in rule["adjustments"]:
                adjusted_qty_str = rule["adjustments"]["per_person"](guest_count)
                adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
                logger.info(f"Category {category}: Per-person adjustment = {adjusted_qty_str}")

            # Multiple varieties adjustment
            if "multiple_varieties" in rule["adjustments"]:
                adjusted_qty_str = rule["adjustments"]["multiple_varieties"](item_count)
                adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
                logger.info(f"Category {category}: Multiple varieties adjustment ({item_count}) = {adjusted_qty_str}")

            # Variety count adjustment
            if "variety_count" in rule["adjustments"]:
                adjusted_qty_str = rule["adjustments"]["variety_count"](item_count)
                adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
                logger.info(f"Category {category}: Variety count adjustment ({item_count}) = {adjusted_qty_str}")

            # Total items adjustment
            if "total_items" in rule["adjustments"] and "total_items" in kwargs:
                total_items = kwargs.get("total_items")
                adjusted_qty_str = rule["adjustments"]["total_items"](total_items)
                adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
                logger.info(f"Category {category}: Total items adjustment ({total_items}) = {adjusted_qty_str}")

            # Dependency adjustments
            # With biryani
            if "with_biryani" in rule["adjustments"] and kwargs.get("has_biryani", False):
                adjusted_qty_str = rule["adjustments"]["with_biryani"]()
                adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
                logger.info(f"Category {category}: With Biryani adjustment = {adjusted_qty_str}")

            # With rice
            if "with_rice" in rule["adjustments"] and kwargs.get("has_rice", False):
                adjusted_qty_str = rule["adjustments"]["with_rice"]()
                adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
                logger.info(f"Category {category}: With Rice adjustment = {adjusted_qty_str}")

            # With curry
            if "with_curry" in rule["adjustments"] and kwargs.get("has_curry", False):
                adjusted_qty_str = rule["adjustments"]["with_curry"]()
                adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
                logger.info(f"Category {category}: With Curry adjustment = {adjusted_qty_str}")

            # With dal
            if "with_dal" in rule["adjustments"] and kwargs.get("has_dal", False):
                adjusted_qty_str = rule["adjustments"]["with_dal"]()
                adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
                logger.info(f"Category {category}: With Dal adjustment = {adjusted_qty_str}")

        # Apply meal type modifier if specified
        meal_type = kwargs.get("meal_type")
        if meal_type and meal_type in self.meal_type_modifiers:
            orig_qty = adjusted_qty
            adjusted_qty *= self.meal_type_modifiers[meal_type]
            logger.info(f"Category {category}: Meal type ({meal_type}) adjustment: {orig_qty:.2f} → {adjusted_qty:.2f}")

        return adjusted_qty, unit

    def apply_dependency_rules(self, category: str, dependent_category: str, current_qty: float, unit: str) -> float:
        normalized_category = self.normalize_category_name(category)
        if normalized_category not in self.category_rules:
            return current_qty
        rule = self.category_rules[normalized_category]
        if "adjustments" not in rule:
            return current_qty
        adjustments = rule["adjustments"]
        dependent_normalized = dependent_category.lower()

        if dependent_normalized == "breads" and "with_breads" in adjustments:
            adjusted_qty_str = adjustments["with_breads"]()
            adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
            logger.info(
                f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty
        elif dependent_normalized == "rice" and "with_rice" in adjustments:
            adjusted_qty_str = adjustments["with_rice"]()
            adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
            logger.info(
                f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty
        elif dependent_normalized == "curries" and "with_curry" in adjustments:
            adjusted_qty_str = adjustments["with_curry"]()
            adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
            logger.info(
                f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty
        elif dependent_normalized == "biryani" and "with_biryani" in adjustments:
            adjusted_qty_str = adjustments["with_biryani"]()
            adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
            logger.info(
                f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty
        elif dependent_normalized == "dal" and "with_dal" in adjustments:
            adjusted_qty_str = adjustments["with_dal"]()
            adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
            logger.info(
                f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty
        elif dependent_normalized == "rasam" and "with_rasam" in adjustments:
            adjusted_qty_str = adjustments["with_rasam"]()
            adjusted_qty = self.extract_quantity_value(adjusted_qty_str)
            logger.info(
                f"Category {category}: Adjusted for {dependent_category} from {current_qty:.2f} to {adjusted_qty}")
            return adjusted_qty

        return current_qty

    def apply_meal_type_modifier(self, meal_type: str, qty: float) -> float:
        if meal_type in self.meal_type_modifiers:
            modifier = self.meal_type_modifiers[meal_type]
            return qty * modifier
        return qty

    def get_dependent_categories(self, category: str) -> List[str]:
        normalized_category = self.normalize_category_name(category)
        if normalized_category in self.category_dependencies:
            return self.category_dependencies[normalized_category]
        return []


@dataclass
class ItemMetadata:
    category: str
    unit: str
    conversion_factor: float = 1.0
    is_veg: bool = True # add vegetarian Status


class HierarchicalFoodPredictor:
    def __init__(self, category_constraints=None, item_vc_file=None,item_data_file=None):
        logger.info("Initializing HierarchicalFoodPredictor")
        self.category_models = {}
        self.item_models = {}
        self.category_scalers = {}
        self.item_scalers = {}
        self.event_time_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.meal_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.event_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.food_rules = FoodCategoryRules()
        self.custom_category_constraints = category_constraints or {}
        self.item_metadata = {}
        self.item_matcher = FoodItemMatcher()
        self.item_vc_mapping = {}
        self.item_specific_data={}
        self.calibration_config = {
            "global_default": {"ml": 0.2, "rule": 0.8},
            "categories": {
                # Main Course & Complex Items
                "Biryani": {"ml": 0.2, "rule": 0.8},  # Strong historical patterns
                "Curries": {"ml": 0.2, "rule": 0.8},  # Good historical data
                "Rice": {"ml": 0.2, "rule": 0.8},  # Standard, consistent patterns
                "Flavored_Rice": {"ml": 0.2, "rule": 0.8},  # Good patterns
                "Fried_Rice": {"ml": 0.2, "rule": 0.8},  # Good patterns

                # Appetizers & Sides
                "Appetizers": {"ml": 0.2, "rule": 0.8},  # Varied consumption
                "Fried_Items": {"ml": 0.2, "rule": 0.8},  # Similar to appetizers
                "Fry": {"ml": 0.2, "rule": 0.8},  # Similar to appetizers
                "Salad": {"ml": 0.2, "rule": 0.8},  # Balanced approach
                "Salan": {"ml": 0.1, "rule": 0.9},  # Balanced approach
                "Raitha": {"ml": 0.1, "rule": 0.9},  # Balanced approach

                # Soups & Liquids
                "Soups": {"ml": 0.55, "rule": 0.45},  # Fairly consistent
                "Liquids(Less_Dense)": {"ml": 0.5, "rule": 0.5},  # Complex rules
                "Liquids(High_Dense)": {"ml": 0.5, "rule": 0.5},  # Complex rules
                "Dal": {"ml": 0.55, "rule": 0.45},  # Fairly predictable

                # Condiments & Accompaniments - Rule-heavy
                "Chutneys": {"ml": 0.3, "rule": 0.7},  # Rule-driven
                "Podi": {"ml": 0.3, "rule": 0.7},  # Standard serving
                "Ghee": {"ml": 0.2, "rule": 0.8},  # Very standardized
                "Pickle": {"ml": 0.3, "rule": 0.7},  # Standard serving
                "Roti_Pachadi": {"ml": 0.3, "rule": 0.7},  # Standard serving
                "Curd": {"ml": 0.4, "rule": 0.6},  # Fairly standardized
                "Crispers": {"ml": 0.3, "rule": 0.7},  # Standard serving
                "Fryums": {"ml": 0.3, "rule": 0.7},  # Standard serving

                # Bread items - Rule-heavy
                "Breads": {"ml": 0.3, "rule": 0.7},  # Usually piece-based
                "Paan": {"ml": 0.2, "rule": 0.8},  # Standard serving

                # Breakfast items
                "Breakfast": {"ml": 0.5, "rule": 0.5},  # Varied patterns
                "Sandwich": {"ml": 0.5, "rule": 0.5},  # Standard portion
                "Omlette": {"ml": 0.4, "rule": 0.6},  # Usually piece-based

                # Desserts & Sweet items
                "Desserts": {"ml": 0.1, "rule": 0.9},  # Rule-driven
                "Cakes": {"ml": 0.1, "rule": 0.9},  # Size-dependent
                "Cup_Cakes": {"ml": 0.3, "rule": 0.7},  # Standard portion
                "Fruits": {"ml": 0.5, "rule": 0.5},  # Fairly standard

                # Beverages
                "Welcome_Drinks": {"ml": 0.4, "rule": 0.6},  # Standard serving
                "Hot_Beverages": {"ml": 0.4, "rule": 0.6},  # Standard serving

                # International categories
                "Italian": {"ml": 0.5, "rule": 0.5},  # Balanced approach
                "Pizza": {"ml": 0.5, "rule": 0.5}  # Balanced approach
            },
            "confidence_thresholds": {
                "high": {"threshold": 0.8, "ml": 0.7, "rule": 0.3},
                "medium": {"threshold": 0.5, "ml": 0.5, "rule": 0.5},
                "low": {"threshold": 0.0, "ml": 0.3, "rule": 0.7}
            }
        }
        if item_vc_file:
            self.load_item_vc_data(item_vc_file)
        if item_data_file:
            self.load_item_specific_data(item_data_file)

    def load_item_specific_data(self, item_data_file):
        logger.info(f"Loading item-specific data from {item_data_file}")
        try:
            item_data = pd.read_csv(item_data_file)
            required_columns = ['item_name', 'category', 'preferred_unit', 'per_guest_ratio', 'base_price_per_piece', 'base_price_per_kg']
            missing_columns = [col for col in required_columns if col not in item_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in {item_data_file}: {missing_columns}")

            for _, row in item_data.iterrows():
                item_name = self.standardize_Item_name(row['item_name'])
                self.item_specific_data[item_name] = {
                    'category': row['category'],
                    'preferred_unit': row['preferred_unit'],
                    'per_guest_ratio': float(row['per_guest_ratio']) if pd.notna(row['per_guest_ratio']) else None,
                    'base_price_per_piece': float(row['base_price_per_piece']) if pd.notna(row['base_price_per_piece']) else None,
                    'base_price_per_kg': float(row['base_price_per_kg']) if pd.notna(row['base_price_per_kg']) else None
                }
            logger.info(f"Loaded item-specific data for {len(self.item_specific_data)} items")
        except Exception as e:
            logger.error(f"Failed to load item-specific data: {e}")
            raise

    def load_item_vc_data(self, item_vc_file):
        logger.info(f"Loading item VC data from {item_vc_file}")
        try:
            vc_data = pd.read_excel(item_vc_file)
            for _, row in vc_data.iterrows():
                item_name = self.standardize_Item_name(row['Item_name'])
                self.item_vc_mapping[item_name] = {
                    'VC': float(row['VC']),
                    'p_value': float(row.get('Power Factor (p)', 0.18))  # Default to 0.18 if P_value is missing
                }
            logger.info(f"Loaded VC and P_value data for {len(self.item_vc_mapping)} items")
        except Exception as e:
            logger.error(f"Failed to load item VC data: {e}")
            raise

    def extract_quantity_value(self, quantity_str):
        return self.food_rules.extract_quantity_value(quantity_str)

    def extract_unit(self, quantity_str):
        return self.food_rules.extract_unit(quantity_str)

    def prepare_features(self, data):
        logger.info("Preparing features")
        if not hasattr(self, 'encoders_fitted') or not self.encoders_fitted:
            self.event_time_encoder.fit(data[['Event_Time']])
            self.meal_type_encoder.fit(data[['Meal_Time']])
            self.event_type_encoder.fit(data[['Event_Type']])
            self.encoders_fitted = True

        event_time_encoded = self.event_time_encoder.transform(data[['Event_Time']])
        meal_type_encoded = self.meal_type_encoder.transform(data[['Meal_Time']])
        event_type_encoded = self.event_type_encoder.transform(data[['Event_Type']])
        features = np.hstack([event_time_encoded, meal_type_encoded, event_type_encoded])
        return features

    def standardize_Item_name(self, Item_name):
        if pd.isna(Item_name):
            return ""
        Item_name = str(Item_name).strip()
        standardized = Item_name.lower()
        standardized = " ".join(standardized.split())
        return standardized

    def determine_item_properties(self, item_name):
        """
        More robust determination of item properties using metadata and pattern matching.
        """
        # Handle '>' prefix automatically
        clean_item_name = item_name.replace('> ', '').replace('>', '') if isinstance(item_name, str) else item_name

        # Start with defaults
        properties = {
            "category": None,
            "unit": "g",
            "is_veg": True
        }

        # Try direct match in metadata first (most reliable)
        if clean_item_name in self.item_metadata:
            return {
                "category": self.item_metadata[clean_item_name].category,
                "unit": self.item_metadata[clean_item_name].unit,
                "is_veg": getattr(self.item_metadata[clean_item_name], "is_veg", True)
            }

        # Try finding closest item in metadata using enhanced matcher
        mapped_item, _ = self.item_matcher.find_item(clean_item_name, self.item_metadata) if hasattr(self,
                                                                                                     'item_matcher') else (
        None, None)
        if mapped_item and mapped_item in self.item_metadata:
            return {
                "category": self.item_metadata[mapped_item].category,
                "unit": self.item_metadata[mapped_item].unit,
                "is_veg": getattr(self.item_metadata[mapped_item], "is_veg", True)
            }

        # If no match in metadata, use enhanced category detection
        item_lower = clean_item_name.lower() if isinstance(clean_item_name, str) else ""
        properties["category"] = self.guess_item_category(clean_item_name)

        # Enhanced unit detection based on both category and item name patterns
        if properties["category"] in self.food_rules.category_rules:
            rule = self.food_rules.category_rules[properties["category"]]
            default_qty = rule.get("default_quantity", "0g")
            unit = self.food_rules.extract_unit(default_qty)
            if unit:
                properties["unit"] = unit

        # Specific patterns for determining unit type
        # Desserts with piece-based units
        if properties["category"] == "Desserts":
            piece_patterns = ["jamun", "gulab", "rasgulla", "laddu", "burfi", "jalebi", "poornam", "buralu",
                              "delight", "mysore pak", "badusha"]
            if any(pattern in item_lower for pattern in piece_patterns):
                properties["unit"] = "pcs"

        # Appetizers with piece-based units
        elif properties["category"] == "Appetizers":
            piece_patterns = ["samosa", "tikka", "kebab", "roll", "cutlet", "patty", "vada", "bonda",
                              "pakora", "spring roll"]
            if any(pattern in item_lower for pattern in piece_patterns):
                properties["unit"] = "pcs"

        # Breakfast items with piece-based units
        elif properties["category"] == "Breakfast":
            piece_patterns = ["idli", "vada", "dosa", "uttapam", "poori", "paratha", "sandwich", "bun"]
            if any(pattern in item_lower for pattern in piece_patterns):
                properties["unit"] = "pcs"

        # Breads are always pieces
        elif properties["category"] in ["Breads", "Bread"]:
            properties["unit"] = "pcs"

        # Liquids are ml
        elif "Liquids" in properties["category"] or properties["category"] in ["Welcome_Drinks", "Soups"]:
            properties["unit"] = "ml"

        # Determine veg/non-veg status with more patterns
        non_veg_indicators = ["chicken", "mutton", "fish", "prawn", "beef", "pork", "egg", "meat",
                              "non veg", "kodi", "murg", "lamb", "goat", "seafood", "keema", "crab"]
        if any(indicator in item_lower for indicator in non_veg_indicators):
            properties["is_veg"] = False

        return properties

    def find_closest_item(self,Item_name):
        """
        Find closest item using hash table system or fallback to original algorithm.

        Args:
            Item_name: The item name to search for

        Returns:
            matched_item: The best matching item name, or None if no match found
        """
        # Use new hash table system if available
        if hasattr(self, 'item_matcher') and self.item_metadata:
            matched_item, _ = self.item_matcher.find_item(Item_name, self.item_metadata)
            if matched_item:
                logger.info(f"Hash table matched '{Item_name}' to '{matched_item}'")
                return matched_item

        # Fallback to original fuzzy matching logic
        if not self.item_metadata:
            return None

        std_name = self.standardize_Item_name(Item_name)
        if std_name in self.Item_name_mapping:
            return self.Item_name_mapping[std_name]

        best_match = None
        best_score = 0
        for known_name in self.item_metadata:
            std_known = self.standardize_Item_name(known_name)
            if std_name in std_known or std_known in std_name:
                similarity = min(len(std_name), len(std_known)) / max(len(std_name), len(std_known))
                if similarity > best_score:
                    best_score = similarity
                    best_match = known_name
            else:
                std_name_words = set(std_name.split())
                std_known_words = set(std_known.split())
                common_words = std_name_words.intersection(std_known_words)
                if common_words:
                    similarity = len(common_words) / max(len(std_name_words), len(std_known_words))
                    if similarity > best_score:
                        best_score = similarity
                        best_match = known_name

        if best_score >= 0.88:
            logger.info(f"Fuzzy matched '{Item_name}' to '{best_match}' with score {best_score:.2f}")
            return best_match

        return None

    def guess_item_category(self, Item_name):
        """More robust category detection for unknown items."""
        item_lower = Item_name.lower()

        # Expanded category keywords with more variations
        category_keywords = {
            "Welcome_Drinks": ["punch", "Packed Juice","fresh fruit juice", "juice", "mojito", "drinks","milk", "tea", "coffee", "juice", "butter milk", "lassi", "soda", "voda", "water melon juice"],
            "Appetizers": ["tikka","chilli","chilli garlic", "65","Sauteed Grilled Chicken Sausage", "panner","Fish Fingers","Mashed Potatos","Cheese Fries","french fires","Potato Skins","Pepper Chicken (Oil Fry)","Lemon Chicken","kabab", "hariyali kebab", "tangdi", "drumsticks", "nuggets","majestic","roll", "poori", "masala vada", "alasanda vada", "veg bullets", "veg spring rolls", "hara bara kebab", "kebab", "lollipop", "chicken lollipop", "pakora", "kodi", "cut", "bajji", "vepudu", "roast", "kurkure", "afghani kebab", "corn", "manchuria", "manchurian", "gobi"],
            #"Appetizers": ["tikka", "65","chiili,"chilli garlic","chilli garlic fish","Sauteed Grilled Chicken Sausage", "panner","Fish Fingers","Mashed Potatos","Cheese Fries","french fires","Potato Skins","Pepper Chicken (Oil Fry)","Lemon Chicken","kabab", "hariyali kebab", "tangdi", "drumsticks", "nuggets","majestic","roll", "poori", "masala vada", "alasanda vada", "veg bullets", "veg spring rolls", "hara bara kebab", "kebab", "lollipop", "chicken lollipop", "pakora", "kodi", "cut", "bajji", "vepudu", "roast", "kurkure", "afghani kebab", "corn", "manchuria", "manchurian", "gobi"],
            "Soups": ["soup", "shorba","mutton marag","broth","cream of chicken","paya","hot and sour",],
            "Fried": ["fried rice"],
            "Italian": ["pasta", "noodles", "white pasta", "veg garlic soft noodles","macroni"],
            "Fry": ["fry", "bendi kaju","Dondakaya Fry","Bhindi Fry","Aloo Fry","Cabbage Fry"],
            "Liquids(Less_Dense)": ["rasam","Pachi pulusu","sambar", "charu", "majjiga pulusu", "Miriyala Rasam","chintapandu rasam","lemon pappucharu","mulakaya pappucharu"],
            "Liquids(High_Dense)": ["ulavacharu"],
            "Curries": ["iguru","Paneer Butter Masala","Chicken Chettinad","gutti vankaya curry","kadai","scrambled egg curry","baigan bartha","bendi do pyaza","boiled egg cury","chana masala","curry", "gravy", "masala", "kurma", "butter","pulusu","mutton rogan josh curry","kadai", "tikka masala", "dal tadka", "boti", "murgh", "methi", "bhurji", "chatapata", "pulsu", "vegetable curry", "dum aloo curry"],
            "Rice": ["steamed rice", "kaju ghee rice", "bagara rice"],
            "Flavored_Rice": ["muddapappu avakai annam","Ragi Sangati", "pudina rice","temple style pulihora", "pappucharu annam","pappu charu annam","cocount rice","cocunut rice","pulihora", "curd rice", "jeera rice", "gongura rice", "Muddapappu Avakaya Annam", "sambar rice", "Muddapappu avakai Annam", "annam"],
            "Pulav": ["pulav","Mutton Fry Piece Pulav","Natukodi Pulav","jeera mutter pulav", "fry piece pulav", "ghee pulav","Green Peas Pulav","Meal Maker Pulav","Paneer Pulav"],
            "Biryani": ["biryani", "biriyani", "Mutton Kheema Biryani","biriani", "kaju panner biryani","panner biryani","panaspottu biryani","egg biryani","chicken chettinad biryani","ulavacharu chicken biryani", "mushroom biryani", "veg biryani", "chicken dum biryani"],
            "Breads": ["naan", "paratha", "kulcha", "pulka", "chapati", "rumali roti","laccha paratha","masala kulcha","panner kulcha","butter garlic naan","roti,pudina naan","tandoori roti"],
            "Dal": ["dal", "lentil", "pappu", "Mamidikaya Pappu (Mango)","Dal Makhani","Dal Tadka","sorakaya pappu", "thotakura pappu", "tomato pappu", "yellow dal""chintakaya papu","palakura pappu","thotakura pappu","tomato pappu","yellow dal tadka"],
            "Chutney":["peanut chutney","allam chutney","green chutney","pudina chutney","dondakay chutney"],
            "Ghee": ["ghee"],
            "Podi": ["podi"],
            "Pickle": ["pickle"],
            "Paan": ["paan", "pan"],
            #"Dips": ["dip","Sour cream Dip","jam","Tandoori Mayo","Mayonnaise Dip","Hummus Dip","Garlic Mayonnaise Dip"],
            "Roti_Pachadi": ["Beerakaya Pachadi", "roti pachadi", "Tomato Pachadi","Vankaya Pachadi","Roti Pachadi", "pachadi","gongura onion chutney"],
            "Crispers": ["fryums", "papad", "crispers"],
            "Raitha": ["raitha", "Raitha", "boondi raitha"],
            "Salan": ["salan", "Salan", "Mirchi Ka Salan"],
            "Fruits": ["seaonsal", "mixed", "cut", "fruit", "Fruit"],
            "Salad": ["salad", "Salad", "ceasar", "green salad", "boiled peanut salad","boiled peanut salad","mexican corn salad","Laccha Pyaaz","Cucumber Salad"],
            "Curd": ["curd", "set curd"],
            "Desserts": ["brownie", "walnut brownie", "Gajar Ka Halwa","Chocolate Brownie","Assorted Pastries","halwa","Semiya Payasam (Kheer)","Sabudana Kheer","Kesari Bath","Double Ka Meetha", "carrot halwa", "shahi ka tukda", "gulab jamun", "apricot delight", "baked gulab jamun", "bobbattu", "bobbatlu", "kalajamun", "rasagulla", "laddu", "poornam", "apricot delight", "gulab jamun", "rasammaiah"],
            "Breakfast": ["idly", "dosa", "vada", "upma","Rava Khichdi","Bisi Bela Bath","Sabudana Khichdi","Upma","Millet Upma", "pongal", "mysore bonda", "idly"],
            "Sandwich": ["sandwich","Veg Sandwitch"],
            "Cup_Cakes": ["cup cake", "cupcake"]
        }

        # Direct matches first (full item name)
        for category, keywords in category_keywords.items():
            if item_lower in keywords:
                logger.info(f"Categorized '{Item_name}' as '{category}' based on exact match")
                return category

        # Partial keyword matching with improved logic
        best_match = None
        best_score = 0

        for category, keywords in category_keywords.items():
            category_score = 0

            # Check for keyword presence
            for keyword in keywords:
                if keyword in item_lower:
                    # Longer keywords are more specific, so give them higher weight
                    weight = len(keyword) / 20  # Normalize by typical max keyword length
                    category_score += weight

                    # If keyword is at the beginning or end of item name, it's likely more relevant
                    if item_lower.startswith(keyword) or item_lower.endswith(keyword):
                        category_score += 0.5

            # If category has better score than current best, update
            if category_score > best_score:
                best_score = category_score
                best_match = category

        if best_match and best_score > 0.5:  # Threshold to ensure reasonable match
            logger.info(
                f"Categorized '{Item_name}' as '{best_match}' based on keyword matching with score {best_score:.2f}")
            return best_match

        # Default fallback with warning
        logger.warning(f"Using default category 'Curries' for '{Item_name}'")
        return "Curries"


    def build_Item_name_mapping(self, data):
        """Build Item_name mapping and populate direct lookup hash table."""
        logger.info("Building Item_name mapping")
        self.Item_name_mapping = {}
        Item_groups = data.groupby(['Item_name'])

        for Item_name, _ in Item_groups:
            std_name = self.standardize_Item_name(Item_name)
            if std_name:
                self.Item_name_mapping[std_name] = Item_name
                self.Item_name_mapping[Item_name] = Item_name

                # Also add to direct lookup if matcher exists
                if hasattr(self, 'item_matcher'):
                    self.item_matcher.direct_lookup[std_name] = Item_name
                    self.item_matcher.direct_lookup[Item_name] = Item_name

        logger.info(f"Built mapping for {len(self.Item_name_mapping)} item variations")

    def build_item_metadata(self, data):
        logger.info("Building item metadata from training data")
        if not hasattr(self, 'Item_name_mapping'):
            self.build_Item_name_mapping(data)

        Item_groups = data.groupby(['Item_name'])
        for Item_name, group in Item_groups:
            # Convert to string if it's a tuple
            if isinstance(Item_name, tuple):
                Item_name = Item_name[0]

            category = group['Category'].iloc[0]
            unit = self.extract_unit(group['Per_person_quantity'].iloc[0])
            if not unit:
                unit = self.food_rules.infer_default_unit(category)
            unit = self.food_rules.validate_unit(category, unit)

            # Store as string
            self.item_metadata[str(Item_name)] = ItemMetadata(category=category, unit=unit)

            is_veg = True
            item_lower = str(Item_name).lower()
            non_veg_indicators = ["chicken", "mutton", "fish", "prawn", "shrimp", "beef",
                                  "pork", "egg", "meat", "non veg", "kodi"]
            if any(indicator in item_lower for indicator in non_veg_indicators):
                is_veg = False

            # Store with veg/non-veg information
            self.item_metadata[str(Item_name)] = ItemMetadata(
                category=category,
                unit=unit,
                is_veg=is_veg
            )
            std_name = self.standardize_Item_name(Item_name)
            if std_name != Item_name:
                self.item_metadata[std_name] = ItemMetadata(category=category, unit=unit)

    def initialize_matcher(self):
        """Initialize the item matcher with current metadata."""
        if self.item_metadata:
            self.item_matcher.build_hash_tables(self.item_metadata)
        logger.info("Initialized food item matcher with two-tier hash tables")

    def get_calibration_weights(self, category, confidence=None):
        """Determine calibration weights based on category and confidence."""
        # Start with global defaults
        weights = self.calibration_config["global_default"].copy()

        # Apply category-specific weights if available
        if category in self.calibration_config["categories"]:
            weights = self.calibration_config["categories"][category].copy()

        # Apply confidence-based adjustment if provided
        if confidence is not None:
            for level, config in self.calibration_config["confidence_thresholds"].items():
                if confidence >= config["threshold"]:
                    # Blend category weights with confidence weights (70-30 split)
                    ml_weight = 0.7 * weights["ml"] + 0.3 * config["ml"]
                    rule_weight = 0.7 * weights["rule"] + 0.3 * config["rule"]

                    # Normalize to ensure sum is 1.0
                    total = ml_weight + rule_weight
                    weights["ml"] = ml_weight / total
                    weights["rule"] = rule_weight / total
                    break

        return weights

    def apply_calibrated_prediction(self, ml_prediction, rule_prediction, category, confidence=None, unit=None):
        """
        Apply calibration with unit normalization to combine ML and rule-based predictions.
        """
        # Normalize ML prediction based on unit type
        normalized_ml = ml_prediction

        # If we're dealing with a piece-based item but ML predicts in grams
        if unit == "pcs" and ml_prediction > 10:  # Threshold to detect likely gram prediction
            # Convert from grams to pieces using category-specific conversion rates
            conversion_rates = {
                "Appetizers": 35,  # ~35g per appetizer piece
                "Desserts": 40,  # ~40g per dessert piece
                "Breakfast": 50,  # ~50g per breakfast item
                "Breads": 30  # ~30g per bread piece
            }

            # Apply conversion with fallback to default rate
            conversion_rate = conversion_rates.get(category, 40)
            normalized_ml = ml_prediction / conversion_rate
            logger.info(
                f"Unit conversion: ML prediction {ml_prediction}g converted to {normalized_ml}pcs for {category}")

        # Get weights based on category and confidence
        weights = self.get_calibration_weights(category, confidence)

        # Apply reasonableness checks specific to each unit type
        if unit == "pcs":
            # Piece-based items typically have reasonable limits per person
            max_reasonable = {
                "Appetizers": 2,  # Max 4 pieces per person
                "Desserts": 2,  # Max 3 dessert pieces per person
                "Breakfast": 2,  # Max 5 breakfast items
                "Breads": 2  # Max 4 bread pieces
            }

            reasonable_limit = max_reasonable.get(category, 3)

            # If ML prediction still exceeds reasonable limits after normalization
            if normalized_ml > reasonable_limit * 1.5:
                # Further reduce ML weight
                weights["ml"] *= 0.2
                weights["rule"] = 1.0 - weights["ml"]
                logger.warning(
                    f"Further adjusted weights for {category} due to unreasonable ML prediction: {normalized_ml} vs reasonable {reasonable_limit}")

        # Apply weighted combination with normalized ML prediction
        calibrated_value = (normalized_ml * weights["ml"]) + (rule_prediction * weights["rule"])

        # Apply final sanity bounds based on unit type
        if unit == "pcs":
            item_type_max = {
                "Desserts": 2,
                "Appetizers": 2,
                "Breads": 2
            }.get(category, 2)

            # Hard cap at reasonable maximum
            calibrated_value = min(calibrated_value, item_type_max)

        logger.info(f"Calibration for {category}: ML={normalized_ml:.2f}*{weights['ml']:.2f}, "
                    f"Rule={rule_prediction:.2f}*{weights['rule']:.2f}, Result={calibrated_value:.2f}")

        return calibrated_value

    def fit(self, data):
        logger.info("Starting model training")
        if not self.item_metadata:
            self.build_item_metadata(data)

        data['quantity_value'] = data['Per_person_quantity'].apply(self.extract_quantity_value)
        X = self.prepare_features(data)
        self.initialize_matcher()
        categories = data['Category'].unique()
        for category in categories:
            logger.info(f"Training model for category: {category}")
            category_data = data[data['Category'] == category]
            if category_data.empty:
                continue
            category_data_subset = category_data[['Order_Number', 'quantity_value', 'Guest_Count']]
            # Sum the per-person quantities for the category within each order
            category_per_person = category_data_subset.groupby('Order_Number').agg({
                'quantity_value': 'sum',  # This is already per-person
                'Guest_Count': 'first'
            }).reset_index()
            # No need to divide by Guest_Count; quantity_value is already per person
            category_per_person = category_per_person.rename(columns={'quantity_value': 'per_person_quantity'})
            category_per_person = category_per_person[['Order_Number', 'per_person_quantity']]
            category_train = pd.merge(
                category_per_person,
                data[['Order_Number', 'Event_Time', 'Meal_Time', 'Event_Type']].drop_duplicates(),
                on='Order_Number',
                how='left'
            )
            if category_train.empty:
                continue
            if category not in self.category_models:
                self.category_models[category] = XGBRegressor(
                    n_estimators=500, learning_rate=0.1, max_depth=6, objective='reg:squarederror', random_state=42
                )
                self.category_scalers[category] = RobustScaler()
            X_cat = self.prepare_features(category_train)
            X_cat_scaled = self.category_scalers[category].fit_transform(X_cat)
            self.category_models[category].fit(X_cat_scaled, category_train['per_person_quantity'])

        training_items = set(data['Item_name'].unique())
        items_to_train = {item_name: metadata for item_name, metadata in self.item_metadata.items() if
                          item_name in training_items}
        for item_name, metadata in items_to_train.items():
            logger.info(f"Training model for item: {item_name}")
            item_data = data[data['Item_name'] == item_name]
            item_data_subset = item_data[['Order_Number', 'quantity_value', 'Guest_Count']]
            # Sum the per-person quantities for the item within each order
            item_per_person = item_data_subset.groupby('Order_Number').agg({
                'quantity_value': 'sum',  # This is already per-person
                'Guest_Count': 'first'
            }).reset_index()
            # No need to divide by Guest_Count; quantity_value is already per person
            item_per_person = item_per_person.rename(columns={'quantity_value': 'per_person_quantity'})
            item_per_person = item_per_person[['Order_Number', 'per_person_quantity']]
            item_train = pd.merge(
                item_per_person,
                data[['Order_Number', 'Event_Time', 'Meal_Time', 'Event_Type']].drop_duplicates(),
                on='Order_Number',
                how='left'
            )
            if item_train.empty:
                continue
            if item_name not in self.item_models:
                self.item_models[item_name] = XGBRegressor(
                    n_estimators=500, learning_rate=0.05, max_depth=8, min_child_weight=5,
                    subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror',
                    reg_lambda=1, reg_alpha=0.5, random_state=42
                )
                self.item_scalers[item_name] = RobustScaler()
            X_item = self.prepare_features(item_train)
            X_item_scaled = self.item_scalers[item_name].fit_transform(X_item)
            self.item_models[item_name].fit(X_item_scaled, item_per_person['per_person_quantity'])
        logger.info("Model training completed")

    def build_menu_context(self, event_time, meal_type, event_type, guest_count, selected_items):
        """Build comprehensive menu context to enhance prediction accuracy"""
        # Initialize context
        menu_context = {
            'categories': [],
            'items': selected_items,
            'total_items': len(selected_items),
            'meal_type': meal_type,
            'event_time': event_time,
            'event_type': event_type,
            'guest_count': guest_count,
            'items_by_category': {},
            'item_properties': {},
            'special_categories': ["Appetizers", "Desserts", "Breakfast"],
            'items_by_unit_type': {}
        }

        # Process each item to determine properties and categories
        for item in selected_items:
            # Get item properties
            properties = self.determine_item_properties(item)
            category = properties["category"]
            unit = properties["unit"]

            # Store item properties
            menu_context['item_properties'][item] = properties

            # Organize by category
            if category not in menu_context['items_by_category']:
                menu_context['items_by_category'][category] = []
                menu_context['categories'].append(category)
            menu_context['items_by_category'][category].append(item)

            # Organize by category and unit type (for special categories)
            if category in menu_context['special_categories']:
                if category not in menu_context['items_by_unit_type']:
                    menu_context['items_by_unit_type'][category] = {'g': [], 'ml': [], 'pcs': []}
                menu_context['items_by_unit_type'][category][unit].append(item)

        # Add context flags for dependencies
        menu_context['has_biryani'] = 'Biryani' in menu_context['categories']
        menu_context['has_rice'] = 'Rice' in menu_context['categories']
        menu_context['has_curries'] = 'Curries' in menu_context['categories']
        menu_context['has_dal'] = 'Dal' in menu_context['categories']

        return menu_context

    def predict(self, event_time, meal_type, event_type, guest_count, selected_items):
        """
        Predict quantities for the given food items with enhanced handling for special categories
        and menu context integration.

        Args:
            event_time: Time of the event (Morning, Afternoon, Evening, Night)
            meal_type: Type of meal (Breakfast, Lunch, Dinner, Hi-Tea)
            event_type: Type of event (Wedding, Birthday Party, etc.)
            guest_count: Number of guests
            selected_items: List of food items

        Returns:
            predictions: Dictionary mapping items to their predicted quantities
        """
        logger.info(f"Making prediction for event: {event_type}, {event_time}, {meal_type}, {guest_count} guests")

        # Build menu context
        menu_context = self.build_menu_context(event_time, meal_type, event_type, guest_count, selected_items)

        # Make sure matcher is initialized if we've built metadata
        if not hasattr(self, 'item_matcher') or not self.item_matcher.direct_lookup:
            if self.item_metadata:
                self.initialize_matcher()

        # Prepare input data
        input_data = pd.DataFrame({'Event_Time': [event_time], 'Meal_Time': [meal_type], 'Event_Type': [event_type]})
        X = self.prepare_features(input_data)

        # Initialize results
        predictions = {}
        items_by_category = {}
        unmapped_items = []
        original_to_mapped = {}
        total_items = len(selected_items)

        # Match items using two-tier hash table approach
        for item in selected_items:
            # Use two-tier hash table matching
            mapped_item = None

            # Try the matcher first if available
            if hasattr(self, 'item_matcher') and self.item_metadata:
                mapped_item, metadata = self.item_matcher.find_item(item, self.item_metadata)

            # Fall back to traditional mapping if needed
            if not mapped_item:
                std_item = self.standardize_Item_name(item)
                mapped_item = getattr(self, 'Item_name_mapping', {}).get(std_item, None)
                if not mapped_item:
                    mapped_item = self.find_closest_item(item)

            # Process the mapped item
            if mapped_item and mapped_item in self.item_metadata:
                category = self.item_metadata[mapped_item].category
                if category not in items_by_category:
                    items_by_category[category] = []
                items_by_category[category].append(mapped_item)
                original_to_mapped[item] = mapped_item
                if item != mapped_item:
                    logger.info(f"Mapped '{item}' to '{mapped_item}'")
            else:
                unmapped_items.append(item)
                logger.warning(f"No match found for item: {item}")

        # Handle unmapped items with category guessing
        if unmapped_items:
            for item in unmapped_items:
                # Determine properties using our new function
                properties = self.determine_item_properties(item)
                category = properties["category"]
                unit = properties["unit"]
                is_veg = properties["is_veg"]

                if category not in items_by_category:
                    items_by_category[category] = []
                items_by_category[category].append(item)
                original_to_mapped[item] = item

                std_item = self.standardize_Item_name(item)
                self.item_metadata[item] = ItemMetadata(category=category, unit=unit, is_veg=is_veg)
                self.item_metadata[std_item] = ItemMetadata(category=category, unit=unit, is_veg=is_veg)

        # Special processing for special categories
        special_categories = ["Appetizers", "Desserts", "Breakfast"]
        special_category_items = {}

        for category in special_categories:
            if category in items_by_category:
                items = items_by_category[category]
                # Group items by unit type
                items_by_unit = {"g": [], "ml": [], "pcs": []}

                for item in items:
                    if item in self.item_metadata:
                        unit = self.item_metadata[item].unit
                        if unit in items_by_unit:
                            items_by_unit[unit].append(item)

                # Store for later use
                special_category_items[category] = items_by_unit

        # Apply calibrated prediction for each category
        category_per_person = {}
        category_quantities = {}

        # First process special categories
        for category, unit_groups in special_category_items.items():
            for unit_type, items in unit_groups.items():
                if not items:  # Skip empty groups
                    continue

                # Special handling parameters
                special_params = {
                    "unit_type": unit_type,
                    "total_items": total_items,
                    "meal_type": meal_type,
                    "has_biryani": "Biryani" in items_by_category,
                    "has_rice": "Rice" in items_by_category,
                    "has_curry": "Curries" in items_by_category,
                    "has_dal": "Dal" in items_by_category
                }

                # Get ML prediction if model exists
                ml_prediction = 0.0
                ml_confidence = 0.5  # Default confidence
                if category in self.category_models:
                    X_cat_scaled = self.category_scalers[category].transform(X)
                    ml_prediction = float(self.category_models[category].predict(X_cat_scaled)[0])

                    # Estimate confidence
                    if hasattr(self.category_models[category], 'n_estimators'):
                        ml_confidence = min(0.9, 0.4 + (len(items) / 20))

                # Get rule-based prediction with unit type
                rule_prediction = 0.0
                qty, unit = self.food_rules.apply_category_rules(
                    category, guest_count, len(items), **special_params
                )

                if qty > 0:
                    rule_prediction = qty
                else:
                    # Fall back to default if needed
                    default_qty, _ = self.food_rules.get_default_quantity(category)
                    rule_prediction = default_qty if default_qty > 0 else 0.0

                # Apply calibration
                calibrated_qty = self.apply_calibrated_prediction(
                    ml_prediction, rule_prediction, category, ml_confidence,unit
                )

                # Store for each item in this unit group
                for item in items:
                    # Store in special item tracking
                    if category not in category_per_person:
                        category_per_person[category] = {}
                    category_per_person[category][item] = (calibrated_qty, unit)

                    logger.info(f"Special category {category}, item {item}: {calibrated_qty:.2f}{unit} per person")

        # Process standard categories
        for category, items in items_by_category.items():
            # Skip already processed special categories
            if category in special_categories:
                continue

            # Special case for breads - use rules directly
            if category == "Breads":
                qty, unit = self.food_rules.get_default_quantity(category)
                category_per_person[category] = qty
                category_quantities[category] = {"value": qty, "unit": unit}
                continue

            # Standard category parameters
            category_params = {
                "total_items": total_items,
                "meal_type": meal_type,
                "has_biryani": "Biryani" in items_by_category,
                "has_rice": "Rice" in items_by_category,
                "has_curry": "Curries" in items_by_category,
                "has_dal": "Dal" in items_by_category
            }

            # Get ML prediction if model exists
            ml_prediction = 0.0
            ml_confidence = 0.5  # Default confidence
            if category in self.category_models:
                X_cat_scaled = self.category_scalers[category].transform(X)
                ml_prediction = float(self.category_models[category].predict(X_cat_scaled)[0])

                # Estimate confidence
                if hasattr(self.category_models[category], 'n_estimators'):
                    ml_confidence = min(0.9, 0.4 + (len(items) / 20))

            # Get rule-based prediction
            rule_prediction = 0.0
            qty, unit = self.food_rules.apply_category_rules(
                category, guest_count, len(items), **category_params
            )

            if qty > 0:
                rule_prediction = qty
            else:
                # Fall back to default if needed
                default_qty, _ = self.food_rules.get_default_quantity(category)
                rule_prediction = default_qty if default_qty > 0 else 0.0

            # Apply calibration
            calibrated_qty = self.apply_calibrated_prediction(
                ml_prediction, rule_prediction, category, ml_confidence
            )

            # Store the calibrated quantity
            category_per_person[category] = calibrated_qty
            unit = unit or (self.item_metadata[items[0]].unit if items else "g")
            category_quantities[category] = {"value": calibrated_qty, "unit": unit}

            logger.info(f"Category {category}: ML={ml_prediction:.2f}, Rule={rule_prediction:.2f}, "
                        f"Calibrated={calibrated_qty:.2f}, Unit={unit}")

        # Apply category dependency rules
        for category, items in items_by_category.items():
            # Skip special categories with item-specific quantities
            if category in special_categories:
                continue

            dependent_categories = self.food_rules.get_dependent_categories(category)
            for dep in dependent_categories:
                dep_normalized = dep.strip()
                matching_categories = [c for c in items_by_category.keys() if c.lower() == dep_normalized.lower()]
                if matching_categories:
                    dep_category = matching_categories[0]
                    orig_qty = category_per_person[category]
                    unit = category_quantities[category]["unit"]
                    adjusted_qty = self.food_rules.apply_dependency_rules(category, dep_category, orig_qty, unit)
                    category_per_person[category] = adjusted_qty
                    category_quantities[category]["value"] = adjusted_qty

        # Apply meal type modifiers for standard categories
        for category in category_quantities:
            # Skip special categories
            if category in special_categories:
                continue

            orig_qty = category_per_person[category]
            modified_qty = self.food_rules.apply_meal_type_modifier(meal_type, orig_qty)
            category_per_person[category] = modified_qty
            category_quantities[category]["value"] = modified_qty

        # Determine per-item quantities based on category
        item_per_person = {}

        # First process special category items
        for category, unit_groups in special_category_items.items():
            for unit_type, items in unit_groups.items():
                for item in items:
                    if category in category_per_person and item in category_per_person[category]:
                        qty, unit = category_per_person[category][item]
                        item_per_person[item] = qty  # Store just the quantity

        # Process standard category items
        for category, items in items_by_category.items():
            # Skip special categories already processed
            if category in special_categories:
                continue

            if category in category_per_person and not isinstance(category_per_person[category], dict):
                category_qty = category_per_person[category]
                for item in items:
                    item_per_person[item] = category_qty

        # Unit conversion helper function
        def auto_convert(quantity: float, unit: str) -> Tuple[float, str]:
            unit_lower = unit.lower()
            if unit_lower == "g" and quantity >= 1000:
                return quantity / 1000, "kg"
            elif unit_lower == "ml" and quantity >= 1000:
                return quantity / 1000, "l"
            return quantity, unit

        # Generate final predictions
        for orig_item in selected_items:
            mapped_item = original_to_mapped.get(orig_item, orig_item)

            if mapped_item in item_per_person:
                per_person_qty = item_per_person[mapped_item]

                # Get category and unit
                category = self.item_metadata[mapped_item].category

                # Get unit based on category or item type
                if category in special_categories and category in category_per_person and mapped_item in \
                        category_per_person[category]:
                    _, unit = category_per_person[category][mapped_item]
                else:
                    unit = category_quantities.get(category, {}).get("unit", "g")
                    if not unit and mapped_item in self.item_metadata:
                        unit = self.item_metadata[mapped_item].unit
                    unit = self.food_rules.validate_unit(category, unit)

                # Calculate total quantity
                total_qty = per_person_qty * guest_count

                # Round piece-based items
                if unit == "pcs":
                    total_qty = max(1, round(total_qty))

                # Convert to appropriate unit scale
                converted_qty, converted_unit = auto_convert(total_qty, unit)

                # Format prediction with 2 decimal places for kg/L
                if converted_unit in ["kg", "l"]:
                    predictions[orig_item] = f"{converted_qty:.2f}{converted_unit}"
                else:
                    predictions[orig_item] = f"{converted_qty}{converted_unit}"

            else:
                logger.warning(f"Item {orig_item} missing from predictions, applying fallback")
                if mapped_item in self.item_metadata:
                    category = self.item_metadata[mapped_item].category
                    unit = self.item_metadata[mapped_item].unit

                    # Use category level quantities if available
                    if category in category_per_person and not isinstance(category_per_person[category], dict):
                        per_person_qty = category_per_person[category]
                        unit = category_quantities.get(category, {}).get("unit", unit)
                        unit = self.food_rules.validate_unit(category, unit)
                        total_qty = per_person_qty * guest_count

                        # Round piece-based items
                        if unit == "pcs":
                            total_qty = max(1, round(total_qty))

                        # Convert to appropriate unit scale
                        converted_qty, converted_unit = auto_convert(total_qty, unit)

                        # Format prediction with 2 decimal places for kg/L
                        if converted_unit in ["kg", "l"]:
                            predictions[orig_item] = f"{converted_qty:.2f}{converted_unit}"
                        else:
                            predictions[orig_item] = f"{converted_qty}{converted_unit}"
                    else:
                        # Fallback to default quantities
                        if unit == "pcs":
                            predictions[orig_item] = f"{2 * guest_count}pcs"
                        elif unit == "ml":
                            total_ml = 100 * guest_count
                            predictions[orig_item] = f"{total_ml / 1000:.2f}L" if total_ml >= 1000 else f"{total_ml}ml"
                        else:  # 'g'
                            total_g = 100 * guest_count
                            predictions[orig_item] = f"{total_g / 1000:.2f}kg" if total_g >= 1000 else f"{total_g}g"

        return predictions

    def calculate_price(self, converted_qty, category, guest_count, item_name, predicted_unit=None):
        """More robust price calculation with better handling of units and special cases."""
        # Normalize category name
        normalized_category = self.food_rules.normalize_category_name(category)
        rule = self.food_rules.category_rules.get(normalized_category, {})

        # Parameters with bounds checks
        FC = 6000  # Fixed cost
        Qmin_static = 50  # Static minimum quantity
        beta = 0.5  # Exponent for dynamic Qmin

        # Get item-specific data with better standardization
        std_item_name = self.standardize_Item_name(item_name)
        item_data = self.item_specific_data.get(std_item_name, {})

        # Try alternative standardization if not found
        if not item_data and '>' in item_name:
            alt_std_name = self.standardize_Item_name(item_name.replace('>', '').strip())
            item_data = self.item_specific_data.get(alt_std_name, {})

        preferred_unit = item_data.get('preferred_unit', None)
        base_price_per_piece = item_data.get('base_price_per_piece', None)
        base_price_per_kg = item_data.get('base_price_per_kg', rule.get("vc_price", 220))

        # Get item-specific VC and p_value with better fallbacks
        vc_data = self.item_vc_mapping.get(std_item_name, {})
        if not vc_data and '>' in item_name:
            alt_std_name = self.standardize_Item_name(item_name.replace('>', '').strip())
            vc_data = self.item_vc_mapping.get(alt_std_name, {})

        # Determine category-appropriate VC
        category_vc = rule.get("vc_price", 220)

        # Special category default VCs
        category_defaults = {
            "Desserts": 45,
            "Appetizers": 60,
            "Biryani": 320,
            "Curries": 300,
            "Breads": 10
        }

        if normalized_category in category_defaults:
            category_vc = category_defaults[normalized_category]

        VC = vc_data.get('VC', category_vc)  # Fallback to category VC
        p_value = vc_data.get('p_value', 0.19)  # Fallback to 0.19

        # Ensure p_value is within realistic bounds
        p_value = max(0.05, min(0.35, p_value))

        # Use the predicted unit if provided, otherwise fall back to default
        default_qty, default_unit = self.food_rules.get_default_quantity(normalized_category)
        unit = predicted_unit if predicted_unit else default_unit
        unit = self.food_rules.validate_unit(normalized_category, unit)
        if preferred_unit:
            unit = preferred_unit

        # Ensure proper numeric conversion with error handling
        try:
            converted_qty = float(converted_qty)
        except (ValueError, TypeError):
            logger.warning(f"Invalid quantity for {item_name}: {converted_qty}, falling back to 100")
            converted_qty = 100

        # Enforce minimum quantity
        converted_qty = max(1, converted_qty)

        # Calculate total quantity (Q) correctly
        Q = converted_qty  # Total quantity

        # Calculate per person quantity
        per_person_qty = Q / guest_count if guest_count > 0 else 0

        # Robust threshold handling
        if unit == "pcs":
            # Scale thresholds based on guest count for piece-based items
            Q_threshold = 45
            Q_in_threshold_unit = Q
            base_price_per_unit = base_price_per_piece if base_price_per_piece is not None else VC
            #base_threshold = 45
            #Q_threshold = base_threshold * (guest_count / 100) if guest_count > 100 else base_threshold
            #Q_in_threshold_unit = Q
            #base_price_per_unit = base_price_per_piece if base_price_per_piece is not None else VC
        else:
            if unit == "kg":
                base_threshold = 50
                Q_in_threshold_unit = Q
            elif unit == "g":
                base_threshold = 50000  # 50 kg in g
                Q_in_threshold_unit = Q / 1000  # Convert g to kg
            elif unit == "l":
                base_threshold = 60
                Q_in_threshold_unit = Q
            elif unit == "ml":
                base_threshold = 60000  # 60 L in ml
                Q_in_threshold_unit = Q / 1000  # Convert ml to l
            else:
                base_threshold = 50
                Q_in_threshold_unit = Q / 1000

            # Scale thresholds based on guest count
            Q_threshold = base_threshold * (1 + (guest_count / 1000))
            base_price_per_unit = base_price_per_kg

        # Calculate Qmin with safeguards against excessive values
        Qmin = min(max(Qmin_static, Q_in_threshold_unit ** beta), Q_threshold)

        # Calculate base rate with protection against division by zero
        base_rate = FC / max(Qmin, 0.1) + VC

        # Determine final prices with more robust approach
        if unit == "pcs":
            # Piece-based pricing simplified
            price_per_piece = base_price_per_unit
            total_price = price_per_piece * Q
            price_per_person = price_per_piece * per_person_qty
        else:
            # Weight-based pricing with smoother curve
            if Q_in_threshold_unit <= 0.1:  # Very small quantity
                unit_price = base_rate  # Higher price for very small quantities
            elif Q_in_threshold_unit <= Q_threshold:
                # Normal scale pricing
                unit_price = base_rate * (Q_in_threshold_unit ** (-p_value))
            else:
                # Large scale pricing - fixed unit price after threshold
                unit_price = base_rate * (Q_threshold ** (-p_value))

            total_price = unit_price * Q_in_threshold_unit
            price_per_person = unit_price * per_person_qty

        # Sanity check on final pricing
        if total_price < 0 or math.isnan(total_price) or math.isinf(total_price):
            logger.warning(f"Invalid price calculation for {item_name}: {total_price}. Using fallback.")
            total_price = VC * Q_in_threshold_unit  # Simple fallback

        # Ensure reasonable per-person pricing
        if price_per_person < 0 or math.isnan(price_per_person) or math.isinf(price_per_person):
            price_per_person = total_price / guest_count

        # Return results with float conversion for consistency
        return float(total_price), float(base_price_per_unit), float(price_per_person)
# [Your existing code for FoodItemMatcher, FoodCategoryRules, ItemMetadata, and HierarchicalFoodPredictor remains unchanged]

def load_and_train_model(data_path, item_vc_file="Book1.xlsx", item_data_file="item_data.csv", category_constraints=None):
    logger.info(f"Loading data from {data_path}")
    data = pd.read_excel(data_path)
    predictor = HierarchicalFoodPredictor(category_constraints=category_constraints, item_vc_file=item_vc_file, item_data_file=item_data_file)
    predictor.fit(data)
    return predictor

def save_model(predictor, filename="food_predictor.dill"):
    """Save the trained predictor model to a .dill file."""
    try:
        with open(filename, 'wb') as f:
            dill.dump(predictor, f)
        logger.info(f"Model saved successfully to {filename}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

def load_model(filename="adouble.dill"):
    """Load the trained predictor model from a .dill file."""
    try:
        with open(filename, 'rb') as f:
            predictor = dill.load(f)
        logger.info(f"Model loaded successfully from {filename}")
        return predictor
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def  get_predictions_from_terminal(predictor):
    print("\n==== Food Quantity Prediction System ====\n")
    print("Welcome to the Food Quantity Prediction System")
    print("This tool will help you determine how much food to prepare for your event.")

    event_time = input("Enter event time (e.g., Morning, Afternoon, Evening, Night): ").strip()
    meal_type = input("Enter meal type (e.g., Breakfast, Lunch, Dinner, Hi-Tea): ").strip()
    event_type = input("Enter event type (e.g., Wedding, Birthday Party): ").strip()

    while True:
        try:
            guest_count = int(input("Enter number of guests: ").strip())
            if guest_count > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    print("\nEnter menu items (one per line, enter 'done' when finished):")
    menu_items = []
    while True:
        item = input("> ").strip()
        if item.lower() == 'done':
            break
        if item:
            menu_items.append(item)

    print("\nProcessing your request...")
    predictions = predictor.predict(event_time, meal_type, event_type, guest_count, menu_items)

    print("\n==== Quantity Predictions ====\n")
    for item, qty_str in predictions.items():
        print(f"  {item}: {qty_str}")

    print("\n==== Price Predictions (Item-specific) ====\n")
    total_event_cost = 0
    for item, qty_str in predictions.items():
        total_qty_val = predictor.extract_quantity_value(qty_str)
        unit = predictor.extract_unit(qty_str)

        std_item = predictor.standardize_Item_name(item)
        mapped_item = getattr(predictor, 'Item_name_mapping', {}).get(std_item, item)
        if mapped_item in predictor.item_metadata:
            category = predictor.item_metadata[mapped_item].category
        else:
            category = predictor.guess_item_category(item)
            unit_inferred = predictor.food_rules.infer_default_unit(category)
            predictor.item_metadata[mapped_item] = ItemMetadata(category=category, unit=unit_inferred)
            predictor.item_metadata[std_item] = ItemMetadata(category=category, unit=unit_inferred)

        # Unpack three values from calculate_price (total_price, base_price_per_unit, price_per_person)
        total_price, base_price_per_unit, price_per_person = predictor.calculate_price(
            total_qty_val, category, guest_count, item, predicted_unit=unit
        )

        # Convert NumPy array to scalar if needed
        total_price = total_price.item() if isinstance(total_price, np.ndarray) else total_price
        base_price_per_unit = base_price_per_unit.item() if isinstance(base_price_per_unit,
                                                                       np.ndarray) else base_price_per_unit
        price_per_person = price_per_person.item() if isinstance(price_per_person, np.ndarray) else price_per_person

        total_event_cost += total_price

        # Calculate per person quantity for display
        per_person_qty = total_qty_val / guest_count if guest_count > 0 else 0

        print(f"  {item}: Total Price = ₹{total_price:.2f}, Price per {unit} = ₹{base_price_per_unit:.2f}, "
              f"Price per Person = ₹{price_per_person:.2f}")

    print("\n==== Event Summary ====\n")
    print(f"  Total Event Cost: ₹{total_event_cost:.2f}")
    price_per_guest = total_event_cost / guest_count if guest_count > 0 else 0
    print(f"  Price per Guest: ₹{price_per_guest:.2f}")

    return predictions
if __name__ == "__main__":
    # Train the model
    predictor = load_and_train_model("DB23.xlsx",
                                     item_vc_file="Book1.xlsx",
                                     item_data_file="item_data.csv")
    
    # Save the trained model to a .dill file
    save_model(predictor, "adouble.dill")
    
    # Optionally, test loading and using the model
    loaded_predictor = load_model("adouble.dill")
    predictions = get_predictions_from_terminal(loaded_predictor)
    
    print("\nThank you for using the Food Quantity Prediction System!")