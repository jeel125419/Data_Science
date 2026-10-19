import joblib
import pandas as pd
import streamlit as st
from datetime import date
from pathlib import Path
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Old Phone Price Prediction", page_icon="📱", layout="centered")

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "model.pkl"
CATALOG_PATH = APP_DIR / "brand_model_catalog.csv"

CATALOG_BRAND_COLUMN = "brand"
CATALOG_MODEL_COLUMN = "model"


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data
def load_catalog(path: str):
    if not Path(path).exists():
        return None
    catalog = pd.read_csv(path)
    if CATALOG_BRAND_COLUMN not in catalog.columns or CATALOG_MODEL_COLUMN not in catalog.columns:
        return None
    return catalog[[CATALOG_BRAND_COLUMN, CATALOG_MODEL_COLUMN]].dropna().drop_duplicates()


def norm(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def find_feature(features, aliases):
    aliases = {norm(alias) for alias in aliases}
    return next((feature for feature in features if norm(feature) in aliases), None)


def get_columns(preprocessor, branch_name):
    try:
        branch = preprocessor.named_transformers_[branch_name]
        if hasattr(branch, "feature_names_in_"):
            return list(branch.feature_names_in_)
    except Exception:
        pass
    for name, transformer, columns in preprocessor.transformers_:
        if name == branch_name and columns is not None:
            return list(columns)
    return []


st.title("📱 Old Phone Price Prediction")
st.write("Enter the phone details below to estimate its resale price.")

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found. Place `{MODEL_PATH}` beside `app.py`.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

if not isinstance(model, Pipeline):
    st.error("The saved file must contain the complete fitted scikit-learn Pipeline.")
    st.stop()
if "preprocessor" not in model.named_steps:
    st.error("The pipeline must contain a step named 'preprocessor'.")
    st.stop()

preprocessor = model.named_steps["preprocessor"]
numeric_columns = get_columns(preprocessor, "num")
categorical_columns = get_columns(preprocessor, "cat")
all_features = numeric_columns + categorical_columns

if not all_features:
    st.error("Could not identify model input columns. Check that your preprocessor has 'num' and 'cat' branches.")
    st.stop()


def feature(*aliases):
    return find_feature(all_features, list(aliases))


ram_feature = feature("ram", "ram_gb", "ram_size", "memory_ram")
storage_feature = feature("storage", "internal_storage", "storage_gb", "rom", "rom_gb", "internal_memory")
brand_feature = feature("brand", "phone_brand", "manufacturer", "brand_name")
model_feature = feature("model", "phone_model", "device_model", "model_name", "phone_model_name")
launch_year_feature = feature("launch_year", "year_of_launch", "release_year", "phone_launch_year")
purchase_year_feature = feature("purchase_year", "year_purchased", "year_of_purchase")
battery_capacity_feature = feature("battery_capacity", "battery_capacity_mah", "battery_mah")
battery_health_feature = feature("battery_health", "battery_health_percent", "battery_health_percentage")
months_since_purchase_feature = feature(
    "months_since_purchase", "month_since_purchase", "purchase_age_months", "months_owned", "age_months"
)

yes_no_aliases = {
    "Screen cracked": ["screen_cracked", "screen_crack", "cracked_screen", "screen_damage", "display_cracked", "display_damage", "cracked_display"],
    "Body damage": ["body_damage", "body_damaged", "physical_damage", "body_condition_damage"],
    "Water damage": ["water_damage", "water_damaged", "liquid_damage"],
    "Repair history": ["repair_history", "repaired", "previously_repaired", "ever_repaired", "repair_done"],
    "Box available": ["box_available", "original_box", "box_included", "has_box"],
    "Charger available": ["charger_available", "charger_included", "has_charger", "charger"],
}
yes_no_features = {}
for label, aliases in yes_no_aliases.items():
    found = find_feature(all_features, aliases)
    if found:
        yes_no_features[found] = label

for col in all_features:
    if col in yes_no_features:
        continue
    name = norm(col)
    if any(word in name for word in ["5g", "4g", "3g", "nfc", "wifi", "bluetooth", "dualsim", "touchscreen"]):
        yes_no_features[col] = str(col).replace("_", " ").title()

categorical_encoder = preprocessor.named_transformers_.get("cat")
categorical_categories = {}
if categorical_encoder is not None and hasattr(categorical_encoder, "categories_"):
    categorical_categories = dict(zip(categorical_columns, categorical_encoder.categories_))

ram_options = [4, 6, 8, 12, 16]
storage_options = [64, 128, 256, 512, 1024]
today = date.today()
catalog = load_catalog(CATALOG_PATH)

# These dropdowns intentionally live OUTSIDE st.form so changing the brand
# immediately reruns Streamlit and refreshes the available phone models.
if not brand_feature or not model_feature:
    st.error(
        "Could not identify the brand and model columns from the trained pipeline. "
        "Check the column names used when training model.pkl."
    )
    st.stop()

if catalog is None:
    st.error(
        f"Brand/model catalog not found or invalid: `{CATALOG_PATH}`. "
        "Create it from your dataset with columns named `brand` and `model`, "
        "then place it beside `app.py`."
    )
    st.code(
        'df[["brand", "model"]].dropna().drop_duplicates().to_csv(\n'
        '    "brand_model_catalog.csv", index=False\n'
        ')',
        language="python",
    )
    st.stop()

catalog_for_features = catalog.rename(columns={
    CATALOG_BRAND_COLUMN: brand_feature,
    CATALOG_MODEL_COLUMN: model_feature,
})
brand_options = sorted(catalog_for_features[brand_feature].dropna().unique().tolist())

if not brand_options:
    st.error("No brands were found in the brand/model catalog.")
    st.stop()

selected_brand = st.selectbox("Brand", brand_options, key="brand_select")
available_models = sorted(
    catalog_for_features.loc[
        catalog_for_features[brand_feature] == selected_brand, model_feature
    ].dropna().unique().tolist()
)

if not available_models:
    st.error("No phone models were found for the selected brand.")
    st.stop()

selected_model = st.selectbox(
    "Phone Model",
    available_models,
    key=f"model_select_{selected_brand}",
)

with st.form("prediction_form"):
    st.subheader("Phone details")
    input_values = {
        brand_feature: selected_brand,
        model_feature: selected_model,
    }

    if ram_feature:
        st.markdown("#### RAM")
        input_values[ram_feature] = st.selectbox("RAM (GB)", ram_options, index=2)

    if storage_feature and storage_feature != ram_feature:
        st.markdown("#### Internal storage")
        input_values[storage_feature] = st.selectbox("Storage (GB)", storage_options, index=2)

    # Ask for purchase month/year and calculate months automatically, if the model has that feature.
    if months_since_purchase_feature:
        st.markdown("#### Purchase date")
        st.caption(
            "Select when the current owner bought the phone. The app calculates "
            "the number of months since purchase automatically."
        )
        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        col1, col2 = st.columns(2)
        with col1:
            purchase_month = st.selectbox(
                "Purchase month", list(range(1, 13)),
                format_func=lambda m: month_names[m - 1],
                index=today.month - 1,
            )
        with col2:
            purchase_year = st.selectbox(
                "Purchase year", list(range(today.year, 1999, -1)), index=0
            )
        if (purchase_year, purchase_month) > (today.year, today.month):
            st.error("Purchase date cannot be in the future.")
            input_values[months_since_purchase_feature] = None
        else:
            months_elapsed = (today.year - purchase_year) * 12 + today.month - purchase_month
            input_values[months_since_purchase_feature] = months_elapsed
            # If the trained model also expects the purchase year, provide it too.
            if purchase_year_feature:
                input_values[purchase_year_feature] = purchase_year
            st.caption(f"Calculated time since purchase: {months_elapsed} months")

    binary_features = [f for f in all_features if f in yes_no_features and f not in input_values]
    if binary_features:
        st.markdown("#### Condition & accessories")
        cols = st.columns(2)
        for idx, col in enumerate(binary_features):
            with cols[idx % 2]:
                answer = st.radio(
                    yes_no_features[col], ["No", "Yes"], index=0,
                    horizontal=True, key=f"yes_no_{col}",
                )
                input_values[col] = 1 if answer == "Yes" else 0

    if launch_year_feature and launch_year_feature not in input_values:
        st.markdown("#### Launch information")
        input_values[launch_year_feature] = st.number_input(
            "Phone Launch Year", min_value=1990, max_value=today.year,
            value=min(2023, today.year), step=1,
            help="Enter the year this phone model was first launched. Future years are not allowed.",
        )

    if battery_capacity_feature and battery_capacity_feature not in input_values:
        input_values[battery_capacity_feature] = st.number_input(
            "Battery Capacity (mAh)", min_value=1, value=5000, step=100,
            help="Battery capacity must be positive.",
        )

    if battery_health_feature and battery_health_feature not in input_values:
        input_values[battery_health_feature] = st.slider(
            "Battery Health (%)", min_value=0, max_value=100, value=100,
            help="Battery health must be between 0% and 100%.",
        )

    special_features = {
        f for f in [
            launch_year_feature, purchase_year_feature, battery_capacity_feature,
            battery_health_feature, months_since_purchase_feature,
        ] if f is not None
    }
    remaining_numeric = [f for f in numeric_columns if f not in input_values and f not in special_features]

    if remaining_numeric:
        st.markdown("#### Other specifications")
        cols = st.columns(2)
        integer_keywords = ["year", "age", "battery", "capacity", "cores", "weight", "height", "width", "count", "price", "month"]
        for idx, col in enumerate(remaining_numeric):
            with cols[idx % 2]:
                label = str(col).replace("_", " ").title()
                if "price" in norm(col):
                    label += " (₹)"
                is_integer = any(word in norm(col) for word in integer_keywords)
                if is_integer:
                    value = st.number_input(
                        label, min_value=0, value=None, step=1,
                        placeholder="Enter a non-negative whole number", key=f"num_{col}",
                    )
                else:
                    value = st.number_input(
                        label, min_value=0.0, value=None, step=0.1,
                        placeholder="Enter a non-negative value", key=f"num_{col}",
                    )
                input_values[col] = value

    remaining_categorical = [f for f in categorical_columns if f not in input_values]
    if remaining_categorical:
        st.markdown("#### Other details")
        cols = st.columns(2)
        for idx, col in enumerate(remaining_categorical):
            with cols[idx % 2]:
                options = categorical_categories.get(col)
                if options is not None and len(options) > 0:
                    input_values[col] = st.selectbox(
                        str(col).replace("_", " ").title(), list(options), key=f"cat_{col}"
                    )
                else:
                    input_values[col] = st.text_input(
                        str(col).replace("_", " ").title(),
                        placeholder="Enter details", key=f"text_{col}",
                    )

    submitted = st.form_submit_button(
        "Predict resale price", type="primary", use_container_width=True
    )

if submitted:
    input_df = pd.DataFrame([input_values])

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        missing = [f for f in expected_features if f not in input_df.columns]
        if missing:
            st.error("Missing input controls for model features: " + ", ".join(map(str, missing)))
            st.stop()
        input_df = input_df[expected_features]

    if input_df.isnull().any().any():
        empty_fields = input_df.columns[input_df.isnull().any()].tolist()
        st.warning("Please complete or correct these fields: " + ", ".join(map(str, empty_fields)))
        st.stop()

    for col in input_df.select_dtypes(include="number").columns:
        value = input_df.iloc[0][col]
        if pd.isna(value) or value < 0:
            st.error(f"{col} must be a non-negative value.")
            st.stop()

    if battery_health_feature and battery_health_feature in input_df.columns:
        health = input_df.iloc[0][battery_health_feature]
        if not 0 <= health <= 100:
            st.error("Battery health must be between 0 and 100.")
            st.stop()

    if launch_year_feature and launch_year_feature in input_df.columns:
        if input_df.iloc[0][launch_year_feature] > today.year:
            st.error("Launch year cannot be in the future.")
            st.stop()

    try:
        prediction = float(model.predict(input_df)[0])
        st.success("Prediction complete!")
        st.metric("Estimated resale price", f"₹ {prediction:,.2f}")
        st.caption("This is an estimate, not a guaranteed selling price.")
        with st.expander("Review submitted details"):
            st.dataframe(input_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info("Check that the selected options and feature columns match your training data.")
