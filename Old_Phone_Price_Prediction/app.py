import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Old Phone Price Prediction", page_icon="📱", layout="centered")
MODEL_PATH = "model.pkl"


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


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


def get_columns(branch_name):
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


def norm(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def find_feature(features, aliases):
    aliases = {norm(alias) for alias in aliases}
    return next((feature for feature in features if norm(feature) in aliases), None)


numeric_columns = get_columns("num")
categorical_columns = get_columns("cat")
all_features = numeric_columns + categorical_columns

if not all_features:
    st.error("Could not identify model input columns. Check that your preprocessor has 'num' and 'cat' branches.")
    st.stop()

ram_feature = find_feature(all_features, ["ram", "ram_gb", "ram_size", "memory_ram"])
storage_feature = find_feature(
    all_features, ["storage", "internal_storage", "storage_gb", "rom", "rom_gb", "internal_memory"]
)

# Match these aliases to the exact column names used in your training dataset.
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
    feature = find_feature(all_features, aliases)
    if feature is not None:
        yes_no_features[feature] = label

# Other common binary features such as 4G/5G are also displayed as Yes/No.
for feature in all_features:
    if feature in yes_no_features:
        continue
    name = norm(feature)
    if any(word in name for word in ["5g", "4g", "3g", "nfc", "wifi", "bluetooth", "dualsim", "touchscreen"]):
        yes_no_features[feature] = str(feature).replace("_", " ").title()

categorical_encoder = preprocessor.named_transformers_.get("cat")
categorical_categories = {}
if categorical_encoder is not None and hasattr(categorical_encoder, "categories_"):
    categorical_categories = dict(zip(categorical_columns, categorical_encoder.categories_))

ram_options = [4, 6, 8, 12, 16]
storage_options = [64, 128, 256, 512, 1024]

with st.form("prediction_form"):
    st.subheader("Phone details")
    input_values = {}

    if ram_feature:
        st.markdown("#### RAM")
        input_values[ram_feature] = st.selectbox("RAM (GB)", ram_options, index=2)

    if storage_feature and storage_feature != ram_feature:
        st.markdown("#### Internal storage")
        input_values[storage_feature] = st.selectbox("Storage (GB)", storage_options, index=2)

    binary_features = [f for f in all_features if f in yes_no_features]
    if binary_features:
        st.markdown("#### Condition & accessories")
        cols = st.columns(2)
        for idx, feature in enumerate(binary_features):
            with cols[idx % 2]:
                answer = st.radio(
                    yes_no_features[feature],
                    ["No", "Yes"],
                    index=0,
                    horizontal=True,
                    key=f"yes_no_{feature}",
                )
                # Convert the answer to 0/1 for the model; the UI shows only Yes/No.
                input_values[feature] = 1 if answer == "Yes" else 0

    remaining_numeric = [f for f in numeric_columns if f not in input_values]
    if remaining_numeric:
        st.markdown("#### Other specifications")
        cols = st.columns(2)
        integer_keywords = ["year", "age", "battery", "capacity", "cores", "weight", "height", "width", "count", "price"]
        for idx, feature in enumerate(remaining_numeric):
            with cols[idx % 2]:
                is_integer = any(word in norm(feature) for word in integer_keywords)
                if is_integer:
                    value = st.number_input(
                        str(feature).replace("_", " ").title(),
                        min_value=0, value=None, step=1,
                        placeholder="Enter a whole number", key=f"num_{feature}"
                    )
                else:
                    value = st.number_input(
                        str(feature).replace("_", " ").title(),
                        min_value=0.0, value=None, step=0.1,
                        placeholder="Enter a value", key=f"num_{feature}"
                    )
                input_values[feature] = value

    remaining_categorical = [f for f in categorical_columns if f not in input_values]
    if remaining_categorical:
        st.markdown("#### Other details")
        cols = st.columns(2)
        for idx, feature in enumerate(remaining_categorical):
            with cols[idx % 2]:
                options = categorical_categories.get(feature)
                if options is not None and len(options) > 0:
                    input_values[feature] = st.selectbox(
                        str(feature).replace("_", " ").title(),
                        list(options), key=f"cat_{feature}"
                    )
                else:
                    input_values[feature] = st.text_input(
                        str(feature).replace("_", " ").title(),
                        placeholder="Enter details", key=f"text_{feature}"
                    )

    submitted = st.form_submit_button("Predict resale price", type="primary", use_container_width=True)

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
        st.warning("Please complete these fields: " + ", ".join(map(str, empty_fields)))
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
