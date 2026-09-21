import streamlit as st
import pandas as pd
import pickle


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #f8f9fa;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 25px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 25px;
        font-weight: bold;
        margin-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.markdown(
    '<div class="title">🏠 House Price Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict house prices using a trained Linear Regression model'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# PROJECT INFORMATION
# ---------------------------------------------------------

st.info(
    "Enter the house details below. "
    "The trained Machine Learning model will predict the estimated house price."
)


# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🏡 House Details</div>',
    unsafe_allow_html=True
)


# Create two columns
col1, col2 = st.columns(2)


# ---------------------------------------------------------
# COLUMN 1 INPUTS
# ---------------------------------------------------------

with col1:

    Square_Footage = st.number_input(
        "Square Footage",
        min_value=0.0,
        value=1500.0,
        step=100.0
    )

    Num_Bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=0,
        value=3,
        step=1
    )

    Num_Bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=0,
        value=2,
        step=1
    )

    Year_Built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2010,
        step=1
    )


# ---------------------------------------------------------
# COLUMN 2 INPUTS
# ---------------------------------------------------------

with col2:

    Lot_Size = st.number_input(
        "Lot Size",
        min_value=0.0,
        value=0.5,
        step=0.1
    )

    Garage_Size = st.number_input(
        "Garage Size",
        min_value=0,
        value=2,
        step=1
    )

    Neighborhood_Quality = st.number_input(
        "Neighborhood Quality",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )


# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------

st.markdown("---")

predict_button = st.button(
    "🔮 Predict House Price",
    use_container_width=True
)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if predict_button:

    # Create DataFrame with the same feature names
    # used while training the model

    input_data = pd.DataFrame({
        "Square_Footage": [Square_Footage],
        "Num_Bedrooms": [Num_Bedrooms],
        "Num_Bathrooms": [Num_Bathrooms],
        "Year_Built": [Year_Built],
        "Lot_Size": [Lot_Size],
        "Garage_Size": [Garage_Size],
        "Neighborhood_Quality": [Neighborhood_Quality]
    })


    # Make prediction

    prediction = model.predict(input_data)


    # Get predicted value

    predicted_price = prediction[0]


    # Display result

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )

    st.success(
        f"🏠 Estimated House Price: ${predicted_price:,.2f}"
    )


    # Show entered values

    st.markdown(
        '<div class="section-title">📋 Entered House Details</div>',
        unsafe_allow_html=True
    )

    display_data = pd.DataFrame({
        "Feature": [
            "Square Footage",
            "Number of Bedrooms",
            "Number of Bathrooms",
            "Year Built",
            "Lot Size",
            "Garage Size",
            "Neighborhood Quality"
        ],
        "Value": [
            Square_Footage,
            Num_Bedrooms,
            Num_Bathrooms,
            Year_Built,
            Lot_Size,
            Garage_Size,
            Neighborhood_Quality
        ]
    })

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "House Price Prediction | Machine Learning Regression Project"
)