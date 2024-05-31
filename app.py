import streamlit as st
import joblib
import numpy as np

st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    .header {
        color: red;
    }
    .full-width {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app title with red heading
st.write("<h1 class='header'>AwaasVaani.com</h1>", unsafe_allow_html=True)

# Try to load the trained model
model = None
try:
    model = joblib.load('GB_reg_model.pkl')
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Ensure model is loaded before proceeding
if model:
    # User input for features
    st.header('Input Features')

    # Input fields for numeric features
    area = st.number_input('Area (in sq ft)', value=0.0, min_value=0.0)
    bhk = st.number_input('BHK', value=1, min_value=1)
    bathroom = st.number_input('Bathroom', value=1, min_value=1)
    per_sqft = st.number_input('Price per Sqft', value=0.0, min_value=0.0)

    # Dropdowns for categorical features
    furnishing = st.selectbox('Furnishing', ['Unfurnished', 'Semi-Furnished', 'Furnished'])

    # Corrected 'Locality' dropdown
    locality_dict = {


        'rohini': 3,
        'saket': 4,
        'vasant kunj': 5,
        'pitampura': 6,
        'mayur vihar': 7,
        'dwarka sector 23': 8,
        'greater kailash': 9,
        'malviya nagar': 10,
        'karol bagh': 11,
        'south extension': 12,
        'green park': 13,
        'laxmi nagar': 14,
        'vasant vihar': 15,
        'noida': 16,
        'gurgaon': 17,
        'faridabad': 18,
        'rajouri garden': 19,
        'paschim vihar': 20,
        'vasundhara enclave': 21,
        'punjabi bagh': 22,
        'new friends colony': 23,
        'east of kailash': 24,
        'nehru place': 25,
        'shalimar bagh': 26,
        'indirapuram': 27
    }
    locality = st.selectbox('Locality', list(locality_dict.keys()))

    parking = st.selectbox('Parking', [0, 1, 2, 3, 4, 5])
    status = st.selectbox('Status', ['Ready to Move', 'Under Construction'])
    transaction = st.selectbox('Transaction', ['New', 'Resale'])
    type_ = st.selectbox('Type', ['Apartment', 'Independent House', 'Villa'])

    # Convert categorical inputs to numerical values if necessary
    furnishing_dict = {'Unfurnished': 0, 'Semi-Furnished': 1, 'Furnished': 2}
    status_dict = {'Ready to Move': 0, 'Under Construction': 1}
    transaction_dict = {'New': 0, 'Resale': 1}
    type_dict = {'Apartment': 0, 'Independent House': 1, 'Villa': 2}

    # Prepare input data for prediction
    input_data = np.array([
        area,
        bhk,
        bathroom,
        furnishing_dict[furnishing],
        locality_dict[locality],
        parking,
        status_dict[status],
        transaction_dict[transaction],
        type_dict[type_],
        per_sqft
    ]).reshape(1, -1)

    # Button for making predictions
    if st.button('Predict'):
        try:
            prediction = model.predict(input_data)
            st.success(f'The predicted price is: ₹{prediction[0]:,.2f}')
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    # Display model metrics (optional)
    st.header('Model Metrics')
    try:
        # Assuming you have access to x_train, y_train, x_test, y_test in your script
        # Placeholder data
        x_train, y_train, x_test, y_test = None, None, None, None
        # Replace the above None values with your actual data
        train_acc = model.score(x_train, y_train)
        test_acc = model.score(x_test, y_test)
        st.text(f'Training Accuracy: {train_acc}')
        st.text(f'Test Accuracy: {test_acc}')
    except Exception as e:
        st.error(f"Error displaying model metrics: {e}")
else:
    st.warning(
        "Model is not loaded. Ensure the model file is available and compatible with the current scikit-learn version.")
