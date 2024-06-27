
import numpy as np
import pickle
import streamlit as st

def load_model():
 
  try:
    with open('bill.sav', 'rb') as file:
      loaded_model = pickle.load(file)
    return loaded_model
  except FileNotFoundError:
    st.error("Model file not found. Please ensure 'model.sav' is in the correct location.")
    return None  # Indicate model loading failure

def predict_bill(amount, model):
  if model is None:
    return None  # Model loading failed

  try:
    # Convert amount to float, handling potential exceptions
    amount_as_float = float(amount)
  except ValueError:
    st.error("Invalid input. Please enter a numerical value for the amount.")
    return None

  # Reshape the input data for prediction
  input_data_reshaped = np.asarray([amount_as_float]).reshape(1, -1)

  prediction = model.predict(input_data_reshaped)[0]
  return prediction

def main():
  st.title('Electricity Bill Prediction Web App')

  # Load the model (handle potential loading errors)
  loaded_model = load_model()
  if loaded_model is None:
    return  # Stop execution if model loading failed

  # Get user input
  amount = st.text_input('Enter Last Month Bill')

  # Predict bill and display results
  if st.button('Predict Bill'):
    prediction = predict_bill(amount, loaded_model)
    if prediction is not None:
      st.success(f"Predicted Electricity Bill: {prediction:.2f}")  # Format prediction with 2 decimal places

if __name__ == '__main__':
  main()