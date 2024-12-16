import pandas as pd
import streamlit as st
import joblib
from huggingface_hub import InferenceClient

# Load the best model and the vectorizer
best_model = joblib.load('best_disease_predictor_model_svm.pkl')

# Load disease information
disease_info = pd.read_csv("test_ds_2.csv")

# Initialize Hugging Face client
client = InferenceClient(api_key="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")

# Function to get disease information based on prediction
def get_disease_info(disease):
    info = disease_info[disease_info['Disease'] == disease].iloc[0]
    return {
        "description": info['Description'],
        "precautions": info['Precautions']
    }

# Function to generate concise statements about the disease using Hugging Face
def generate_disease_statements(disease):
    system_prompt = "Provide very concise and informative statements about the disease. No introductory statements, and no incomplete sentences. Just give small, clear points about foods to eat, foods to avoid, and precautions."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Provide simple advice on what to eat, avoid, and key precautions for {disease}."}
    ]

    responses = []
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=messages,
            max_tokens=250  # Limit the response length
        )

        # Extract the response content correctly
        if hasattr(response, 'choices') and response.choices:
            response_text = response.choices[0].message['content']
            responses.append(response_text)
        else:
            responses.append('No response available.')
    except Exception as e:
        print(f"Error while fetching response from Hugging Face: {e}")
        responses.append(f"Error retrieving information for {disease}.")

    return responses

# Main function for the Streamlit app
def main():
    st.title("Disease Prediction Based on Symptoms and Additional Factors")

    # Symptom selection
    symptoms = disease_info['Symptoms'].str.split(', ').explode().unique().tolist()
    selected_symptoms = st.multiselect("Select Symptoms", symptoms)

    # Additional inputs
    severity = st.selectbox("Select Severity", ["1", "2", "3", "4", "5"])
    duration = st.selectbox("Select Duration", ["Short-term", "Medium-term", "Long-term"])
    frequency = st.selectbox("Select Frequency", ["Often", "Sometimes", "Rare"])

    # Button to predict
    if st.button("Predict Disease"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            # Prepare the input for prediction
            symptoms_input = ", ".join(selected_symptoms)
            input_data = pd.DataFrame({
                'Symptoms': [symptoms_input],
                'Severity': [severity],
                'Duration': [duration],
                'Frequency': [frequency]
            })

            # Predict the disease using the best model
            try:
                predicted_disease = best_model.predict(input_data)[0]

                # Get disease information
                disease_details = get_disease_info(predicted_disease)

                # Generate additional statements using Hugging Face after prediction
                additional_statements = generate_disease_statements(predicted_disease)

                # Display results
                st.success(f"Disease Predicted: **{predicted_disease}**")
                st.subheader("Disease Description")
                st.write(disease_details["description"])
                st.subheader("Precautions")
                st.write(disease_details["precautions"])
                st.subheader("Additional Statements")
                for statement in additional_statements:
                    st.write(statement)

            except Exception as e:
                st.error(f"Error in prediction: {e}")

# Run the Streamlit app
if __name__ == '__main__':
    main()
