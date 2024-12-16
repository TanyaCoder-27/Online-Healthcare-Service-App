import streamlit as st
from pathlib import Path
import google.generativeai as genai
from api_key import api_key
import os

genai.configure(api_key=api_key)
# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

# apply safety settings
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]
model = genai.GenerativeModel(
  model_name="gemini-1.5-pro-002",
  generation_config=generation_config,
  safety_settings = safety_settings
)

system_prompt="""
As a highly skilled medical practitioner specializing in image analysis, you are tasked with:

Your Responsibilities include:

1. Detailed Analysis: Thoroughly analyze each image, focusing on identifying any abnormal findings.
2. Findings Report: Document all observed anomalies or signs of disease. Clearly articulate these findings in a structured format.
3. Recommendations and Next Steps: Based on your analysis, suggest potential next steps, including follow-up tests or treatment plans as applicable.
4. Treatment Suggestions: If appropriate, recommend possible treatment options or interventions.

Important Notes:

1. Scope of Response: Only respond if the image pertains to human health issues.
2. Clarity of Image: In cases where the image quality impedes clear analysis, note that certain aspects are 'Unable to be determined based on the provided image.'
3. Disclaimer: Accompany your analysis with the disclaimer: "Consult with a Doctor before making any decisions."
4. Your insights are invaluable in guiding clinical decisions. Please proceed with the analysis,adhering to structured approach outlined above.
Please provide me an output response with these 4 headings: Detailed Analysis, Findings Report, Recommendations and Next Steps, Treatment Suggestions
"""


st.set_page_config(page_title ="Disease prediction",page_icon=":robot:")
st.title("🔍 Vital Image Analytics 📊")

uploaded_file=st.file_uploader("Upload the medical image for analysis",type=["png","jpg","jpeg"])

submit_button=st.button("Generate the analysis")

if submit_button:
    if uploaded_file is not None:
        try:
            temp_file_path = "image0.jpeg"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())  # Write directly from the uploaded file

            st.image(temp_file_path)

            # Correct way to create image_parts: read from the saved file
            image_parts = [
                {
                    "mime_type": "image/jpeg",  # Or the correct MIME type based on file upload
                    "data": Path(temp_file_path).read_bytes(),
                }
            ]

            prompt_parts = [image_parts[0], system_prompt]

            response = model.generate_content(prompt_parts)
            if response:
                st.title("Analysis:")
                st.write(response.text)
            else:
                st.error("Model did not generate a response.")

        except Exception as e:
            st.error(f"Error processing image: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    else:
        st.warning("Please upload an image.")