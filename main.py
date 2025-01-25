import streamlit as st
import google.generativeai as genai
from api import api_key

# Configure the Generative AI API
genai.configure(api_key=api_key)

# Define the AI generation configuration
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Initialize models
text_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

vision_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Set Streamlit page configuration
st.set_page_config(page_title="CIKITSA AI", page_icon=":doctor:")

# Header
st.title("CIKITSA AI  🩺")
st.subheader("Your one-stop solution for Medical Report Summarization and X-Ray Disease Prediction")

# Define system prompts for both use cases
report_system_prompt = """"As a skilled medical report summarizer, your task is to simplify complex medical reports into clear, easy-to-understand language suitable for patients or non-medical individuals. You are to carefully interpret medical terminology, diagnoses, and procedures in a way that preserves the essential information and intended meaning, while removing unnecessary technical jargon.

Your Responsibilities:

Simplified Summary: Summarize the report in a way that a person without a medical background can easily understand, focusing on the key points of the diagnosis, findings, and any relevant conditions or symptoms.

Clear Explanation of Medical Terms: Explain any complex medical terms in simple language or provide definitions where necessary.

Main Takeaways: Highlight the most important findings, potential health implications, and any instructions or precautions for the patient.

Next Steps: Provide a clear summary of the recommended next steps, such as additional tests, follow-up appointments, or lifestyle changes.

Gentle Disclaimer: Add a friendly reminder to consult their doctor if they have further questions or if they are unsure about any part of the report.

Important Notes:

Scope of Response: Only respond if the content pertains to a medical or health-related report.
Patient Sensitivity: Use respectful and reassuring language that acknowledges the patient's concerns or anxiety.
Format of Output:

Please summarize the report under the following sections:

1. Key Findings: A brief explanation of the main health issue(s) or conditions identified. 2. Explanation of Terms: A list of complex medical terms with their definitions, if necessary. 3. Next Steps and Recommendations: Clear instructions on what the patient should do next, including further tests, treatments, or lifestyle adjustments. 4. Patient Guidance: Any important advice for the patient regarding managing their condition or seeking support.
3.Mention the patients name at the beginning of the report and the date of the report if available.

At the end, include this disclaimer: "Please consult your healthcare provider for any questions or additional guidance about your condition or treatment options."
"""

disease_list = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 
    'Nodule', 'Pleural Thickening', 'Pneumonia', 'Pneumothorax'
]

xray_system_prompt =f"""As a medical image analyst, your task is to evaluate chest X-rays 
    for the presence of the following diseases:
    {', '.join(disease_list)}.
    Predict the likelihood of each disease being present and provide a summary 
    in patient-friendly language. 

    Provide the output in the following format:
    1. **Disease Probabilities:** A ranked list of diseases with their predicted likelihoods (as percentages).
    2. **Key Findings:** Highlight the most significant abnormalities detected.
    3. **Next Steps:** Offer recommendations for further testing, consultation, or treatment if needed.

    Add this disclaimer at the end: 
    "These results are predictions and should not replace professional medical advice. Please consult a healthcare provider for a definitive diagnosis."
    """

# Sidebar with options for users
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose the Analysis Type", ("Medical Report Summarizer", "Chest X-Ray Analysis"))

if option == "Medical Report Summarizer":
    # Medical Report Summarizer Section
    st.subheader("Summarize Your Medical Report")
    uploaded_report_file = st.file_uploader("Upload your medical report image (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

    if uploaded_report_file:
        st.image(uploaded_report_file, width=300, caption="Uploaded Medical Report Image")
    
    submit_button = st.button("Generate Report Summary")
    
    if submit_button and uploaded_report_file:
        with st.spinner('Processing...'):
            # Prepare the image for analysis
            image_data = uploaded_report_file.getvalue()
            image_parts = [{"mime_type": "image/jpeg", "data": image_data}]
            prompt_parts = [image_parts[0], report_system_prompt]

            # Generate the report summary
            response = text_model.generate_content(prompt_parts)

            if response:
                st.subheader("Here is the Summary of Your Report:")
                st.write(response.text)
            else:
                st.error("Sorry, there was an error generating the report.")

elif option == "Chest X-Ray Analysis":
    # Chest X-Ray Disease Prediction Section
    st.subheader("Analyze Your Chest X-Ray for Disease Prediction")
    uploaded_xray_file = st.file_uploader("Upload your chest X-ray image (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

    if uploaded_xray_file:
        st.image(uploaded_xray_file, width=400, caption="Uploaded Chest X-Ray Image")
    
    xray_submit_button = st.button("Analyze X-Ray")
    
    if xray_submit_button and uploaded_xray_file:
        with st.spinner('Analyzing...'):
            # Prepare the image for analysis
            image_data = uploaded_xray_file.getvalue()
            image_parts = [{"mime_type": "image/jpeg", "data": image_data}]
            prompt_parts = [xray_system_prompt, image_parts[0]]

            # Generate the X-ray disease analysis
            response = vision_model.generate_content(prompt_parts)

            if response:
                st.subheader("X-Ray Analysis Results")
                st.write(response.text)
            else:
                st.error("Sorry, there was an error analyzing the X-ray.")

