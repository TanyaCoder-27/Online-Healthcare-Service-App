import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

# Custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header
st.markdown("<div class='header'>", unsafe_allow_html=True)
st.markdown("<h1>About Our Service</h1>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# About content
st.markdown("""
## Our Mission : 
📢We aim to make preliminary health information more accessible to everyone through the power of artificial intelligence. Our disease prediction system uses advanced machine learning algorithms to analyze symptoms and provide potential insights about various health conditions. 🧠💡

## Important Note :
⚠️ This tool is for educational purposes only to check out your medical condition at the very moment and should not be used as a substitute for professional medical advice, diagnosis, or treatment for long run. Always consult with a qualified healthcare provider about any medical concerns. 🩺👨‍⚕️👩‍⚕️""" )