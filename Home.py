'''import streamlit as st
import requests
from streamlit_lottie import st_lottie

# Page configuration
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load assets
load_css()
lottie_health = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_5njp3vgg.json")

# Header section
st.markdown("<div class='header'>", unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h1>AI-Powered Health Assistant</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='subtitle'>
        Your intelligent companion for preliminary disease prediction based on symptoms
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 2])
    with col_btn1:
        st.button("Get Started →", key="get_started")
    with col_btn2:
        st.button("Learn More ↓", key="learn_more")

with col2:
    st_lottie(lottie_health, height=300)

st.markdown("</div>", unsafe_allow_html=True)

# Features section
st.markdown("<div class='features-section'>", unsafe_allow_html=True)
st.markdown("<h2>Key Features</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🔍 Smart Symptom Analysis
    Advanced AI algorithms analyze your symptoms to provide preliminary disease predictions
    """)

with col2:
    st.markdown("""
    ### 📋 Comprehensive Information
    Detailed descriptions, precautions, and dietary recommendations for various conditions
    """)

with col3:
    st.markdown("""
    ### 🤖 Easy to Use
    User-friendly interface making health information accessible to everyone
    """)

st.markdown("</div>", unsafe_allow_html=True)

# How it works section
st.markdown("<div class='how-it-works'>", unsafe_allow_html=True)
st.markdown("<h2>How It Works</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 1. Select Symptoms
    Choose from a comprehensive list of symptoms you're experiencing
    """)

with col2:
    st.markdown("""
    #### 2. Get Prediction
    Our AI model analyzes your symptoms and provides potential conditions
    """)

with col3:
    st.markdown("""
    #### 3. View Details
    Access detailed information about the predicted condition
    """)

st.markdown("</div>", unsafe_allow_html=True)
'''

import streamlit as st
import requests
from streamlit_lottie import st_lottie

# Page configuration
st.set_page_config(
    page_title="Health Genie",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load assets
load_css()
lottie_health = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_tutvdkg0.json")

# Header section
st.markdown("<div class='header'>", unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h1>Health Genie</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='subtitle'>
        Your intelligent companion for preliminary disease prediction based on symptoms
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 2])
    with col_btn1:
        st.button("Get Started →", key="get_started")
    with col_btn2:
        st.button("Learn More ↓", key="learn_more")

with col2:
    st_lottie(lottie_health, height=300, key="lottie")

st.markdown("</div>", unsafe_allow_html=True)

# Features section
st.markdown("<div class='features-section'>", unsafe_allow_html=True)
st.markdown("<h2>Key Features</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🔍 Smart Symptom Analysis
    Advanced AI algorithms analyze your symptoms to provide preliminary disease predictions
    """)

with col2:
    st.markdown("""
    ### 📋 Comprehensive Information
    Detailed descriptions, precautions, and dietary recommendations for various conditions
    """)

with col3:
    st.markdown("""
    ### 🤖 Easy to Use
    User-friendly interface making health information accessible to everyone
    """)

st.markdown("</div>", unsafe_allow_html=True)

# How it works section
st.markdown("<div class='how-it-works'>", unsafe_allow_html=True)
st.markdown("<h2>How It Works</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1. Select Symptoms
    Choose from a comprehensive list of symptoms you're experiencing
    """)

with col2:
    st.markdown("""
    ### 2. Get Prediction
    Our AI model analyzes your symptoms and provides potential conditions
    """)

with col3:
    st.markdown("""
    ### 3. View Details
    Access detailed information about the predicted condition
    """)

st.markdown("</div>", unsafe_allow_html=True)