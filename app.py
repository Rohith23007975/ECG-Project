import streamlit as st
import pandas as pd
from Scripts.sampledep import run_ecg_inference
import matplotlib.pyplot as plt

# ----------------------------
# ECG Class Details
# ----------------------------
ecg_class_details = {
    "N": {
        "label": "Normal Heartbeat",
        "description": "The heart rhythm appears normal and regular.",
        "icon": "✅",
        "color": "#A7D7C5"  # soft watercolor green
    },
    "S": {
        "label": "Supraventricular Premature Beat",
        "description": "An early heartbeat originating from the upper chambers of the heart.",
        "icon": "⚠️",
        "color": "#F6D6A8"  # soft watercolor amber
    },
    "V": {
        "label": "Ventricular Premature Beat",
        "description": "An abnormal early heartbeat originating from the lower chambers.",
        "icon": "🔴",
        "color": "#E6A6A6"  # soft watercolor red
    },
    "F": {
        "label": "Fusion Beat",
        "description": "A heartbeat formed by a mix of normal and ventricular impulses.",
        "icon": "🔶",
        "color": "#F2C6A0"  # soft watercolor orange
    }
}


# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="ECG Arrhythmia Detection",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# Custom CSS for Medical Minimalistic Design
# ----------------------------
st.markdown("""
    <style>
    /* Import Medical-friendly Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Title Styling */
    h1 {
        color: #1e3a5f;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    /* Subtitle Styling */
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Upload Section Card */
    .upload-card {
        background: white;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 10px 20px rgba(0, 0, 0, 0.03);
        margin: 2rem 0;
        border-left: 4px solid #3b82f6;
    }
    
    /* Results Card */
    .results-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 10px 20px rgba(0, 0, 0, 0.03);
        margin: 1.5rem 0;
    }
    
    /* Prediction Result Box */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
    }
    
    .prediction-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .prediction-value {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    /* Confidence Box */
    .confidence-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 16px rgba(245, 87, 108, 0.3);
    }
    
    /* Distribution Card */
    .distribution-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-top: 1.5rem;
    }
    
    .distribution-title {
        color: #1e3a5f;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Beat Item */
    .beat-item {
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-left: 3px solid #3b82f6;
        transition: transform 0.2s ease;
    }
    
    .beat-item:hover {
        transform: translateX(5px);
    }
    
    .beat-type {
        color: #475569;
        font-weight: 500;
    }
    
    .beat-count {
        color: #1e3a5f;
        font-weight: 700;
        font-size: 1.2rem;
    }
    
    /* File Uploader Styling */
    .stFileUploader > div > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Primary Button Styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 3rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4);
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }
    
    .stButton > button[kind="primary"]:active {
        transform: translateY(-1px);
    }
    
    /* Spinner Styling */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Section Headers */
    .section-header {
        color: #1e3a5f;
        font-size: 2rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Info Box */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        color: #1e3a5f;
        margin: 1rem 0;
    }
    
    /* Error Box */
    .error-box {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        color: #991b1b;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Header Section
# ----------------------------
st.markdown("# 🫀 ECG Arrhythmia Detection")
st.markdown('<p class="subtitle">Advanced cardiac rhythm analysis powered by machine learning</p>', unsafe_allow_html=True)

# ----------------------------
# Information Box
# ----------------------------
st.markdown("""
    <div class="info-box">
        <strong>📋 Instructions:</strong> Upload a CSV file containing ECG signal data. 
        The system will automatically analyze the recording and detect arrhythmia patterns.
    </div>
""", unsafe_allow_html=True)

# ----------------------------
# Upload Section
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose ECG CSV File",
    type=["csv"],
    help="Upload a CSV file with ECG signal data"
)

# Show analyze button only if file is uploaded
if uploaded_file is not None:
    st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
    analyze_button = st.button("🔬 Start Analysis", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    analyze_button = False

# ----------------------------
# Analysis and Results
# ----------------------------
if uploaded_file is not None and analyze_button:
    with st.spinner("🔬 Analyzing ECG signal..."):
        result = run_ecg_inference(uploaded_file)
    
    if "error" in result:
        st.markdown(f"""
            <div class="error-box">
                <strong>⚠️ Error:</strong> {result["error"]}
            </div>
        """, unsafe_allow_html=True)
    else:
        # Get class details
        predicted_class = result['predicted_label']
        class_info = ecg_class_details.get(predicted_class, {
            "label": predicted_class,
            "description": "Unknown classification",
            "icon": "❓",
            "color": "#64748b"
        })
        
        # Prediction Result
        st.markdown('<div class="results-card">', unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="prediction-box" style="background: linear-gradient(135deg, {class_info['color']} 0%, {class_info['color']}dd 100%);">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">{class_info['icon']}</div>
                <div class="prediction-label">Detected Arrhythmia Type</div>
                <div class="prediction-value">{predicted_class}</div>
                <div style="font-size: 1.3rem; font-weight: 600; margin-top: 0.5rem; opacity: 0.95;">
                    {class_info['label']}
                </div>
                <div style="font-size: 0.95rem; margin-top: 1rem; opacity: 0.9; line-height: 1.5;">
                    {class_info['description']}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Confidence Score
        st.markdown(f"""
            <div class="confidence-box">
                <div class="prediction-label">Confidence Level</div>
                <div class="prediction-value">{result['confidence']}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Beat Distribution
        st.markdown('<div class="section-header"> Beat Distribution Analysis</div>', unsafe_allow_html=True)
        
        for beat_type, count in result["beat_distribution"].items():
            st.markdown(f"""
                <div class="beat-item">
                    <span class="beat-type">{beat_type}</span>
                    <span class="beat-count">{count}</span>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    ""