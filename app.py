import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(
    page_title="🧠 Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Bigger, bold, gradient title with shadow and slight animation */
    .title {
        font-size: 64px;
        font-weight: 900;
        background: linear-gradient(90deg, #004aad, #0073ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.25rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: shine 3s linear infinite;
    }
    @keyframes shine {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }

    .subtitle {
        font-size: 22px;
        font-weight: 600;
        color: #004aad;
        text-align: center;
        margin-top: 0;
        margin-bottom: 2rem;
        letter-spacing: 1.2px;
        font-style: italic;
    }

    /* Button styling */
    .stButton>button {
        background-color: #004aad;
        color: white;
        font-weight: 600;
        padding: 10px 20px;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #003080;
    }

    /* Confidence bar */
    .confidence-bar {
        height: 24px;
        border-radius: 12px;
        background: #e0e0e0;
        margin-top: 5px;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 12px;
        background: linear-gradient(90deg, #004aad, #0073ff);
        text-align: center;
        color: white;
        font-weight: 600;
        line-height: 24px;
    }

    /* Badge colors */
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 16px;
        color: white;
        margin-top: 12px;
    }
    .Glioma {background-color: #d9534f;}
    .Meningioma {background-color: #5bc0de;}
    .Pituitary {background-color: #f0ad4e;}
    .NoTumor {background-color: #5cb85c;}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("About this App")
    st.write("""
    This application uses a deep learning model trained on brain MRI scans to detect the presence and type of brain tumor.
    Upload a clear MRI scan image (JPG, PNG) and get a prediction with confidence score.

    **Tumor Types:**
    - Glioma Tumor
    - Meningioma Tumor
    - Pituitary Tumor
    - No Tumor (healthy scan)

    **Disclaimer:** This tool is for research/demo purposes only and does not replace professional medical diagnosis.
    """)

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Upload an MRI brain scan image.
    2. Wait for the model to analyze.
    3. See the prediction and confidence score.
    4. Consult a medical professional for confirmation.
    """)

st.markdown('<p class="title">🧠 Brain Tumor Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by Deep Learning - Upload your MRI scan to get started</p>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_brain_model():
    return load_model("model.keras")

model = load_brain_model()
class_names = ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"]

def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

        if hasattr(uploaded_file, "name"):
            st.write(f"**File Name:** {uploaded_file.name}")
        if hasattr(uploaded_file, "size"):
            size_kb = uploaded_file.size / 1024
            st.write(f"**File Size:** {size_kb:.2f} KB")

        input_img = preprocess_image(image)

        with st.spinner("Analyzing scan..."):
            prediction = model.predict(input_img)

        pred_index = np.argmax(prediction)
        pred_class = class_names[pred_index]
        confidence = prediction[0][pred_index] * 100

        badge_class = {
            "Glioma Tumor": "Glioma",
            "Meningioma Tumor": "Meningioma",
            "Pituitary Tumor": "Pituitary",
            "No Tumor": "NoTumor"
        }[pred_class]

        if confidence > 90:
            st.markdown(f'<span class="badge {badge_class}">{pred_class}</span>', unsafe_allow_html=True)

            st.write("**Confidence**")
            st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width:{confidence}%; min-width:50px;">
                        {confidence:.2f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(
                "The MRI scan could not be confidently recognized. "
                "Please consult a medical professional for an accurate diagnosis."
            )


    except Exception as e:
        st.error(f"Error processing the image: {e}")
else:
    st.info("Please upload an MRI scan image to get started.")

st.markdown("---")
st.markdown(
    """
    <p style="font-size:12px; color:#999; text-align:center;">
    © 2025 Brain Tumor Detection - For educational and research purposes only. Not a substitute for professional medical advice.
    </p>
    """,
    unsafe_allow_html=True
)
