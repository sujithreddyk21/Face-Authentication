import streamlit as st
import requests
from PIL import Image
import io

# ---------------------------
# CONFIG
# ---------------------------
FASTAPI_URL = "http://localhost:8000/verify"   # Change if deployed elsewhere

st.title("🧑‍🦰 Face Verification App (Streamlit Frontend)")
st.write("Upload two images — the app will call FastAPI and verify if they belong to the same person.")

# ---------------------------
# IMAGE UPLOAD SECTION
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    img_file_a = st.file_uploader("Upload Image A", type=["jpg", "jpeg", "png"])

with col2:
    img_file_b = st.file_uploader("Upload Image B", type=["jpg", "jpeg", "png"])

threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.5)

# ---------------------------
# SHOW PREVIEW
# ---------------------------
if img_file_a:
    st.image(img_file_a, caption="Image A", width=250)

if img_file_b:
    st.image(img_file_b, caption="Image B", width=250)

# ---------------------------
# VERIFY BUTTON
# ---------------------------
if st.button("Verify Faces"):
    if not img_file_a or not img_file_b:
        st.error("Please upload both images.")
    else:
        with st.spinner("Verifying... 🔍"):
            # Prepare files for API
            files = {
                "image_a": (img_file_a.name, img_file_a.getvalue(), img_file_a.type),
                "image_b": (img_file_b.name, img_file_b.getvalue(), img_file_b.type)
            }

            params = {"threshold": threshold}

            try:
                response = requests.post(FASTAPI_URL, files=files, params=params)

                if response.status_code != 200:
                    st.error(f"API Error: {response.text}")
                else:
                    data = response.json()

                    # ---------------------------
                    # DISPLAY RESULTS
                    # ---------------------------
                    st.subheader("🔎 Verification Result")
                    st.write(f"**Prediction:** {data['verification']}")
                    st.write(f"**Similarity Score:** `{data['similarity']:.4f}`")
                    st.write(f"**Threshold Used:** `{data['threshold']}`")
                    st.write(f"**Embedding Dimensions:** `{data['embedding_dim']}`")

                    # ---------------------------
                    # FACE DETECTION INFO
                    # ---------------------------
                    st.subheader("Detected Faces Info")

                    st.json({
                        "Image A Faces": data["faces"]["image_a"],
                        "Image B Faces": data["faces"]["image_b"],
                        "Used Face Index": data["used_faces"],
                    })

            except Exception as e:
                st.error(f"Failed: {e}")
