import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(page_title="Chest X-ray AI Assistant", layout="wide")

st.title("Chest X-ray AI Assistant")
st.caption("Upload a chest X-ray image to get disease prediction, generated report, and Grad-CAM visualization.")
st.warning("This tool is for research and educational use only. It is not a clinical diagnosis system.")


def build_probability_table(probabilities: dict, threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for label, prob in probabilities.items():
        status = "Likely Positive" if prob >= threshold else "Low Confidence / Negative"
        rows.append(
            {
                "Condition": label,
                "Probability": round(float(prob), 4),
                "Status": status,
            }
        )
    df = pd.DataFrame(rows).sort_values("Probability", ascending=False).reset_index(drop=True)
    return df


def highlight_probability_rows(row):
    if row["Probability"] >= 0.5:
        return ["background-color: #ffe6e6; font-weight: bold;"] * len(row)
    return [""] * len(row)


uploaded_file = st.file_uploader(
    "Upload one chest X-ray image",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    help="Please upload one image at a time."
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    if st.button("Analyze Image", use_container_width=True):
        with st.spinner("Running inference..."):
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "image/png")
            }

            try:
                response = requests.post(BACKEND_URL, files=files, timeout=120)
            except Exception as e:
                st.error(f"Backend connection failed: {e}")
                st.stop()

        if response.status_code != 200:
            st.error(f"Backend error: {response.status_code}")
            st.text(response.text)
        else:
            result = response.json()

            # Summary metrics
            st.subheader("Prediction Summary")
            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("Top Predicted Label", result["top_label"])

            with col_b:
                st.metric("Top Probability", f"{result['top_probability']:.4f}")

            # Images
            st.subheader("Images")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Uploaded Image**")
                st.image(image, use_container_width=True)

            with col2:
                st.markdown("**Grad-CAM Visualization**")
                gradcam_base64 = result.get("gradcam_base64")
                if gradcam_base64:
                    st.image(f"data:image/png;base64,{gradcam_base64}", use_container_width=True)
                else:
                    st.info("No Grad-CAM image returned.")

            # Report section
            st.subheader("Generated Report")
            st.info(
                f"Highest-confidence finding: **{result['top_label']}** "
                f"with probability **{result['top_probability']:.4f}**"
            )
            st.markdown("**Model-generated summary**")
            st.text_area(
                "Report",
                result["report"],
                height=220,
                disabled=True,
                label_visibility="collapsed"
            )

            # Probability table
            st.subheader("Prediction Probabilities")
            probs = result["probabilities"]
            prob_df = build_probability_table(probs, threshold=0.5)

            st.caption("Rows highlighted in red indicate probability ≥ 0.50.")
            st.dataframe(
                prob_df.style.apply(highlight_probability_rows, axis=1),
                use_container_width=True
            )

            # Optional progress bars too
            with st.expander("See probability bars"):
                for _, row in prob_df.iterrows():
                    st.write(f"**{row['Condition']}** — {row['Status']}")
                    st.progress(
                        min(max(float(row["Probability"]), 0.0), 1.0),
                        text=f"{row['Probability']:.4f}"
                    )