import streamlit as st
import os
import sys
import tempfile

# Fix path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Now these will work
from segmentation.segment_image import segment_image
from object_analysis.detect_objects import classify_images
from text_extraction.extract_text import extract_text_from_images
from summarization.summarize_data import summarize_objects
from video_mode.video_pipeline import process_video


# Streamlit UI settings
st.set_page_config(page_title="AI Vision System", layout="centered")

st.title("🧠 AI Image & Video Analyzer")
st.markdown("Upload an **image** or **video** for object detection, text extraction, and summarization.")

# --- IMAGE MODE ---
st.header("📷 Image Upload")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_img_path = tmp_file.name

    st.image(tmp_img_path, caption="Original Image", use_column_width=True)

    with st.spinner("🔍 Processing image..."):
        segmented_dir = "data/segmented"
        os.makedirs(segmented_dir, exist_ok=True)
        segmented_paths = segment_image(tmp_img_path, output_dir=segmented_dir)

    if not segmented_paths:
        st.error("❌ No objects detected.")
    else:
        with st.spinner("📦 Classifying & extracting text..."):
            classes = classify_images(segmented_dir)
            texts = extract_text_from_images(segmented_dir)
            summaries = summarize_objects(classes, texts)

        st.success("✅ Done!")

        st.subheader("🧩 Segmented Objects")
        for file in os.listdir(segmented_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(segmented_dir, file)
                st.image(path, caption=file, width=150)

        st.subheader("🧠 Summarized Output")
        for img, summary in summaries.items():
            st.markdown(f"**{img}**")
            st.write(summary)

# --- VIDEO MODE ---
st.header("🎥 Video Upload")
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"], key="video")

if video_file:
    st.video(video_file)

    if st.button("▶️ Process Video"):
        with st.spinner("⏳ Running YOLOv8 + EasyOCR..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_vid:
                temp_vid.write(video_file.read())
                video_path = temp_vid.name

            output_path = "data/results/streamlit_output.avi"
            os.makedirs("data/results", exist_ok=True)

            process_video(video_path, output_path)

        st.success("✅ Video processing complete!")
        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Annotated Video", f, file_name="annotated_video.avi", mime="video/avi")
