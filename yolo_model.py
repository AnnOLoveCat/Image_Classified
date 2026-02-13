import numpy as np                     # 數值運算
import streamlit as st                 # UI 框架
from PIL import Image                  # 影像處理
import cv2                             # OpenCV
from ultralytics import YOLOWorld      # YOLO 模型
from deep_translator import GoogleTranslator # 翻譯工具 (穩定版)

# -------------------------
# Model
# -------------------------
@st.cache_resource
def load_model():
    model_world = YOLOWorld("yolov8x-worldv2.pt")
    # 預設標籤，避免初始化報錯
    model_world.set_classes(["object", "item", "person", "tool", "equipment", "facility", "structure"])
    return model_world

# -------------------------
# Preprocess & Helpers
# -------------------------
def preprocess_image(image: Image.Image):
    img_pil = image.convert("RGB")
    img = np.array(img_pil)
    return img

def enhance_image(img_np):
    """CLAHE 影像增強"""
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced

def translate_text(text):
    """中文轉英文翻譯"""
    if not text: return ""
    # 若輸入純英文則跳過
    if all(ord(c) < 128 for c in text.replace(",", "").replace(" ", "")):
        return text
    try:
        translator = GoogleTranslator(source='auto', target='en')
        translated = translator.translate(text)
        return translated
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text

# -------------------------
# Inference
# -------------------------
def detect_objects(model, img_np: np.ndarray, conf: float = 0.25):
    results = model.predict(
        source=img_np,
        conf=conf,
        imgsz=1280,
        verbose=False,
    )
    return results

def render_result(result):
    plotted = result.plot()
    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    return plotted

# -------------------------
# Main App
# -------------------------
def main():
    st.set_page_config(page_title="YOLO Model AI Image Classifying", layout="centered")
    st.title("YOLO Model Image Classifying")
    st.write("Upload an Image and let AI tell you what's in it (mostly COCO category)")

    model = load_model()

    # --- Sidebar ---
    with st.sidebar:
        st.header("Optimization Tools")
        st.write("Export model for deployment (e.g., Jetson Orin Nano).")
        export_format = st.selectbox("Format", ["ONNX", "TensorRT (.engine)"])
        if st.button("Export Model"):
            try:
                with st.spinner(f"Exporting to {export_format}..."):
                    fmt = "onnx" if "ONNX" in export_format else "engine"
                    path = model.export(format=fmt)
                st.success(f"Exported to: {path}")
            except Exception as e:
                st.error(f"Export failed: {e}")

    # --- Main UI ---
    with st.container():
        uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])
        
        user_classes = st.text_input(
            "Wanted Object Classes (Input Chinese or English, Comma-Separated)",
            value="person, chair, bottle, phone",
            help="Example: 安全帽, 人 (will be auto-translated)"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        with col2:
            st.write("")
            st.write("") 
            use_enhance = st.checkbox("Enhance Image", value=False, help="Use CLAHE for better details")

    if uploaded_file is not None:
        img_pil = Image.open(uploaded_file).convert("RGB")
        img_np = preprocess_image(img_pil)
        
        if use_enhance:
            img_np = enhance_image(img_np)
            st.image(img_np, caption="Uploaded Image (Enhanced)", use_container_width=True)
        else:
            st.image(img_pil, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect"):
            if not user_classes:
                st.warning("At Least One Class Name！")
                return
            
            with st.spinner("Analyzing Image (Translating tags & Detecting)..."):
                # 1. 翻譯
                translated_classes = translate_text(user_classes)
                class_list = [c.strip() for c in translated_classes.split(",") if c.strip()]
                
                # 2. 設定類別
                model.set_classes(class_list)

                # 3. 偵測
                results = detect_objects(model, img_np, conf=conf)
                
                if results and len(results[0].boxes) > 0:
                    result = results[0]
                    plotted_rgb = render_result(result)
                    st.image(plotted_rgb, caption=f"Detections (Targets: {translated_classes})", use_container_width=True)
                    
                    st.subheader("Detections (Top)")
                    for b in result.boxes:
                        cls_id = int(b.cls[0])
                        label = result.names[cls_id]
                        score = float(b.conf[0])
                        st.write(f"- **{label}** (Confidence: {score:.2f})")
                else:
                    st.info("No Objects Detected Above The Confidence Threshold.")

                # 統計圖表
                if results:
                    counts = {}
                    for b in results[0].boxes:
                        label = results[0].names[int(b.cls[0])]
                        counts[label] = counts.get(label, 0) + 1
                    
                    if counts:
                        st.write("### Statistical Results")
                        st.bar_chart(counts)

if __name__ == "__main__":
    main()