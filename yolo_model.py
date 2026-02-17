# --- 偵測環境是否正常安裝 ---
try:
    import onnx
    import onnxsim
    import onnxruntime
    # st.toast("Environment check passed: ONNX packages found.", icon="✅") 
except ImportError as e:
    st.error(f"環境安裝失敗，缺少套件: {e}")
    st.info("正在嘗試最後手段：強制執行 pip install...")
    # 這是最後的救命稻草，但在雲端通常會失敗，僅作為除錯訊息
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "onnxslim", "onnxruntime"])
        st.success("強制安裝成功！請重新整理頁面。")
    except Exception as install_error:
        st.error(f"無法修復環境: {install_error}")
        st.stop()

import numpy as np                     # 數值運算
import streamlit as st                 # UI 框架
from PIL import Image                  # 影像處理
import cv2                             # OpenCV
from ultralytics import YOLOWorld      # YOLO 模型
import os

# -------------------------
# Model
# -------------------------
@st.cache_resource
def load_model():
    # 使用 YOLO-World 模型
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
    """影像增強"""
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced

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
    st.write("Upload an Image and let AI tell you what's in it.")

    model = load_model()

    # --- Sidebar: Export Tools ---
    with st.sidebar:
        st.header("Optimization Tools")
        st.write("Export model for deployment (e.g., Jetson Orin Nano).")
        export_format = st.selectbox("Format", ["ONNX", "TensorRT (.engine)"])

        # 1. 初始化 session_state，確保下載按鈕狀態被記住
        if "export_file" not in st.session_state:
            st.session_state['export_file'] = None

        if st.button("Export Model"):
            try:
                with st.spinner(f"Exporting to {export_format}..."):
                    fmt = "onnx" if "ONNX" in export_format else "engine"
                    path = model.export(format=fmt)
                st.success(f"Exported to: {path}")
            except Exception as e:
                st.error(f"Export failed: {e}")
                
        # 2. 匯出按鈕邏輯
        if st.button("Generate Export File"):
            try:
                with st.spinner(f"Exporting to {export_format}... (This may take a while)"):
                    fmt = "onnx" if "ONNX" in export_format else "engine"
                    
                    # 執行匯出，並取得伺服器上的檔案路徑
                    path = model.export(format=fmt) 
                    
                    # 將路徑存入 session_state
                    st.session_state['export_file'] = path
                    
                st.success(f"Export successful! File is ready.")
                
            except Exception as e:
                st.error(f"Export failed: {e}")
                st.session_state['export_file'] = None # 失敗時重置

        # 3. 如果 session_state 裡有檔案，顯示下載按鈕
        # 這是讓檔案能從「雲端」傳回「使用者電腦」的唯一橋樑
        if st.session_state['export_file'] and os.path.exists(st.session_state['export_file']):
            file_path = st.session_state['export_file']
            file_name = os.path.basename(file_path)
            
            with open(file_path, "rb") as f:
                st.download_button(
                    label=f"⬇️ Click to Download {file_name}",
                    data=f,
                    file_name=file_name,
                    mime="application/octet-stream",
                    type="primary" # 讓按鈕變明顯的顏色
                )
    # --- Main UI ---
    with st.container():
        uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])
        
        # 修改提示：要求使用者直接輸入英文
        user_classes = st.text_input(
            "Wanted Object Classes (Input English Only, Comma-Separated)",
            value="person, helmet, vest, machinery",
            help="Example: person, hardhat, steel beam"
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
                st.warning("Please enter at least one class name.")
                return
            
            with st.spinner("Analyzing Image..."):
                class_list = [c.strip() for c in user_classes.split(",") if c.strip()] # 直接處理字串，不再翻譯
                model.set_classes(class_list)# 設定類別
                results = detect_objects(model, img_np, conf=conf)# 偵測
                
                if results and len(results[0].boxes) > 0:
                    result = results[0]
                    plotted_rgb = render_result(result)
                    st.image(plotted_rgb, caption=f"Detections", use_container_width=True)
                    
                    st.subheader("Detections List")
                    # 統計數量
                    counts = {}
                    for b in result.boxes:
                        cls_id = int(b.cls[0])
                        label = result.names[cls_id]
                        score = float(b.conf[0])
                        counts[label] = counts.get(label, 0) + 1
                        st.write(f"- **{label}** (Confidence: {score:.2f})")
                    
                    st.write("### Statistical Results")
                    st.bar_chart(counts)
                else:
                    st.info("No Objects Detected Above The Confidence Threshold.")

if __name__ == "__main__":
    main()