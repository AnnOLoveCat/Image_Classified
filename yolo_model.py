import numpy as np                     # 數值運算：影像會轉成 numpy 陣列以便處理/推論
import streamlit as st                 # 建 UI 的框架（上傳圖片、按鈕、顯示結果）
from PIL import Image                  # 讀圖、色彩空間轉換（特別是統一成 RGB）
import cv2                             # OpenCV：畫圖、色彩轉換（YOLO 內建 plot 回傳 BGR 要轉 RGB）

from ultralytics import YOLO           # Ultralytics YOLO介面（支援v11與自訓練的 best.pt）
from ultralytics import YOLOWorld
# from googletrans import Translator   #最新:因為此套濺可能是過於老舊必須換掉 新增：Google 翻譯套件，用於將中文標籤轉為英文
from deep_translator import GoogleTranslator

# -------------------------
# Model
# -------------------------
@st.cache_resource                     # 把模型載入結果快取起來，避免每次重繪 UI 都重新下載/載入模型
def load_model():
    # model = "yolo11l.pt"  # 預訓練模型檔名（可改為 'best.pt' 或其他）
    model_world = YOLOWorld("yolov8x-worldv2.pt")  # 世界模型檔名
    # 預設給予一組通用標籤，避免初始化報錯
    model_world.set_classes(["object", "item", "person", "tool", "equipment", "facility", "structure"])
    return model_world    # 也可用 YOLOWorld

# -------------------------
# Preprocess & Helpers (新增功能區)
# -------------------------
def preprocess_image(image: Image.Image):
    img_pil = image.convert("RGB")          # 強制轉成 RGB（三通道）
    img = np.array(img_pil)                 # 轉成 numpy 格式
    return img

def enhance_image(img_np):
    """
    新增功能：影像增強 (CLAHE)
    針對工廠或低光源環境，強化邊緣細節，提升 YOLO 辨識率
    """
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced

def translate_text(text):
    """
    新增功能：標籤翻譯
    讓使用者輸入中文 (如: 安全帽)，自動轉為英文 (helmet) 給模型讀取
    """
    if not text: return ""
    # 若輸入純英文則跳過翻譯
    if all(ord(c) < 128 for c in text.replace(",", "").replace(" ", "")):
        return text
    try:
        # 改用 deep_translator 的寫法，功能一模一樣但不會報錯
        translator = GoogleTranslator(source='auto', target='en')
        translated = translator.translate(text)
        return translated
    except Exception as e:
        st.warning(f"Translation failed: {e}") # 顯示錯誤但不讓程式崩潰
        return text

# -------------------------
# Inference
# -------------------------
def detect_objects(model, img_np: np.ndarray, conf: float = 0.25):
    """
    物件偵測主函式
    """
    # YOLO 的 predict 可直接接受 numpy / PIL / 檔案路徑
    results = model.predict(
        source=img_np,                  # 輸入影像來源（這裡是 numpy）
        conf=conf,                      # 只保留置信度 >= conf 的框
        imgsz=1280,                     # 保持 1280 可確保較高準確度
        verbose=False,                  # 關閉冗長日誌，讓前端乾淨
    )
    return results                      # 回傳 Results 列表

# -------------------------
# Draw / Render
# -------------------------
def render_result(result):
    plotted = result.plot()                             # 內建畫框、標籤、信心分數（回傳 BGR）
    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)  # 轉回 RGB 給 st.image 顯示
    return plotted

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="YOLO Model AI Image Classifying", layout="centered")     # 頁籤標題, 置中排版
    st.title("YOLO Model Image Classifying")                                                # 頁面主標
    st.write("Upload an Image and let AI tell you what's in it（mostly COCO category）")    # 簡介

    model = load_model()    # 載入並快取模型

    # --- 新增：側邊欄 (效能優化工具) ---
    with st.sidebar:
        st.header("Optimization Tools")
        st.write("Export model for deployment (e.g., Jetson Orin Nano).")
        export_format = st.selectbox("Format", ["ONNX", "TensorRT (.engine)"])
        if st.button("Export Model"):
            try:
                with st.spinner(f"Exporting to {export_format}..."):
                    fmt = "onnx" if "ONNX" in export_format else "engine"
                    path = model.export(format=fmt) # 執行匯出
                st.success(f"Exported to: {path}")
            except Exception as e:
                st.error(f"Export failed: {e}")

    # --- 主畫面 UI ---
    with st.container():
        uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])
        
        user_classes = st.text_input(
            "Wanted Object Classes (Input Chinese or English, Comma-Separated)",
            value="person, chair, bottle, phone",
            help="Example: 安全帽, 人 (will be auto-translated)"
        )
        
        # 新增：影像增強開關 (排版調整，與 Slider 放在一起)
        col1, col2 = st.columns([3, 1])
        with col1:
            conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        with col2:
            st.write("") # 佔位，讓 Checkbox 對齊
            st.write("") 
            use_enhance = st.checkbox("Enhance Image", value=False, help="Use CLAHE for better details")

    if uploaded_file is not None:
        # 影像讀取與處理
        img_pil = Image.open(uploaded_file).convert("RGB")  # 安全起見再轉 RGB 一次
        img_np = preprocess_image(img_pil)                  # 轉 numpy
        
        # 根據開關決定是否進行影像增強
        if use_enhance:
            img_np = enhance_image(img_np)
            st.image(img_np, caption="Uploaded Image (Enhanced)", use_container_width=True)
        else:
            st.image(img_pil, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect"):                        # 點擊後才進行推論
            if not user_classes:
                st.warning("At Least One Class Name！")
                return
            
            with st.spinner("Analyzing Image (Translating tags & Detecting)..."):     # 顯示載入中畫面

                # 1. 翻譯標籤 (中文 -> 英文)
                translated_classes = translate_text(user_classes)
                class_list = [c.strip() for c in translated_classes.split(",") if c.strip()]
                
                # 2. 設定使用者的指定的類別 (餵給模型的必須是英文)
                model.set_classes(class_list)                       

                # 3. 執行偵測 (使用處理過的 img_np)
                results = detect_objects(model, img_np, conf=conf)
                
                if results and len(results[0].boxes) > 0:
                    result = results[0]
                    # 畫圖並顯示
                    plotted_rgb = render_result(result)
                    st.image(plotted_rgb, caption=f"Detections (Targets: {translated_classes})", use_container_width=True)
                    
                    # 顯示文字清單
                    st.subheader("Detections (Top)")
                    for b in result.boxes:
                        cls_id = int(b.cls[0])
                        label = result.names[cls_id]
                        score = float(b.conf[0])
                        st.write(f"- **{label}** (Confidence: {score:.2f})")
                else:
                    st.info("No Objects Detected Above The Confidence Threshold.")

                if results:
                    counts = {}
                    for b in results[0].boxes:
                        label = results[0].names[int(b.cls[0])]
                        counts[label] = counts.get(label, 0) + 1
                    
                    st.write("### Statistical Results")
                    st.bar_chart(counts) # 畫長條圖
                    
if __name__ == "__main__":
    main()