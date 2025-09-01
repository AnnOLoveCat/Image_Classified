import numpy as np                     # 數值運算：影像會轉成 numpy 陣列以便處理/推論
import streamlit as st                 # 建 UI 的框架（上傳圖片、按鈕、顯示結果）
from PIL import Image                  # 讀圖、色彩空間轉換（特別是統一成 RGB）
import cv2                             # OpenCV：畫圖、色彩轉換（YOLO 內建 plot 回傳 BGR 要轉 RGB）

from ultralytics import YOLO           # Ultralytics YOLO介面（支援v11與自訓練的 best.pt）

# -------------------------
# Model
# -------------------------
@st.cache_resource                     # 把模型載入結果快取起來，避免每次重繪 UI 都重新下載/載入模型
def load_model():
    # 預設載入 COCO 上預訓練的 YOLO11-L 偵測模型（精度高，但相對慢）
    # 若改用你自己的模型，將路徑改為 'best.pt'（同資料夾或填絕對路徑）
    return YOLO("yolo11l.pt")          # 也可寫 YOLO("best.pt") 使用自訓練權重

# -------------------------
# Preprocess (optional 但保留轉 RGB 以兼容各種圖片)
# -------------------------
def preprocess_image(image: Image.Image):
    img_pil = image.convert("RGB")          # 強制轉成 RGB（三通道），避免 RGBA（透明度）/ 灰階 / CMYK 造成通道數不符
    
    img = np.array(img_pil)                 # 轉成 numpy 格式；YOLO 可以吃 PIL/numpy/檔案路徑，這裡用 numpy 最直觀
    return img

# -------------------------
# Inference
# -------------------------
def detect_objects(model: YOLO, img_np: np.ndarray, conf: float = 0.25):
    """
    物件偵測主函式：
      - model : 已載入的 YOLO 模型
      - img_np: numpy 影像（RGB）
      - conf  : 置信度門檻（過低會有雜訊，過高會漏檢；依需求調整）
    """
    # YOLO 的 predict 可直接接受 numpy / PIL / 檔案路徑
    results = model.predict(
        source=img_np,                  # 輸入影像來源（這裡是 numpy）
        conf=conf,                      # 只保留置信度 >= conf 的框
        verbose=False                   # 關閉冗長日誌，讓前端乾淨
    )
    return results                      # 回傳 Results 列表（通常單張圖就是長度 1）

# -------------------------
# Draw / Render
# -------------------------
def render_result(result):
    """
    把 YOLO 偵測結果畫成可視化圖片：
      - result.plot() 會回傳 BGR 圖（OpenCV 格式）
      - Streamlit 需要 RGB，因此要做色彩空間轉換
    """
    plotted = result.plot()                             # 內建畫框、標籤、信心分數（回傳 BGR）
    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)  # 轉回 RGB 給 st.image 顯示
    return plotted

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="YOLO Model AI Image Classifying", layout="centered")   #頁籤標題, 置中排版

    st.title("YOLO11 Model Image Classifying")  # 頁面主標（此範例為「偵測」，文字可自行改為 Detection）
    st.write("Upload an Image and let AI tell you what's in it（mostly COCO category）")# 簡介；COCO 是預訓練資料集（80 類）

    model = load_model()    # 載入並快取模型（首次耗時，之後很快）

    uploaded_file = st.file_uploader("Choose an Image",type=["jpg", "jpeg", "png"])    # 限定可上傳的副檔名
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05) # 置信度門檻：最小、最大、預設、步進

    if uploaded_file is not None:
        # 影像預覽（原圖）
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect"):                        # 點擊後才進行推論，避免一上傳就跑
            with st.spinner("Analyzing Image..."):     # 顯示載入中指示（避免卡住的感覺）
                img_pil = Image.open(uploaded_file).convert("RGB")  # 安全起見再轉 RGB 一次
                img_np = preprocess_image(img_pil)     # 轉 numpy（如果要自定前處理可集中改這裡）

                results = detect_objects(model, img_np, conf=conf)  # 進行偵測
                if not results:                         
                    st.error("No result returned.")
                    return

                result = results[0] # 單張圖通常只有一個 result
                plotted_rgb = render_result(result)                                     # 畫框 + 轉 RGB
                st.image(plotted_rgb, caption="Detections", use_container_width=True)   # 顯示畫好框的圖

                
                if result.boxes is not None and len(result.boxes) > 0:              # 以下把每個偵測到的物件用「文字列表」列出（類別、信心分數、方框座標）
                    st.subheader("Detections (Top)")
                    for b in result.boxes:                                          # 逐一讀取每個框（Box）
                        cls_id = int(b.cls[0]) if b.cls is not None else -1         # 類別 ID（對應 result.names 字典）
                        score = float(b.conf[0]) if b.conf is not None else 0.0     # 置信度（0~1）
                        xyxy = b.xyxy[0].tolist() if b.xyxy is not None else []     # 邊框座標（左上 x,y 到 右下 x,y）
                        name = result.names.get(cls_id, str(cls_id))                # 類別名稱（COCO 預訓練 80 類；自訓練則為你的類別）
                
                        st.write(f"- **{name}** | conf: {score:.2f} | box: {list(map(lambda x: round(x, 1), xyxy))}")   # 四捨五入座標方便閱讀
                else:
                    st.info("No objects detected above the confidence threshold.") 
                    
if __name__ == "__main__":
    main()                                             