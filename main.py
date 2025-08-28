import cv2
import numpy as np
import streamlit as st
from PIL import Image
from keras.applications.mobilenet_v2 import (
    MobileNetV2,        # 載入 MobileNetV2 模型結構與權重
    preprocess_input,   # 將原始影像轉換為模型可接受的格式
    decode_predictions  # 將模型輸出的向量（機率分布）轉換為可讀的標籤名稱與機率
)
#  mobilenet_v2這是較輕量型模型，適合行動裝置與邊緣計算，因為模型參數量小、推論速度快、對計算資源需求低
#  但你可以去訓練模型，標記物件、給AI訓練給自己用

#   部分問題請參見docs/troubleshooting.md

def load_model():
    model = MobileNetV2(weights="imagenet") # 載入ImageNet（1000 類別、1400 萬張圖片）上已訓練過的模型。適合做分類或轉移學習。
                                            # 輕量級 CNN 模型，專為行動裝置設計，計算量低、效率高。
                                            # 會跑出類似這種結果[0.9, 0.1, 0.05, 0.0]依照機率排序出來
                                            # 這種就在表示說此物件以機率表示說"此物件可能XXX為何"
    return model

def preprocess_image(image):
    img_pil = image.convert("RGB")                                        # 先用 PIL 強制轉成 RGB（三通道），避免 RGBA/灰階/CMYK 問題

    img = np.array(img_pil)                                               # 1. 轉換成數值矩陣
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)       # 2. Resize 到模型需要的尺寸

    img = img.astype(np.float32)                                          # 3. dtype 與模型前處理, 做像素標準化，符合訓練分佈
    img = preprocess_input(img)                                           # 5. MobileNetV2 的 preprocess_input
    img = np.expand_dims(img, axis=0)                                     # 6. 增加 batch 維度, 輸入格式是 (batch_size, height, width, channels)
    
    return img

def classify_image(model, img):
    try:
        processed_image = preprocess_image(img)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]  # 這邊是只取前3個物件， 
                                                                        # 模型一般會處裡多張圖片，但我只給予一張圖片, 
                                                                        # 所以也只需一個回應
        return decoded_predictions
    
    except Exception as e:  
        st.error(f"Error classifying image, {e}")

def main():
    st.set_page_config(page_title="AI Image Classifying",page_icon="", layout="centered")

    st.title("AI Image Classifying")
    st.write("Upload an Image and let AI tell you what's in it")

    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "png"])

    if uploaded_file is not None:
        preview = st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)    # 這裡只負責顯示圖片

        btn = st.button("Classify Image")

        if btn:
            with st.spinner("Analyzing Image...take a moment"):
                img_pil = Image.open(uploaded_file).convert("RGB")      # 這裡負責讀檔，轉成 RGB 圖片，供模型做推論
                predictions = classify_image(model, img_pil)

                if predictions:
                    st.subheader("Predictions")
                    for _, label, score in predictions:                 #idx部分可以用"_"代替，但為了看懂用idx代替
                        st.write(f"**{label}**: {score:.2%}")

if __name__ == "__main__":
    main()