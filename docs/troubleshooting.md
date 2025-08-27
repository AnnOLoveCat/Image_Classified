## Troubleshooting

**Q1: 為什麼明明已經安裝了 `Streamlit` ，但 VS Code 裡沒有馬上被偵測到，甚至需要重開多次 VS Code 才能用**  
- A:在 VS Code 按 Ctrl/Cmd + Shift + P → 找 Python: Select Interpreter → 選有裝套件的那個環境。
- 再按 Ctrl/Cmd + Shift + P → Developer: Reload Window。
- 若還不行，執行 Pylance: Restart Language Server。
---

**Q2: 為什麼 `tensorflow.keras.applications.mobilenet_v2` 匯入不了？**  
- A: 因為 Keras 3 與 TensorFlow 脫鉤，改用 `keras.applications.mobilenet_v2`，或檢查 TF/Keras 版本相容性。
---

**Q3: `img = np.array(image)`這部分為什麼圖片要轉換格式?**

- A:在深度學習(deep learning)裡，模型只能處理 數值型張量（tensors），而不是原始圖片檔（JPEG、PNG）。
- 轉換的目的：
> - 統一資料格式：不同圖片大小、顏色模式（RGB、灰階）必須轉成相同維度的數值矩陣。
> - 數值標準化：把像素值（0–255）縮放到特定範圍（如 0–1 或 -1–1），確保模型收斂更穩定。
> - 方便 GPU 並行運算：模型在張量上進行矩陣運算，統一格式才能批次計算。
```bash
    類似這樣，當然會比較複雜
    [[0, 1, 2],
    [1, 1, 0],
    [1, 2, 1]]   
```
---

**Q4: `img.resize(img, (224, 224))` 這部分為什麼要調整尺寸?**

- A: 深度學習模型的卷積層(Convolutional Layer)與全連接層(Fully Connected Layer)都需要固定的輸入維度，否則權重矩陣無法對應。

過程中的細節：
- 插值（Interpolation）：
> - 如果原圖比目標尺寸大 → 會縮小，像素被壓縮。
> - 如果原圖比目標尺寸小 → 會放大，像素會被「估算」出來。

- 資訊丟失：無論放大或縮小，原圖的某些細節（例如非常小的物件）可能會模糊或失真。
- 常見做法：先 resize 到模型尺寸，再做 資料增強（data augmentation）或保持長寬比（aspect ratio） 以減少失真。
---

**Q5: `img = preprocess_input(img)` 轉換成模型訓練時所需的標準化格式?**
- A: 模型在訓練時通常會把像素做 縮放 或 正規化，例如把 0-255 的像素值轉成 0~1 區間。

- 好處：
> - 加快收斂速度
> - 減少數值過大導致的梯度不穩定
> - 保持跟預訓練模型一致的輸入格式
---

**Q6: `img = np.expand_dims(img, axis=0)` 為甚麼需要加批次維度?**

- A: 深度學習模型的輸入格式是 
> - batch_size(一次輸入的圖片數量),
> - height, width(影像大小)
> - channels(顏色通道（RGB → 3）)

即使只想預測一張圖片，模型架構也要求輸入維度要跟訓練時一致。
- 例如，一張 224x224x3 的圖需要變成 (1, 224, 224, 3)，
- 這裡的 1 表示一個 batch，讓模型的矩陣運算流程保持一致，無論是一張圖還是多張圖都能通用。

1. 統一資料格式：訓練與推論同一套輸入規則，減少維度錯誤。
2. 支援向量化計算：GPU/TPU 能一次處理多張圖片，加速運算。
3. 程式碼一致性：一張圖也視為 batch → 1，避免多寫額外的分支邏輯。
---