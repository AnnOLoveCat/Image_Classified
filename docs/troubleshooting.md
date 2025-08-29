## Troubleshooting

**Q1: 為什麼明明已經安裝了 `Streamlit` ，但 VS Code 裡沒有馬上被偵測到，甚至需要重開多次 VS Code 才能用**  
- A:在 VS Code 按 Ctrl + Shift + P → 找 Python: Select Interpreter → 選有裝套件的那個環境。
- 再按 Ctrl + Shift + P → Developer: Reload Window。
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
&nbsp;&nbsp;1. 加快收斂速度  
&nbsp;&nbsp;2. 減少數值過大導致的梯度不穩定  
&nbsp;&nbsp;3. 保持跟預訓練模型一致的輸入格式  
---
**Q6: `img = np.expand_dims(img, axis=0)` 為甚麼需要加批次維度?**

- A: 深度學習模型的輸入格式是 
> - batch_size(一次輸入的圖片數量),
> - height, width(影像大小)
> - channels(顏色通道（RGB → 3）)

即使只想預測一張圖片，模型架構也要求輸入維度要跟訓練時一致。
- 例如，一張 224x224x3 的圖需要變成 (1, 224, 224, 3)，<br>這裡的 1 表示一個 batch，讓模型的矩陣運算流程保持一致，無論是一張圖還是多張圖都能通用。</br>
- 好處:<br>
&nbsp;1. 統一資料格式：訓練與推論同一套輸入規則，減少維度錯誤。</br>
&nbsp;2. 支援向量化計算：GPU/TPU 能一次處理多張圖片，加速運算。</br>
&nbsp;3. 程式碼一致性：一張圖也視為 batch → 1，避免多寫額外的分支邏輯。</br>
---
**Q7： 報錯`Depth of input must be a multiple of depth of filter: 4 vs 3`**

- A: 輸入影像是 RGBA（4 通道），模型只接受 RGB（3 通道）。
- 在`preprocess_image`中用`image.convert("RGB")`強制轉為3通道。<br>如果仍遇到，確認你送入的是`PIL.Image`物件而非`np.ndarray`（避免跳過轉換）。</br>
---
**Q8: `use_column_width` 被棄用？**

- A: 是 Streamlit 的 API 變更, 已改用 `use_container_width=True`。
---
**Q9: 需要用`tensorflow.keras`還是`keras`？**

- A: 此專案採用`from keras.applications.mobilenet_v2`，相容性更好。<br>改成`from tensorflow.keras...`後匯入失敗，多半是 TF/Keras 版本不匹配</br>
---

**Q10: 為什麼要`preprocess_input`？**

- A: 把像素值轉成模型訓練時的分佈（例如 [-1, 1]），否則預測品質會下降。<br>**重點**：不要重複呼叫；先 astype(np.float32) 再呼叫一次就好（程式已修正）。
---

**Q11: JPG/PNG/CMYK/灰階都可以嗎？**

- A: 可以，程式會：<br>用 PIL 的 convert("RGB") 三通道 + OpenCV resize 到 224×224，避免通道數不對或尺寸不合的問題。
---

**Q12: OpenCV 的 BGR / RGB 會不會搞混？**

- A: 不會，因為我們使用 PIL.Image → np.array 的流程，得到的是 RGB。<br>若你未來改用 cv2.imread，一定要轉 cv2.cvtColor(img, cv2.COLOR_BGR2RGB)。
---

**Q13: 為什麼只顯示前 3 個預測？**

- A: 用 decode_predictions(predictions, top=3) 取最有意義的 3 筆；你可以改 top=5 或直接秀完整 softmax 向量。
---

**Q14: 需要 GPU 嗎？**

- A: 不強制。MobileNetV2 是輕量模型，CPU 也可跑。<br>Windows/NVIDIA 想用 GPU → 安裝符合版本的 CUDA/CuDNN 與對應 TensorFlow；macOS Apple Silicon → tensorflow-macos + tensorflow-metal。
---

**Q15: 可以一次分類多張圖嗎？**

- A: 可以，把多張圖各自 preprocess_image 後以 np.vstack 或 np.concatenate 組成 (N, 224, 224, 3)，再 model.predict(batch)。<br>Streamlit 端可用 accept_multiple_files=True，再迴圈處理。