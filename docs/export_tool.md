# YOLO-World 物件偵測與模型匯出工具

這是一個基於 Streamlit 的 AI 應用程式，使用 YOLO-World (Real-Time Open-Vocabulary Object Detection) 模型。

使用者可以上傳圖片，透過文字輸入自定義想要偵測的物件（例如 "person, helmet, machinery"），並進行即時偵測。

這裡是專注於「優化工具 (Optimization Tools)」，允許使用者直接在雲端將客製化的模型匯出為 ONNX 格式，
以便部署到邊緣裝置（如 NVIDIA Jetson Orin Nano）。

## 「優化工具」開發歷程與技術問題

在 Streamlit Cloud 這種唯讀且資源受限的環境中，實作模型匯出 (Model Export) 功能，經歷了一連串的技術挑戰。

以下是遇到的關鍵問題與解決方案的完整紀錄。
### 1. 權限地獄：Permission Denied 與 AutoUpdate 迴圈

    問題： 當使用者點擊"Export Model"時，Ultralytics套件偵測到環境缺少onnx，並試圖在執行期間 (Runtime) 自動執行 pip install。

    錯誤訊息：Permission denied(os error 13)以及attempting AutoUpdate... failed。

    原因：Streamlit Cloud的執行環境是唯讀(Read-Only)。
    
    程式啟動後無法動態寫入系統目錄，所有套件必須在 部署階段 (Build Phase) 就安裝完畢。

    解決方案：
        放棄自動生成的鎖定檔 (uv.lock, pyproject.toml)，因為它們鎖定了錯誤的環境狀態。
        改用手動維護的 requirements.txt，並明確列出匯出所需的依賴項，強迫 Streamlit 在部署時就將它們安裝好。

### 2. 幽靈模組：缺少 CLIP (Missing Module)

    問題：修復ONNX安裝後，程式崩潰並顯示ModuleNotFoundError: clip。
    原因：YOLO-World需要OpenAI的CLIP模型來理解文字提示。
    
    標準PyPI版本的clip套件往往版本過舊或與ultralytics不相容。

    解決方案：
        在requirements.txt中指定CLIP的GitHub 來源，確保安裝正確的版本：
        git+https://github.com/ultralytics/CLIP.git

### 3. 編譯失敗：onnxslim 簡化錯誤

    問題： 匯出過程失敗，出現警告 WARNING ⚠️ ONNX: simplifier failure。
    原因： Ultralytics 預設會使用 onnxslim 來優化模型。
    
    此函式庫在 Linux 雲端環境中通常需要 C++ 編譯器 (GCC) 支援，容易導致安裝或執行失敗。

    解決方案：
        我們在程式碼中明確 關閉了簡化功能，避開了這個依賴問題：
        model.export(format="onnx", simplify=False)

### 4. 記憶體不足：Connection Reset (OOM Crash)

    問題： 匯出過程中，應用程式畫面變灰並重新啟動 (connection reset by peer)。
    原因： 我們最初使用的是 yolov8x-worldv2.pt (Extra Large) 模型。這個巨型模型（7200 萬參數）在匯出時需要的記憶體遠超 Streamlit Cloud 免費版提供的額度（約 1GB - 3GB）。

    解決方案：
        我們將預設模型切換為Small版本 (yolov8s-worldv2.pt)，既保留了功能性，又能在雲端記憶體限制內順利完成匯出。
        (額外: 目前暫時改回yolov8x-worldv2.pt，測試是否可繼續使用)

# 最終環境配置 (Final Configuration)

若要重現此專案的成功部署，請確保您的環境配置如下：
## 1. requirements.txt (關鍵清單)

在 Streamlit Cloud 上，請暫時移除 uv.lock 或 pyproject.toml，並使用這份乾淨的清單：
Plaintext

### 1. 匯出工具 (Export Tools) - 必須優先安裝 ---
```bash
onnx>=1.12.0
onnxruntime
```

### 2. 核心套件 (Core Dependencies) ---
```bash
ultralytics
streamlit
opencv-python-headless
pillow
numpy
```

### 3. 文字編碼器 (Text Encoder) - 必須指定來源 ---
```bash
git+https://github.com/ultralytics/CLIP.git
```

## 2. app.py (關鍵程式碼)

```bash
A. 載入雲端友善的模型：

@st.cache_resource
def load_model():
    # 使用 's' (Small) 版本以避免雲端記憶體不足 (OOM)
    return YOLOWorld("yolov8s-worldv2.pt")

B. 穩健的匯出功能：

if st.button("Export Model"):
    try:
        with st.spinner(f"Exporting to ONNX..."):
            # 關鍵：設定 simplify=False 以避免 onnxslim 錯誤
            path = model.export(format="onnx", simplify=False)
            
            # 將路徑存入 session state，確保下載按鈕不會消失
            st.session_state['export_file'] = path
        st.success(f"Export successful!")
    except Exception as e:
        st.error(f"Export failed: {e}")
```
如何使用

    設定偵測類別： 在文字輸入框中輸入您想偵測的物件（例如 "cat, dog, car"）。
    上傳圖片： 上傳一張 JPG 或 PNG 圖片。
    開始偵測： 點擊 "Detect" 按鈕查看標註結果。
    匯出模型 (側邊欄)：
        前往側邊欄的 "Optimization Tools"。
        選擇 ONNX 格式。
        點擊 "Export Model"。
        等待處理完成後，點擊綠色的 "⬇️ Click to Download" 按鈕，即可將 .onnx 檔案下載至您的電腦。

## 部署目標 (Deployment Target)

匯出的 .onnx 檔案已經過優化，適合部署於邊緣運算裝置：

    NVIDIA Jetson Orin Nano/AGX Orin(需再轉為TensorRT Engine)
    Raspberry Pi(使用ONNX Runtime)
    本機PC(CPU/GPU推論)

    模型來源： Ultralytics YOLO-World
    應用框架： Streamlit