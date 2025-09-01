# 使用 uv 安裝與管理 Python 套件（專案快速上手）

參考 -> https://docs.astral.sh/uv/getting-started/installation/#shell-autocompletion

## 文件
- [安裝教學](installation.md)
- [常見問題與解法](docs/troubleshooting.md)
---

## 1) 先安裝 uv

```bash
pip install uv
```

**確認版本**
```bash
uv --version
```

---

## 2) 建立與使用虛擬環境

在專案根目錄執行：
```bash
uv venv
```

**啟用虛擬環境**
- Windows（PowerShell）：
  ```powershell
  .venv\Scripts\activate
  ```

> 之後在此資料夾下使用 `uv` 指令，套件會安裝到 `.venv` 內，與系統環境隔離。

---

## 3) 安裝套件（Example : `uv pip` 與 `requirements.txt`）

**安裝單一/多個套件**
```bash
uv pip install requests
uv pip install numpy pandas
```

**安裝特定版本**
```bash
uv pip install requests==2.32.3
```

**由需求檔安裝**
```bash
uv pip install -r requirements.txt
```

**輸出目前環境套件清單**
```bash
uv pip freeze > requirements.txt
```

> 備註：`uv pip` 是對 pip/pip-tools 的高速相容介面，常見子指令（`install`、`freeze`、`compile`、`sync` 等）皆可用。

---

## 4) 執行程式與指令

**在專案環境中執行 Python/模組/指令**（自動確保相依就緒）：
```bash
uv run python main.py
uv run python -m your_module
```

**一次性執行工具（不用事先安裝到環境）**
```bash
uvx ruff
uvx pycowsay "hello from uv"
```

---

## 5) 以 `requirements.in` → `requirements.txt` 鎖定（可選）

若你慣用 `requirements.in`（僅列出高階需求），可編譯為鎖定版 `requirements.txt`：
```bash
# 例：requirements.in 內容
# httpx
# ruff>=0.3.0

# 產生鎖定檔
uv pip compile requirements.in -o requirements.txt

# 依鎖定檔安裝
uv pip install -r requirements.txt
```

---

## 6) 常用對照表

| 目的 | uv 指令 | 傳統做法(virtualenv) |
|---|---|---|
| 建立 venv | `uv venv` | `python -m venv .venv` / `virtualenv .venv` |
| 啟用 venv | `source .venv/bin/activate`（或 `.venv\\Scripts\\activate`） | 同左 |
| 安裝套件 | `uv pip install pkg` | `pip install pkg` |
| 指定版本 | `uv pip install pkg==1.2.3` | `pip install pkg==1.2.3` |
| 從需求檔安裝 | `uv pip install -r requirements.txt` | 同左 |
| 匯出需求檔 | `uv pip freeze > requirements.txt` | `pip freeze > requirements.txt` |
| 執行程式 | `uv run python main.py` | 先啟用 venv 再 `python main.py` |
| 執行工具（臨時） | `uvx ruff` | `pipx run ruff` |

---

## 7) 更新 / 升級

**自我更新（standalone 安裝者）**
```bash
uv self update
```

---

## 8) 專案初始化（可選，最小範例）

若要快速建立新專案骨架與相依：
```bash
# 建立 venv 並安裝相依
uv venv
uv pip install httpx ruff

# 新增檔案
printf "import httpx\nprint('HTTPX', httpx.__version__)\n" > main.py

# 執行
uv run python main.py
uv run streamlit run main.py
```

---

## 9) 疑難排解 Tips

- **未啟用 `.venv` 卻安裝到系統環境？<br>請改用 `uv run ...` 或先啟用 `.venv` 再執行指令。
- **需求檔含 URL/路徑相依**：請遵循 `package_name @ URL` 格式；<br>或改用 `pyproject.toml`/`uv lock` 方案。
- **Windows 權限**：PowerShell 需要允許執行腳本（`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`）。

---

### 參考
- uv 官方 CLI 參考與用法：`uv venv`、`uv pip`、`uv run`、`uvx` 等。
- 更完整的專案/鎖檔工作流，請參考 Astral 官方文件。

