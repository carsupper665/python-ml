# ML Trainer CLI

這是一個針對內建資料集 `heart` 與 `winequality-red` 的機器學習命令列工具，可執行模型訓練、結果比較與推薦模型優化。

目前已實作功能：

- `list-models`
- `download --dataset ...`
- `download --all`
- `train --dataset ...`
- `compare --dataset ...`
- `optimize --dataset ... --method {random,grid}`

## 環境需求

- Windows
- Python 3.12+
- 專案使用 repo 內的虛擬環境：`.venv`

## 安裝依賴

若 `.venv` 已存在，可直接執行：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
```

安裝完成後，可使用以下任一種方式執行：

```powershell
.venv\Scripts\python.exe -m mltrainer list-models
.venv\Scripts\mltrainer.exe list-models
.venv\Scripts\python.exe main.py list-models
```

## 支援資料集

### `heart`

- 任務類型：Classification
- 標籤欄位：`target`
- 排名指標：`F1-score`
- 模型：`LR`、`DT`、`SVM`、`RF`、`KNN`
- 推薦優化模型：`Random Forest Classifier`

### `winequality-red`

- 任務類型：Regression
- 標籤欄位：`quality`
- 檔案分隔符號：`;`
- 排名指標：`RMSE`
- 模型：`Regression`、`RANSAC`、`Ridge`、`LASSO`、`ElasticNet`、`DTR`、`RFR`
- 推薦優化模型：`Random Forest Regressor`

## 指令用法

### 1. 顯示支援模型

```powershell
.venv\Scripts\python.exe -m mltrainer list-models
```

### 2. 下載資料集

```powershell
.venv\Scripts\python.exe -m mltrainer download --dataset heart
.venv\Scripts\python.exe -m mltrainer download --dataset winequality-red
.venv\Scripts\python.exe -m mltrainer download --all
```

說明：

- 若本機沒有資料集，可先用這個指令下載
- 下載過程會顯示進度條
- `winequality-red` 會直接從 UCI 下載
- `heart` 會透過 Kaggle 下載，因此需要先配置 Kaggle API 憑證

### 3. 訓練指定資料集

```powershell
.venv\Scripts\python.exe -m mltrainer train --dataset heart
.venv\Scripts\python.exe -m mltrainer train --dataset winequality-red
```

說明：

- 若缺少資料集，會先提示並自動下載
- 會訓練該資料集對應的全部模型
- 會依規格排序結果
- 會將實際排名第一名的模型存成 `best_model.joblib`

### 4. 讀取既有結果並比較

```powershell
.venv\Scripts\python.exe -m mltrainer compare --dataset heart
.venv\Scripts\python.exe -m mltrainer compare --dataset winequality-red
```

說明：

- `compare` 不會重跑訓練
- 只會讀取既有 `metrics.json`
- 若找不到結果檔，會提示你先執行 `train`

### 5. 優化推薦模型

```powershell
.venv\Scripts\python.exe -m mltrainer optimize --dataset heart --method random
.venv\Scripts\python.exe -m mltrainer optimize --dataset heart --method grid

.venv\Scripts\python.exe -m mltrainer optimize --dataset winequality-red --method random
.venv\Scripts\python.exe -m mltrainer optimize --dataset winequality-red --method grid
```

說明：

- 若缺少資料集，會先提示並自動下載
- `optimize` 只優化規格指定的推薦模型
- `heart` 只優化 `Random Forest Classifier`
- `winequality-red` 只優化 `Random Forest Regressor`
- 優化後會覆蓋 `best_model.joblib`
- 優化資訊會寫回 `metrics.json` 的 `optimization` 欄位

## 輸出檔案

執行 `train` 或 `optimize` 後，結果會輸出到 `outputs/`：

```text
outputs/
  heart/
    metrics.json
    best_model.joblib
  winequality-red/
    metrics.json
    best_model.joblib
```

### `metrics.json` 內容

包含：

- dataset 名稱
- 任務類型
- 排名指標
- 資料切分設定
- 最佳模型資訊
- 各模型評估結果與排名
- 優化結果（若有執行 optimize）

## 實作細節

- 固定 `random_state = 42`
- `test_size = 0.2`
- `heart` 使用 stratified split
- `winequality-red` 使用一般 train/test split
- 終端輸出使用 `rich` 與 `colorama`

## 建議操作順序

```powershell
.venv\Scripts\python.exe -m mltrainer list-models
.venv\Scripts\python.exe -m mltrainer train --dataset heart
.venv\Scripts\python.exe -m mltrainer compare --dataset heart
.venv\Scripts\python.exe -m mltrainer optimize --dataset heart --method random
```

再依同樣方式處理 `winequality-red`。

## 常見問題

### PowerShell 找不到 `mltrainer`

如果你直接輸入 `mltrainer ...` 出現找不到指令，通常是因為目前 shell 沒有啟用 `.venv`。

可改用：

```powershell
.venv\Scripts\mltrainer.exe list-models
```

或：

```powershell
.venv\Scripts\python.exe -m mltrainer list-models
```

### `compare` 顯示找不到 `metrics.json`

先執行對應資料集的訓練：

```powershell
.venv\Scripts\python.exe -m mltrainer train --dataset heart
```

### `heart` 下載失敗

`heart` 使用 Kaggle 資料來源。若尚未設定 Kaggle API 憑證，下載會失敗。

請先準備 Kaggle 的 `kaggle.json` 憑證，再重新執行：

```powershell
.venv\Scripts\python.exe -m mltrainer download --dataset heart
```

## 專案結構

```text
main.py
pyproject.toml
requirements.txt
mltrainer/
  cli.py
  core/
  ui/
dataset/
outputs/
```
