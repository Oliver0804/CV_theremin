# Theremin 手部追蹤專案

此專案使用手部追蹤來控制類似 Theremin 的音效輸出。左手控制音調，右手控制音量。此外，會在視頻畫面上顯示一個鏡像的 PNG 圖像。

## 環境建置

請按照以下步驟使用 conda 安裝所需的 Python 庫並建置環境：

```bash
# 創建新的 conda 環境
conda create -n theremin_env python=3.9

# 激活新創建的環境
conda activate theremin_env

# 安裝所需的 Python 庫
pip install opencv-python numpy mediapipe pyaudio
```

## 使用說明

確保您的電腦連接了攝像頭。
將 play.png 圖像文件放在與腳本相同的目錄中。
運行 Python 腳本。

```
python main.py
```

## 注意事項

左手控制音調：當左手移向左邊時音調升高，移向右邊時音調降低。
右手控制音量：當右手移向上方時音量增大，移向下方時音量減小。
