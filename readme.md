一个监听麦克风或读取本地音频文件（`wav/mp3/m4a/...`），将泰语语音识别并翻译成英文的脚本。

### 运行

```bash
pip install -r requirements.txt
python app.py
```

### 注意事项（Windows）

- **处理 `.m4a`**：`pydub` 依赖 **ffmpeg**。请先安装 ffmpeg 并确保 `ffmpeg` 在 PATH 中可用。
- **无 GPU 的机器**：已兼容 CPU（transformers pipeline 使用 `device=-1`）。