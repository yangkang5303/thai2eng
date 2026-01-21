#an app that listen to micophone or mp3 file and translate the voice to english in realtime

import warnings
import torch
import sounddevice as sd
import numpy as np
import queue
import threading
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import time
import soundfile as sf
import os
from pydub import AudioSegment
import io
from collections import deque

# 静默已知的 transformers 告警（不影响结果，但会刷屏）
warnings.filterwarnings(
    "ignore",
    message=r".*input name `inputs` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*forced_decoder_ids.*creates a conflict.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*attention mask is not set and cannot be inferred.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*chunk_length_s.*is very experimental.*",
)

# 初始化语音识别模型
MODEL_NAME = "biodatlab/whisper-th-medium-combined"
lang = "th"
# transformers pipeline 的 device 参数：GPU 用 0/1/...；CPU 用 -1（不要传 "cpu" 字符串）
device = 0 if torch.cuda.is_available() else -1
# 如果你想强制使用 GPU，并在 GPU 不可用时报错，可以直接设置 device=0
# device = 0 

asr_pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
    batch_size=8,  # 启用批处理以提高 GPU 利用率
    ignore_warning=True,  # 抑制 "chunk_length_s is experimental" 警告
)

# 避免 language 参数与 forced_decoder_ids 的冲突告警
try:
    if hasattr(asr_pipe, "model") and hasattr(asr_pipe.model, "generation_config"):
        asr_pipe.model.generation_config.forced_decoder_ids = None
except Exception:
    pass

# 初始化翻译模型（使用更先进的 NLLB-200 泰英翻译模型）
NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
translator = pipeline(
    "translation",
    model=NLLB_MODEL_ID,
    tokenizer=NLLB_MODEL_ID,
    src_lang="tha_Thai",
    tgt_lang="eng_Latn",
    device=device,
    max_length=512,  # 增加最大长度以避免警告并处理长句子
)

# 音频参数
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
# CHUNK_DURATION 已不再用于麦克风模式的固定触发，仅作为文件处理或默认参考
CHUNK_DURATION = 10  
audio_queue = queue.Queue()

# VAD (语音活动检测) 参数
VAD_SILENCE_THRESHOLD = 0.01  # 能量阈值，低于此值视为静音
VAD_SILENCE_DURATION = 0.8    # 静音持续多久（秒）则触发翻译
VAD_MAX_DURATION = 15.0       # 单次最长录音时长（秒），强制触发翻译
VAD_MIN_DURATION = 0.5        # 最短有效语音时长（秒）


def choose_input_device():
    """
    让用户选择输入设备（比如麦克风 / 立体声混音 / 虚拟声卡），
    这样可以更容易抓到“电脑里其他程序”的声音。
    """
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as e:
        print(f"查询音频设备失败，将使用默认输入设备。错误: {e}")
        return None

    print("\n可用输入设备列表：")
    input_indices = []
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            input_indices.append(idx)
            hostapi_name = hostapis[dev["hostapi"]]["name"]
            print(
                f"{idx}: {dev['name']}  "
                f"(hostapi={hostapi_name}, 输入通道数={dev['max_input_channels']})"
            )

    print("提示：如果你想抓电脑里其他程序的声音，")
    print("请在 Windows 声音设置里启用“立体声混音 / Stereo Mix / What U Hear”等输入设备，")
    print("然后在这里选择对应的设备编号。")

    choice = input("请输入要使用的输入设备编号（直接回车=使用默认设备）: ").strip()
    if choice == "":
        return None

    try:
        idx = int(choice)
    except ValueError:
        print("输入不是有效的数字，将使用默认设备。")
        return None

    if idx not in input_indices:
        print("编号不在可用输入设备范围内，将使用默认设备。")
        return None

    return idx


def audio_callback(indata, frames, time, status):
    """音频回调函数，将音频数据放入队列"""
    if status:
        print(f"状态: {status}")
    audio_queue.put(indata.copy())

def process_audio():
    """处理音频数据，使用 VAD (语音活动检测) 实现更自然的触发"""
    print("\n[实时翻译已就绪] 请说话...")
    
    current_audio_buffer = []
    silence_start_time = None
    voice_start_time = None
    
    while True:
        try:
            # 阻塞等待音频数据，设置超时以便循环检查状态
            audio_chunk = audio_queue.get(timeout=0.1)
            audio_chunk = audio_chunk.flatten()
            
            # 计算当前块的能量 (RMS)
            rms = np.sqrt(np.mean(audio_chunk**2))
            is_silent = rms < VAD_SILENCE_THRESHOLD
            
            current_audio_buffer.append(audio_chunk)
            
            now = time.time()
            
            if not is_silent:
                # 检测到声音
                if voice_start_time is None:
                    voice_start_time = now
                silence_start_time = None # 重置静音计时器
            else:
                # 静音中
                if silence_start_time is None:
                    silence_start_time = now
            
            # 检查触发条件
            should_process = False
            
            if voice_start_time is not None:
                duration = now - voice_start_time
                
                # 条件 1: 持续静音足够久 (一段话讲完了)
                if is_silent and silence_start_time and (now - silence_start_time >= VAD_SILENCE_DURATION):
                    if duration >= VAD_MIN_DURATION:
                        should_process = True
                
                # 条件 2: 达到最大时长 (防止一句话太长或环境噪音导致不触发)
                if duration >= VAD_MAX_DURATION:
                    should_process = True
            
            if should_process:
                # 合并缓冲区数据
                audio_data = np.concatenate(current_audio_buffer, dtype=np.float32)
                
                # 重置状态
                current_audio_buffer = []
                voice_start_time = None
                silence_start_time = None
                
                # 执行 ASR 和翻译
                result = asr_pipe(audio_data, generate_kwargs={"language": "th"})
                thai_text = result["text"].strip()
                
                if thai_text:
                    # 简单的幻觉/重复过滤
                    words = thai_text.split()
                    if len(words) > 10 and len(set(words)) / len(words) < 0.2:
                        continue # 跳过疑似幻觉
                        
                    translation = translator(thai_text)[0]['translation_text']
                    print(f"[{time.strftime('%H:%M:%S')}] Thai: {thai_text}")
                    print(f"[{time.strftime('%H:%M:%S')}] English: {translation}")
                    
        except queue.Empty:
            # 如果队列为空且 buffer 里有东西，并且静音很久了，也可以考虑触发
            continue
        except Exception as e:
            print(f"处理音频时出错: {e}")
            continue

def process_audio_chunk(audio_data):
    """处理音频数据块"""
    # 进行语音识别
    result = asr_pipe(audio_data, generate_kwargs={"language": "th"})
    thai_text = result["text"]
    
    if thai_text.strip():
        # 进行翻译
        translation = translator(thai_text)[0]['translation_text']
        #print(f"\n[{time.strftime('%H:%M:%S')}] 泰语: {thai_text}")
        print(f"[{time.strftime('%H:%M:%S')}] 英语: {translation}")

def process_audio_file(file_path):
    """处理音频文件，利用 pipeline 的内部分段和批处理功能，并分大块处理以显示进度"""
    print(f"正在处理音频文件: {file_path}")
    
    try:
        # 获取文件扩展名
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.m4a':
            # pydub 处理 m4a
            audio = AudioSegment.from_file(file_path, format="m4a")
            audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
            audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        else:
            # soundfile 处理其他格式
            audio_data, sample_rate = sf.read(file_path)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            if sample_rate != SAMPLE_RATE:
                from scipy import signal
                audio_data = signal.resample(audio_data, int(len(audio_data) * SAMPLE_RATE / sample_rate))

        duration_s = len(audio_data) / SAMPLE_RATE
        print(f"音频数据加载完成，总时长: {duration_s:.2f} 秒")
        
        # 使用 15 秒一快进行处理，平衡速度和 GPU 利用率，确保存实时感
        segment_duration = 15
        segment_samples = segment_duration * SAMPLE_RATE
        
        print(f"开始识别并翻译 (每段 {segment_duration} 秒)...")
        
        for i in range(0, len(audio_data), segment_samples):
            segment = audio_data[i:i + segment_samples]
            current_start_time = i / SAMPLE_RATE
            
            # 识别当前段
            result = asr_pipe(
                segment, 
                generate_kwargs={"language": "th"}, 
                return_timestamps=True,
                chunk_length_s=30,
                stride_length_s=5
            )
            
            chunks = result.get("chunks", [])
            if not chunks:
                thai_text = result.get("text", "").strip()
                if thai_text:
                    translation = translator(thai_text)[0]['translation_text']
                    print(f"[{int(current_start_time//60):02d}:{int(current_start_time%60):02d}] Thai: {thai_text}")
                    print(f"[{int(current_start_time//60):02d}:{int(current_start_time%60):02d}] English: {translation}")
                continue

            # 准备批量翻译段内的 chunk
            texts_to_translate = []
            chunk_data_valid = []
            for c in chunks:
                text = c["text"].strip()
                if not text:
                    continue
                
                # 简单的幻觉检测：如果文本过于单一且重复，则跳过
                words = text.split()
                if len(words) > 10 and len(set(words)) / len(words) < 0.2:
                    continue
                
                texts_to_translate.append(text)
                chunk_data_valid.append(c)
            
            if texts_to_translate:
                translations = translator(texts_to_translate, batch_size=8, max_length=512)
                for c, trans in zip(chunk_data_valid, translations):
                    thai_text = c["text"].strip()
                    translation_text = trans['translation_text']
                    start, end = c["timestamp"]
                    
                    # 进一步检测翻译中的重复
                    words_trans = translation_text.split()
                    if len(words_trans) > 20 and len(set(words_trans)) / len(words_trans) < 0.1:
                         translation_text = " ".join(words_trans[:10]) + "... (filtered)"

                    # 调整时间戳到全局时间
                    if start is not None and end is not None:
                        global_start = current_start_time + start
                        global_end = current_start_time + end
                        time_str = f"[{int(global_start//60):02d}:{int(global_start%60):02d} -> {int(global_end//60):02d}:{int(global_end%60):02d}]"
                    else:
                        time_str = f"[{int(current_start_time//60):02d}:{int(current_start_time%60):02d}]"
                    
                    print(f"{time_str} Thai: {thai_text}")
                    print(f"{time_str} English: {translation_text}")
            
            # 显示总体进度
            progress = min(100, (i + segment_samples) / len(audio_data) * 100)
            current_pos_s = min(duration_s, (i + segment_samples) / SAMPLE_RATE)
            print(f">>> 进度: {progress:.1f}% (已处理: {current_pos_s:.1f}/{duration_s:.1f}s)")
        
        print("\n处理全部完成！")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n处理音频文件时出错: {str(e)}")
        print("支持的音频格式: wav, mp3, m4a, flac, ogg")

def main():
    print("请选择音频输入源：")
    print("1. 麦克风实时录音")
    print("2. 本地音频文件")
    
    choice = input("请输入选项（1或2）: ")
    
    if choice == "1":
        print("开始录音，按 Ctrl+C 停止...")
        print("每10秒将输出一次识别和翻译结果")

        # 让用户选择具体的输入设备（比如立体声混音），没有选择就用默认输入设备
        device_index = choose_input_device()
        
        # 启动音频处理线程
        process_thread = threading.Thread(target=process_audio, daemon=True)
        process_thread.start()
        
        try:
            stream_kwargs = dict(
                callback=audio_callback,
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
            )
            if device_index is not None:
                stream_kwargs["device"] = device_index

            with sd.InputStream(**stream_kwargs):
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n程序已停止")
            
    elif choice == "2":
        file_path = input("请输入音频文件路径: ")
        if os.path.exists(file_path):
            process_audio_file(file_path)
        else:
            print("错误：文件不存在！")
    else:
        print("无效的选项！")

if __name__ == "__main__":
    main()