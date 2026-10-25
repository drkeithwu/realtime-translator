import io
import os
import time
import wave
import threading

import gradio as gr
import numpy as np

from google import genai
from google.genai import types

# ==========================================
# 1. Gemini API 設定與初始化
# ==========================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    try:
        with open("GEMINI_API_KEY.txt", "r", encoding="utf-8") as f:
            API_KEY = f.read().strip()
    except FileNotFoundError:
        pass

if not API_KEY:
    raise ValueError("未找到 API Key！請設定 GEMINI_API_KEY 環境變數，或於同目錄提供 GEMINI_API_KEY.txt 檔案。")

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(timeout=10000)
)

# 使用目前官方支援且反應最快速的模型
MODEL_NAME = "gemini-2.5-flash"

# 省額度與穩定切分設定（避免頻繁觸發 429 額度限制）
MIN_SEGMENT_SECONDS = 3.0    # 累積滿 3.0 秒才檢查斷句
MAX_SEGMENT_SECONDS = 7.0    # 最長 7.0 秒強制切分
PAUSE_THRESHOLD = 0.8        # 停頓超過 0.8 秒判定一句結束
SILENCE_RMS = 0.015          # 靜音判定門檻

RETRY_DELAY = 15             # 觸發 429 錯誤時的冷卻時間拉長至 15 秒
audio_lock = threading.Lock()


# ==========================================
# 2. 音訊轉成 WAV 格式
# ==========================================

def audio_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """將 numpy 陣列格式的音訊轉成 16-bit PCM WAV"""
    audio = np.asarray(audio)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.floating):
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
    else:
        audio = audio.astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(audio.tobytes())

    return buffer.getvalue()


# ==========================================
# 3. Gemini 語音辨識 + 翻譯
# ==========================================

def translate_audio(audio: np.ndarray, sample_rate: int, history_context: str = ""):
    """將音訊片段傳送至 Gemini 進行語音辨識與翻譯"""
    wav_data = audio_to_wav(audio, sample_rate)

    prompt = f"""你是專業的台灣繁體中文與印尼文口譯員。
任務：將音訊轉寫為繁體中文，並翻譯為印尼文。

前文對話：
{history_context or "無"}

請嚴格僅輸出以下兩行：
中文：[繁體中文轉寫]
印尼文：[印尼文翻譯]

若音訊無語音或僅雜音，請直接輸出「無語音」。"""

    try:
        start_time = time.time()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(
                    data=wav_data,
                    mime_type="audio/wav"
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1
            )
        )

        result = (response.text or "").strip()
        print(f"[DEBUG] API 耗時: {time.time() - start_time:.2f}s | 結果: {result[:40]}...", flush=True)

        if not result or "無語音" in result:
            return None, None

        chinese = ""
        indonesian = ""

        for line in result.splitlines():
            line = line.strip()
            if line.startswith(("中文：", "中文:")):
                chinese = line[3:].strip()
            elif line.startswith(("印尼文：", "印尼文:")):
                indonesian = line[4:].strip()

        if not chinese and not indonesian:
            return None, None

        return chinese or "（無文字）", indonesian or "（無翻譯）"

    except Exception as e:
        print(f"[ERROR] 翻譯失敗: {e}", flush=True)
        raise e


# ==========================================
# 4. 音訊串流處理邏輯
# ==========================================

def process_audio(audio, state):
    if state is None:
        state = {
            "buffer": np.array([], dtype=np.float32),
            "sample_rate": 0,
            "chinese": "",
            "indonesian": "",
            "history": [],
            "next_retry_at": 0,
            "silence_counter": 0.0
        }

    if audio is None:
        return state["chinese"], state["indonesian"], state, "等待麥克風語音..."

    sample_rate, chunk = audio
    chunk = np.asarray(chunk)

    if chunk.ndim == 2:
        chunk = chunk.mean(axis=1)

    if np.issubdtype(chunk.dtype, np.integer):
        chunk = chunk.astype(np.float32) / 32768.0
    else:
        chunk = chunk.astype(np.float32)

    with audio_lock:
        if state["sample_rate"] != sample_rate:
            state["buffer"] = np.array([], dtype=np.float32)
            state["sample_rate"] = sample_rate
            state["silence_counter"] = 0.0

        state["buffer"] = np.concatenate([state["buffer"], chunk])
        total_seconds = len(state["buffer"]) / sample_rate

        chunk_rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0
        chunk_seconds = len(chunk) / sample_rate

        if chunk_rms < SILENCE_RMS:
            state["silence_counter"] += chunk_seconds
        else:
            state["silence_counter"] = 0.0

        # 省額度觸發判定：累積 3.0 秒以上且停頓 0.8 秒，或滿 7.0 秒
        is_speech_ended = (total_seconds >= MIN_SEGMENT_SECONDS and state["silence_counter"] >= PAUSE_THRESHOLD)
        is_too_long = (total_seconds >= MAX_SEGMENT_SECONDS)

        if not (is_speech_ended or is_too_long):
            return (
                state["chinese"],
                state["indonesian"],
                state,
                f"收音中... ({total_seconds:.1f} 秒 / 停頓 {state['silence_counter']:.1f}s)"
            )

        now = time.time()
        if now < state["next_retry_at"]:
            remain_cool = int(state["next_retry_at"] - now)
            return (
                state["chinese"],
                state["indonesian"],
                state,
                f"API 額度滿載冷卻中，請等待 {remain_cool} 秒..."
            )

        segment_to_process = state["buffer"].copy()
        history_context = "\n".join(state["history"][-2:])

        # 清空緩衝區
        state["buffer"] = np.array([], dtype=np.float32)
        state["silence_counter"] = 0.0

        try:
            chinese, indonesian = translate_audio(
                segment_to_process,
                sample_rate,
                history_context=history_context
            )

            state["next_retry_at"] = 0

            if chinese and indonesian:
                state["chinese"] += f"{chinese}\n"
                state["indonesian"] += f"{indonesian}\n"
                state["history"].append(f"中文: {chinese} -> 印尼文: {indonesian}")

                if len(state["history"]) > 6:
                    state["history"].pop(0)

            return (
                state["chinese"],
                state["indonesian"],
                state,
                "持續監聽中..."
            )

        except Exception as e:
            state["next_retry_at"] = time.time() + RETRY_DELAY
            error_msg = str(e)
            if "429" in error_msg or "QUOTA" in error_msg:
                status_str = f"已達 API 額度上限 (429)，自動冷卻 {RETRY_DELAY} 秒中..."
            else:
                status_str = f"請求失敗，{RETRY_DELAY} 秒後可重試: {error_msg[:40]}"

            return (
                state["chinese"],
                state["indonesian"],
                state,
                status_str
            )


# ==========================================
# 5. Gradio 使用者介面
# ==========================================

with gr.Blocks(title="中文 → 印尼文即時字幕") as demo:

    gr.Markdown(
        """
        # 🎙️ 中文 → 印尼文即時口譯字幕
        ### 已調整模型為 `gemini-2.5-flash` 並加上發送頻率防暴發機制。
        """
    )

    state = gr.State({
        "buffer": np.array([], dtype=np.float32),
        "sample_rate": 0,
        "chinese": "",
        "indonesian": "",
        "history": [],
        "next_retry_at": 0,
        "silence_counter": 0.0
    })

    with gr.Row():
        chinese_output = gr.Textbox(
            label="中文語音辨識 (繁體中文)",
            lines=10,
            placeholder="辨識內容將顯示於此..."
        )

        indonesian_output = gr.Textbox(
            label="印尼文翻譯 (Indonesian)",
            lines=10,
            placeholder="印尼文字幕將顯示於此..."
        )

    status = gr.Textbox(
        label="系統狀態",
        value="等待麥克風語音...",
        interactive=False
    )

    mic = gr.Audio(
        sources=["microphone"],
        type="numpy",
        streaming=True,
        label="麥克風輸入"
    )

    mic.stream(
        fn=process_audio,
        inputs=[mic, state],
        outputs=[
            chinese_output,
            indonesian_output,
            state,
            status
        ],
        stream_every=0.5,
        time_limit=3600,
        concurrency_limit=1,
        queue=False
    )

    gr.Markdown(
        """
        **💡 升級說明**
        1. **修正 404**：改用最新的 `gemini-2.5-flash` 官方模型。
        2. **防止 429 爆額度**：設定說話滿 3 秒且停頓 0.8 秒才送出，大幅減少不必要的請求。
        """
    )


# ==========================================
# 6. 主程式進入點
# ==========================================

if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )