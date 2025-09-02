import cv2
import torch
import os
import numpy as np
from PIL import Image
import easyocr
import gradio as gr
import tempfile
import shutil
import re
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

# ──────────────────────────────────────────────
# Model and Processor Setup
# ──────────────────────────────────────────────
model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Load quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Specify memory limits for auto device mapping
max_memory = {
    0: "8GiB",       # GPU 0
    "cpu": "16GiB"   # CPU fallback
}

# Load LLaVA-NeXT model
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
    max_memory=max_memory,
    torch_dtype=torch.float16
)

# Load processor
processor = LlavaNextProcessor.from_pretrained(model_id)

# Load OCR
reader = easyocr.Reader(['en'], gpu=False)

# ──────────────────────────────────────────────
# Core VLM Frame Analysis
# ──────────────────────────────────────────────
def frame_to_vlm_response(frame, question):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # LLaVA chat format
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image"}
        ],
    }]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True)
    decoded = processor.decode(outputs[0], skip_special_tokens=True).strip()

    return decoded

# ──────────────────────────────────────────────
# Timestamp Extraction via OCR
# ──────────────────────────────────────────────
def get_timestamp_from_frame(frame):
    result = reader.readtext(frame)
    for _, text, conf in result:
        matches = re.findall(r'\d{1,2}:\d{2}:\d{2}', text)
        if matches:
            return matches[0]
    return None

# ──────────────────────────────────────────────
# Video Analyzer
# ──────────────────────────────────────────────
def analyze_video_stream(video_path, question, fps=2):
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, int(frame_rate / fps))

    suspicious_times = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            try:
                frame = cv2.resize(frame, (448, 448))
                # Refined prompt
                refined_prompt = "Only answer yes or no. Is there a person clearly holding a gun in this frame?"
                response = frame_to_vlm_response(frame, refined_prompt)

                # Clean and evaluate response
                if '?' in response:
                    response = response.split('?', 1)[1].strip()

                if any(word in response.lower() for word in ["yes", "gun", "weapon", "armed"]):
                    ts = get_timestamp_from_frame(frame)
                    if ts:
                        suspicious_times.append(ts)
            except Exception as e:
                print(f"⚠️ Frame error: {e}")

        frame_id += 1

    cap.release()
    return suspicious_times

# ──────────────────────────────────────────────
# Gradio Video Handler
# ──────────────────────────────────────────────
def handle_video_input(video_file, question):
    if video_file is None:
        return "⚠️ Please upload or record a video."

    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "input_video.mp4")
    shutil.copy(video_file, video_path)

    try:
        result = analyze_video_stream(video_path, question)
    except Exception as e:
        result = f"Error during analysis: {e}"

    shutil.rmtree(temp_dir)

    if isinstance(result, list) and result:
        return f"🕵️ Suspicious activity detected at: {', '.join(result)}"
    elif isinstance(result, list):
        return "✅ No suspicious activity detected."
    else:
        return result

# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────
with gr.Blocks(title="Retail Surveillance Analyzer") as demo:
    gr.Markdown("## 🛒 Retail Surveillance Analyzer")
    gr.Markdown("Upload a CCTV video or record using your webcam. The AI will analyze it for suspicious activity.")

    with gr.Row():
        video_input = gr.Video(label="Upload or Record Video", format="mp4")
        question_input = gr.Textbox(label="Question for AI", value="Is anyone holding a gun?")

    analyze_btn = gr.Button("Analyze Video")
    output_text = gr.Textbox(label="AI Output", lines=5)

    analyze_btn.click(fn=handle_video_input, inputs=[video_input, question_input], outputs=output_text)

if __name__ == "__main__":
    demo.launch()
