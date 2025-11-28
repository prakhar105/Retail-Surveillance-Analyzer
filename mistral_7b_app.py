import cv2
import torch
import os
import numpy as np
from PIL import Image
import easyocr
import gradio as gr
import tempfile
import os
import shutil
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"


# Quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # more accurate
    bnb_4bit_compute_dtype=torch.float16
)

# Load processor
processor = LlavaNextProcessor.from_pretrained(model_id)

# Load VLM and processor
model =LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",  # will automatically split across GPU/CPU if needed
    quantization_config=quant_config,
    torch_dtype=torch.float16
).to(device)

# Load OCR
reader = easyocr.Reader(['en'] ,gpu=False)

def frame_to_vlm_response(frame, question):
    """Send an OpenCV frame directly to LLaVA-NeXT VLM."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Chat-style prompt template
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image"}
        ],
    }]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=70)
    return processor.decode(outputs[0], skip_special_tokens=True)
    del inputs, outputs
    torch.cuda.empty_cache()


def get_timestamp_from_frame(frame):
    """Extract timestamp from video frame using OCR."""
    result = reader.readtext(frame)
    for _, text, conf in result:
        if ":" in text:  # crude filter for HH:MM:SS
            return text.strip()
    return None

def analyze_video_stream(video_path, question, fps=1):
    """Scan video for suspicious activity using VLM and OCR."""
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
            response = frame_to_vlm_response(frame, question)
            # Optional: Split off after prompt
            if '?' in response:
                response = response.split('?', 1)[1].strip()
            if "yes" in response.lower():
                ts = get_timestamp_from_frame(frame)
                if ts:
                    suspicious_times.append(ts)

        frame_id += 1
        cap.release()
    return suspicious_times, response


def handle_video_input(video_file, question):
    if video_file is None:
        return " Please upload or record a video."
    
    # Save the uploaded video to a temporary location
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "input_video.mp4")
    
    shutil.copy(video_file, video_path)
    
    # Call your model pipeline
    result = analyze_video_stream(video_path, question)
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    return result



# Gradio UI
with gr.Blocks(title="Retail CCTV Analyzer") as demo:
    gr.Markdown("## Retail Surveillance Analyzer")
    gr.Markdown("Upload a CCTV video or record using your webcam. The AI will analyze it for suspicious activity.")

    with gr.Row():
        video_input = gr.Video(label="Upload or Record Video", format="mp4")
        question_input = gr.Textbox(label="Question for AI", value="Is anyone holding a gun?")
    
    analyze_btn = gr.Button("Analyze Video")
    output_text = gr.Textbox(label="AI Output")

    analyze_btn.click(fn=handle_video_input, 
                      inputs=[video_input, question_input],
                      outputs=output_text)
    
# ─── Optional: Local CLI batch test for all test_videos/ ─── #
if __name__ == "__main__":
    demo.launch()

# # Analyze all videos
# video_dir = "test_videos"
# question = "Is anyone holding a gun?"

# for video in os.listdir(video_dir):
#     video_path = os.path.join(video_dir, video)
#     print(video_path)
#     timestamps = analyze_video_stream(video_path, question)
#     print(f"\nVideo: {video}")
#     print("Suspicious activity at:", timestamps if timestamps else "None detected")
