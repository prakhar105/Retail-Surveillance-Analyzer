# 🛒 Retail Surveillance Analyzer
![](https://github.com/prakhar105/supermarket-item-monitor/blob/main/assests/logo.png)
A computer vision-powered surveillance tool that uses a **Visual Language Model (VLM)** to detect suspicious behavior (e.g., shoplifting, weapon possession) from CCTV footage. Built with [LLaVA-NeXT](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf), EasyOCR, and Gradio.

![](https://github.com/prakhar105/supermarket-item-monitor/blob/main/assests/Screenshot%202025-09-02%20112218.png)


---

## 🔍 Key Features

-  **Video Upload or Webcam Recording** – Analyze pre-recorded videos or live camera input.
-  **AI-Powered Understanding** – Uses a quantized LLaVA-NeXT VLM to understand visual context.
-  **Suspicious Activity Detection** – Prompts like _“Is anyone holding a gun?”_ or _“Is anyone shoplifting?”_ are analyzed.
-  **Timestamp Detection** – Automatically extracts visual timestamps using OCR.
-  **4-bit Quantized Inference** – Efficient GPU usage with [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) quantization.

![](https://github.com/prakhar105/supermarket-item-monitor/blob/main/assests/flowchart.png)
---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/retail-surveillance-analyzer.git
cd retail-surveillance-analyzer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

Make sure you have `ffmpeg` installed and available in your system path.

---

##  Model Used

- **LLaVA-NeXT v1.6 Mistral 7B (Quantized)**  
  [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)

---

##  How to Use

### 🔗 Gradio Web UI

```bash
python app.py  # or your main file name
```

- Upload a video or record using webcam.
- Ask a question like:  
  _“Is anyone holding a gun?”_  
  _“Is anyone shoplifting?”_
- View timestamps of suspicious activity.

###  Optional CLI Batch Mode

To analyze all videos in a folder (`test_videos/`), uncomment the section at the end of the script and run:

```bash
python app.py
```

---

##  Project Structure

```
.
├── app.py                # Main application
├── test_videos/         # Optional test video folder
├── requirements.txt      # Python dependencies
└── README.md
```

---

##  Tech Stack

- 🖼️ **LLaVA-NeXT** – Visual-Language model
- 🔤 **EasyOCR** – Extract timestamps from video frames
- 🎛️ **BitsAndBytes** – 4-bit quantized inference
- 🎛️ **Transformers** – Model hub and utilities
- 🧪 **Gradio** – UI for video upload & interaction
- 🎥 **OpenCV** – Frame extraction and resizing

---

##  Example Prompt

```text
Is anyone holding a gun?
```

💬 AI might respond:  
_“Yes, at around 0:26 in the video, a woman dressed in black appears to be placing an item into her black handbag.”_

---

##  To-Do / Improvements

- [ ] Integrate motion detection for faster frame skipping
- [ ] Use video metadata timestamps (instead of OCR)
- [ ] Add person tracking / ID persistence
- [ ] Export suspicious frames to disk

---

##  License

MIT License – use freely with attribution.

---

##  Acknowledgements

- Hugging Face 🤗 – for LLaVA and Transformers
- BitsAndBytes by Tim Dettmers
- EasyOCR by Jaided AI
- Gradio by Hugging Face