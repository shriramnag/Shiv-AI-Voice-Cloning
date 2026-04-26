import os
import sys
import logging
import tempfile
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch
import scipy.io.wavfile as wavfile
import re
import uuid

# --- Shiv AI Directory Setup ---
temp_audio_dir="./Shiv_Audio"
os.makedirs(temp_audio_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Setup path to import components
# ---------------------------------------------------------------------------
OmniVoice_path = f"{os.getcwd()}/OmniVoice/"
sys.path.append(OmniVoice_path)
from subtitle import subtitle_maker

try:
    from subtitle import LANGUAGE_CODE as WHISPER_LANGUAGE_CODE
except ImportError:
    WHISPER_LANGUAGE_CODE = None

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# ---------------------------------------------------------------------------
# Logging & Brand Identity
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")
print("🚀 Shiv AI System is starting by Shri Ram Nag...")

# ---------------------------------------------------------------------------
# Model Loading (Linking to your Repository)
# ---------------------------------------------------------------------------
# यहाँ आपकी रिपॉजिटरी का उपयोग किया गया है
SHIV_AI_REPO = "Shriramnag/Shiv-AI-Voice-Cloning"

try:
    print(f"Loading Shiv AI model from {SHIV_AI_REPO}...")
    model = OmniVoice.from_pretrained(
        SHIV_AI_REPO,
        device_map="cuda",
        dtype=torch.float16,
        load_asr=False,
    )
except Exception as e:
    print(f"Falling back to local model path for Shiv AI...")
    model = OmniVoice.from_pretrained(
        "./", 
        device_map="cuda",
        dtype=torch.float16,
        load_asr=False,
    )

sampling_rate = model.sampling_rate
print("✅ Shiv AI Model Loaded Successfully!")

# ---------------------------------------------------------------------------
# Event Tags & JS Functions
# ---------------------------------------------------------------------------
EVENT_TAGS = [
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]", 
    "[surprise-wa]", "[dissatisfaction-hnn]"
]

# JS for Voice Tag Insertion
INSERT_TAG_JS = """
(tag_val, current_text) => {
    const textarea = document.querySelector('.shiv-textbox textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    return current_text.slice(0, start) + " " + tag_val + " " + current_text.slice(end);
}
"""

# ---------------------------------------------------------------------------
# Core Logic & Helpers
# ---------------------------------------------------------------------------
def tts_file_name(text, language="hi"):
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)[:20].strip().replace(" ", "_")
    rand = uuid.uuid4().hex[:6].upper()
    return f"{temp_audio_dir}/ShivAI_{clean_text}_{rand}.wav"

def _gen_core(text, language, ref_audio, instruct, mode, ref_text=None, **kwargs):
    if not text or not text.strip():
        return None, "कृपया टेक्स्ट लिखें।"

    gen_config = OmniVoiceGenerationConfig(
        num_step=32,
        guidance_scale=2.0,
        denoise=True,
        preprocess_prompt=True,
        postprocess_output=True,
    )

    kw = dict(text=text.strip(), language=language if language != "Auto" else None, generation_config=gen_config)
    
    if mode == "clone":
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    
    audio = model.generate(**kw)
    waveform = (audio[0] * 32767).astype(np.int16)
    return (sampling_rate, waveform), "सफलतापूर्वक जनरेट हुआ!"

# ---------------------------------------------------------------------------
# Gradio UI Construction (Fully Branded)
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(primary_hue="orange", font=["Inter", "Arial", "sans-serif"])

css = """
.gradio-container {max-width: 100% !important;}
footer {visibility: hidden !important;}
.shiv-header {text-align: center; margin: 20px auto; padding: 10px; border-bottom: 2px solid #ff6600;}
.tag-btn {background: #fff3e0 !important; border: 1px solid #ffcc80 !important; color: #e65100 !important;}
"""

with gr.Blocks(theme=theme, css=css, title="Shiv AI - Shri Ram Nag") as demo:
    gr.HTML("""
        <div class="shiv-header">
            <h1 style="font-size: 2.8em; color: #ff6600; margin-bottom: 0;">🔱 Shiv AI Voice Cloning</h1>
            <p style="font-size: 1.2em; color: #555;">Advanced Multilingual Speech Engine | <b>Owner: Shri Ram Nag</b></p>
        </div>
    """)

    with gr.Tabs():
        with gr.TabItem("Voice Clone (आवाज़ क्लोनिंग)"):
            with gr.Row():
                with gr.Column():
                    vc_text = gr.Textbox(label="Text to Synthesize", lines=5, elem_classes="shiv-textbox", placeholder="यहाँ अपना टेक्स्ट लिखें...")
                    
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            btn.click(fn=None, inputs=[btn, vc_text], outputs=vc_text, js=INSERT_TAG_JS)
                    
                    vc_lang = gr.Dropdown(label="भाषा (Language)", choices=["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES), value="Auto")
                    vc_ref_audio = gr.Audio(label="अपनी आवाज़ अपलोड करें (Reference Audio)", type="filepath")
                    vc_btn = gr.Button("शिव आवाज़ जनरेट करें", variant="primary")
                
                with gr.Column():
                    vc_audio = gr.Audio(label="Shiv AI Output", type="numpy")
                    gr.Markdown("### **प्रोजेक्ट डेवलपर: श्री राम नाग**")
                    gr.Markdown("यह सिस्टम आपकी आवाज़ को 600+ भाषाओं में क्लोन कर सकता है।")

    gr.HTML("<div style='text-align:center; padding: 20px; color: #888;'>© 2026 Shiv AI | Designed for Shri Ram Nag</div>")

    # इवेंट हैंडलिंग
    def start_clone(text, lang, ref_aud):
        res, status = _gen_core(text, lang, ref_aud, None, mode="clone")
        return res

    vc_btn.click(fn=start_clone, inputs=[vc_text, vc_lang, vc_ref_audio], outputs=[vc_audio])

if __name__ == "__main__":
    demo.launch(share=True)
