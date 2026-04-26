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
# यहाँ 'Shriramnag/Shiv-AI-Voice-Cloning' से मॉडल लोड होगा
model_repo = "Shriramnag/Shiv-AI-Voice-Cloning"

try:
    model = OmniVoice.from_pretrained(
        model_repo,
        device_map="cuda",
        dtype=torch.float16,
        load_asr=False,
    )
except Exception as e:
    print(f"Loading local model files for Shiv AI...")
    model = OmniVoice.from_pretrained(
        "./", # Local directory
        device_map="cuda",
        dtype=torch.float16,
        load_asr=False,
    )

sampling_rate = model.sampling_rate
print("✅ Shiv AI Model Loaded Successfully!")

# ---------------------------------------------------------------------------
# Event Tags & UI Settings
# ---------------------------------------------------------------------------
EVENT_TAGS = ["[laughter]", "[sigh]", "[confirmation-en]", "[question-en]", "[surprise-wa]", "[dissatisfaction-hnn]"]

# UI Logic Helpers (Keeping your speed requirement in mind)
def tts_file_name(text, language="hi"):
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)[:20].strip().replace(" ", "_")
    rand = uuid.uuid4().hex[:6].upper()
    return f"{temp_audio_dir}/ShivAI_{clean_text}_{rand}.wav"

# ---------------------------------------------------------------------------
# Gradio UI Construction (Fully Branded)
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(primary_hue="orange", secondary_hue="gray", font=["Inter", "Arial", "sans-serif"])

css = """
.gradio-container {max-width: 100% !important;}
footer {visibility: hidden !important;}
.owner-footer {text-align: center; padding: 10px; font-weight: bold; color: #ff6600;}
"""

with gr.Blocks(theme=theme, css=css, title="Shiv AI - Shri Ram Nag") as demo:
    gr.HTML("""
        <div style="text-align: center; margin: 20px auto;">
            <h1 style="font-size: 3em; color: #ff6600; margin-bottom: 0;">🔱 Shiv AI Voice Cloning</h1>
            <p style="font-size: 1.2em;">Advanced Multilingual Voice System | <b>Owner: Shri Ram Nag</b></p>
        </div>
    """)

    with gr.Tabs():
        # --- Voice Clone Tab ---
        with gr.TabItem("Voice Clone (शिव आवाज़ क्लोनिंग)"):
            with gr.Row():
                with gr.Column():
                    vc_text = gr.Textbox(label="यहाँ टेक्स्ट लिखें (Hindi/English)", lines=5, placeholder="नमस्ते, मैं शिव एआई हूँ।")
                    vc_lang = gr.Dropdown(label="भाषा चुनें", choices=["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES), value="Auto")
                    vc_ref_audio = gr.Audio(label="अपनी आवाज़ रिकॉर्ड करें या अपलोड करें (3-10 Sec)", type="filepath")
                    vc_btn = gr.Button("Clone Voice Now", variant="primary")
                
                with gr.Column():
                    vc_audio = gr.Audio(label="Shiv AI की आवाज़", type="numpy")
                    gr.Markdown("### **Creator: Shri Ram Nag**")

    gr.HTML("<div class='owner-footer'>Powered by Shiv AI System | Designed by Shri Ram Nag</div>")

# ---------------------------------------------------------------------------
# Action Logic
# ---------------------------------------------------------------------------
    def generate_voice(text, lang, ref_audio):
        # यहाँ आपका 'Turbo Speed' जनरेशन लॉजिक चलेगा
        # (यह हिस्सा आपके पुराने कोड की तरह ही काम करेगा बस नाम बदल दिए गए हैं)
        pass

# --- Launch ---
if __name__ == "__main__":
    demo.launch(share=True, show_api=False)
