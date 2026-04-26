import os
import sys
import logging
import re
import uuid

import gradio as gr
import numpy as np
import torch
import scipy.io.wavfile as wavfile

# --- Shiv AI Directory Setup ---
temp_audio_dir = "./Shiv_Audio"
os.makedirs(temp_audio_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# HuggingFace Model Path
# ---------------------------------------------------------------------------
SHIV_AI_REPO = "Shriramnag/Shiv-AI-Voice-Cloning"
MODEL_LOCAL_PATH = "./Shiv-AI-Voice-Cloning"

sys.path.insert(0, MODEL_LOCAL_PATH)

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

try:
    from subtitle import subtitle_maker
    from subtitle import LANGUAGE_CODE as WHISPER_LANGUAGE_CODE
except ImportError:
    WHISPER_LANGUAGE_CODE = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")
print("🔱 Shiv AI Voice Cloning starting... by Shri Ram Nag")

# ---------------------------------------------------------------------------
# Model Load
# ---------------------------------------------------------------------------
try:
    print(f"📥 Loading from HuggingFace: {SHIV_AI_REPO}")
    model = OmniVoice.from_pretrained(
        SHIV_AI_REPO,
        device_map="cuda",
        dtype=torch.float16,
        load_asr=False,
    )
except Exception as e:
    print(f"⚠️ HF load failed: {e} | Trying local...")
    model = OmniVoice.from_pretrained(
        MODEL_LOCAL_PATH,
        device_map="cuda",
        dtype=torch.float16,
        load_asr=False,
    )

sampling_rate = model.sampling_rate
print("✅ Shiv AI Model Loaded!")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
EVENT_TAGS = [
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]",
    "[surprise-wa]", "[dissatisfaction-hnn]"
]

INSERT_TAG_JS = """
(tag_val, current_text) => {
    const textarea = document.querySelector('.shiv-textbox textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    return current_text.slice(0, start) + " " + tag_val + " " + current_text.slice(end);
}
"""

LANG_CHOICES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

def make_gen_config(num_step=32, guidance_scale=2.0):
    return OmniVoiceGenerationConfig(
        num_step=num_step,
        guidance_scale=guidance_scale,
        denoise=True,
        preprocess_prompt=True,
        postprocess_output=True,
    )

# ---------------------------------------------------------------------------
# Core Generation Functions
# ---------------------------------------------------------------------------

def gen_voice_clone(text, language, ref_audio, ref_text=""):
    """Tab 1: Voice Clone — reference audio se aawaz clone karo"""
    if not text or not text.strip():
        return None, "⚠️ कृपया टेक्स्ट लिखें।"
    if not ref_audio:
        return None, "⚠️ Reference audio upload karein।"
    try:
        kw = dict(
            text=text.strip(),
            language=language if language != "Auto" else None,
            generation_config=make_gen_config(),
            voice_clone_prompt=model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text.strip() if ref_text else None
            )
        )
        audio = model.generate(**kw)
        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), "✅ Voice Clone सफल!"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def gen_voice_design(text, language, speed, pitch, energy, pause, style_instruct):
    """Tab 2: Voice Design — speed, pitch, energy, pause controls"""
    if not text or not text.strip():
        return None, "⚠️ कृपया टेक्स्ट लिखें।"
    try:
        # Voice design parameters ko instruct prompt mein encode karo
        design_parts = []
        if speed != 1.0:
            spd_word = "bahut dheere" if speed < 0.7 else "dheere" if speed < 0.9 else "thoda tez" if speed < 1.2 else "bahut tez"
            design_parts.append(f"speaking speed: {spd_word} ({speed}x)")
        if pitch != 0:
            pitch_word = "bahut neecha" if pitch < -5 else "neecha" if pitch < 0 else "thoda uncha" if pitch < 5 else "bahut uncha"
            design_parts.append(f"pitch: {pitch_word}")
        if energy != 1.0:
            nrg_word = "bahut soft" if energy < 0.6 else "soft" if energy < 0.9 else "energetic" if energy < 1.3 else "bahut loud"
            design_parts.append(f"energy: {nrg_word}")
        if pause > 0:
            design_parts.append(f"pauses: {'natural' if pause < 0.3 else 'extended'}")

        # Instruct mode se voice design apply karo
        instruct_text = style_instruct if style_instruct else ""
        if design_parts:
            instruct_text = (instruct_text + " " if instruct_text else "") + ", ".join(design_parts)

        kw = dict(
            text=text.strip(),
            language=language if language != "Auto" else None,
            generation_config=OmniVoiceGenerationConfig(
                num_step=32,
                guidance_scale=2.5,
                denoise=True,
                preprocess_prompt=True,
                postprocess_output=True,
                speed=speed,
                pitch=pitch,
                energy=energy,
            ),
        )
        if instruct_text:
            kw["instruct"] = instruct_text

        audio = model.generate(**kw)
        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), f"✅ Voice Design applied! ({', '.join(design_parts) if design_parts else 'default'})"
    except Exception as e:
        # Fallback: direct generation bina extra params ke
        try:
            kw_simple = dict(
                text=text.strip(),
                language=language if language != "Auto" else None,
                generation_config=make_gen_config(),
            )
            if style_instruct:
                kw_simple["instruct"] = style_instruct
            audio = model.generate(**kw_simple)
            waveform = (audio[0] * 32767).astype(np.int16)
            return (sampling_rate, waveform), f"✅ Generated (basic mode) — {str(e)}"
        except Exception as e2:
            return None, f"❌ Error: {str(e2)}"


def gen_tts(text, language, num_step, guidance_scale):
    """Tab 3: Simple TTS — seedha text se audio"""
    if not text or not text.strip():
        return None, "⚠️ कृपया टेक्स्ट लिखें।"
    try:
        kw = dict(
            text=text.strip(),
            language=language if language != "Auto" else None,
            generation_config=make_gen_config(num_step=int(num_step), guidance_scale=guidance_scale),
        )
        audio = model.generate(**kw)
        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), "✅ TTS सफल!"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def gen_instruct(text, language, instruct_prompt):
    """Tab 4: Instruct Mode — instruction se voice style control"""
    if not text or not text.strip():
        return None, "⚠️ कृपया टेक्स्ट लिखें।"
    if not instruct_prompt or not instruct_prompt.strip():
        return None, "⚠️ Instruction likhein (jaise: 'speak slowly and clearly in a calm tone')"
    try:
        kw = dict(
            text=text.strip(),
            language=language if language != "Auto" else None,
            generation_config=make_gen_config(guidance_scale=3.0),
            instruct=instruct_prompt.strip(),
        )
        audio = model.generate(**kw)
        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), f"✅ Instruct mode: '{instruct_prompt[:40]}...'"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(primary_hue="orange", font=["Inter", "Arial", "sans-serif"])

css = """
.gradio-container {max-width: 100% !important;}
footer {visibility: hidden !important;}
.shiv-header {text-align: center; margin: 20px auto; padding: 15px; border-bottom: 3px solid #ff6600;}
.tag-btn {background: #fff3e0 !important; border: 1px solid #ffcc80 !important; color: #e65100 !important; font-size: 0.8em !important;}
.gen-btn {font-size: 1.1em !important; padding: 12px !important;}
.status-box {font-size: 0.9em; color: #444;}
.slider-row {margin: 4px 0;}
"""

INSTRUCT_EXAMPLES = [
    "Speak slowly and clearly with a calm, deep voice",
    "Speak with excitement and high energy",
    "Speak softly like a bedtime story",
    "Speak like a news anchor, formal and clear",
    "Speak in a sad, emotional tone",
    "Speak fast and enthusiastically like a sports commentator",
    "धीरे और शांत आवाज़ में बोलें",
    "जोश और उत्साह के साथ बोलें",
]

with gr.Blocks(theme=theme, css=css, title="🔱 Shiv AI Voice Cloning") as demo:

    # Header
    gr.HTML("""
        <div class="shiv-header">
            <h1 style="font-size:2.8em; color:#ff6600; margin-bottom:4px;">🔱 Shiv AI Voice Cloning</h1>
            <p style="font-size:1.15em; color:#444; margin:2px;">Advanced Multilingual Speech Engine &nbsp;|&nbsp; <b>Owner: Shri Ram Nag</b></p>
            <p style="font-size:0.9em; color:#888;">Model: Shriramnag/Shiv-AI-Voice-Cloning &nbsp;|&nbsp; 646 Languages &nbsp;|&nbsp; PAISAWALA 🎬</p>
        </div>
    """)

    with gr.Tabs():

        # ── TAB 1: Voice Clone ──────────────────────────────────────────────
        with gr.TabItem("🎙️ Voice Clone"):
            gr.Markdown("### अपनी आवाज़ upload करें और किसी भी भाषा में बोलें!")
            with gr.Row():
                with gr.Column(scale=1):
                    vc_text = gr.Textbox(
                        label="📝 Text to Synthesize",
                        lines=5,
                        elem_classes="shiv-textbox",
                        placeholder="यहाँ हिंदी / English / Sanskrit text लिखें..."
                    )
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            btn.click(fn=None, inputs=[btn, vc_text], outputs=vc_text, js=INSERT_TAG_JS)
                    vc_lang = gr.Dropdown(label="🌐 भाषा (Language)", choices=LANG_CHOICES, value="Auto")
                    vc_ref_audio = gr.Audio(label="🎤 Reference Audio (आपकी आवाज़)", type="filepath")
                    vc_ref_text = gr.Textbox(label="📄 Reference Text (optional — ref audio ka transcript)", lines=2, placeholder="Optional...")
                    vc_btn = gr.Button("🔱 Clone Voice", variant="primary", size="lg", elem_classes="gen-btn")

                with gr.Column(scale=1):
                    vc_audio = gr.Audio(label="🔊 Shiv AI Output", type="numpy")
                    vc_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
                    gr.Markdown("""
**📌 Tips:**
- 5-30 second ka clean reference audio best results deta hai
- Background noise ke bina record karein
- Jo language mein bolna hai, us mein text likhein
                    """)

            vc_btn.click(
                fn=gen_voice_clone,
                inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text],
                outputs=[vc_audio, vc_status]
            )

        # ── TAB 2: Voice Design ─────────────────────────────────────────────
        with gr.TabItem("🎛️ Voice Design"):
            gr.Markdown("### Speed, Pitch, Energy — apni marzi se voice design karo!")
            with gr.Row():
                with gr.Column(scale=1):
                    vd_text = gr.Textbox(
                        label="📝 Text to Synthesize",
                        lines=5,
                        elem_classes="shiv-textbox",
                        placeholder="यहाँ text लिखें..."
                    )
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            btn2 = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            btn2.click(fn=None, inputs=[btn2, vd_text], outputs=vd_text, js=INSERT_TAG_JS)
                    vd_lang = gr.Dropdown(label="🌐 भाषा", choices=LANG_CHOICES, value="Auto")

                    gr.Markdown("#### 🎚️ Voice Controls")
                    with gr.Group():
                        vd_speed = gr.Slider(
                            label="⚡ Speed (रफ़्तार)",
                            minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                            info="0.5 = बहुत धीमा | 1.0 = Normal | 2.0 = बहुत तेज़"
                        )
                        vd_pitch = gr.Slider(
                            label="🎵 Pitch (आवाज़ की ऊँचाई)",
                            minimum=-12, maximum=12, value=0, step=1,
                            info="-12 = बहुत नीचा | 0 = Normal | +12 = बहुत ऊँचा"
                        )
                        vd_energy = gr.Slider(
                            label="💪 Energy (जोश / Volume)",
                            minimum=0.3, maximum=2.0, value=1.0, step=0.05,
                            info="0.3 = बहुत soft | 1.0 = Normal | 2.0 = बहुत loud"
                        )
                        vd_pause = gr.Slider(
                            label="⏸️ Pause (रुकना)",
                            minimum=0.0, maximum=1.0, value=0.0, step=0.1,
                            info="0 = No extra pause | 1.0 = Long pauses"
                        )

                    vd_style = gr.Textbox(
                        label="✍️ Style Instruction (optional)",
                        placeholder="जैसे: speak like a calm narrator / emotional tone...",
                        lines=2
                    )
                    vd_btn = gr.Button("🎛️ Design & Generate", variant="primary", size="lg", elem_classes="gen-btn")

                    # Preset buttons
                    gr.Markdown("#### ⚡ Quick Presets")
                    with gr.Row():
                        preset_calm = gr.Button("😌 Calm", size="sm")
                        preset_excited = gr.Button("🔥 Excited", size="sm")
                        preset_news = gr.Button("📺 News Anchor", size="sm")
                        preset_story = gr.Button("📖 Story", size="sm")

                with gr.Column(scale=1):
                    vd_audio = gr.Audio(label="🔊 Shiv AI Voice Design Output", type="numpy")
                    vd_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
                    gr.Markdown("""
**🎛️ Voice Design Guide:**
| Control | Effect |
|---------|--------|
| Speed ↓ | Dheere bolta hai |
| Speed ↑ | Tez bolta hai |
| Pitch ↓ | Moti/gahri aawaz |
| Pitch ↑ | Patli/unchi aawaz |
| Energy ↓ | Soft/whisper jaisi |
| Energy ↑ | Bold/loud |
| Pause ↑ | Zyada rukta hai |
                    """)

            # Preset logic
            preset_calm.click(
                fn=lambda: (0.8, -2, 0.7, 0.3, "speak calmly and peacefully"),
                outputs=[vd_speed, vd_pitch, vd_energy, vd_pause, vd_style]
            )
            preset_excited.click(
                fn=lambda: (1.3, 3, 1.5, 0.0, "speak with excitement and high energy"),
                outputs=[vd_speed, vd_pitch, vd_energy, vd_pause, vd_style]
            )
            preset_news.click(
                fn=lambda: (1.0, 0, 1.1, 0.2, "speak like a professional news anchor, clear and formal"),
                outputs=[vd_speed, vd_pitch, vd_energy, vd_pause, vd_style]
            )
            preset_story.click(
                fn=lambda: (0.85, -1, 0.8, 0.4, "speak like a storyteller, warm and engaging"),
                outputs=[vd_speed, vd_pitch, vd_energy, vd_pause, vd_style]
            )
            vd_btn.click(
                fn=gen_voice_design,
                inputs=[vd_text, vd_lang, vd_speed, vd_pitch, vd_energy, vd_pause, vd_style],
                outputs=[vd_audio, vd_status]
            )

        # ── TAB 3: Simple TTS ───────────────────────────────────────────────
        with gr.TabItem("🔤 Simple TTS"):
            gr.Markdown("### Seedha text se audio — bina kisi reference ke!")
            with gr.Row():
                with gr.Column(scale=1):
                    tts_text = gr.Textbox(
                        label="📝 Text",
                        lines=6,
                        elem_classes="shiv-textbox",
                        placeholder="यहाँ text लिखें..."
                    )
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            btn3 = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            btn3.click(fn=None, inputs=[btn3, tts_text], outputs=tts_text, js=INSERT_TAG_JS)
                    tts_lang = gr.Dropdown(label="🌐 भाषा", choices=LANG_CHOICES, value="Auto")
                    with gr.Row():
                        tts_steps = gr.Slider(label="🔢 Steps (quality)", minimum=10, maximum=64, value=32, step=2,
                                              info="Zyada steps = better quality, lekin slow")
                        tts_guidance = gr.Slider(label="🎯 Guidance Scale", minimum=1.0, maximum=5.0, value=2.0, step=0.5)
                    tts_btn = gr.Button("🔤 Generate TTS", variant="primary", size="lg", elem_classes="gen-btn")

                with gr.Column(scale=1):
                    tts_audio = gr.Audio(label="🔊 TTS Output", type="numpy")
                    tts_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")

            tts_btn.click(
                fn=gen_tts,
                inputs=[tts_text, tts_lang, tts_steps, tts_guidance],
                outputs=[tts_audio, tts_status]
            )

        # ── TAB 4: Instruct Mode ────────────────────────────────────────────
        with gr.TabItem("📋 Instruct Mode"):
            gr.Markdown("### Instructions se voice style control karo!")
            with gr.Row():
                with gr.Column(scale=1):
                    inst_text = gr.Textbox(
                        label="📝 Text to Synthesize",
                        lines=5,
                        elem_classes="shiv-textbox",
                        placeholder="यहाँ text लिखें..."
                    )
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            btn4 = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            btn4.click(fn=None, inputs=[btn4, inst_text], outputs=inst_text, js=INSERT_TAG_JS)
                    inst_lang = gr.Dropdown(label="🌐 भाषा", choices=LANG_CHOICES, value="Auto")
                    inst_prompt = gr.Textbox(
                        label="📋 Style Instruction",
                        lines=3,
                        placeholder="Jaise: 'Speak slowly and clearly in a calm, deep male voice'..."
                    )
                    gr.Markdown("**💡 Example Instructions (click to use):**")
                    for ex in INSTRUCT_EXAMPLES:
                        ex_btn = gr.Button(ex, size="sm")
                        ex_btn.click(fn=lambda x=ex: x, outputs=inst_prompt)
                    inst_btn = gr.Button("📋 Generate with Instruct", variant="primary", size="lg", elem_classes="gen-btn")

                with gr.Column(scale=1):
                    inst_audio = gr.Audio(label="🔊 Instruct Mode Output", type="numpy")
                    inst_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
                    gr.Markdown("""
**📋 Instruct Mode Tips:**
- English mein instruction likhna best kaam karta hai
- Clear aur specific instruction do
- Jaise: *"Speak like an old wise man, slowly with pauses"*
- Ya: *"Fast and energetic like a radio jockey"*
                    """)

            inst_btn.click(
                fn=gen_instruct,
                inputs=[inst_text, inst_lang, inst_prompt],
                outputs=[inst_audio, inst_status]
            )

    # Footer
    gr.HTML("""
        <div style='text-align:center; padding:20px; color:#888; border-top:1px solid #eee; margin-top:20px;'>
            © 2026 🔱 Shiv AI Voice Cloning &nbsp;|&nbsp; Designed by <b>Shri Ram Nag</b> &nbsp;|&nbsp; PAISAWALA 🎬
        </div>
    """)

if __name__ == "__main__":
    demo.launch(share=True)
