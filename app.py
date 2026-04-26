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
# Constants
# ---------------------------------------------------------------------------
EVENT_TAGS = [
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]",
    "[surprise-wa]", "[dissatisfaction-hnn]"
]

LANG_CHOICES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

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
# ✅ LONG TEXT CHUNKING — Main Fix
# ---------------------------------------------------------------------------
# Max characters per chunk (tune karo agar zaroorat ho)
MAX_CHUNK_CHARS = 150

def split_text_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list:
    """
    Long script ko natural chunks mein todta hai:
    1. Pehle '…' aur newline pe todta hai (natural pauses)
    2. Phir bade chunks ko sentence boundary pe todta hai
    3. Har chunk max_chars se zyada nahi hoga
    """
    # Step 1: Split on ellipsis + newline (script style)
    # '…\n' ya '\n' pe split karo
    raw_lines = re.split(r'(?:…\n|…|\n)', text)
    raw_lines = [l.strip() for l in raw_lines if l.strip()]

    chunks = []
    current = ""

    for line in raw_lines:
        # Agar line khud badi hai, use sentence boundary pe todo
        if len(line) > max_chars:
            # Pehle current flush karo
            if current.strip():
                chunks.append(current.strip())
                current = ""
            # Bade line ko sentences pe todo
            sentences = re.split(r'(?<=[।.!?])\s+', line)
            temp = ""
            for sent in sentences:
                if len(temp) + len(sent) + 1 <= max_chars:
                    temp = (temp + " " + sent).strip()
                else:
                    if temp:
                        chunks.append(temp.strip())
                    temp = sent.strip()
            if temp:
                chunks.append(temp.strip())
        else:
            # Normal line — current mein add karo
            if len(current) + len(line) + 1 <= max_chars:
                current = (current + " " + line).strip()
            else:
                if current:
                    chunks.append(current.strip())
                current = line.strip()

    if current.strip():
        chunks.append(current.strip())

    # Empty chunks hata do
    chunks = [c for c in chunks if c.strip()]
    return chunks


def generate_audio_for_chunk(chunk_text, language, gen_config, voice_clone_prompt=None, instruct=None):
    """Single chunk ka audio generate karo"""
    kw = dict(
        text=chunk_text,
        language=language if language != "Auto" else None,
        generation_config=gen_config,
    )
    if voice_clone_prompt is not None:
        kw["voice_clone_prompt"] = voice_clone_prompt
    if instruct is not None:
        kw["instruct"] = instruct

    audio = model.generate(**kw)
    return audio[0]  # numpy array, float


def join_audio_chunks(chunks_audio: list, silence_ms: int = 300) -> np.ndarray:
    """
    Saare audio chunks ko ek saath jodo.
    Beech mein thodi si silence (natural pauses ke liye).
    """
    silence_samples = int(sampling_rate * silence_ms / 1000)
    silence = np.zeros(silence_samples, dtype=np.float32)

    result = []
    for i, chunk in enumerate(chunks_audio):
        result.append(chunk)
        if i < len(chunks_audio) - 1:
            result.append(silence)

    return np.concatenate(result)


def make_gen_config(num_step=32, guidance_scale=2.0, speed=1.0, pitch=0, energy=1.0):
    """Generation config banao — speed/pitch/energy optional"""
    try:
        return OmniVoiceGenerationConfig(
            num_step=num_step,
            guidance_scale=guidance_scale,
            denoise=True,
            preprocess_prompt=True,
            postprocess_output=True,
            speed=speed,
            pitch=pitch,
            energy=energy,
        )
    except TypeError:
        # Agar model speed/pitch/energy support nahi karta
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
    """Tab 1: Voice Clone with long-text support"""
    if not text or not text.strip():
        return None, "⚠️ कृपया टेक्स्ट लिखें।"
    if not ref_audio:
        return None, "⚠️ Reference audio upload karein।"

    try:
        # Voice clone prompt ek baar banao (efficient)
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text.strip() if ref_text else None
        )
        gen_config = make_gen_config()

        # Long text chunking
        chunks = split_text_into_chunks(text)
        print(f"📝 Total chunks: {len(chunks)}")
        for i, c in enumerate(chunks):
            print(f"  Chunk {i+1}: '{c[:60]}...' ({len(c)} chars)")

        # Har chunk generate karo
        audio_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"🎙️ Generating chunk {i+1}/{len(chunks)}...")
            chunk_audio = generate_audio_for_chunk(
                chunk, language, gen_config,
                voice_clone_prompt=voice_clone_prompt
            )
            audio_chunks.append(chunk_audio)

        # Saare chunks join karo
        final_audio = join_audio_chunks(audio_chunks, silence_ms=250)
        waveform = (final_audio * 32767).astype(np.int16)

        total_sec = len(waveform) / sampling_rate
        return (sampling_rate, waveform), f"✅ Voice Clone सफल! {len(chunks)} chunks | {total_sec:.1f} sec audio"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def gen_voice_design(text, language, speed, pitch, energy, pause, style_instruct):
    """Tab 2: Voice Design with chunking"""
    if not text or not text.strip():
        return None, "⚠️ कृपया टेक्स्ट लिखें।"

    try:
        gen_config = make_gen_config(guidance_scale=2.5, speed=speed, pitch=pitch, energy=energy)
        instruct = style_instruct.strip() if style_instruct else None
        chunks = split_text_into_chunks(text)
        print(f"📝 Voice Design chunks: {len(chunks)}")

        audio_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"🎛️ Chunk {i+1}/{len(chunks)}: {chunk[:50]}")
            chunk_audio = generate_audio_for_chunk(chunk, language, gen_config, instruct=instruct)
            audio_chunks.append(chunk_audio)

        silence_ms = int(pause * 600)  # pause slider → silence duration
        final_audio = join_audio_chunks(audio_chunks, silence_ms=max(200, silence_ms))
        waveform = (final_audio * 32767).astype(np.int16)
        total_sec = len(waveform) / sampling_rate

        params = f"speed={speed} pitch={pitch} energy={energy}"
        return (sampling_rate, waveform), f"✅ Voice Design! {len(chunks)} chunks | {total_sec:.1f}s | {params}"

    except Exception as e:
        # Fallback: basic mode
        try:
            gen_config = make_gen_config()
            chunks = split_text_into_chunks(text)
            audio_chunks = []
            for chunk in chunks:
                chunk_audio = generate_audio_for_chunk(chunk, language, gen_config, instruct=style_instruct or None)
                audio_chunks.append(chunk_audio)
            final_audio = join_audio_chunks(audio_chunks)
            waveform = (final_audio * 32767).astype(np.int16)
            return (sampling_rate, waveform), f"✅ Basic mode ({len(chunks)} chunks) — {str(e)}"
        except Exception as e2:
            return None, f"❌ Error: {str(e2)}"


def gen_tts(text, language, num_step, guidance_scale):
    """Tab 3: Simple TTS with chunking"""
    if not text or not text.strip():
        return None, "⚠️ कृपया टेक्स्ट लिखें।"

    try:
        gen_config = make_gen_config(num_step=int(num_step), guidance_scale=guidance_scale)
        chunks = split_text_into_chunks(text)
        print(f"📝 TTS chunks: {len(chunks)}")

        audio_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"🔤 Chunk {i+1}/{len(chunks)}: {chunk[:50]}")
            chunk_audio = generate_audio_for_chunk(chunk, language, gen_config)
            audio_chunks.append(chunk_audio)

        final_audio = join_audio_chunks(audio_chunks)
        waveform = (final_audio * 32767).astype(np.int16)
        total_sec = len(waveform) / sampling_rate
        return (sampling_rate, waveform), f"✅ TTS सफल! {len(chunks)} chunks | {total_sec:.1f} sec"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def gen_instruct(text, language, instruct_prompt):
    """Tab 4: Instruct Mode with chunking"""
    if not text or not text.strip():
        return None, "⚠️ कृपया टेक्स्ट लिखें।"
    if not instruct_prompt or not instruct_prompt.strip():
        return None, "⚠️ Instruction likhein।"

    try:
        gen_config = make_gen_config(guidance_scale=3.0)
        instruct = instruct_prompt.strip()
        chunks = split_text_into_chunks(text)
        print(f"📝 Instruct chunks: {len(chunks)}")

        audio_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"📋 Chunk {i+1}/{len(chunks)}: {chunk[:50]}")
            chunk_audio = generate_audio_for_chunk(chunk, language, gen_config, instruct=instruct)
            audio_chunks.append(chunk_audio)

        final_audio = join_audio_chunks(audio_chunks)
        waveform = (final_audio * 32767).astype(np.int16)
        total_sec = len(waveform) / sampling_rate
        return (sampling_rate, waveform), f"✅ Instruct: '{instruct[:40]}' | {len(chunks)} chunks | {total_sec:.1f}s"

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
"""

with gr.Blocks(theme=theme, css=css, title="🔱 Shiv AI Voice Cloning") as demo:

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
            gr.Markdown("### अपनी आवाज़ upload करें — Long script भी पूरी पढ़ेगा! ✅")
            with gr.Row():
                with gr.Column(scale=1):
                    vc_text = gr.Textbox(
                        label="📝 Text / Script (लंबी script भी चलेगी!)",
                        lines=10,
                        elem_classes="shiv-textbox",
                        placeholder="यहाँ पूरी script paste करें... छोटी हो या बड़ी, सब चलेगा।"
                    )
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            btn.click(fn=None, inputs=[btn, vc_text], outputs=vc_text, js=INSERT_TAG_JS)
                    vc_lang = gr.Dropdown(label="🌐 भाषा (Language)", choices=LANG_CHOICES, value="Auto")
                    vc_ref_audio = gr.Audio(label="🎤 Reference Audio (आपकी आवाज़)", type="filepath")
                    vc_ref_text = gr.Textbox(
                        label="📄 Reference Transcript (optional)",
                        lines=2,
                        placeholder="Reference audio mein jo bola gaya hai woh likhein (optional)..."
                    )
                    vc_btn = gr.Button("🔱 Clone & Generate Full Audio", variant="primary", size="lg", elem_classes="gen-btn")

                with gr.Column(scale=1):
                    vc_audio = gr.Audio(label="🔊 Shiv AI Output (Full Audio)", type="numpy")
                    vc_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
                    gr.Markdown("""
**📌 Long Script Fix — Kaise kaam karta hai:**
- Script automatically **chhote chunks** mein toot ti hai
- Har chunk alag generate hota hai (koi gap nahi)
- Sab chunks seamlessly **join** ho jaate hain
- 2 minute ka gap **ab nahi aayega** ✅

**💡 Tips:**
- `…` (ellipsis) natural pause point ban jaata hai
- 5–30 sec ka saaf reference audio best hai
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
                        label="📝 Text / Script",
                        lines=8,
                        elem_classes="shiv-textbox",
                        placeholder="यहाँ text लिखें..."
                    )
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            btn2 = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            btn2.click(fn=None, inputs=[btn2, vd_text], outputs=vd_text, js=INSERT_TAG_JS)
                    vd_lang = gr.Dropdown(label="🌐 भाषा", choices=LANG_CHOICES, value="Auto")

                    gr.Markdown("#### 🎚️ Voice Controls")
                    vd_speed = gr.Slider(
                        label="⚡ Speed (रफ़्तार)",
                        minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                        info="0.5=बहुत धीमा | 1.0=Normal | 2.0=बहुत तेज़"
                    )
                    vd_pitch = gr.Slider(
                        label="🎵 Pitch (आवाज़ की ऊँचाई)",
                        minimum=-12, maximum=12, value=0, step=1,
                        info="-12=बहुत नीचा | 0=Normal | +12=बहुत ऊँचा"
                    )
                    vd_energy = gr.Slider(
                        label="💪 Energy (जोश / Volume)",
                        minimum=0.3, maximum=2.0, value=1.0, step=0.05,
                        info="0.3=बहुत soft | 1.0=Normal | 2.0=बहुत loud"
                    )
                    vd_pause = gr.Slider(
                        label="⏸️ Pause Between Chunks",
                        minimum=0.0, maximum=1.0, value=0.3, step=0.1,
                        info="0=No pause | 1.0=Long pauses between chunks"
                    )
                    vd_style = gr.Textbox(
                        label="✍️ Style Instruction (optional)",
                        placeholder="जैसे: speak like a calm narrator...",
                        lines=2
                    )
                    gr.Markdown("#### ⚡ Quick Presets")
                    with gr.Row():
                        preset_calm    = gr.Button("😌 Calm",        size="sm")
                        preset_excited = gr.Button("🔥 Excited",     size="sm")
                        preset_news    = gr.Button("📺 News Anchor", size="sm")
                        preset_story   = gr.Button("📖 Story",       size="sm")

                    vd_btn = gr.Button("🎛️ Design & Generate", variant="primary", size="lg", elem_classes="gen-btn")

                with gr.Column(scale=1):
                    vd_audio = gr.Audio(label="🔊 Voice Design Output", type="numpy")
                    vd_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
                    gr.Markdown("""
| Control | Effect |
|---------|--------|
| Speed ↓ | Dheere bolta hai |
| Speed ↑ | Tez bolta hai |
| Pitch ↓ | Moti/Gahri aawaz |
| Pitch ↑ | Patli/Unchi aawaz |
| Energy ↓ | Soft/Whisper |
| Energy ↑ | Bold/Loud |
| Pause ↑ | Chunks ke beech zyada gap |
                    """)

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
                        label="📝 Text / Script",
                        lines=8,
                        elem_classes="shiv-textbox",
                        placeholder="यहाँ text लिखें..."
                    )
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            btn3 = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            btn3.click(fn=None, inputs=[btn3, tts_text], outputs=tts_text, js=INSERT_TAG_JS)
                    tts_lang     = gr.Dropdown(label="🌐 भाषा", choices=LANG_CHOICES, value="Auto")
                    tts_steps    = gr.Slider(label="🔢 Steps (Quality)", minimum=10, maximum=64, value=32, step=2,
                                             info="Zyada steps = better, lekin slow")
                    tts_guidance = gr.Slider(label="🎯 Guidance Scale", minimum=1.0, maximum=5.0, value=2.0, step=0.5)
                    tts_btn = gr.Button("🔤 Generate TTS", variant="primary", size="lg", elem_classes="gen-btn")

                with gr.Column(scale=1):
                    tts_audio  = gr.Audio(label="🔊 TTS Output", type="numpy")
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
                        label="📝 Text / Script",
                        lines=8,
                        elem_classes="shiv-textbox",
                        placeholder="यहाँ text लिखें..."
                    )
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            btn4 = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            btn4.click(fn=None, inputs=[btn4, inst_text], outputs=inst_text, js=INSERT_TAG_JS)
                    inst_lang   = gr.Dropdown(label="🌐 भाषा", choices=LANG_CHOICES, value="Auto")
                    inst_prompt = gr.Textbox(
                        label="📋 Style Instruction",
                        lines=3,
                        placeholder="Speak slowly and clearly in a calm, deep male voice..."
                    )
                    gr.Markdown("**💡 Examples (click karein):**")
                    for ex in INSTRUCT_EXAMPLES:
                        ex_btn = gr.Button(ex, size="sm")
                        ex_btn.click(fn=lambda x=ex: x, outputs=inst_prompt)
                    inst_btn = gr.Button("📋 Generate with Instruct", variant="primary", size="lg", elem_classes="gen-btn")

                with gr.Column(scale=1):
                    inst_audio  = gr.Audio(label="🔊 Instruct Output", type="numpy")
                    inst_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
                    gr.Markdown("""
**📋 Instruct Mode Tips:**
- English mein instruction best kaam karta hai
- Clear aur specific rakho
- Examples: *"Speak like an old wise man, slowly with pauses"*
- Ya: *"Fast and energetic like a radio jockey"*
                    """)

            inst_btn.click(
                fn=gen_instruct,
                inputs=[inst_text, inst_lang, inst_prompt],
                outputs=[inst_audio, inst_status]
            )

    gr.HTML("""
        <div style='text-align:center; padding:20px; color:#888; border-top:1px solid #eee; margin-top:20px;'>
            © 2026 🔱 Shiv AI Voice Cloning &nbsp;|&nbsp; Designed by <b>Shri Ram Nag</b> &nbsp;|&nbsp; PAISAWALA 🎬
        </div>
    """)

if __name__ == "__main__":
    demo.launch(share=True)
