import os
import sys
import logging
import re

import gradio as gr
import numpy as np
import torch

# ── Setup ──────────────────────────────────────────────────────────────────
os.makedirs("./Shiv_Audio", exist_ok=True)
SHIV_AI_REPO = "Shriramnag/Shiv-AI-Voice-Cloning"
MODEL_LOCAL  = "./Shiv-AI-Voice-Cloning"
sys.path.insert(0, MODEL_LOCAL)

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
try:
    from subtitle import LANGUAGE_CODE as WHISPER_LANGUAGE_CODE
except ImportError:
    WHISPER_LANGUAGE_CODE = None

logging.basicConfig(level=logging.WARNING)
print("🔱 Shiv AI starting…")

try:
    model = OmniVoice.from_pretrained(SHIV_AI_REPO, device_map="cuda", dtype=torch.float16, load_asr=False)
except Exception:
    model = OmniVoice.from_pretrained(MODEL_LOCAL, device_map="cuda", dtype=torch.float16, load_asr=False)

SR = model.sampling_rate
print(f"✅ Model loaded | SR={SR}")

# ── Constants ──────────────────────────────────────────────────────────────
LANG_CHOICES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)
EVENT_TAGS   = ["[laughter]","[sigh]","[confirmation-en]","[question-en]","[surprise-wa]","[dissatisfaction-hnn]"]
INSTRUCT_EX  = [
    "Speak slowly and clearly with a calm, deep voice",
    "Speak with excitement and high energy",
    "Speak softly like a bedtime story narrator",
    "Speak like a professional news anchor, formal and clear",
    "Speak in a sad, emotional tone with pauses",
    "Fast and enthusiastic like a radio jockey",
    "धीरे, शांत और गहरी आवाज़ में बोलें",
    "जोश और उत्साह के साथ तेज़ आवाज़ में बोलें",
]

INSERT_TAG_JS = """
(tag_val, current_text) => {
    const ta = document.querySelector('.shiv-tb textarea');
    if (!ta) return current_text + ' ' + tag_val;
    const s = ta.selectionStart, e = ta.selectionEnd;
    return current_text.slice(0,s) + ' ' + tag_val + ' ' + current_text.slice(e);
}
"""

# ── Chunking ───────────────────────────────────────────────────────────────
MAX_CH = 120

def split_chunks(text, max_ch=MAX_CH):
    lines = re.split(r'(?:…\n?|।\n|\n)', text)
    lines = [l.strip() for l in lines if l.strip()]
    chunks, cur = [], ""
    for line in lines:
        if len(line) > max_ch:
            if cur: chunks.append(cur); cur = ""
            for sent in re.split(r'(?<=[।.!?])\s+', line):
                if len(cur)+len(sent)+1 <= max_ch: cur = (cur+" "+sent).strip()
                else:
                    if cur: chunks.append(cur)
                    cur = sent.strip()
        else:
            if len(cur)+len(line)+1 <= max_ch: cur = (cur+" "+line).strip()
            else:
                if cur: chunks.append(cur)
                cur = line.strip()
    if cur: chunks.append(cur)
    return [c for c in chunks if c.strip()]

def join_chunks(audios, silence_ms=0):
    if silence_ms > 0:
        sil = np.zeros(int(SR*silence_ms/1000), dtype=np.float32)
        parts = []
        for i, a in enumerate(audios):
            parts.append(a)
            if i < len(audios)-1: parts.append(sil)
        return np.concatenate(parts)
    return np.concatenate(audios)

def make_cfg(steps=32, gs=2.0, speed=1.0, pitch=0, energy=1.0):
    try:
        return OmniVoiceGenerationConfig(num_step=steps, guidance_scale=gs,
            denoise=True, preprocess_prompt=True, postprocess_output=True,
            speed=speed, pitch=pitch, energy=energy)
    except TypeError:
        return OmniVoiceGenerationConfig(num_step=steps, guidance_scale=gs,
            denoise=True, preprocess_prompt=True, postprocess_output=True)

def run_chunk(text, lang, cfg, vcp=None, instruct=None):
    kw = dict(text=text, language=lang if lang!="Auto" else None, generation_config=cfg)
    if vcp:     kw["voice_clone_prompt"] = vcp
    if instruct: kw["instruct"] = instruct
    return model.generate(**kw)[0]

def to_wav(a): return (SR, (a*32767).astype(np.int16))

# ── Tab Functions ──────────────────────────────────────────────────────────
def fn_clone(text, lang, ref, ref_text):
    if not text or not text.strip(): return None, "⚠️ Text likhein"
    if not ref: return None, "⚠️ Reference audio upload karein"
    try:
        vcp = model.create_voice_clone_prompt(ref_audio=ref, ref_text=ref_text.strip() or None)
        cfg = make_cfg(); chunks = split_chunks(text)
        print(f"[Clone] {len(chunks)} chunks")
        audio = join_chunks([run_chunk(c, lang, cfg, vcp=vcp) for c in chunks], silence_ms=0)
        return to_wav(audio), f"✅ Done! {len(chunks)} chunks | {len(audio)/SR:.1f}s audio"
    except Exception as e: return None, f"❌ Error: {e}"

def fn_design(text, lang, speed, pitch, energy, pause_ms, style):
    if not text or not text.strip(): return None, "⚠️ Text likhein"
    try:
        cfg = make_cfg(gs=2.5, speed=speed, pitch=pitch, energy=energy)
        inst = style.strip() or None; chunks = split_chunks(text)
        print(f"[Design] {len(chunks)} chunks")
        audio = join_chunks([run_chunk(c, lang, cfg, instruct=inst) for c in chunks], silence_ms=int(pause_ms))
        return to_wav(audio), f"✅ Done! speed={speed} | pitch={pitch} | energy={energy} | {len(audio)/SR:.1f}s"
    except Exception as e:
        try:
            cfg = make_cfg(gs=2.5); chunks = split_chunks(text)
            audio = join_chunks([run_chunk(c, lang, cfg, instruct=style.strip() or None) for c in chunks])
            return to_wav(audio), f"✅ Done (basic)! {len(audio)/SR:.1f}s"
        except Exception as e2: return None, f"❌ Error: {e2}"

def fn_tts(text, lang, steps, gs):
    if not text or not text.strip(): return None, "⚠️ Text likhein"
    try:
        cfg = make_cfg(int(steps), float(gs)); chunks = split_chunks(text)
        print(f"[TTS] {len(chunks)} chunks")
        audio = join_chunks([run_chunk(c, lang, cfg) for c in chunks])
        return to_wav(audio), f"✅ Done! {len(chunks)} chunks | {len(audio)/SR:.1f}s audio"
    except Exception as e: return None, f"❌ Error: {e}"

def fn_instruct(text, lang, prompt):
    if not text or not text.strip(): return None, "⚠️ Text likhein"
    if not prompt or not prompt.strip(): return None, "⚠️ Instruction likhein"
    try:
        cfg = make_cfg(gs=3.0); chunks = split_chunks(text)
        print(f"[Instruct] {len(chunks)} chunks")
        audio = join_chunks([run_chunk(c, lang, cfg, instruct=prompt.strip()) for c in chunks])
        return to_wav(audio), f"✅ Done! '{prompt[:35]}…' | {len(audio)/SR:.1f}s"
    except Exception as e: return None, f"❌ Error: {e}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS — Fixed: text visibility, audio player, animations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

/* ── RESET & BASE ── */
*, *::before, *::after { box-sizing: border-box !important; margin: 0; }

:root {
    --saffron: #FF6B00;
    --gold:    #FFB347;
    --glow:    rgba(255,107,0,0.4);
    --dark:    #0A0A0C;
    --surf:    #111115;
    --card:    #18181E;
    --border:  rgba(255,107,0,0.25);
    --text:    #EDEAE2;
    --muted:   #7A7870;
    --green:   #4CAF50;
    --r:       12px;
}

body, .gradio-container, .gradio-container * {
    background-color: transparent;
}
.gradio-container {
    background: var(--dark) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    max-width: 100% !important;
    min-height: 100vh;
    padding: 0 !important;
}
footer, .footer { display: none !important; }
.main { padding: 0 !important; }

/* ── HEADER ── */
.shiv-header {
    background: linear-gradient(160deg, #0f0800 0%, #1a0d00 40%, #0A0A0C 100%);
    border-bottom: 1px solid var(--border);
    text-align: center;
    padding: 36px 24px 28px;
    position: relative;
    overflow: hidden;
}
.shiv-header::before {
    content: '';
    position: absolute; top: -60px; left: 50%; transform: translateX(-50%);
    width: 500px; height: 200px;
    background: radial-gradient(ellipse, rgba(255,107,0,0.2) 0%, transparent 70%);
    pointer-events: none;
    animation: pulse-glow 4s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%,100% { opacity: 0.6; transform: translateX(-50%) scaleX(1); }
    50%      { opacity: 1;   transform: translateX(-50%) scaleX(1.2); }
}
.shiv-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(2em, 5vw, 3em);
    font-weight: 800;
    background: linear-gradient(90deg, #FF6B00 0%, #FFD580 50%, #FF6B00 100%);
    background-size: 200% auto;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    animation: shimmer 3s linear infinite;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin-bottom: 10px;
}
@keyframes shimmer { to { background-position: 200% center; } }
.shiv-header .sub { color: var(--muted); font-size: 0.88em; line-height: 1.8; }
.shiv-header .sub b { color: var(--gold); font-weight: 500; }
.badge {
    display: inline-block;
    background: rgba(255,107,0,0.15);
    border: 1px solid rgba(255,107,0,0.35);
    color: var(--gold);
    font-size: 0.72em;
    font-weight: 500;
    letter-spacing: 0.06em;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 6px 4px 0;
    vertical-align: middle;
}

/* ── TABS ── */
.tabs { background: var(--surf) !important; }
.tab-nav { 
    background: var(--surf) !important; 
    border-bottom: 1px solid var(--border) !important; 
    padding: 0 20px !important; 
    gap: 0 !important;
}
.tab-nav button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88em !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 14px 18px !important;
    border-radius: 0 !important;
    cursor: pointer !important;
    transition: color 0.2s, border-color 0.2s !important;
    white-space: nowrap !important;
}
.tab-nav button:hover { color: var(--gold) !important; }
.tab-nav button.selected {
    color: var(--saffron) !important;
    border-bottom-color: var(--saffron) !important;
    background: transparent !important;
}
.tabitem { 
    background: var(--dark) !important; 
    padding: 24px 20px !important; 
}

/* ── SECTION TITLE ── */
.sec-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.85em !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--saffron) !important;
    margin: 0 0 14px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}
.sec-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── LABELS ── */
label span, .label-wrap span, .svelte-1ipelgc {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.75em !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--gold) !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   TEXTBOX — CRITICAL FIX (white text issue)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.shiv-tb textarea,
.shiv-tb textarea:focus,
textarea,
textarea:focus,
.block textarea,
div[data-testid="textbox"] textarea {
    background: #1C1C22 !important;
    color: #EDEAE2 !important;
    border: 1.5px solid rgba(255,107,0,0.2) !important;
    border-radius: var(--r) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95em !important;
    line-height: 1.7 !important;
    padding: 14px 16px !important;
    caret-color: var(--saffron) !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
    resize: vertical !important;
    outline: none !important;
}
textarea:focus,
div[data-testid="textbox"] textarea:focus {
    border-color: var(--saffron) !important;
    box-shadow: 0 0 0 3px rgba(255,107,0,0.15), 0 0 20px rgba(255,107,0,0.08) !important;
}
textarea::placeholder { color: #555550 !important; font-style: italic !important; }
/* Scrollbar inside textarea */
textarea::-webkit-scrollbar { width: 4px; }
textarea::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

/* ── DROPDOWN ── */
.wrap .wrap-inner, 
div[data-testid="dropdown"] .wrap,
.dropdown-arrow,
select,
.svelte-select {
    background: #1C1C22 !important;
    border: 1.5px solid rgba(255,107,0,0.2) !important;
    border-radius: var(--r) !important;
    color: var(--text) !important;
}
div[data-testid="dropdown"] ul {
    background: #1C1C22 !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
div[data-testid="dropdown"] li:hover { background: rgba(255,107,0,0.12) !important; }
div[data-testid="dropdown"] li { color: var(--text) !important; }

/* ── SLIDERS ── */
input[type=range] {
    accent-color: var(--saffron) !important;
    height: 4px !important;
}
.range-wrap { background: transparent !important; }
.range-value { 
    color: var(--gold) !important; 
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85em !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   AUDIO PLAYER — CRITICAL FIX
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
/* Container */
div[data-testid="audio"],
.audio-wrap,
.gr-audio,
div[class*="audio"] {
    background: #18181E !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    padding: 12px !important;
    min-height: 80px !important;
}
/* Play/Pause button */
div[data-testid="audio"] button,
div[data-testid="audio"] button svg,
.audio-player button {
    color: var(--text) !important;
    fill: var(--text) !important;
    background: var(--saffron) !important;
    border-radius: 50% !important;
    padding: 8px !important;
    width: 38px !important; height: 38px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    border: none !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
div[data-testid="audio"] button:hover {
    transform: scale(1.08) !important;
    box-shadow: 0 0 16px var(--glow) !important;
}
/* Waveform / progress bar */
div[data-testid="audio"] .waveform,
div[data-testid="audio"] .playback,
div[data-testid="audio"] progress,
div[data-testid="audio"] [class*="waveform"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}
/* Time display */
div[data-testid="audio"] [class*="time"],
div[data-testid="audio"] span {
    color: var(--muted) !important;
    font-size: 0.8em !important;
    font-family: 'DM Sans', sans-serif !important;
}
/* Download button (keep visible) */
div[data-testid="audio"] [aria-label="Download"] {
    background: transparent !important;
    color: var(--gold) !important;
    width: auto !important; height: auto !important;
    padding: 4px !important;
    border-radius: 6px !important;
}

/* ── STATUS BOX ── */
.status-box textarea,
.status-box div[data-testid="textbox"] textarea {
    background: #111118 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    color: var(--gold) !important;
    font-size: 0.82em !important;
    border-radius: 8px !important;
    font-style: normal !important;
    min-height: 52px !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   BUTTONS
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
/* Generate button */
.btn-gen {
    position: relative !important;
    background: linear-gradient(135deg, #FF6B00 0%, #CC4400 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--r) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1em !important;
    letter-spacing: 0.03em !important;
    padding: 15px 28px !important;
    width: 100% !important;
    cursor: pointer !important;
    overflow: hidden !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 24px rgba(255,107,0,0.4) !important;
}
.btn-gen::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, transparent 60%);
    pointer-events: none;
}
.btn-gen::after {
    content: '';
    position: absolute; inset: -1px;
    border-radius: calc(var(--r) + 1px);
    background: linear-gradient(135deg, rgba(255,180,70,0.6), transparent);
    opacity: 0;
    transition: opacity 0.2s;
    pointer-events: none;
}
.btn-gen:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 32px rgba(255,107,0,0.6) !important;
}
.btn-gen:hover::after { opacity: 1; }
.btn-gen:active { transform: translateY(-1px) !important; }

/* Ripple animation on click */
@keyframes ripple {
    0%   { transform: scale(0); opacity: 0.5; }
    100% { transform: scale(4); opacity: 0; }
}

/* Tag buttons */
.tag-btn {
    background: rgba(255,107,0,0.08) !important;
    border: 1px solid rgba(255,107,0,0.25) !important;
    color: var(--gold) !important;
    border-radius: 20px !important;
    font-size: 0.72em !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 5px 10px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    white-space: nowrap !important;
}
.tag-btn:hover {
    background: rgba(255,107,0,0.22) !important;
    border-color: var(--saffron) !important;
    transform: scale(1.05) !important;
    box-shadow: 0 2px 10px rgba(255,107,0,0.2) !important;
}
.tag-btn:active { transform: scale(0.97) !important; }

/* Preset buttons */
.preset-btn {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82em !important;
    padding: 9px 14px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    flex: 1 !important;
}
.preset-btn:hover {
    background: rgba(255,107,0,0.14) !important;
    border-color: rgba(255,107,0,0.5) !important;
    color: var(--gold) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 14px rgba(255,107,0,0.15) !important;
}
.preset-btn:active { transform: translateY(0px) !important; }

/* Example instruction buttons */
.ex-btn {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    color: var(--muted) !important;
    border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8em !important;
    padding: 7px 12px !important;
    text-align: left !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.18s !important;
    margin-bottom: 5px !important;
    display: block !important;
}
.ex-btn:hover {
    background: rgba(255,107,0,0.10) !important;
    border-color: rgba(255,107,0,0.4) !important;
    color: var(--gold) !important;
    padding-left: 16px !important;
}

/* ── INFO CARDS ── */
.info-card {
    background: rgba(255,107,0,0.05);
    border: 1px solid rgba(255,107,0,0.15);
    border-radius: var(--r);
    padding: 16px 18px;
    margin-top: 14px;
    font-size: 0.84em;
    color: var(--muted);
    line-height: 1.8;
    animation: fade-in 0.4s ease;
}
.info-card strong { color: var(--gold); }
.info-card code {
    background: rgba(255,107,0,0.12);
    color: var(--saffron);
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 0.9em;
}
@keyframes fade-in { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }

/* ── DIVIDER ── */
.divider {
    border: none !important;
    border-top: 1px solid rgba(255,107,0,0.12) !important;
    margin: 18px 0 !important;
}

/* ── FOOTER ── */
.shiv-footer {
    text-align: center;
    padding: 20px;
    color: var(--muted);
    font-size: 0.8em;
    border-top: 1px solid var(--border);
    background: var(--surf);
    letter-spacing: 0.04em;
}
.shiv-footer span { color: var(--saffron); }
.shiv-footer a { color: var(--gold); text-decoration: none; }

/* ── LOADING STATE on generate ── */
.generating .btn-gen {
    animation: pulse-btn 1s ease-in-out infinite !important;
    pointer-events: none !important;
}
@keyframes pulse-btn {
    0%,100% { box-shadow: 0 4px 24px rgba(255,107,0,0.4); }
    50%      { box-shadow: 0 4px 40px rgba(255,107,0,0.8); }
}

/* ── RESPONSIVE ── */
@media (max-width: 640px) {
    .tabitem { padding: 16px 12px !important; }
    .shiv-header { padding: 24px 16px 20px; }
    .tab-nav button { padding: 12px 12px !important; font-size: 0.8em !important; }
}
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUILD UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with gr.Blocks(css=CSS, title="🔱 Shiv AI Voice Cloning") as demo:

    # ── HEADER ──
    gr.HTML("""
    <div class="shiv-header">
      <h1>🔱 Shiv AI Voice Cloning</h1>
      <div class="sub">
        Advanced Multilingual Neural Speech Engine · 646 Languages<br>
        <b>Shri Ram Nag</b> · PAISAWALA 🎬
        <br>
        <span class="badge">v4.0</span>
        <span class="badge">T4 GPU</span>
        <span class="badge">OmniVoice</span>
      </div>
    </div>
    """)

    with gr.Tabs(elem_classes="tab-nav"):

        # ════════════════════════════════════════
        # TAB 1 — Voice Clone
        # ════════════════════════════════════════
        with gr.TabItem("🎙️ Voice Clone"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=11, min_width=300):
                    gr.HTML('<div class="sec-title">📝 Script</div>')
                    vc_text = gr.Textbox(
                        lines=9, elem_classes="shiv-tb",
                        placeholder="यहाँ अपनी पूरी script paste करें — लंबी script भी चलेगी…",
                        label="", show_label=False
                    )
                    with gr.Row(elem_classes="tag-row"):
                        for tag in EVENT_TAGS:
                            b = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            b.click(fn=None, inputs=[b, vc_text], outputs=vc_text, js=INSERT_TAG_JS)

                    with gr.Row():
                        vc_lang     = gr.Dropdown(LANG_CHOICES, value="Auto", label="🌐 Language", scale=1)
                        vc_ref_text = gr.Textbox(label="📄 Reference Transcript (optional)", lines=1,
                                                  placeholder="Reference audio ka transcript — optional", scale=2)
                    vc_ref = gr.Audio(label="🎤 Reference Audio  (5–30 sec, saaf awaaz)",
                                       type="filepath", elem_classes="audio-wrap")
                    vc_btn = gr.Button("🔱  Clone Voice & Generate Full Audio",
                                        elem_classes="btn-gen", variant="primary")

                with gr.Column(scale=9, min_width=260):
                    gr.HTML('<div class="sec-title">🔊 Output</div>')
                    vc_out    = gr.Audio(type="numpy", label="Generated Audio",
                                         elem_classes="audio-wrap", show_download_button=True)
                    vc_status = gr.Textbox(label="Status", interactive=False,
                                            elem_classes="status-box", lines=2)
                    gr.HTML("""
                    <div class="info-card">
                      <strong>✅ Hakla issue fixed!</strong><br>
                      Script → auto chunks → generate each → join seamlessly.<br>
                      Zero extra silence = no stutter ✅<br><br>
                      <strong>💡 Best results:</strong><br>
                      · 5–30 sec saaf reference audio<br>
                      · Background noise na ho<br>
                      · <code>…</code> use karo natural pause ke liye
                    </div>
                    """)

            vc_btn.click(fn_clone, [vc_text, vc_lang, vc_ref, vc_ref_text], [vc_out, vc_status])

        # ════════════════════════════════════════
        # TAB 2 — Voice Design
        # ════════════════════════════════════════
        with gr.TabItem("🎛️ Voice Design"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=11, min_width=300):
                    gr.HTML('<div class="sec-title">📝 Script</div>')
                    vd_text = gr.Textbox(
                        lines=6, elem_classes="shiv-tb",
                        placeholder="यहाँ text paste करें…",
                        label="", show_label=False
                    )
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            b2 = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            b2.click(fn=None, inputs=[b2, vd_text], outputs=vd_text, js=INSERT_TAG_JS)
                    vd_lang = gr.Dropdown(LANG_CHOICES, value="Auto", label="🌐 Language")

                    gr.HTML('<hr class="divider"><div class="sec-title">🎚️ Voice Controls</div>')
                    vd_speed  = gr.Slider(0.5, 2.0, value=1.0, step=0.05,
                                           label="⚡ Speed        0.5 = Slow  |  1.0 = Normal  |  2.0 = Fast")
                    vd_pitch  = gr.Slider(-12, 12,  value=0,   step=1,
                                           label="🎵 Pitch         -12 = Deep  |  0 = Normal  |  +12 = High")
                    vd_energy = gr.Slider(0.3, 2.0, value=1.0, step=0.05,
                                           label="💪 Energy       0.3 = Whisper  |  1.0 = Normal  |  2.0 = Loud")
                    vd_pause  = gr.Slider(0, 400, value=0, step=50,
                                           label="⏸️ Extra Pause (ms between chunks)  —  0 = recommended")
                    vd_style  = gr.Textbox(label="✍️ Style Instruction (optional)",
                                            placeholder="e.g. speak like a calm narrator with warm tone…",
                                            lines=2, elem_classes="shiv-tb")

                    gr.HTML('<hr class="divider"><div class="sec-title">⚡ Quick Presets</div>')
                    with gr.Row():
                        pc = gr.Button("😌  Calm",        elem_classes="preset-btn")
                        pe = gr.Button("🔥  Excited",     elem_classes="preset-btn")
                        pn = gr.Button("📺  News Anchor", elem_classes="preset-btn")
                        ps = gr.Button("📖  Story",       elem_classes="preset-btn")

                    vd_btn = gr.Button("🎛️  Design & Generate Audio",
                                        elem_classes="btn-gen", variant="primary")

                with gr.Column(scale=9, min_width=260):
                    gr.HTML('<div class="sec-title">🔊 Output</div>')
                    vd_out    = gr.Audio(type="numpy", label="Voice Design Output",
                                         elem_classes="audio-wrap", show_download_button=True)
                    vd_status = gr.Textbox(label="Status", interactive=False,
                                            elem_classes="status-box", lines=2)
                    gr.HTML("""
                    <div class="info-card">
                      <strong>🎚️ Controls Guide:</strong><br>
                      Speed ↓ = dheere &nbsp;|&nbsp; Speed ↑ = tez<br>
                      Pitch ↓ = moti/gahri &nbsp;|&nbsp; Pitch ↑ = patli/unchi<br>
                      Energy ↓ = soft/whisper &nbsp;|&nbsp; Energy ↑ = bold/loud<br>
                      Pause = silence between chunks (0 = best)<br><br>
                      <strong>⚡ Preset guide:</strong><br>
                      😌 Calm → meditation, yoga, sleep<br>
                      🔥 Excited → promos, ads, reels<br>
                      📺 News → formal reads, documentary<br>
                      📖 Story → kids, audiobook, folklore
                    </div>
                    """)

            pc.click(fn=lambda: (0.8,  -2, 0.7,   0, "speak calmly and peacefully, gentle tone"),       outputs=[vd_speed,vd_pitch,vd_energy,vd_pause,vd_style])
            pe.click(fn=lambda: (1.3,   3, 1.5,   0, "speak with excitement and high energy"),          outputs=[vd_speed,vd_pitch,vd_energy,vd_pause,vd_style])
            pn.click(fn=lambda: (1.0,   0, 1.1,   0, "speak like a professional news anchor, formal"),  outputs=[vd_speed,vd_pitch,vd_energy,vd_pause,vd_style])
            ps.click(fn=lambda: (0.85, -1, 0.8, 100, "speak like a warm storyteller, engaging pace"),   outputs=[vd_speed,vd_pitch,vd_energy,vd_pause,vd_style])
            vd_btn.click(fn_design,
                         [vd_text,vd_lang,vd_speed,vd_pitch,vd_energy,vd_pause,vd_style],
                         [vd_out, vd_status])

        # ════════════════════════════════════════
        # TAB 3 — Simple TTS
        # ════════════════════════════════════════
        with gr.TabItem("🔤 Simple TTS"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=11, min_width=300):
                    gr.HTML('<div class="sec-title">📝 Script</div>')
                    tts_text = gr.Textbox(
                        lines=9, elem_classes="shiv-tb",
                        placeholder="यहाँ text paste करें — reference audio ki zaroorat nahi…",
                        label="", show_label=False
                    )
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            b3 = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            b3.click(fn=None, inputs=[b3, tts_text], outputs=tts_text, js=INSERT_TAG_JS)
                    tts_lang = gr.Dropdown(LANG_CHOICES, value="Auto", label="🌐 Language")

                    gr.HTML('<hr class="divider"><div class="sec-title">🎯 Quality Settings</div>')
                    with gr.Row():
                        tts_steps = gr.Slider(10, 64, value=40, step=2, scale=1,
                                               label="🔢 Steps  —  ↑ quality, ↓ speed")
                        tts_gs    = gr.Slider(1.0, 5.0, value=3.0, step=0.5, scale=1,
                                               label="🎯 Guidance Scale  —  ↑ realistic")
                    tts_btn = gr.Button("🔤  Generate HD Audio",
                                         elem_classes="btn-gen", variant="primary")

                with gr.Column(scale=9, min_width=260):
                    gr.HTML('<div class="sec-title">🔊 Output</div>')
                    tts_out    = gr.Audio(type="numpy", label="TTS Output",
                                          elem_classes="audio-wrap", show_download_button=True)
                    tts_status = gr.Textbox(label="Status", interactive=False,
                                             elem_classes="status-box", lines=2)
                    gr.HTML("""
                    <div class="info-card">
                      <strong>🎯 HD Quality Tips:</strong><br>
                      Steps=40, Guidance=3.0 → best balance<br>
                      Steps=60, Guidance=4.0 → max quality (slow)<br>
                      Steps=20, Guidance=2.0 → fast draft<br><br>
                      <strong>📝 Script tips:</strong><br>
                      · <code>…</code> = natural pause<br>
                      · <code>[laughter]</code> = hasne ki awaaz<br>
                      · <code>[sigh]</code> = aah bharni<br>
                      · Naya line = naya chunk
                    </div>
                    """)

            tts_btn.click(fn_tts, [tts_text, tts_lang, tts_steps, tts_gs], [tts_out, tts_status])

        # ════════════════════════════════════════
        # TAB 4 — Instruct Mode
        # ════════════════════════════════════════
        with gr.TabItem("📋 Instruct Mode"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=11, min_width=300):
                    gr.HTML('<div class="sec-title">📝 Script</div>')
                    inst_text = gr.Textbox(
                        lines=7, elem_classes="shiv-tb",
                        placeholder="यहाँ text paste करें…",
                        label="", show_label=False
                    )
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            b4 = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            b4.click(fn=None, inputs=[b4, inst_text], outputs=inst_text, js=INSERT_TAG_JS)
                    inst_lang   = gr.Dropdown(LANG_CHOICES, value="Auto", label="🌐 Language")
                    inst_prompt = gr.Textbox(
                        label="📋 Style Instruction",
                        placeholder="Speak slowly and clearly in a calm, deep voice with pauses…",
                        lines=3, elem_classes="shiv-tb"
                    )
                    gr.HTML('<hr class="divider"><div class="sec-title">💡 Example Instructions</div>')
                    for ex in INSTRUCT_EX:
                        eb = gr.Button(ex, elem_classes="ex-btn", size="sm")
                        eb.click(fn=lambda x=ex: x, outputs=inst_prompt)

                    inst_btn = gr.Button("📋  Generate with Instruction",
                                          elem_classes="btn-gen", variant="primary")

                with gr.Column(scale=9, min_width=260):
                    gr.HTML('<div class="sec-title">🔊 Output</div>')
                    inst_out    = gr.Audio(type="numpy", label="Instruct Output",
                                           elem_classes="audio-wrap", show_download_button=True)
                    inst_status = gr.Textbox(label="Status", interactive=False,
                                              elem_classes="status-box", lines=2)
                    gr.HTML("""
                    <div class="info-card">
                      <strong>📋 Instruct Mode Tips:</strong><br>
                      · English mein instructions best kaam karte hain<br>
                      · Specific aur descriptive rakho<br><br>
                      <strong>🎬 Examples:</strong><br>
                      <em>"Speak like a Bollywood trailer narrator"</em><br>
                      <em>"Old wise grandfather telling a story"</em><br>
                      <em>"Soft and emotional like reading a love letter"</em><br>
                      <em>"Energetic sports commentary style"</em>
                    </div>
                    """)

            inst_btn.click(fn_instruct, [inst_text, inst_lang, inst_prompt], [inst_out, inst_status])

    # ── FOOTER ──
    gr.HTML("""
    <div class="shiv-footer">
      © 2026 &nbsp;<span>🔱 Shiv AI Voice Cloning</span>&nbsp; ·
      &nbsp;Designed by <span>Shri Ram Nag</span>&nbsp; · &nbsp;PAISAWALA 🎬
    </div>
    """)

if __name__ == "__main__":
    demo.launch(share=True)
