import os
import sys
import logging
import re
import uuid

import gradio as gr
import numpy as np
import torch

# ── Directory Setup ────────────────────────────────────────────────────────
os.makedirs("./Shiv_Audio", exist_ok=True)

SHIV_AI_REPO   = "Shriramnag/Shiv-AI-Voice-Cloning"
MODEL_LOCAL    = "./Shiv-AI-Voice-Cloning"

sys.path.insert(0, MODEL_LOCAL)
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
try:
    from subtitle import LANGUAGE_CODE as WHISPER_LANGUAGE_CODE
except ImportError:
    WHISPER_LANGUAGE_CODE = None

logging.basicConfig(level=logging.WARNING)
print("🔱 Shiv AI starting…")

# ── Model Load ─────────────────────────────────────────────────────────────
try:
    model = OmniVoice.from_pretrained(SHIV_AI_REPO, device_map="cuda", dtype=torch.float16, load_asr=False)
except Exception:
    model = OmniVoice.from_pretrained(MODEL_LOCAL,  device_map="cuda", dtype=torch.float16, load_asr=False)

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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHUNKING — Fixed hakla issue
# Max chars kept small so model doesn't rush; silence=0ms (model adds own)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAX_CH = 120

def split_chunks(text: str, max_ch: int = MAX_CH) -> list[str]:
    """Split on ellipsis/newline first, then sentence boundaries."""
    lines = re.split(r'(?:…\n?|।\n|\n)', text)
    lines = [l.strip() for l in lines if l.strip()]
    chunks, cur = [], ""
    for line in lines:
        if len(line) > max_ch:
            if cur: chunks.append(cur); cur = ""
            for sent in re.split(r'(?<=[।.!?])\s+', line):
                if len(cur) + len(sent) + 1 <= max_ch:
                    cur = (cur + " " + sent).strip()
                else:
                    if cur: chunks.append(cur)
                    cur = sent.strip()
        else:
            if len(cur) + len(line) + 1 <= max_ch:
                cur = (cur + " " + line).strip()
            else:
                if cur: chunks.append(cur)
                cur = line.strip()
    if cur: chunks.append(cur)
    return [c for c in chunks if c.strip()]


def join_chunks(audios: list, silence_ms: int = 0) -> np.ndarray:
    """
    Join chunks. silence_ms=0 by default because model already adds
    natural pauses at sentence ends — extra silence causes hakla effect.
    """
    if silence_ms > 0:
        sil = np.zeros(int(SR * silence_ms / 1000), dtype=np.float32)
        parts = []
        for i, a in enumerate(audios):
            parts.append(a)
            if i < len(audios)-1: parts.append(sil)
        return np.concatenate(parts)
    return np.concatenate(audios)


def make_cfg(steps=32, gs=2.0, speed=1.0, pitch=0, energy=1.0):
    try:
        return OmniVoiceGenerationConfig(
            num_step=steps, guidance_scale=gs,
            denoise=True, preprocess_prompt=True, postprocess_output=True,
            speed=speed, pitch=pitch, energy=energy,
        )
    except TypeError:
        return OmniVoiceGenerationConfig(
            num_step=steps, guidance_scale=gs,
            denoise=True, preprocess_prompt=True, postprocess_output=True,
        )


def run_chunk(text, lang, cfg, vcp=None, instruct=None):
    kw = dict(text=text, language=lang if lang!="Auto" else None, generation_config=cfg)
    if vcp      is not None: kw["voice_clone_prompt"] = vcp
    if instruct is not None: kw["instruct"]           = instruct
    return model.generate(**kw)[0]   # float32 array


def to_wav(arr: np.ndarray):
    return (SR, (arr * 32767).astype(np.int16))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fn_clone(text, lang, ref_audio, ref_text):
    if not text  or not text.strip():       return None, "⚠️ Text likhein"
    if not ref_audio:                        return None, "⚠️ Reference audio upload karein"
    try:
        vcp    = model.create_voice_clone_prompt(ref_audio=ref_audio,
                                                  ref_text=ref_text.strip() or None)
        cfg    = make_cfg(steps=32, gs=2.0)
        chunks = split_chunks(text)
        print(f"[Clone] {len(chunks)} chunks")
        audio  = join_chunks([run_chunk(c, lang, cfg, vcp=vcp) for c in chunks], silence_ms=0)
        dur    = len(audio)/SR
        return to_wav(audio), f"✅ Done | {len(chunks)} chunks | {dur:.1f}s"
    except Exception as e:
        return None, f"❌ {e}"


def fn_design(text, lang, speed, pitch, energy, pause_ms, style):
    if not text or not text.strip(): return None, "⚠️ Text likhein"
    try:
        # Try with speed/pitch/energy params
        cfg  = make_cfg(steps=32, gs=2.5, speed=speed, pitch=pitch, energy=energy)
        inst = style.strip() or None
        chunks = split_chunks(text)
        print(f"[Design] {len(chunks)} chunks | speed={speed} pitch={pitch} energy={energy}")
        audio  = join_chunks([run_chunk(c, lang, cfg, instruct=inst) for c in chunks],
                             silence_ms=int(pause_ms))
        dur = len(audio)/SR
        return to_wav(audio), f"✅ Done | speed={speed} pitch={pitch} energy={energy} | {dur:.1f}s"
    except Exception as e:
        print(f"[Design] Param error: {e} — falling back")
        try:
            cfg  = make_cfg(steps=32, gs=2.5)
            inst = style.strip() or None
            chunks = split_chunks(text)
            audio  = join_chunks([run_chunk(c, lang, cfg, instruct=inst) for c in chunks], silence_ms=0)
            return to_wav(audio), f"✅ Done (basic) | {len(audio)/SR:.1f}s"
        except Exception as e2:
            return None, f"❌ {e2}"


def fn_tts(text, lang, steps, gs):
    if not text or not text.strip(): return None, "⚠️ Text likhein"
    try:
        # Higher guidance for more realistic/HD output
        cfg    = make_cfg(steps=int(steps), gs=float(gs))
        chunks = split_chunks(text)
        print(f"[TTS] {len(chunks)} chunks")
        audio  = join_chunks([run_chunk(c, lang, cfg) for c in chunks], silence_ms=0)
        return to_wav(audio), f"✅ Done | {len(chunks)} chunks | {len(audio)/SR:.1f}s"
    except Exception as e:
        return None, f"❌ {e}"


def fn_instruct(text, lang, prompt):
    if not text   or not text.strip():   return None, "⚠️ Text likhein"
    if not prompt or not prompt.strip(): return None, "⚠️ Instruction likhein"
    try:
        cfg    = make_cfg(steps=32, gs=3.0)
        chunks = split_chunks(text)
        print(f"[Instruct] {len(chunks)} chunks | '{prompt[:40]}'")
        audio  = join_chunks([run_chunk(c, lang, cfg, instruct=prompt.strip()) for c in chunks],
                             silence_ms=0)
        return to_wav(audio), f"✅ Done | '{prompt[:35]}…' | {len(audio)/SR:.1f}s"
    except Exception as e:
        return None, f"❌ {e}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODERN UI  —  dark glassmorphism with saffron/gold accent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --saffron:  #FF6B00;
    --gold:     #FFB347;
    --dark:     #0D0D0F;
    --surface:  #16161A;
    --card:     rgba(255,255,255,0.04);
    --border:   rgba(255,107,0,0.2);
    --text:     #F0EEE8;
    --muted:    #888880;
    --radius:   14px;
    --success:  #4CAF50;
}

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: var(--dark) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    min-height: 100vh;
}
footer { display: none !important; }
.gradio-container { max-width: 100% !important; padding: 0 !important; }

/* ── Header ── */
.shiv-header {
    background: linear-gradient(135deg, #0D0D0F 0%, #1a0d00 50%, #0D0D0F 100%);
    border-bottom: 1px solid var(--border);
    text-align: center;
    padding: 32px 20px 24px;
    position: relative;
    overflow: hidden;
}
.shiv-header::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 80% at 50% -20%, rgba(255,107,0,0.15), transparent);
    pointer-events: none;
}
.shiv-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(2em, 5vw, 3.2em);
    font-weight: 800;
    background: linear-gradient(90deg, var(--saffron), var(--gold), var(--saffron));
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    margin: 0 0 8px;
    letter-spacing: -0.5px;
}
@keyframes shimmer { to { background-position: 200% center; } }
.shiv-header p { color: var(--muted); margin: 4px 0; font-size: 0.9em; }
.shiv-header b { color: var(--gold); }

/* ── Tabs ── */
.tab-nav {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 24px !important;
}
.tab-nav button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 14px 20px !important;
    border-radius: 0 !important;
    background: transparent !important;
    transition: all .2s !important;
}
.tab-nav button.selected, .tab-nav button:hover {
    color: var(--saffron) !important;
    border-bottom-color: var(--saffron) !important;
}

/* ── Tab content ── */
.tabitem { padding: 24px !important; }

/* ── Cards / panels ── */
.shiv-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    backdrop-filter: blur(12px);
}

/* ── Labels ── */
label, .label-wrap span {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78em !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--gold) !important;
}

/* ── Textboxes ── */
.shiv-tb textarea, textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95em !important;
    padding: 12px !important;
    resize: vertical !important;
    transition: border-color .2s !important;
}
.shiv-tb textarea:focus, textarea:focus {
    border-color: var(--saffron) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(255,107,0,0.12) !important;
}
textarea::placeholder { color: var(--muted) !important; }

/* ── Dropdowns ── */
.wrap .wrap-inner, select, .dropdown {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── Sliders ── */
input[type=range] { accent-color: var(--saffron) !important; }
.wrap span { color: var(--text) !important; }

/* ── Buttons ── */
.btn-primary {
    background: linear-gradient(135deg, var(--saffron), #e65000) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1em !important;
    padding: 13px 24px !important;
    cursor: pointer !important;
    transition: transform .15s, box-shadow .15s !important;
    box-shadow: 0 4px 20px rgba(255,107,0,0.35) !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
}
.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(255,107,0,0.5) !important;
}
.btn-primary:active { transform: translateY(0) !important; }

/* ── Tag buttons ── */
.tag-btn {
    background: rgba(255,107,0,0.1) !important;
    border: 1px solid rgba(255,107,0,0.3) !important;
    color: var(--gold) !important;
    border-radius: 6px !important;
    font-size: 0.72em !important;
    padding: 4px 8px !important;
    cursor: pointer !important;
    transition: all .15s !important;
    font-family: 'DM Sans', sans-serif !important;
}
.tag-btn:hover {
    background: rgba(255,107,0,0.25) !important;
    border-color: var(--saffron) !important;
}

/* ── Preset buttons ── */
.preset-btn {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-size: 0.82em !important;
    padding: 7px 12px !important;
    cursor: pointer !important;
    transition: all .15s !important;
    font-family: 'DM Sans', sans-serif !important;
}
.preset-btn:hover {
    background: rgba(255,107,0,0.15) !important;
    border-color: var(--saffron) !important;
    color: var(--gold) !important;
}

/* ── Example instruction buttons ── */
.ex-btn {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: var(--muted) !important;
    border-radius: 6px !important;
    font-size: 0.78em !important;
    padding: 5px 10px !important;
    text-align: left !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: all .15s !important;
    margin-bottom: 4px !important;
}
.ex-btn:hover {
    border-color: var(--saffron) !important;
    color: var(--gold) !important;
    background: rgba(255,107,0,0.08) !important;
}

/* ── Status box ── */
.status-box textarea {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    color: var(--gold) !important;
    font-size: 0.82em !important;
    border-radius: 8px !important;
}

/* ── Audio player ── */
.audio-wrap {
    background: rgba(255,107,0,0.06) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Info cards ── */
.info-card {
    background: rgba(255,107,0,0.06);
    border: 1px solid rgba(255,107,0,0.15);
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 12px;
    font-size: 0.85em;
    color: var(--muted);
    line-height: 1.7;
}
.info-card strong { color: var(--gold); }

/* ── Section title ── */
.sec-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1em;
    font-weight: 700;
    color: var(--gold);
    margin: 0 0 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 16px 0;
}

/* ── Footer ── */
.shiv-footer {
    text-align: center;
    padding: 20px;
    color: var(--muted);
    font-size: 0.8em;
    border-top: 1px solid var(--border);
    background: var(--surface);
}
.shiv-footer span { color: var(--saffron); }

/* ── Responsive ── */
@media (max-width: 768px) {
    .tabitem { padding: 16px !important; }
    .shiv-header { padding: 24px 16px 18px; }
}
"""

HEADER_HTML = """
<div class="shiv-header">
  <h1>🔱 Shiv AI Voice Cloning</h1>
  <p>Advanced Multilingual Neural Speech Engine &nbsp;·&nbsp; 646 Languages</p>
  <p><b>Shri Ram Nag</b> &nbsp;·&nbsp; PAISAWALA 🎬 &nbsp;·&nbsp; <code style="color:#666;font-size:0.85em">Shriramnag/Shiv-AI-Voice-Cloning</code></p>
</div>
"""

FOOTER_HTML = """
<div class="shiv-footer">
  © 2026 <span>🔱 Shiv AI Voice Cloning</span> &nbsp;·&nbsp;
  Designed by <span>Shri Ram Nag</span> &nbsp;·&nbsp; PAISAWALA 🎬
</div>
"""

# ── Build UI ───────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="🔱 Shiv AI Voice Cloning") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs(elem_classes="tab-nav"):

        # ══════════════════════════════════════════════════
        # TAB 1 — Voice Clone
        # ══════════════════════════════════════════════════
        with gr.TabItem("🎙️ Voice Clone"):
            with gr.Row(equal_height=False):
                # Left panel
                with gr.Column(scale=6):
                    gr.HTML('<div class="sec-title">📝 Script</div>')
                    vc_text = gr.Textbox(
                        lines=9, elem_classes="shiv-tb",
                        placeholder="पूरी script paste करें — लंबी script भी चलेगी…",
                        label="", show_label=False
                    )
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            b = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            b.click(fn=None, inputs=[b, vc_text], outputs=vc_text, js=INSERT_TAG_JS)

                    with gr.Row():
                        vc_lang = gr.Dropdown(LANG_CHOICES, value="Auto",
                                              label="🌐 Language", scale=1)
                        vc_ref_text = gr.Textbox(
                            label="📄 Reference Transcript (optional)", lines=1,
                            placeholder="Reference audio mein jo bola — optional", scale=2
                        )
                    vc_ref = gr.Audio(label="🎤 Reference Audio (5–30 sec, saaf awaaz)", type="filepath")
                    vc_btn = gr.Button("🔱  Clone Voice & Generate", elem_classes="btn-primary")

                # Right panel
                with gr.Column(scale=5):
                    gr.HTML('<div class="sec-title">🔊 Output</div>')
                    vc_out    = gr.Audio(type="numpy", label="", elem_classes="audio-wrap")
                    vc_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
                    gr.HTML("""
                    <div class="info-card">
                      <strong>✅ Hakla issue fixed</strong><br>
                      Script auto-chunks → har chunk generate → silently joined<br>
                      No extra silence = no stutter between words<br><br>
                      <strong>💡 Best results:</strong><br>
                      · 5–30 sec saaf reference audio<br>
                      · Background noise na ho<br>
                      · <code>…</code> (ellipsis) natural pause point banta hai
                    </div>
                    """)

            vc_btn.click(fn_clone, [vc_text, vc_lang, vc_ref, vc_ref_text], [vc_out, vc_status])

        # ══════════════════════════════════════════════════
        # TAB 2 — Voice Design
        # ══════════════════════════════════════════════════
        with gr.TabItem("🎛️ Voice Design"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=6):
                    gr.HTML('<div class="sec-title">📝 Script</div>')
                    vd_text = gr.Textbox(
                        lines=7, elem_classes="shiv-tb",
                        placeholder="यहाँ text लिखें…",
                        label="", show_label=False
                    )
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            b2 = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            b2.click(fn=None, inputs=[b2, vd_text], outputs=vd_text, js=INSERT_TAG_JS)

                    vd_lang = gr.Dropdown(LANG_CHOICES, value="Auto", label="🌐 Language")

                    gr.HTML('<hr class="divider"><div class="sec-title">🎚️ Voice Controls</div>')
                    vd_speed  = gr.Slider(0.5, 2.0, value=1.0, step=0.05,
                                          label="⚡ Speed — 0.5 (slow) → 1.0 (normal) → 2.0 (fast)")
                    vd_pitch  = gr.Slider(-12, 12,  value=0,   step=1,
                                          label="🎵 Pitch — -12 (deep) → 0 (normal) → +12 (high)")
                    vd_energy = gr.Slider(0.3, 2.0, value=1.0, step=0.05,
                                          label="💪 Energy — 0.3 (whisper) → 1.0 (normal) → 2.0 (loud)")
                    vd_pause  = gr.Slider(0,   400, value=0,   step=50,
                                          label="⏸️ Extra Pause between chunks (ms) — 0 recommended")

                    vd_style = gr.Textbox(label="✍️ Style Instruction (optional)",
                                          placeholder="e.g. speak like a calm narrator…", lines=2)

                    gr.HTML('<hr class="divider"><div class="sec-title">⚡ Quick Presets</div>')
                    with gr.Row():
                        pc = gr.Button("😌  Calm",        elem_classes="preset-btn")
                        pe = gr.Button("🔥  Excited",     elem_classes="preset-btn")
                        pn = gr.Button("📺  News Anchor", elem_classes="preset-btn")
                        ps = gr.Button("📖  Story",       elem_classes="preset-btn")

                    vd_btn = gr.Button("🎛️  Design & Generate", elem_classes="btn-primary")

                with gr.Column(scale=5):
                    gr.HTML('<div class="sec-title">🔊 Output</div>')
                    vd_out    = gr.Audio(type="numpy", label="", elem_classes="audio-wrap")
                    vd_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
                    gr.HTML("""
                    <div class="info-card">
                      <strong>🎛️ Controls Guide</strong><br>
                      Speed ↓ = dheere &nbsp;|&nbsp; Speed ↑ = tez<br>
                      Pitch ↓ = moti/gahri &nbsp;|&nbsp; Pitch ↑ = patli/unchi<br>
                      Energy ↓ = soft/whisper &nbsp;|&nbsp; Energy ↑ = loud/bold<br>
                      Pause = chunks ke beech silence (0 = best)<br><br>
                      <strong>⚡ Preset tips:</strong> Calm → meditation/yoga<br>
                      Excited → promos | News → formal reads | Story → kids
                    </div>
                    """)

            pc.click(fn=lambda: (0.8,  -2, 0.7,   0, "speak calmly and peacefully"),                        outputs=[vd_speed,vd_pitch,vd_energy,vd_pause,vd_style])
            pe.click(fn=lambda: (1.3,   3, 1.5,   0, "speak with excitement and high energy"),              outputs=[vd_speed,vd_pitch,vd_energy,vd_pause,vd_style])
            pn.click(fn=lambda: (1.0,   0, 1.1,   0, "speak like a professional news anchor, formal"),      outputs=[vd_speed,vd_pitch,vd_energy,vd_pause,vd_style])
            ps.click(fn=lambda: (0.85, -1, 0.8, 100, "speak like a warm storyteller with gentle pauses"),   outputs=[vd_speed,vd_pitch,vd_energy,vd_pause,vd_style])
            vd_btn.click(fn_design, [vd_text,vd_lang,vd_speed,vd_pitch,vd_energy,vd_pause,vd_style], [vd_out,vd_status])

        # ══════════════════════════════════════════════════
        # TAB 3 — Simple TTS (HD realistic)
        # ══════════════════════════════════════════════════
        with gr.TabItem("🔤 Simple TTS"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=6):
                    gr.HTML('<div class="sec-title">📝 Script</div>')
                    tts_text = gr.Textbox(
                        lines=9, elem_classes="shiv-tb",
                        placeholder="यहाँ text paste करें — बिना reference audio के बोलेगा…",
                        label="", show_label=False
                    )
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            b3 = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            b3.click(fn=None, inputs=[b3, tts_text], outputs=tts_text, js=INSERT_TAG_JS)

                    tts_lang = gr.Dropdown(LANG_CHOICES, value="Auto", label="🌐 Language")

                    with gr.Row():
                        tts_steps = gr.Slider(10, 64, value=40, step=2,
                                              label="🔢 Steps — ↑ better quality, ↓ faster", scale=1)
                        tts_gs    = gr.Slider(1.0, 5.0, value=3.0, step=0.5,
                                              label="🎯 Guidance Scale — ↑ more realistic", scale=1)

                    tts_btn = gr.Button("🔤  Generate HD Audio", elem_classes="btn-primary")

                with gr.Column(scale=5):
                    gr.HTML('<div class="sec-title">🔊 Output</div>')
                    tts_out    = gr.Audio(type="numpy", label="", elem_classes="audio-wrap")
                    tts_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
                    gr.HTML("""
                    <div class="info-card">
                      <strong>🎯 HD Realistic Tips:</strong><br>
                      · Steps = 40, Guidance = 3.0 → best realism<br>
                      · Steps = 20, Guidance = 2.0 → fast draft<br>
                      · Steps = 60, Guidance = 4.0 → highest quality<br><br>
                      <strong>📝 Script tips:</strong><br>
                      · <code>…</code> add karo pauses ke liye<br>
                      · <code>[laughter]</code> tags se emotions add karo
                    </div>
                    """)

            tts_btn.click(fn_tts, [tts_text, tts_lang, tts_steps, tts_gs], [tts_out, tts_status])

        # ══════════════════════════════════════════════════
        # TAB 4 — Instruct Mode
        # ══════════════════════════════════════════════════
        with gr.TabItem("📋 Instruct Mode"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=6):
                    gr.HTML('<div class="sec-title">📝 Script</div>')
                    inst_text = gr.Textbox(
                        lines=7, elem_classes="shiv-tb",
                        placeholder="यहाँ text लिखें…",
                        label="", show_label=False
                    )
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            b4 = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            b4.click(fn=None, inputs=[b4, inst_text], outputs=inst_text, js=INSERT_TAG_JS)

                    inst_lang   = gr.Dropdown(LANG_CHOICES, value="Auto", label="🌐 Language")
                    inst_prompt = gr.Textbox(
                        label="📋 Style Instruction",
                        placeholder="Speak slowly and clearly in a calm, deep male voice…",
                        lines=3
                    )

                    gr.HTML('<hr class="divider"><div class="sec-title">💡 Example Instructions</div>')
                    for ex in INSTRUCT_EX:
                        eb = gr.Button(ex, elem_classes="ex-btn", size="sm")
                        eb.click(fn=lambda x=ex: x, outputs=inst_prompt)

                    inst_btn = gr.Button("📋  Generate with Instruction", elem_classes="btn-primary")

                with gr.Column(scale=5):
                    gr.HTML('<div class="sec-title">🔊 Output</div>')
                    inst_out    = gr.Audio(type="numpy", label="", elem_classes="audio-wrap")
                    inst_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
                    gr.HTML("""
                    <div class="info-card">
                      <strong>📋 Instruct Mode Tips:</strong><br>
                      · English instructions best work karte hain<br>
                      · Specific aur clear rakho<br><br>
                      <strong>🎯 Examples:</strong><br>
                      <em>"Speak like an old wise man, slowly with dramatic pauses"</em><br>
                      <em>"Fast and energetic like a Bollywood trailer narrator"</em><br>
                      <em>"Soft and emotional like reading a love letter"</em>
                    </div>
                    """)

            inst_btn.click(fn_instruct, [inst_text, inst_lang, inst_prompt], [inst_out, inst_status])

    gr.HTML(FOOTER_HTML)

if __name__ == "__main__":
    demo.launch(share=True)
