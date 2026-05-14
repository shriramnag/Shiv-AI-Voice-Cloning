"""
🔱 Shiv AI Voice Cloning
Developer: Shri Ram Nag | PAISAWALA
Model: Shriramnag/Shiv-AI-Voice-Cloning (HuggingFace)
"""
import os, sys, re, logging
import numpy as np
import torch
import gradio as gr

# ── Model path (same folder as app.py) ────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

os.makedirs(os.path.join(BASE_DIR, "Shiv_Audio"), exist_ok=True)
logging.basicConfig(level=logging.WARNING)

# ── Import omnivoice (pip install omnivoice se aata hai) ──────────────────
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

print("🔱 Shiv AI Voice Cloning loading...")
model = OmniVoice.from_pretrained(
    BASE_DIR,               # local repo folder se load
    device_map="cuda",
    dtype=torch.float16,
    load_asr=False,
)
SR = model.sampling_rate
print(f"✅ Shiv AI loaded! SR={SR}")

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
(tag_val, cur) => {
    const ta = document.querySelector('.shiv-tb textarea');
    if (!ta) return cur + ' ' + tag_val;
    const s = ta.selectionStart, e = ta.selectionEnd;
    return cur.slice(0,s) + ' ' + tag_val + ' ' + cur.slice(e);
}
"""

# ── Chunking (hakla fix) ───────────────────────────────────────────────────
def split_chunks(text, max_ch=120):
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

def join_chunks(audios, sil_ms=0):
    if sil_ms > 0:
        sil = np.zeros(int(SR*sil_ms/1000), dtype=np.float32)
        out = []
        for i,a in enumerate(audios):
            out.append(a)
            if i < len(audios)-1: out.append(sil)
        return np.concatenate(out)
    return np.concatenate(audios)

def make_cfg(steps=32, gs=2.0, speed=1.0, pitch=0, energy=1.0):
    try:
        return OmniVoiceGenerationConfig(num_step=steps, guidance_scale=gs,
            denoise=True, preprocess_prompt=True, postprocess_output=True,
            speed=speed, pitch=pitch, energy=energy)
    except TypeError:
        return OmniVoiceGenerationConfig(num_step=steps, guidance_scale=gs,
            denoise=True, preprocess_prompt=True, postprocess_output=True)

def run_chunk(text, lang, cfg, vcp=None, inst=None):
    kw = dict(text=text, language=lang if lang!="Auto" else None, generation_config=cfg)
    if vcp:  kw["voice_clone_prompt"] = vcp
    if inst: kw["instruct"] = inst
    return model.generate(**kw)[0]

def to_wav(a): return (SR, (a*32767).astype(np.int16))

# ── Tab functions ──────────────────────────────────────────────────────────
def fn_clone(text, lang, ref, ref_text):
    if not text or not text.strip(): return None, "⚠️ Text likhein"
    if not ref:                       return None, "⚠️ Reference audio upload karein"
    try:
        vcp = model.create_voice_clone_prompt(ref_audio=ref, ref_text=ref_text.strip() or None)
        chunks = split_chunks(text)
        audio  = join_chunks([run_chunk(c,lang,make_cfg(),vcp=vcp) for c in chunks])
        return to_wav(audio), f"✅ Done! {len(chunks)} chunks | {len(audio)/SR:.1f}s"
    except Exception as e: return None, f"❌ {e}"

def fn_design(text, lang, speed, pitch, energy, pause_ms, style):
    if not text or not text.strip(): return None, "⚠️ Text likhein"
    try:
        cfg    = make_cfg(gs=2.5, speed=speed, pitch=pitch, energy=energy)
        inst   = style.strip() or None
        chunks = split_chunks(text)
        audio  = join_chunks([run_chunk(c,lang,cfg,inst=inst) for c in chunks], sil_ms=int(pause_ms))
        return to_wav(audio), f"✅ Done! speed={speed} pitch={pitch} energy={energy} | {len(audio)/SR:.1f}s"
    except Exception as e:
        try:
            chunks = split_chunks(text)
            audio  = join_chunks([run_chunk(c,lang,make_cfg(gs=2.5),inst=style.strip() or None) for c in chunks])
            return to_wav(audio), f"✅ Done (basic)! {len(audio)/SR:.1f}s"
        except Exception as e2: return None, f"❌ {e2}"

def fn_tts(text, lang, steps, gs):
    if not text or not text.strip(): return None, "⚠️ Text likhein"
    try:
        chunks = split_chunks(text)
        audio  = join_chunks([run_chunk(c,lang,make_cfg(int(steps),float(gs))) for c in chunks])
        return to_wav(audio), f"✅ Done! {len(chunks)} chunks | {len(audio)/SR:.1f}s"
    except Exception as e: return None, f"❌ {e}"

def fn_instruct(text, lang, prompt):
    if not text   or not text.strip():   return None, "⚠️ Text likhein"
    if not prompt or not prompt.strip(): return None, "⚠️ Instruction likhein"
    try:
        chunks = split_chunks(text)
        audio  = join_chunks([run_chunk(c,lang,make_cfg(gs=3.0),inst=prompt.strip()) for c in chunks])
        return to_wav(audio), f"✅ Done! {len(chunks)} chunks | {len(audio)/SR:.1f}s"
    except Exception as e: return None, f"❌ {e}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
:root{--sf:#FF6B00;--gd:#FFB347;--dk:#0A0A0C;--sf2:#111115;--bd:rgba(255,107,0,0.25);--tx:#EDEAE2;--mt:#7A7870;--r:12px;}
body,.gradio-container{background:var(--dk)!important;color:var(--tx)!important;font-family:'DM Sans',sans-serif!important;max-width:100%!important;padding:0!important;}
footer{display:none!important;}
/* Header */
.shiv-hdr{background:linear-gradient(160deg,#0f0800,#1a0d00 40%,#0A0A0C);border-bottom:1px solid var(--bd);text-align:center;padding:32px 20px 24px;position:relative;overflow:hidden;}
.shiv-hdr::before{content:'';position:absolute;top:-60px;left:50%;transform:translateX(-50%);width:500px;height:200px;background:radial-gradient(ellipse,rgba(255,107,0,0.18),transparent 70%);animation:pglow 4s ease-in-out infinite;}
@keyframes pglow{0%,100%{opacity:.6}50%{opacity:1}}
.shiv-hdr h1{font-family:'Syne',sans-serif!important;font-size:clamp(1.9em,5vw,3em);font-weight:800;background:linear-gradient(90deg,#FF6B00,#FFD580,#FF6B00);background-size:200%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:shimmer 3s linear infinite;margin-bottom:8px;}
@keyframes shimmer{to{background-position:200% center}}
.shiv-hdr p{color:var(--mt);font-size:.88em;margin:3px 0;}
.shiv-hdr b{color:var(--gd);}
.badge{display:inline-block;background:rgba(255,107,0,.12);border:1px solid rgba(255,107,0,.3);color:var(--gd);font-size:.7em;padding:2px 9px;border-radius:20px;margin:5px 3px 0;}
/* Tabs */
.tab-nav{background:var(--sf2)!important;border-bottom:1px solid var(--bd)!important;padding:0 16px!important;}
.tab-nav button{font-family:'DM Sans',sans-serif!important;font-size:.85em!important;color:var(--mt)!important;background:transparent!important;border:none!important;border-bottom:2px solid transparent!important;padding:13px 16px!important;border-radius:0!important;transition:all .2s!important;}
.tab-nav button.selected,.tab-nav button:hover{color:var(--sf)!important;border-bottom-color:var(--sf)!important;}
.tabitem{background:var(--dk)!important;padding:20px!important;}
/* Labels */
.sec-title{font-family:'Syne',sans-serif!important;font-size:.8em!important;font-weight:700!important;letter-spacing:.12em!important;text-transform:uppercase!important;color:var(--sf)!important;margin:0 0 12px!important;}
label span,.label-wrap span{font-size:.75em!important;font-weight:500!important;letter-spacing:.08em!important;text-transform:uppercase!important;color:var(--gd)!important;}
/* Textboxes — FIXED white text issue */
.shiv-tb textarea,textarea,div[data-testid="textbox"] textarea{background:#1C1C22!important;color:#EDEAE2!important;border:1.5px solid rgba(255,107,0,.2)!important;border-radius:var(--r)!important;font-family:'DM Sans',sans-serif!important;font-size:.95em!important;padding:12px 14px!important;caret-color:var(--sf)!important;transition:border-color .25s,box-shadow .25s!important;}
textarea:focus,div[data-testid="textbox"] textarea:focus{border-color:var(--sf)!important;box-shadow:0 0 0 3px rgba(255,107,0,.15)!important;outline:none!important;}
textarea::placeholder{color:#55554f!important;font-style:italic!important;}
/* Dropdown */
.wrap .wrap-inner,select,div[data-testid="dropdown"] .wrap{background:#1C1C22!important;border:1.5px solid rgba(255,107,0,.2)!important;border-radius:var(--r)!important;color:var(--tx)!important;}
div[data-testid="dropdown"] ul{background:#1C1C22!important;border:1px solid var(--bd)!important;border-radius:10px!important;}
div[data-testid="dropdown"] li{color:var(--tx)!important;}
div[data-testid="dropdown"] li:hover{background:rgba(255,107,0,.12)!important;}
/* Slider */
input[type=range]{accent-color:var(--sf)!important;}
/* Audio player — FIXED play button */
div[data-testid="audio"]{background:#18181E!important;border:1px solid var(--bd)!important;border-radius:var(--r)!important;padding:10px!important;min-height:72px!important;}
div[data-testid="audio"] button{background:var(--sf)!important;border-radius:50%!important;width:36px!important;height:36px!important;border:none!important;color:#fff!important;fill:#fff!important;display:flex!important;align-items:center!important;justify-content:center!important;transition:transform .15s,box-shadow .15s!important;cursor:pointer!important;}
div[data-testid="audio"] button:hover{transform:scale(1.1)!important;box-shadow:0 0 16px rgba(255,107,0,.5)!important;}
div[data-testid="audio"] button svg{fill:#fff!important;color:#fff!important;}
div[data-testid="audio"] [class*="time"],div[data-testid="audio"] span{color:var(--mt)!important;font-size:.8em!important;}
/* Status */
.status-box textarea,div.status-box div[data-testid="textbox"] textarea{background:#111118!important;border:1px solid rgba(255,255,255,.06)!important;color:var(--gd)!important;font-size:.82em!important;border-radius:8px!important;}
/* Generate button */
.btn-gen{background:linear-gradient(135deg,#FF6B00,#CC4400)!important;color:#fff!important;border:none!important;border-radius:var(--r)!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:1em!important;padding:14px 24px!important;width:100%!important;cursor:pointer!important;box-shadow:0 4px 24px rgba(255,107,0,.4)!important;transition:transform .2s,box-shadow .2s!important;letter-spacing:.02em!important;}
.btn-gen:hover{transform:translateY(-3px)!important;box-shadow:0 8px 32px rgba(255,107,0,.65)!important;}
.btn-gen:active{transform:translateY(-1px)!important;}
/* Tag buttons */
.tag-btn{background:rgba(255,107,0,.08)!important;border:1px solid rgba(255,107,0,.25)!important;color:var(--gd)!important;border-radius:20px!important;font-size:.72em!important;padding:5px 10px!important;transition:all .2s!important;cursor:pointer!important;}
.tag-btn:hover{background:rgba(255,107,0,.22)!important;transform:scale(1.06)!important;border-color:var(--sf)!important;}
.tag-btn:active{transform:scale(.97)!important;}
/* Preset buttons */
.preset-btn{background:rgba(255,255,255,.04)!important;border:1px solid rgba(255,255,255,.08)!important;color:var(--mt)!important;border-radius:8px!important;font-size:.82em!important;padding:9px 12px!important;flex:1!important;transition:all .2s!important;cursor:pointer!important;}
.preset-btn:hover{background:rgba(255,107,0,.14)!important;border-color:rgba(255,107,0,.5)!important;color:var(--gd)!important;transform:translateY(-2px)!important;}
.preset-btn:active{transform:translateY(0)!important;}
/* Example buttons */
.ex-btn{background:transparent!important;border:1px solid rgba(255,255,255,.07)!important;color:var(--mt)!important;border-radius:7px!important;font-size:.8em!important;padding:7px 12px!important;text-align:left!important;width:100%!important;transition:all .18s!important;margin-bottom:5px!important;cursor:pointer!important;}
.ex-btn:hover{background:rgba(255,107,0,.1)!important;border-color:rgba(255,107,0,.4)!important;color:var(--gd)!important;padding-left:18px!important;}
/* Info card */
.info-card{background:rgba(255,107,0,.05);border:1px solid rgba(255,107,0,.15);border-radius:var(--r);padding:14px 16px;margin-top:12px;font-size:.84em;color:var(--mt);line-height:1.8;}
.info-card strong{color:var(--gd);}
.info-card code{background:rgba(255,107,0,.12);color:var(--sf);padding:1px 5px;border-radius:4px;}
.divider{border:none!important;border-top:1px solid rgba(255,107,0,.12)!important;margin:16px 0!important;}
/* Footer */
.shiv-ftr{text-align:center;padding:18px;color:var(--mt);font-size:.8em;border-top:1px solid var(--bd);background:var(--sf2);}
.shiv-ftr span{color:var(--sf);}
"""

# ── Build UI ───────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="🔱 Shiv AI Voice Cloning") as demo:

    gr.HTML("""
    <div class='shiv-hdr'>
      <h1>🔱 Shiv AI Voice Cloning</h1>
      <p>Advanced Multilingual Neural Speech Engine &nbsp;·&nbsp; 646 Languages</p>
      <p><b>Shri Ram Nag</b> &nbsp;·&nbsp; PAISAWALA 🎬</p>
      <div>
        <span class='badge'>v2026.1</span>
        <span class='badge'>646 Languages</span>
        <span class='badge'>T4 GPU</span>
      </div>
    </div>""")

    with gr.Tabs(elem_classes="tab-nav"):

        # ── TAB 1: Voice Clone ─────────────────────────────────────────────
        with gr.TabItem("🎙️ Voice Clone"):
            with gr.Row():
                with gr.Column(scale=11):
                    gr.HTML('<div class="sec-title">📝 Script</div>')
                    vc_text = gr.Textbox(lines=9, elem_classes="shiv-tb", label="", show_label=False,
                                          placeholder="पूरी script paste करें — लंबी script भी चलेगी…")
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            b = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            b.click(fn=None, inputs=[b, vc_text], outputs=vc_text, js=INSERT_TAG_JS)
                    with gr.Row():
                        vc_lang     = gr.Dropdown(LANG_CHOICES, value="Auto", label="🌐 Language", scale=1)
                        vc_ref_text = gr.Textbox(label="📄 Reference Transcript (optional)", lines=1, scale=2)
                    vc_ref = gr.Audio(label="🎤 Reference Audio  (5–30 sec, saaf awaaz)", type="filepath")
                    vc_btn = gr.Button("🔱  Clone Voice & Generate Full Audio", elem_classes="btn-gen", variant="primary")
                with gr.Column(scale=9):
                    gr.HTML('<div class="sec-title">🔊 Output</div>')
                    vc_out    = gr.Audio(type="numpy", label="Generated Audio", show_download_button=True)
                    vc_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box", lines=2)
                    gr.HTML("""<div class='info-card'>
                      <strong>✅ Hakla issue fixed!</strong><br>
                      Script → auto chunks → seamless join<br><br>
                      <strong>💡 Tips:</strong><br>
                      · 5–30 sec saaf reference audio<br>
                      · <code>…</code> = natural pause<br>
                      · <code>[laughter]</code> = hasna
                    </div>""")
            vc_btn.click(fn_clone, [vc_text, vc_lang, vc_ref, vc_ref_text], [vc_out, vc_status])

        # ── TAB 2: Voice Design ────────────────────────────────────────────
        with gr.TabItem("🎛️ Voice Design"):
            with gr.Row():
                with gr.Column(scale=11):
                    gr.HTML('<div class="sec-title">📝 Script</div>')
                    vd_text = gr.Textbox(lines=6, elem_classes="shiv-tb", label="", show_label=False,
                                          placeholder="यहाँ text paste करें…")
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            b2 = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            b2.click(fn=None, inputs=[b2, vd_text], outputs=vd_text, js=INSERT_TAG_JS)
                    vd_lang = gr.Dropdown(LANG_CHOICES, value="Auto", label="🌐 Language")
                    gr.HTML('<hr class="divider"><div class="sec-title">🎚️ Voice Controls</div>')
                    vd_speed  = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="⚡ Speed  —  0.5=Slow  |  1.0=Normal  |  2.0=Fast")
                    vd_pitch  = gr.Slider(-12, 12,  value=0,   step=1,    label="🎵 Pitch  —  -12=Deep  |  0=Normal  |  +12=High")
                    vd_energy = gr.Slider(0.3, 2.0, value=1.0, step=0.05, label="💪 Energy  —  0.3=Whisper  |  1.0=Normal  |  2.0=Loud")
                    vd_pause  = gr.Slider(0, 400, value=0, step=50,       label="⏸️ Extra Pause between chunks (ms)  —  0=best")
                    vd_style  = gr.Textbox(label="✍️ Style Instruction (optional)", lines=2, elem_classes="shiv-tb",
                                            placeholder="e.g. speak like a calm narrator…")
                    gr.HTML('<hr class="divider"><div class="sec-title">⚡ Quick Presets</div>')
                    with gr.Row():
                        pc = gr.Button("😌  Calm",        elem_classes="preset-btn")
                        pe = gr.Button("🔥  Excited",     elem_classes="preset-btn")
                        pn = gr.Button("📺  News Anchor", elem_classes="preset-btn")
                        ps = gr.Button("📖  Story",       elem_classes="preset-btn")
                    vd_btn = gr.Button("🎛️  Design & Generate", elem_classes="btn-gen", variant="primary")
                with gr.Column(scale=9):
                    gr.HTML('<div class="sec-title">🔊 Output</div>')
                    vd_out    = gr.Audio(type="numpy", label="Voice Design Output", show_download_button=True)
                    vd_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box", lines=2)
                    gr.HTML("""<div class='info-card'>
                      <strong>🎚️ Guide:</strong><br>
                      Speed ↓=dheere &nbsp;|&nbsp; ↑=tez<br>
                      Pitch ↓=gahri &nbsp;|&nbsp; ↑=patli<br>
                      Energy ↓=soft &nbsp;|&nbsp; ↑=loud<br><br>
                      <strong>⚡ Presets:</strong><br>
                      😌 Calm → meditation, yoga<br>
                      🔥 Excited → reels, ads<br>
                      📺 News → documentary<br>
                      📖 Story → audiobook
                    </div>""")
            pc.click(fn=lambda:(0.8,-2,0.7,0,"speak calmly and peacefully"),          outputs=[vd_speed,vd_pitch,vd_energy,vd_pause,vd_style])
            pe.click(fn=lambda:(1.3,3,1.5,0,"speak with excitement and high energy"), outputs=[vd_speed,vd_pitch,vd_energy,vd_pause,vd_style])
            pn.click(fn=lambda:(1.0,0,1.1,0,"speak like a professional news anchor"), outputs=[vd_speed,vd_pitch,vd_energy,vd_pause,vd_style])
            ps.click(fn=lambda:(0.85,-1,0.8,100,"speak like a warm storyteller"),     outputs=[vd_speed,vd_pitch,vd_energy,vd_pause,vd_style])
            vd_btn.click(fn_design, [vd_text,vd_lang,vd_speed,vd_pitch,vd_energy,vd_pause,vd_style], [vd_out,vd_status])

        # ── TAB 3: Simple TTS ──────────────────────────────────────────────
        with gr.TabItem("🔤 Simple TTS"):
            with gr.Row():
                with gr.Column(scale=11):
                    gr.HTML('<div class="sec-title">📝 Script</div>')
                    tts_text = gr.Textbox(lines=9, elem_classes="shiv-tb", label="", show_label=False,
                                           placeholder="Text paste karein — reference ki zaroorat nahi…")
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            b3 = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            b3.click(fn=None, inputs=[b3, tts_text], outputs=tts_text, js=INSERT_TAG_JS)
                    tts_lang = gr.Dropdown(LANG_CHOICES, value="Auto", label="🌐 Language")
                    gr.HTML('<hr class="divider"><div class="sec-title">🎯 Quality Settings</div>')
                    with gr.Row():
                        tts_steps = gr.Slider(10,64,value=40,step=2, label="🔢 Steps  —  ↑quality  ↓speed", scale=1)
                        tts_gs    = gr.Slider(1.0,5.0,value=3.0,step=0.5, label="🎯 Guidance  —  ↑realistic", scale=1)
                    tts_btn = gr.Button("🔤  Generate HD Audio", elem_classes="btn-gen", variant="primary")
                with gr.Column(scale=9):
                    gr.HTML('<div class="sec-title">🔊 Output</div>')
                    tts_out    = gr.Audio(type="numpy", label="TTS Output", show_download_button=True)
                    tts_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box", lines=2)
                    gr.HTML("""<div class='info-card'>
                      <strong>🎯 HD Tips:</strong><br>
                      Steps=40, Guidance=3.0 → best balance<br>
                      Steps=60, Guidance=4.0 → max quality<br>
                      Steps=20, Guidance=2.0 → fast draft<br><br>
                      <code>…</code> = pause &nbsp;|&nbsp; <code>[laughter]</code> = hansi<br>
                      <code>[sigh]</code> = aah &nbsp;|&nbsp; <code>[question-en]</code> = sawaal
                    </div>""")
            tts_btn.click(fn_tts, [tts_text, tts_lang, tts_steps, tts_gs], [tts_out, tts_status])

        # ── TAB 4: Instruct Mode ───────────────────────────────────────────
        with gr.TabItem("📋 Instruct Mode"):
            with gr.Row():
                with gr.Column(scale=11):
                    gr.HTML('<div class="sec-title">📝 Script</div>')
                    inst_text = gr.Textbox(lines=7, elem_classes="shiv-tb", label="", show_label=False,
                                            placeholder="यहाँ text paste करें…")
                    with gr.Row():
                        for tag in EVENT_TAGS:
                            b4 = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            b4.click(fn=None, inputs=[b4, inst_text], outputs=inst_text, js=INSERT_TAG_JS)
                    inst_lang   = gr.Dropdown(LANG_CHOICES, value="Auto", label="🌐 Language")
                    inst_prompt = gr.Textbox(label="📋 Style Instruction", lines=3, elem_classes="shiv-tb",
                                              placeholder="Speak slowly and clearly in a calm, deep voice…")
                    gr.HTML('<hr class="divider"><div class="sec-title">💡 Examples</div>')
                    for ex in INSTRUCT_EX:
                        eb = gr.Button(ex, elem_classes="ex-btn", size="sm")
                        eb.click(fn=lambda x=ex: x, outputs=inst_prompt)
                    inst_btn = gr.Button("📋  Generate with Instruction", elem_classes="btn-gen", variant="primary")
                with gr.Column(scale=9):
                    gr.HTML('<div class="sec-title">🔊 Output</div>')
                    inst_out    = gr.Audio(type="numpy", label="Instruct Output", show_download_button=True)
                    inst_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box", lines=2)
                    gr.HTML("""<div class='info-card'>
                      <strong>📋 Tips:</strong><br>
                      English instructions best kaam karte hain<br><br>
                      <em>"Bollywood trailer narrator style"</em><br>
                      <em>"Old grandfather telling a story"</em><br>
                      <em>"Soft emotional love letter reader"</em><br>
                      <em>"Energetic sports commentator"</em>
                    </div>""")
            inst_btn.click(fn_instruct, [inst_text, inst_lang, inst_prompt], [inst_out, inst_status])

    gr.HTML("""
    <div class='shiv-ftr'>
      © 2026 &nbsp;<span>🔱 Shiv AI Voice Cloning</span>&nbsp;
      ·&nbsp; Developed by <span>Shri Ram Nag</span> &nbsp;·&nbsp; PAISAWALA 🎬
    </div>""")

if __name__ == "__main__":
    demo.launch(share=True)
