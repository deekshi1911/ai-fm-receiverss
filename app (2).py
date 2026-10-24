import io
import os
import tempfile
import time

import numpy as np
import requests
import streamlit as st
import soundfile as sf
import librosa

# ------------------------------------------------------------------
# SETTINGS (same values you used in training)
# ------------------------------------------------------------------
MODEL_PATH = next(
    (p for p in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_fm_noise_suppression.keras"),
        "/content/fm_work/models/ai_fm_noise_suppression.keras",
    ] if os.path.exists(p)),
    "ai_fm_noise_suppression.keras",
)
SAMPLE_RATE = 16000
N_FFT = 512
HOP = 256
CHUNK_SECONDS = 2                      # CNN was trained on 2-second clips
CHUNK = SAMPLE_RATE * CHUNK_SECONDS
FRAMES = 126                           # model input = (257, 126, 1)

LANGUAGES = {
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "French": "fr",
    "Spanish": "es",
    "English": "en",
}

SPEECH_LANGS = {                       # language spoken in the audio
    "English (India)": "en-IN",
    "English (US)": "en-US",
    "Hindi": "hi-IN",
    "Telugu": "te-IN",
    "Tamil": "ta-IN",
    "Malayalam": "ml-IN",
    "Kannada": "kn-IN",
}

st.set_page_config(page_title="AI-FM Receiver", page_icon="📻", layout="centered")


# ------------------------------------------------------------------
# 1. LOAD CNN MODEL (once)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading CNN model...")
def load_model():
    import tensorflow as tf
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# ------------------------------------------------------------------
# 2. NOISE SUPPRESSION
# ------------------------------------------------------------------
def load_audio(file_bytes):
    audio, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def denoise(audio, model):
    """Run the CNN on 2-second chunks and rebuild audio.
    Uses the NOISY phase (much clearer than Griffin-Lim only)."""
    out = []
    for start in range(0, len(audio), CHUNK):
        seg = audio[start:start + CHUNK]
        real_len = len(seg)
        if real_len < CHUNK:
            seg = np.pad(seg, (0, CHUNK - real_len))

        stft = librosa.stft(seg, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT)
        mag = np.abs(stft)
        phase = np.angle(stft)

        feat = np.log1p(mag).astype(np.float32)[np.newaxis, ..., np.newaxis]
        pred = model.predict(feat, verbose=0)[0, :, :, 0]

        clean_mag = np.maximum(np.expm1(pred), 0)
        clean_mag = clean_mag[:, :phase.shape[1]]

        rebuilt = librosa.istft(
            clean_mag * np.exp(1j * phase),
            hop_length=HOP, win_length=N_FFT, length=CHUNK,
        )
        out.append(rebuilt[:real_len])

    cleaned = np.concatenate(out)
    peak = np.max(np.abs(cleaned))
    if peak > 0:
        cleaned = 0.9 * cleaned / peak
    return cleaned.astype(np.float32)


def to_wav_bytes(audio):
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return buf.getvalue()


# ------------------------------------------------------------------
# 3. SPEECH RECOGNITION  (this is what defines recognized_text)
# ------------------------------------------------------------------
def recognize(audio, language_code):
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    wav_bytes = to_wav_bytes(audio)
    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        data = recognizer.record(source)
    try:
        return recognizer.recognize_google(data, language=language_code)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        raise RuntimeError(f"Speech service error: {e}")


# ------------------------------------------------------------------
# 4. TRANSLATION (3 fallbacks so it keeps working)
# ------------------------------------------------------------------
def translate(text, target, source="auto"):
    if not text.strip():
        return ""

    # Option 1: Google (deep-translator)
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source=source, target=target).translate(text)
        if result:
            return result
    except Exception:
        pass

    # Option 2: MyMemory (needs a real source language, not "auto")
    try:
        src = "en" if source == "auto" else source
        r = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text[:480], "langpair": f"{src}|{target}"},
            timeout=20,
        )
        data = r.json()
        if data.get("responseStatus") == 200:
            result = data["responseData"]["translatedText"]
            if result:
                return result
    except Exception:
        pass

    # Option 3: LibreTranslate public mirror
    try:
        r = requests.post(
            "https://translate.argosopentech.com/translate",
            json={"q": text, "source": "en" if source == "auto" else source,
                  "target": target, "format": "text"},
            timeout=20,
        )
        return r.json().get("translatedText", "")
    except Exception:
        return ""


# ------------------------------------------------------------------
# 5. USER INTERFACE
# ------------------------------------------------------------------
st.title("📻 AI-FM Receiver")
st.subheader("AI Noise Suppression + Speech Recognition + Translation")

model = load_model()
if model is None:
    st.error(
        f"CNN model not found at:\n{MODEL_PATH}\n\n"
        "Run the training cell (Step 14) first, or upload the .keras file there."
    )

with st.sidebar:
    st.header("Settings")
    speech_lang_name = st.selectbox("Language spoken in the audio",
                                    list(SPEECH_LANGS.keys()))
    targets = st.multiselect(
        "Translate into",
        list(LANGUAGES.keys()),
        default=["Telugu", "Hindi", "Tamil"],
    )
    use_cnn = st.checkbox("Apply AI noise suppression", value=True)
    recog_on = st.radio("Recognize speech from",
                        ["Cleaned audio", "Original audio"], index=0)

uploaded = st.file_uploader("Upload FM audio (wav / flac / ogg)",
                            type=["wav", "flac", "ogg"])

if uploaded is not None:
    raw_bytes = uploaded.read()
    st.markdown("**Original audio**")
    st.audio(raw_bytes)

    if st.button("▶ Process audio", type="primary"):
        try:
            audio = load_audio(raw_bytes)
        except Exception as e:
            st.error(f"Could not read this audio file: {e}")
            st.stop()

        st.write(f"Duration: {len(audio) / SAMPLE_RATE:.1f} seconds")

        # ---- Step 1: denoise
        cleaned = audio
        if use_cnn and model is not None:
            with st.spinner("AI is removing noise..."):
                cleaned = denoise(audio, model)
            st.markdown("**AI-cleaned audio**")
            st.audio(to_wav_bytes(cleaned), format="audio/wav")
            st.download_button("⬇ Download cleaned audio",
                               to_wav_bytes(cleaned),
                               file_name="AI_cleaned.wav", mime="audio/wav")

        # ---- Step 2: speech recognition
        source_audio = cleaned if recog_on == "Cleaned audio" else audio
        with st.spinner("Recognizing speech..."):
            try:
                text = recognize(source_audio, SPEECH_LANGS[speech_lang_name])
            except Exception as e:
                st.error(str(e))
                st.stop()

        if not text:
            st.warning(
                "No speech could be recognized. Try 'Original audio' in the "
                "sidebar (CNN output can sound robotic), or check the language."
            )
            st.stop()

        st.markdown("### 🗣 Recognized text")
        st.success(text)

        # ---- Step 3: translation
        if targets:
            st.markdown("### 🌐 Translations")
            src_code = SPEECH_LANGS[speech_lang_name].split("-")[0]
            results = {"recognized_text": text}
            for name in targets:
                with st.spinner(f"Translating to {name}..."):
                    out = translate(text, LANGUAGES[name], source=src_code)
                if out:
                    st.markdown(f"**{name}**")
                    st.info(out)
                    results[name] = out
                else:
                    st.warning(f"{name}: translation service unavailable, try again.")
                time.sleep(0.5)

            import json
            st.download_button(
                "⬇ Download results (JSON)",
                json.dumps(results, ensure_ascii=False, indent=2),
                file_name="speech_results.json", mime="application/json",
            )

st.caption("AI-FM Receiver | CNN noise suppression · Google Speech · Multilingual translation")
