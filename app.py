import os
import streamlit as st
import google.generativeai as genai
import time
from datetime import datetime
from gtts import gTTS
from io import BytesIO
from dotenv import load_dotenv
import speech_recognition as sr
import uuid
import tempfile
import requests
import wave
import io

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Henrietta Lacks Interactive Chatbot",
    page_icon="🧬",
    layout="wide"
)

# Get the Gemini API key from environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("No GEMINI_API_KEY found in the environment. Please set GEMINI_API_KEY in your .env file.")
    st.stop()

# Initialize session state variables if not already present
if "current_audio" not in st.session_state:
    st.session_state.current_audio = None
if "is_speaking" not in st.session_state:
    st.session_state.is_speaking = False
if "audio_timestamp" not in st.session_state:
    st.session_state.audio_timestamp = 0
if "audio_format" not in st.session_state:
    st.session_state.audio_format = 'audio/mp3'
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_processed_audio" not in st.session_state:
    st.session_state.last_processed_audio = None

# Use the new Streamlit audio_input if available
has_audio_input = hasattr(st, "audio_input")

# Custom CSS for enhanced UI styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B0082;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.3rem;
        color: #663399;
        text-align: center;
        margin-bottom: 2rem;
    }
    .voice-btn {
        background-color: #4B0082;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: none;
        cursor: pointer;
    }
    .voice-btn:hover {
        background-color: #663399;
    }
    .audio-status {
        font-size: 1.2rem;
        color: #4B0082;
        text-align: center;
        margin: 1rem 0;
    }
    .speaking-indicator {
        display: inline-block;
        width: 15px;
        height: 15px;
        border-radius: 50%;
        background-color: #4B0082;
        margin-right: 10px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    .chat-message {
        padding: 8px;
        margin-bottom: 4px;
        border-radius: 5px;
    }
    .user-message {
        background-color: #e6e6ff;
    }
    .assistant-message {
        background-color: #f0e6ff;
    }
</style>
""", unsafe_allow_html=True)

# ------------- Function Definitions --------------

def google_speech_recognition(audio_bytes, language_hint=None):
    """
    Process audio bytes using Google Speech Recognition.
    """
    recognizer = sr.Recognizer()
    try:
        with io.BytesIO(audio_bytes) as audio_io:
            with wave.open(audio_io, 'rb') as wave_file:
                frame_rate = wave_file.getframerate()
                sample_width = wave_file.getsampwidth()
                audio_data = sr.AudioData(audio_bytes, frame_rate, sample_width)
        try:
            if language_hint == 'Korean':
                text = recognizer.recognize_google(audio_data, language="ko-KR")
            else:
                text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Error connecting to Google Speech Recognition service"
    except Exception as e:
        st.error("Error processing audio")
        return "Error processing audio file"


def whisper_asr(audio_bytes, api_key=None):
    """
    Recognize speech using OpenAI's Whisper API.
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("Whisper API key not found. Please set OPENAI_API_KEY in your environment.")
            return "Speech recognition service unavailable"
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio.flush()
        try:
            with open(temp_audio.name, "rb") as audio_file:
                files = {
                    "file": audio_file,
                    "model": (None, "whisper-1")
                }
                response = requests.post(url, headers=headers, files=files)
            if response.status_code == 200:
                return response.json().get("text", "")
            else:
                st.error("Error with Whisper speech recognition. Try text input instead.")
                return "Error with speech recognition service"
        except Exception as e:
            st.error("Error processing audio for Whisper ASR.")
            return "Error processing audio"


def text_to_speech(text):
    """
    Convert text to speech using Google TTS.
    Returns audio bytes and the audio format.
    """
    try:
        audio_bytes = BytesIO()
        tts = gTTS(text=text, lang='en', slow=False)
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        audio_data = audio_bytes.read()
        return audio_data, 'audio/mp3'
    except Exception as e:
        st.error(f"Error in text-to-speech: {str(e)}")
        return None, None


def get_gemini_response(prompt, api_key):
    """
    Get response from Gemini API with detailed context for Henrietta Lacks.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        system_prompt = """
You are a conversational AI portraying Henrietta Lacks (1920-1951), an African American woman whose cancer cells 
(known as HeLa cells) were taken without her knowledge or consent in 1951. These cells became the first immortal 
human cell line and have been crucial for countless medical breakthroughs.

IMPORTANT: Keep your responses brief and directly answer the question asked.

Speak in first person as if you are Henrietta, with a warm, dignified tone. Use simple, straightforward language 
appropriate for a woman from rural Virginia in the 1940s/50s. Show wisdom when discussing your legacy.

DETAILED KNOWLEDGE ABOUT HENRIETTA LACKS:

PERSONAL LIFE:
- Born Loretta Pleasant on August 1, 1920, in Roanoke, Virginia, but known as Henrietta
- Raised in Clover, Virginia by her grandfather after her mother died in childbirth
- Married cousin David "Day" Lacks at age 14
- Moved to Turner Station in Baltimore County during WWII when Day got work at Bethlehem Steel
- Had five children: Lawrence, Elsie (who had developmental disabilities and was institutionalized), David Jr. (Sonny), Deborah, and Joseph (Zakariyya)
- Loved to dance, cook, and was known for her red nail polish
- Deeply religious and attended New Shiloh Baptist Church
- Known for generosity and care for family and community

MEDICAL HISTORY & HELA CELLS:
- Felt a "knot" in her womb and experienced abnormal bleeding in January 1951
- Diagnosed with cervical cancer at Johns Hopkins Hospital (one of the few hospitals that treated Black patients)
- During a biopsy, Dr. George Gey took samples without consent, later discovering that they grew unusually well in the lab
- The cell line was named "HeLa" from the first letters of her first and last names
- HeLa cells became the first "immortal" human cell line and were used in groundbreaking research
- Henrietta Lacks died on October 4, 1951, at age 31

IMPACT ON SCIENCE & ETHICS:
- HeLa cells played a key role in developing the polio vaccine, cancer research, AIDS research, and many other scientific breakthroughs
- Over 110,000 scientific publications have relied on HeLa cells
- The use of her cells raised important ethical issues regarding consent and compensation
- Her story became widely known through the book "The Immortal Life of Henrietta Lacks" by Rebecca Skloot
        """
        full_prompt = f"{system_prompt}\n\nUser asks: {prompt}\n\nHenrietta Lacks responds:"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"I'm sorry, there was an error: {str(e)}"

# ---------------- Main App Layout ----------------

st.markdown("<h1 class='main-header'>Henrietta Lacks Interactive Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Ask questions and hear Henrietta's voice</p>", unsafe_allow_html=True)

# If new audio was generated (via TTS), play it automatically
if st.session_state.current_audio and st.session_state.is_speaking:
    st.audio(st.session_state.current_audio, format=st.session_state.audio_format, start_time=0)
    st.markdown("""
    <div class="audio-status">
        <span class="speaking-indicator"></span>Henrietta is speaking...
    </div>
    """, unsafe_allow_html=True)
    # Reset speaking flag after playback
    st.session_state.is_speaking = False

# Sidebar: Select your Speech Recognition Provider
asr_provider = st.sidebar.selectbox(
    "Speech Recognition Provider",
    ["Google Speech Recognition", "OpenAI Whisper"]
)

st.sidebar.markdown("---")
st.sidebar.info("If voice input does not work, please use Text Input or File Upload.")

# Create tabs for different input methods
voice_tab, text_tab, file_tab = st.tabs(["Voice Input", "Text Input", "File Upload"])

# ---------------- Voice Input Tab ----------------
with voice_tab:
    st.markdown("### Voice Input")
    if has_audio_input:
        st.write("Speak now and your question will be transcribed:")
        audio_input = st.audio_input("Record your question here")
        if audio_input is not None:
            # Create a unique key/hash for the audio to avoid reprocessing the same clip
            audio_bytes = audio_input.read()
            current_audio_hash = hash(audio_bytes)
            if current_audio_hash != st.session_state.last_processed_audio:
                st.session_state.last_processed_audio = current_audio_hash
                with st.spinner("Processing audio..."):
                    if asr_provider == "Google Speech Recognition":
                        user_text = google_speech_recognition(audio_bytes)
                    else:
                        user_text = whisper_asr(audio_bytes)
                    
                    # Proceed only if valid text is returned
                    if user_text and user_text not in [
                        "Could not understand audio",
                        "Error processing audio",
                        "Error connecting to Google Speech Recognition service",
                        "Speech recognition service unavailable"
                    ]:
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": user_text
                        })
                        # Get AI (Henrietta) response
                        response_text = get_gemini_response(user_text, gemini_api_key)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response_text
                        })
                        # Convert response to speech
                        audio_data, audio_format = text_to_speech(response_text)
                        if audio_data:
                            st.session_state.current_audio = audio_data
                            st.session_state.audio_format = audio_format
                            st.session_state.is_speaking = True
                        st.rerun()
                    else:
                        st.error("Failed to transcribe audio. Please try again or use another input method.")
    else:
        st.warning("Voice input is not supported in this version of Streamlit. Please use Text Input or File Upload.")

# ---------------- Text Input Tab ----------------
with text_tab:
    st.markdown("### Text Input")
    with st.form(key="text_input_form", clear_on_submit=True):
        user_text = st.text_input("Enter your question:")
        submit = st.form_submit_button("Send")
    if submit and user_text:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_text
        })
        with st.spinner("Henrietta is thinking..."):
            response_text = get_gemini_response(user_text, gemini_api_key)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_text
            })
            audio_data, audio_format = text_to_speech(response_text)
            if audio_data:
                st.session_state.current_audio = audio_data
                st.session_state.audio_format = audio_format
                st.session_state.is_speaking = True
            st.rerun()


