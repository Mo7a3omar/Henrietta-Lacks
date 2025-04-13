import streamlit as st
import speech_recognition as sr
import google.generativeai as genai
import time
from datetime import datetime
from gtts import gTTS
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="Henrietta Lacks Interactive Chatbot",
    page_icon="🧬",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'current_audio' not in st.session_state:
    st.session_state.current_audio = None
if 'is_speaking' not in st.session_state:
    st.session_state.is_speaking = False
if 'audio_timestamp' not in st.session_state:
    st.session_state.audio_timestamp = 0
if 'audio_format' not in st.session_state:
    st.session_state.audio_format = 'audio/mp3'

# Custom CSS for better UI
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
</style>
""", unsafe_allow_html=True)

# Speech recognition function
def recognize_speech():
    """
    Records and recognizes speech from the microphone.
    Returns the recognized text.
    """
    recognizer = sr.Recognizer()
    
    # Display recording status
    recording_placeholder = st.empty()
    recording_placeholder.markdown("""
    <div class="audio-status">
        <span class="speaking-indicator"></span>Recording... Speak now!
    </div>
    """, unsafe_allow_html=True)
    
    text = ""
    
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            
            recording_placeholder.markdown("""
            <div class="audio-status">Processing speech...</div>
            """, unsafe_allow_html=True)
            
            text = recognizer.recognize_google(audio)
            recording_placeholder.empty()
        except sr.WaitTimeoutError:
            recording_placeholder.markdown("""
            <div class="audio-status" style="color: red;">No speech detected. Please try again.</div>
            """, unsafe_allow_html=True)
        except sr.UnknownValueError:
            recording_placeholder.markdown("""
            <div class="audio-status" style="color: red;">Could not understand audio. Please try again.</div>
            """, unsafe_allow_html=True)
        except sr.RequestError as e:
            recording_placeholder.markdown(f"""
            <div class="audio-status" style="color: red;">Error with speech recognition service: {e}</div>
            """, unsafe_allow_html=True)
        except Exception as e:
            recording_placeholder.markdown(f"""
            <div class="audio-status" style="color: red;">An error occurred: {e}</div>
            """, unsafe_allow_html=True)
    
    time.sleep(1)
    recording_placeholder.empty()
    
    return text

# Text-to-speech function using Google TTS
def text_to_speech(text):
    """
    Converts text to speech using Google Text-to-Speech and returns the audio data.
    
    Args:
        text (str): The text to convert to speech
        
    Returns:
        bytes: MP3 audio data
    """
    try:
        # Create a BytesIO object to store the audio data
        audio_bytes = BytesIO()
        
        # Create gTTS object and save to BytesIO
        tts = gTTS(text=text, lang='en', slow=False)
        tts.write_to_fp(audio_bytes)
        
        # Reset position to beginning of BytesIO object
        audio_bytes.seek(0)
        
        # Read bytes
        audio_data = audio_bytes.read()
        
        return audio_data, 'audio/mp3'
    
    except Exception as e:
        st.error(f"Error in text-to-speech: {str(e)}")
        return None, None

# Function to interact with Gemini API
def get_gemini_response(prompt, api_key):
    """
    Gets a brief, focused response from the Gemini API.
    
    Args:
        prompt (str): The user's prompt
        api_key (str): Gemini API key
        
    Returns:
        str: Response from Gemini
    """
    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Set up the model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Enhanced system prompt with comprehensive knowledge about Henrietta Lacks
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
        - Raised in Clover, Virginia by grandfather after mother died in childbirth
        - Married cousin David "Day" Lacks at age 14
        - Moved to Turner Station in Baltimore County during WWII when Day got work at Bethlehem Steel
        - Had five children: Lawrence, Elsie (who had developmental disabilities and was institutionalized), David Jr. (Sonny), Deborah, and Joseph (Zakariyya)
        - Loved to dance, cook, and was known for her red nail polish
        - Deeply religious and attended New Shiloh Baptist Church
        - Known for generosity and care for family and community
        
        MEDICAL HISTORY & HELA CELLS:
        - Felt a "knot" in womb and abnormal bleeding in January 1951
        - Diagnosed with cervical cancer at Johns Hopkins Hospital (the only hospital in the area that treated Black patients)
        - Treated with radium treatments, standard at the time
        - Dr. George Gey took samples without consent during biopsy and discovered they grew unusually well in lab
        - Named the cell line "HeLa" from first letters of Henrietta Lacks
        - HeLa cells became first "immortal" human cell line - they didn't die after a few cell divisions like other cells
        - Died October 4, 1951, at age 31 from aggressive cancer that had metastasized throughout body
        - Buried in unmarked grave in family cemetery in Clover, Virginia (later marked in 2010)
        
        IMPACT OF HELA CELLS ON SCIENCE & MEDICINE:
        - Used in developing polio vaccine in 1952 by Jonas Salk
        - Sent into space to study effects of zero gravity on human cells
        - Used for research on cancer, AIDS, radiation effects, gene mapping
        - First human cells successfully cloned in 1953
        - Used to develop techniques for in vitro fertilization
        - Used to study HPV and develop HPV vaccines
        - Over 110,000 scientific publications have used HeLa cells
        - HeLa cells have been used in research that earned three Nobel Prizes
        
        ETHICAL ISSUES & FAMILY JOURNEY:
        - Family didn't learn about the cells until 1973 when researchers contacted them for blood samples
        - Family couldn't afford healthcare despite Henrietta's cells being commercialized
        - Medical records published without permission, violating privacy
        - HeLa genome sequenced and published in 2013, later restricted after family concerns
        - No compensation ever provided to family despite commercial value of cells
        - Rebecca Skloot established Henrietta Lacks Foundation to provide scholarships and health assistance to descendants
        - NIH reached agreement with family in 2013 for some control over access to HeLa genome data
        - In 2022, estate of Henrietta Lacks settled a lawsuit against biotechnology company Thermo Fisher Scientific for profiting from her cells

        CULTURAL IMPACT:
        - Story told in book "The Immortal Life of Henrietta Lacks" by Rebecca Skloot (2010)
        - HBO film starring Oprah Winfrey as Deborah Lacks (2017)
        - Portraits displayed at Smithsonian and other museums
        - October 4 designated as Henrietta Lacks Day in Baltimore
        - WHO Director-General awarded posthumous award for contributions to medical science (2021)
        - In 2023, family reached settlement with another biotech company over use of cells
        
        If asked about something you wouldn't know as Henrietta (like events after 1951), respond in character by acknowledging the limitation of your perspective like "I passed away in 1951, so I wouldn't know about that, but I'm glad to hear my cells have helped in this way."
        """
        
        # Combine system prompt with user's question
        full_prompt = f"{system_prompt}\n\nUser asks: {prompt}\n\nHenrietta Lacks responds:"
        
        # Generate response
        response = model.generate_content(full_prompt)
        return response.text
    
    except Exception as e:
        return f"I'm sorry, there was an error: {str(e)}"

# Main content
st.markdown("<h1 class='main-header'>Henrietta Lacks Interactive Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Ask questions and hear Henrietta's voice</p>", 
           unsafe_allow_html=True)

# Audio player area
audio_placeholder = st.empty()
status_placeholder = st.empty()

# Handle automatic audio playback when new audio is generated
if st.session_state.current_audio and st.session_state.is_speaking:
    # Display audio
    audio_element = audio_placeholder.audio(
        st.session_state.current_audio, 
        format=st.session_state.audio_format,
        start_time=0
    )
    
    # Add a speaking indicator above the audio element
    status_placeholder.markdown("""
    <div class="audio-status">
        <span class="speaking-indicator"></span>Henrietta is speaking...
    </div>
    """, unsafe_allow_html=True)
    
    # Set speaking to False after displaying once
    st.session_state.is_speaking = False

# Get API key from .env file
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("No API key found in .env file. Please add GEMINI_API_KEY to your .env file.")

# Voice interaction section
st.markdown("---")
if st.button("🎤 Ask Henrietta", disabled=not api_key):
    if not api_key:
        st.error("API key is missing from .env file.")
    else:
        # Record and recognize speech
        user_text = recognize_speech()
        
        if user_text:
            # Get response from Gemini
            bot_text = get_gemini_response(user_text, api_key)
            
            # Convert response to speech
            audio_data, audio_format = text_to_speech(bot_text)
            
            if audio_data:
                # Save current audio to session state for playback
                st.session_state.current_audio = audio_data
                st.session_state.audio_format = audio_format
                st.session_state.is_speaking = True
                st.session_state.audio_timestamp = datetime.now().timestamp()
                
                # Show the bot's text response
                st.markdown(f"**You asked:** {user_text}")
                st.markdown(f"**Henrietta:** {bot_text}")
                
                # Rerun to update UI and play audio
                st.rerun()
            else:
                st.error("Failed to generate audio. Please try again.")