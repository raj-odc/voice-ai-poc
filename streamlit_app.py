import streamlit as st
import time
from voice_assistant import VoiceAssistant

# Initialize the voice assistant
va = VoiceAssistant()

# Set page config
st.set_page_config(
    page_title="Voice Assistant Demo",
    page_icon="🎤",
    layout="centered"
)

# Title and description
st.title("Voice Assistant Demo 🎤")
st.markdown("""
This is a demonstration of the Voice Assistant with the following features:
- User Registration with Voice Profile
- Voice Command Processing
- Speaker Recognition
- Speech-to-Text Transcription
""")

# Sidebar for user ID input
# Add at the top of the file
import logging

# Create a StreamHandler that writes to a string buffer
class StreamlitHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()
        self.logs = []
    
    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)

# Configure logging
streamlit_handler = StreamlitHandler()
streamlit_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger()
logger.addHandler(streamlit_handler)
logger.setLevel(logging.DEBUG)

# Add this in the sidebar
with st.sidebar:
    st.header("User Settings")
    user_id = st.text_input("Enter User ID", key="user_id")
    st.subheader("Debug Logs")
    if st.checkbox("Show Debug Logs"):
        for log in streamlit_handler.logs:
            st.text(log)

# Main content area with tabs
tab1, tab2 = st.tabs(["Register User", "Process Voice Command"])

# Register User Tab
with tab1:
    st.header("User Registration")
    if not user_id:
        st.warning("Please enter a User ID in the sidebar first.")
    else:
        if st.button("Start Registration", key="register"):
            try:
                with st.spinner("Recording voice sample..."):
                    success = va.register_new_user(user_id)
                    if success:
                        st.success(f"Successfully registered user: {user_id}")
                    else:
                        st.error("Failed to register user. Please try again.")
            except Exception as e:
                st.error(f"Registration failed: {str(e)}")

# Process Voice Command Tab
with tab2:
    st.header("Voice Command Processing")
    
    # Create columns for displaying different components
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Voice Command", key="command"):
            try:
                with st.spinner("Listening..."):
                    # Capture and process voice command
                    audio_data = va.capture_voice_sample()
                    if audio_data is not None:
                        # Identify speaker
                        speaker_id = va.identify_speaker(audio_data)
                        if speaker_id:
                            st.success(f"Speaker identified: {speaker_id}")
                            
                            # Transcribe audio
                            text = va.transcribe_audio(audio_data)
                            if text:
                                st.info("Transcribed Text:")
                                st.write(text)
                                
                                # Generate response
                                with st.spinner("Generating response..."):
                                    response = va.generate_response(text)
                                    
                                    st.success("Response:")
                                    st.write(response)
                                    
                                    # Speak the response
                                    va.speak_response(response)
                            else:
                                st.error("Failed to transcribe audio")
                        else:
                            st.error("Could not identify speaker. Please register first.")
                    else:
                        st.error("Failed to capture audio. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    with col2:
        st.subheader("Instructions")
        st.markdown("""
        1. Make sure you've registered your voice profile first
        2. Click 'Start Voice Command' to begin
        3. Speak your command clearly
        4. Wait for the system to process and respond
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Made with ❤️ by Selvaraj</p>
</div>
""", unsafe_allow_html=True)