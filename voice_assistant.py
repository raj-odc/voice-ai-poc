import os
import speech_recognition as sr
import numpy as np
import librosa
from pymongo import MongoClient
import openai
import pyttsx3
from dotenv import load_dotenv
import sounddevice as sd
from scipy.io.wavfile import write
import whisper  # This import will now use openai-whisper

# Load environment variables
load_dotenv()

class VoiceAssistant:
    def __init__(self):
        # Initialize MongoDB connection
        self.mongo_client = MongoClient(os.getenv('MONGODB_URI'))
        self.db = self.mongo_client[os.getenv('DB_NAME')]
        self.voice_profiles = self.db.voice_profiles
        
        # Initialize OpenAI
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        
        # Initialize Whisper model for STT
        self.whisper_model = whisper.load_model('base')
        
        # Audio configuration
        self.sample_rate = int(os.getenv('SAMPLE_RATE', 16000))
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 1024))
        
    def capture_voice_sample(self, duration=5):
        """Capture a voice sample for the specified duration."""
        try:
            print(f"Recording for {duration} seconds...")
            audio_data = sd.rec(int(duration * self.sample_rate),
                               samplerate=self.sample_rate,
                               channels=1,
                               dtype=np.float32)
            sd.wait()
            return audio_data
        except Exception as e:
            print(f"Error capturing voice sample: {str(e)}")
            return None

    def extract_voice_embeddings(self, audio_data):
        """Extract voice embeddings from audio data using librosa."""
        try:
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=audio_data.flatten(),
                                       sr=self.sample_rate,
                                       n_mfcc=13)
            return mfcc.flatten()
        except Exception as e:
            print(f"Error extracting voice embeddings: {str(e)}")
            return None

    def store_voice_profile(self, user_id, embeddings):
        """Store voice embeddings in MongoDB."""
        try:
            profile = {
                'user_id': user_id,
                'embeddings': embeddings.tolist()
            }
            self.voice_profiles.update_one(
                {'user_id': user_id},
                {'$set': profile},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error storing voice profile: {str(e)}")
            return False

    def identify_speaker(self, audio_data):
        """Compare voice embeddings to identify the speaker."""
        try:
            current_embeddings = self.extract_voice_embeddings(audio_data)
            if current_embeddings is None:
                return None

            best_match = None
            min_distance = float('inf')

            for profile in self.voice_profiles.find():
                stored_embeddings = np.array(profile['embeddings'])
                distance = np.linalg.norm(current_embeddings - stored_embeddings)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = profile['user_id']

            return best_match if min_distance < 100 else None
        except Exception as e:
            print(f"Error identifying speaker: {str(e)}")
            return None

    def transcribe_audio(self, audio_data):
        """Convert speech to text using Whisper."""
        try:
            # Save audio data temporarily
            write('temp_audio.wav', self.sample_rate, audio_data)
            
            # Transcribe using Whisper
            result = self.whisper_model.transcribe('temp_audio.wav')
            
            # Clean up temporary file
            os.remove('temp_audio.wav')
            
            return result['text']
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return None

    def query_database(self, query):
        """Search MongoDB for relevant information based on the query."""
        try:
            # Implement your database querying logic here
            # This is a placeholder implementation
            results = self.db.information.find(
                {'$text': {'$search': query}}
            ).limit(5)
            return list(results)
        except Exception as e:
            print(f"Error querying database: {str(e)}")
            return []

    def generate_response(self, query, context):
        """Generate response using OpenAI's GPT."""
        try:
            prompt = f"Context: {context}\nQuery: {query}\nResponse:"
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=150
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I couldn't generate a response."

    def speak_response(self, text):
        """Convert text to speech and play it."""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            return True
        except Exception as e:
            print(f"Error speaking response: {str(e)}")
            return False

    def process_voice_command(self):
        """Main function to process voice commands."""
        try:
            # Capture voice input
            audio_data = self.capture_voice_sample()
            if audio_data is None:
                return

            # Identify speaker
            speaker_id = self.identify_speaker(audio_data)
            if speaker_id is None:
                self.speak_response("Sorry, I couldn't identify you.")
                return

            # Transcribe audio
            text = self.transcribe_audio(audio_data)
            if text is None:
                self.speak_response("Sorry, I couldn't understand that.")
                return

            # Query database
            db_results = self.query_database(text)
            
            # Generate response
            response = self.generate_response(text, db_results)
            
            # Speak response
            self.speak_response(response)

        except Exception as e:
            print(f"Error processing voice command: {str(e)}")
            self.speak_response("Sorry, there was an error processing your request.")

    def register_new_user(self, user_id):
        """Register a new user with their voice profile."""
        try:
            print("Recording voice sample for registration...")
            audio_data = self.capture_voice_sample(duration=5)
            if audio_data is None:
                return False

            embeddings = self.extract_voice_embeddings(audio_data)
            if embeddings is None:
                return False

            return self.store_voice_profile(user_id, embeddings)
        except Exception as e:
            print(f"Error registering new user: {str(e)}")
            return False