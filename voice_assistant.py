import os
import logging
import numpy as np
import librosa
import sounddevice as sd
from pymongo import MongoClient
from scipy.io.wavfile import write
import pyttsx3
import whisper
import openai
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()

# Set up logging
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceAssistant:
    def __init__(self):
        self.sample_rate = int(os.getenv("SAMPLE_RATE", 16000))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 1024))
        self.rec_duration = 5

        self.mongo_client = MongoClient(os.getenv("MONGODB_URI"))
        self.db = self.mongo_client[os.getenv("DB_NAME")]
        self.voice_profiles = self.db.voice_profiles

        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.tts_engine = pyttsx3.init()
        self.whisper_model = whisper.load_model("base")

    def capture_audio(self):
        try:
            logger.info("Capturing audio...")
            audio = sd.rec(int(self.rec_duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype=np.float32)
            sd.wait(10)
            return audio
        except Exception as e:
            logger.error(f"Audio capture failed: {e}")
            return None

    def extract_embeddings(self, audio):
        try:
            logger.info("Starting embedding extraction")
            
            # Normalize and ensure consistent length
            normalized = audio.flatten() / np.max(np.abs(audio))
            target_length = self.sample_rate * self.rec_duration
            
            if len(normalized) < target_length:
                logger.info(f"Padding audio from {len(normalized)} to {target_length}")
                normalized = np.pad(normalized, (0, target_length - len(normalized)))
            else:
                logger.info(f"Truncating audio from {len(normalized)} to {target_length}")
                normalized = normalized[:target_length]

            # Extract MFCC features with fixed parameters
            mfcc = librosa.feature.mfcc(
                y=normalized,
                sr=self.sample_rate,
                n_mfcc=20,
                hop_length=512,
                n_fft=2048
            )
            logger.info(f"MFCC shape: {mfcc.shape}")

            # Compute statistics over time
            mean_features = np.mean(mfcc, axis=1)
            std_features = np.std(mfcc, axis=1)
            
            # Combine features
            embedding = np.concatenate([mean_features, std_features])
            
            # Normalize final embedding
            embedding = (embedding - np.mean(embedding)) / np.std(embedding)
            
            logger.info(f"Final embedding shape: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None

    def identify_user(self, audio):
        try:
            embedding = self.extract_embeddings(audio)
            if embedding is None:
                logger.error("Failed to extract embeddings from input audio")
                return None
    
            best_match = None
            best_score = -1
            threshold = 0.5  # Lowered threshold for better matching
    
            # Debug log for current embedding
            logger.info(f"Current embedding shape: {embedding.shape}")
            logger.info(f"Current embedding norm: {np.linalg.norm(embedding)}")
    
            for profile in self.voice_profiles.find():
                try:
                    stored = np.array(profile['embeddings'])
                    logger.info(f"Comparing with user {profile['user_id']}, stored shape: {stored.shape}")
                    
                    if stored.shape != embedding.shape:
                        logger.warning(f"Shape mismatch: stored {stored.shape} vs current {embedding.shape}")
                        continue
    
                    # Normalize vectors before comparison
                    embedding_norm = embedding / np.linalg.norm(embedding)
                    stored_norm = stored / np.linalg.norm(stored)
                    
                    # Calculate cosine similarity
                    sim = np.dot(embedding_norm, stored_norm)
                    logger.info(f"Similarity score with {profile['user_id']}: {sim}")
                    
                    if sim > best_score:
                        best_score = sim
                        best_match = profile['user_id']
                except Exception as e:
                    logger.error(f"Error comparing with profile {profile['user_id']}: {e}")
                    continue
    
            logger.info(f"Best match: {best_match} with score: {best_score}")
            return best_match if best_score > threshold else None
        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            return None

    def store_user_profile(self, user_id, audio):
        embedding = self.extract_embeddings(audio)
        if embedding is None:
            return False

        self.voice_profiles.update_one(
            {'user_id': user_id},
            {'$set': {'embeddings': embedding.tolist()}},
            upsert=True
        )
        return True

    def transcribe_audio(self, audio):
        try:
            temp_path = "temp.wav"
            write(temp_path, self.sample_rate, audio)
            result = self.whisper_model.transcribe(temp_path)
            os.remove(temp_path)
            return result['text']
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    def generate_response(self, query, context=""):
        try:
            prompt = f"Context: {context}\nQuery: {query}\nResponse:"
            completion = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=150
            )
            return completion.choices[0].text.strip()
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "Sorry, I couldn't generate a response."

    def speak(self, text):
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS failed: {e}")

    def handle_command(self):
        audio = self.capture_audio()
        if audio is None:
            return

        user = self.identify_user(audio)
        if not user:
            self.speak("Sorry, I couldn't identify you.")
            return

        text = self.transcribe_audio(audio)
        if not text:
            self.speak("Sorry, I didn't catch that.")
            return

        response = self.generate_response(text)
        self.speak(response)

    def register_user(self, user_id):
        print(f"Recording audio for user {user_id}...")

        audio = self.capture_audio()

        print("Audio captured.", audio)
        if audio is None:
            return False
        return self.store_user_profile(user_id, audio)

    def capture_voice_sample(self):
        """Alias for capture_audio to match Streamlit app interface"""
        return self.capture_audio()

    def identify_speaker(self, audio):
        """Alias for identify_user to match Streamlit app interface"""
        return self.identify_user(audio)

    def speak_response(self, text):
        """Alias for speak to match Streamlit app interface"""
        return self.speak(text)

    def query_database(self, text):
        """Query MongoDB for relevant context"""
        try:
            # This is a placeholder implementation
            # You can customize this based on your needs
            return ""
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return ""

    def register_new_user(self, user_id):
        """Alias for register_user to match Streamlit app interface"""
        return self.register_user(user_id)
