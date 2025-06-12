import cv2
import numpy as np
import json
import os
import logging
from vosk import Model, KaldiRecognizer
import wave
import librosa
import pytesseract
import face_recognition
import matplotlib.pyplot as plt
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_extraction_vosk.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('feature_extraction')

def validate_audio(audio_path):
    """Validate audio is mono, 16-bit, 16000 Hz."""
    try:
        with wave.open(audio_path, 'rb') as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                raise ValueError(f"Audio {audio_path} must be mono, 16-bit, 16000 Hz; got {wf.getnchannels()} channels, {wf.getsampwidth()*8}-bit, {wf.getframerate()} Hz")
        logger.info(f"Validated audio: {audio_path}")
        return True
    except Exception as e:
        logger.error(f"Audio validation failed: {str(e)}")
        return False

def extract_visual_features(frame):
    """Extract visual features: dominant colors, motion, faces."""
    try:
        # Dominant colors (K-means clustering)
        pixels = frame.reshape(-1, 3)
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant_colors = centers.astype(int).tolist()
        
        # Motion (placeholder, single frame)
        motion_magnitude = 0.0
        
        # Face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        faces = [{'top': loc[0], 'right': loc[1], 'bottom': loc[2], 'left': loc[3]} for loc in face_locations]
        
        return {
            'dominant_colors': dominant_colors,
            'motion_magnitude': motion_magnitude,
            'faces': faces
        }
    except Exception as e:
        logger.error(f"Failed to extract visual features: {str(e)}")
        return {'dominant_colors': [], 'motion_magnitude': 0.0, 'faces': []}

def extract_audio_features(audio_path, start_time, end_time, model_path, duration):
    """Extract audio features: transcript, RMS energy, audio type."""
    try:
        # Adjust time bounds
        start_time = max(0, start_time)
        end_time = min(end_time, duration)
        if end_time <= start_time:
            raise ValueError(f"Invalid time range: start={start_time}s, end={end_time}s")
        
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=end_time - start_time)
        if len(y) == 0:
            raise ValueError("Empty audio segment")
        
        # RMS energy
        rms = np.mean(librosa.feature.rms(y=y)[0])
        
        # Audio type (heuristic)
        audio_type = 'silence' if rms < 0.01 else 'music_or_dialogue'
        
        # Transcription with Vosk
        with wave.open(audio_path, 'rb') as wf:
            model = Model(model_path)
            rec = KaldiRecognizer(model, wf.getframerate())
            transcript = ""
            
            wf.setpos(int(start_time * wf.getframerate()))
            frames_to_read = int((end_time - start_time) * wf.getframerate())
            while frames_to_read > 0:
                data = wf.readframes(min(4000, frames_to_read))
                if not data:
                    break
                frames_to_read -= len(data) // (wf.getsampwidth() * wf.getnchannels())
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    transcript += result.get('text', '') + " "
            
            result = json.loads(rec.FinalResult())
            transcript += result.get('text', '')
            transcript = transcript.strip()
        
        return {
            'transcript': transcript,
            'rms_energy': float(rms),
            'audio_type': audio_type
        }
    except Exception as e:
        logger.error(f"Failed to extract audio features: {str(e)}")
        return {'transcript': '', 'rms_energy': 0.0, 'audio_type': 'unknown'}

def extract_text_features(frame):
    """Extract text from frame using OCR."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config='--psm 11')
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to extract text features: {str(e)}")
        return ""

def main(audio_path, model_path, keyframe_metadata_path, keyframe_folder, video_duration=220.8):
    """Extract multimodal features from Step 1 keyframes."""
    try:
        # Load keyframe metadata
        if not os.path.exists(keyframe_metadata_path):
            raise FileNotFoundError(f"Keyframe metadata not found: {keyframe_metadata_path}")
        with open(keyframe_metadata_path, 'r', encoding='utf-8') as f:
            keyframe_metadata = json.load(f)
        logger.info(f"Loaded {len(keyframe_metadata)} keyframes from {keyframe_metadata_path}")
        
        if not keyframe_metadata:
            raise ValueError("No keyframes in metadata")
        
        # Validate audio
        if not validate_audio(audio_path):
            logger.warning("Audio validation failed; audio features will be empty")
            audio_available = False
        else:
            audio_available = True
        
        features_metadata = []
        rms_energies = []
        
        for kf in keyframe_metadata:
            keyframe_num = kf['keyframe_num']
            frame_num = kf['frame_num']
            original_frame_num = kf['original_frame_num']
            timestamp = kf['timestamp']
            keyframe_path = os.path.join(keyframe_folder, f"keyframe_{keyframe_num}.jpg")
            
            if not os.path.exists(keyframe_path):
                logger.warning(f"Keyframe image not found: {keyframe_path}")
                continue
            
            # Read keyframe image
            frame = cv2.imread(keyframe_path)
            if frame is None:
                logger.warning(f"Cannot read keyframe image: {keyframe_path}")
                continue
            
            # Extract features
            visual_features = extract_visual_features(frame)
            audio_features = (
                extract_audio_features(
                    audio_path,
                    max(0, timestamp - 2.5),
                    min(timestamp + 2.5, video_duration),
                    model_path,
                    video_duration
                )
                if audio_available
                else {'transcript': '', 'rms_energy': 0.0, 'audio_type': 'unknown'}
            )
            text_features = extract_text_features(frame)
            
            features_metadata.append({
                'keyframe_num': keyframe_num,
                'frame_num': frame_num,
                'original_frame_num': original_frame_num,
                'timestamp': timestamp,
                'visual': visual_features,
                'audio': audio_features,
                'text': text_features
            })
            rms_energies.append((timestamp, audio_features['rms_energy']))
        
        # Save features metadata
        output_path = 'features_metadata_vosk.json'
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(features_metadata, f, indent=2)
        logger.info(f"Saved features metadata to {output_path}")
        
        # Plot RMS energy
        if rms_energies and audio_available:
            timestamps, energies = zip(*rms_energies)
            plt.figure(figsize=(10, 4))
            plt.plot(timestamps, energies, 'b-')
            plt.title('RMS Energy Over Time')
            plt.xlabel('Time (s)')
            plt.ylabel('RMS Energy')
            plt.grid(True)
            plt.savefig('rms_energy_vosk.png')
            plt.close()
            logger.info("Saved RMS energy plot to rms_energy_vosk.png")
        
        return features_metadata
    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        return []

if __name__ == "__main__":
    audio_path = "howl_scene_audio_mono.wav"
    model_path = "models/vosk-model-en-us-0.22"
    keyframe_metadata_path = "keyframes/keyframe_metadata.json"
    keyframe_folder = "keyframes"
    
    # Verify inputs
    for path in [audio_path, model_path, keyframe_metadata_path, keyframe_folder]:
        if not os.path.exists(path):
            logger.error(f"Input not found: {path}")
            exit(1)
    
    main(audio_path, model_path, keyframe_metadata_path, keyframe_folder)