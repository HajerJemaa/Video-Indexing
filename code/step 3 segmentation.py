import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
import logging
from pathlib import Path
from vosk import Model, KaldiRecognizer
import wave

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('segmentation.log', mode='w'),
        logging.StreamHandler()
    ]
)
seg_logger = logging.getLogger('segmentation')

def validate_audio(audio_path):
    """Validate audio is mono, 16-bit, 16000 Hz."""
    try:
        with wave.open(audio_path, 'rb') as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                raise ValueError(f"Audio {audio_path} must be mono, 16-bit, 16000 Hz; got {wf.getnchannels()} channels, {wf.getsampwidth()*8}-bit, {wf.getframerate()} Hz")
        seg_logger.info(f"Validated audio: {audio_path}")
        return True
    except Exception as e:
        seg_logger.error(f"Audio validation failed: {str(e)}")
        return False

def transcribe_audio_segment(audio_path, start_time, end_time, model_path, duration=220.8):
    """Transcribe audio segment from start_time to end_time using Vosk."""
    try:
        # Adjust time bounds
        start_time = max(0, start_time)
        end_time = min(end_time, duration)
        if end_time <= start_time:
            raise ValueError(f"Invalid time range: start={start_time}s, end={end_time}s")
        
        # Open audio
        with wave.open(audio_path, 'rb') as wf:
            # Validate audio format
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                raise ValueError("Audio must be mono, 16-bit, 16000 Hz")
            
            # Initialize Vosk
            model = Model(model_path)
            rec = KaldiRecognizer(model, wf.getframerate())
            transcript = ""
            
            # Seek to start_time
            wf.setpos(int(start_time * wf.getframerate()))
            frames_to_read = int((end_time - start_time) * wf.getframerate())
            
            # Read and transcribe audio
            while frames_to_read > 0:
                data = wf.readframes(min(4000, frames_to_read))
                if not data:
                    break
                frames_to_read -= len(data) // (wf.getsampwidth() * wf.getnchannels())
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    transcript += result.get('text', '') + " "
            
            # Finalize transcription
            result = json.loads(rec.FinalResult())
            transcript += result.get('text', '')
            transcript = transcript.strip()
            
            seg_logger.info(f"Transcribed segment t={start_time:.2f}-{end_time:.2f}s: {transcript[:50]}...")
            return transcript
    except Exception as e:
        seg_logger.error(f"Transcription failed for t={start_time:.2f}-{end_time:.2f}s: {str(e)}")
        return ""

def get_shot_boundaries(keyframe_metadata, total_frames=6624):
    """Use Step 1 keyframes as shot boundaries."""
    try:
        if not keyframe_metadata:
            raise ValueError("No keyframe metadata provided")
        
        # Extract frame numbers and add video start/end
        shot_boundaries = [0]  # Start of video
        shot_boundaries.extend(kf['original_frame_num'] for kf in keyframe_metadata)
        shot_boundaries.append(total_frames)  # End of video
        shot_boundaries = sorted(set(shot_boundaries))  # Remove duplicates, sort
        
        seg_logger.info(f"Detected {len(shot_boundaries)-1} shots from {len(keyframe_metadata)} keyframes")
        return shot_boundaries
    except Exception as e:
        seg_logger.error(f"Shot boundary detection failed: {str(e)}")
        return []

def group_shots_into_scenes(shot_boundaries, keyframe_metadata, features_metadata, audio_path, model_path, num_scenes=5):
    """Group shots into scenes using K-means on Step 2 features and transcribe dialogue."""
    try:
        if not features_metadata or not keyframe_metadata:
            raise ValueError("No features or keyframe metadata provided")
        
        # Validate audio
        audio_available = validate_audio(audio_path)
        
        # Prepare features: dominant colors, RMS energy, transcript TF-IDF
        features = []
        shot_timestamps = []
        shot_keyframes = []
        
        # Compute TF-IDF for Step 2 transcripts (for clustering)
        transcripts = [kf['audio']['transcript'] for kf in features_metadata]
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(transcripts).toarray()
        
        for i in range(len(shot_boundaries) - 1):
            start_frame = shot_boundaries[i]
            end_frame = shot_boundaries[i + 1]
            start_time = start_frame / 30.0  # 30 FPS
            
            # Find closest keyframe
            closest_keyframe = min(
                features_metadata,
                key=lambda x: abs(x['timestamp'] - start_time),
                default=None
            )
            if closest_keyframe:
                kf_index = next(
                    i for i, kf in enumerate(features_metadata)
                    if kf['keyframe_num'] == closest_keyframe['keyframe_num']
                )
                colors = np.array(closest_keyframe['visual']['dominant_colors']).flatten()
                rms = closest_keyframe['audio']['rms_energy']
                tfidf = tfidf_matrix[kf_index]
                feature_vector = np.concatenate([colors, [rms], tfidf])
                features.append(feature_vector)
                shot_timestamps.append(start_time)
                shot_keyframes.append(closest_keyframe)
            else:
                features.append(np.zeros(10 + 50))  # 3 colors * 3 RGB + RMS + 50 TF-IDF
                shot_timestamps.append(start_time)
                shot_keyframes.append(None)
        
        if not features:
            raise ValueError("No valid features for clustering")
        
        # Cluster shots into scenes
        features = np.array(features)
        kmeans = KMeans(n_clusters=min(num_scenes, len(features)), random_state=0, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Collect dialogues for each scene
        scenes = []
        for i, label in enumerate(labels):
            start_time = shot_timestamps[i]
            end_time = shot_boundaries[i + 1] / 30.0
            # Transcribe audio segment
            dialogue = (
                transcribe_audio_segment(audio_path, start_time, end_time, model_path)
                if audio_available
                else ""
            )
            scenes.append({
                'shot_index': i,
                'start_frame': shot_boundaries[i],
                'end_frame': shot_boundaries[i + 1],
                'start_time': start_time,
                'end_time': end_time,
                'scene_label': int(label),
                'keyframe_num': shot_keyframes[i]['keyframe_num'] if shot_keyframes[i] else -1,
                'original_frame_num': shot_keyframes[i]['original_frame_num'] if shot_keyframes[i] else -1,
                'dialogue': dialogue
            })
            seg_logger.info(f"Scene {i} (t={start_time:.2f}-{end_time:.2f}s, label={label}): Dialogue: {dialogue[:50]}...")
        
        seg_logger.info(f"Grouped {len(scenes)} shots into {num_scenes} scenes")
        return scenes
    except Exception as e:
        seg_logger.error(f"Scene grouping failed: {str(e)}")
        return []

def detect_events(features_metadata, transcript_word_threshold=3, face_threshold=2):
    """Detect events based on transcript and faces."""
    try:
        events = []
        for kf in features_metadata:
            transcript = kf['audio']['transcript'].strip()
            num_faces = len(kf['visual']['faces'])
            is_event = False
            event_type = []
            
            # Significant dialogue
            if transcript and len(transcript.split()) > transcript_word_threshold:
                is_event = True
                event_type.append('dialogue')
            # Multiple faces
            if num_faces >= face_threshold:
                is_event = True
                event_type.append('social')
            
            if is_event:
                events.append({
                    'keyframe_num': kf['keyframe_num'],
                    'frame_num': kf['frame_num'],
                    'original_frame_num': kf['original_frame_num'],
                    'timestamp': kf['timestamp'],
                    'event_types': event_type,
                    'transcript': transcript,
                    'num_faces': num_faces
                })
                seg_logger.info(f"Event at t={kf['timestamp']:.2f}s: {event_type}, transcript: {transcript[:50]}...")
        
        seg_logger.info(f"Detected {len(events)} events")
        return events
    except Exception as e:
        seg_logger.error(f"Event detection failed: {str(e)}")
        return []

def main(video_path, keyframe_metadata_path, features_metadata_path, audio_path, model_path):
    """Main function for video segmentation: shots, scenes, events."""
    try:
        # Load metadata
        with open(keyframe_metadata_path, 'r', encoding='utf-8') as f:
            keyframe_metadata = json.load(f)
        with open(features_metadata_path, 'r', encoding='utf-8') as f:
            features_metadata = json.load(f)
        
        if not keyframe_metadata or not features_metadata:
            raise ValueError("Empty keyframe or features metadata")
        
        # Video info
        fps = 30
        frame_width = 1920
        frame_height = 1080
        total_frames = 6624
        
        # Detect shots
        shot_boundaries = get_shot_boundaries(keyframe_metadata, total_frames)
        if not shot_boundaries:
            raise ValueError("No shots detected")
        
        # Group shots into scenes
        scenes = group_shots_into_scenes(
            shot_boundaries, keyframe_metadata, features_metadata, audio_path, model_path, num_scenes=5
        )
        
        # Detect events
        events = detect_events(features_metadata, transcript_word_threshold=3, face_threshold=2)
        
        # Combine metadata
        output_metadata = {
            'shots': [{
                'shot_index': i,
                'start_frame': shot_boundaries[i],
                'end_frame': shot_boundaries[i + 1],
                'start_time': shot_boundaries[i] / fps,
                'end_time': shot_boundaries[i + 1] / fps,
                'keyframe_num': keyframe_metadata[i]['keyframe_num'] if i < len(keyframe_metadata) else -1,
                'original_frame_num': keyframe_metadata[i]['original_frame_num'] if i < len(keyframe_metadata) else -1
            } for i in range(len(shot_boundaries) - 1)],
            'scenes': scenes,
            'events': events,
            'video_info': {
                'path': video_path,
                'fps': fps,
                'frame_width': frame_width,
                'frame_height': frame_height,
                'total_frames': total_frames
            }
        }
        
        # Save metadata
        output_path = 'segmentation_metadata.json'
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_metadata, f, indent=2)
        seg_logger.info(f"Saved segmentation metadata to {output_path}")
        
        return output_metadata
    except Exception as e:
        seg_logger.error(f"Segmentation failed: {str(e)}")
        return {}

if __name__ == "__main__":
    video_path = "howl_scene.mp4"
    keyframe_metadata_path = "keyframes/keyframe_metadata.json"
    features_metadata_path = "features_metadata_vosk.json"
    audio_path = "howl_scene_audio_mono.wav"
    model_path = "models/vosk-model-en-us-0.22"
    
    # Verify inputs
    for path in [video_path, keyframe_metadata_path, features_metadata_path, audio_path, model_path]:
        if not os.path.exists(path):
            seg_logger.error(f"Input not found: {path}")
            exit(1)
    
    main(video_path, keyframe_metadata_path, features_metadata_path, audio_path, model_path)