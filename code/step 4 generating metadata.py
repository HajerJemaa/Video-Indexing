import ffmpeg
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import json
import os
import logging
from pathlib import Path
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('metadata_generation.log', mode='w'),
        logging.StreamHandler()
    ]
)
meta_logger = logging.getLogger('metadata_generation')

def extract_technical_metadata(video_path):
    """Extract technical metadata using ffmpeg-python."""
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        technical = {
            'path': video_path,
            'format': probe['format']['format_name'],
            'duration': float(probe['format']['duration']),
            'resolution': f"{video_stream['width']}x{video_stream['height']}",
            'frame_rate': eval(video_stream['r_frame_rate']),
            'codec': video_stream['codec_name'],
            'file_size_mb': round(file_size_mb, 2)
        }
        meta_logger.info(f"Extracted technical metadata: {technical}")
        return technical
    except Exception as e:
        meta_logger.error(f"Technical metadata extraction failed: {str(e)}")
        return {}

def extract_keywords(texts, max_keywords=10):
    """Extract keywords using TF-IDF from a list of texts."""
    try:
        # Filter out empty texts and log count
        valid_texts = [t for t in texts if t.strip()]
        meta_logger.info(f"Processing {len(valid_texts)}/{len(texts)} non-empty texts for keywords")
        if not valid_texts:
            return [[] for _ in texts]
        
        # Use 'english' stop words
        vectorizer = TfidfVectorizer(
            max_features=max_keywords,
            stop_words='english',
            token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'  # ≥3 letters, no numbers
        )
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        feature_names = vectorizer.get_feature_names_out()
        keywords = [[] for _ in texts]
        
        # Assign keywords to valid texts
        valid_idx = 0
        for i, text in enumerate(texts):
            if text.strip():
                scores = tfidf_matrix[valid_idx].toarray().flatten()
                top_indices = scores.argsort()[-max_keywords:][::-1]
                keywords[i] = [feature_names[idx] for idx in top_indices if scores[idx] > 0]
                valid_idx += 1
        
        meta_logger.info(f"Extracted keywords: {keywords[0][:3] if keywords and keywords[0] else 'None'}...")
        return keywords
    except Exception as e:
        meta_logger.error(f"Keyword extraction failed: {str(e)}")
        return [[] for _ in texts]

def analyze_sentiment(texts):
    """Perform sentiment analysis using transformers."""
    try:
        # Filter out empty texts and log count
        valid_texts = [t for t in texts if t.strip()]
        meta_logger.info(f"Processing {len(valid_texts)}/{len(texts)} non-empty texts for sentiment")
        if not valid_texts:
            return [{"label": "NEUTRAL", "score": 0.0} for _ in texts]
        
        # Initialize offline sentiment pipeline
        sentiment_analyzer = pipeline(
            'sentiment-analysis',
            model='distilbert-base-uncased-finetuned-sst-2-english',
            tokenizer='distilbert-base-uncased-finetuned-sst-2-english',
            device=-1  # CPU
        )
        # Truncate texts to 512 tokens
        results = sentiment_analyzer([t[:512] for t in valid_texts])
        
        # Assign sentiments to all texts
        sentiments = [{"label": "NEUTRAL", "score": 0.0} for _ in texts]
        valid_idx = 0
        for i, text in enumerate(texts):
            if text.strip():
                sentiments[i] = results[valid_idx]
                valid_idx += 1
        
        meta_logger.info(f"Analyzed sentiment for {len(texts)} texts: {sentiments[0]}...")
        return sentiments
    except Exception as e:
        meta_logger.error(f"Sentiment analysis failed: {str(e)}")
        return [{"label": "NEUTRAL", "score": 0.0} for _ in texts]

def main(video_path, keyframe_metadata_path, features_metadata_path, segmentation_metadata_path):
    """Main function for metadata generation."""
    try:
        # Load metadata
        with open(keyframe_metadata_path, 'r', encoding='utf-8') as f:
            keyframe_metadata = json.load(f)
        with open(features_metadata_path, 'r', encoding='utf-8') as f:
            features_metadata = json.load(f)
        with open(segmentation_metadata_path, 'r', encoding='utf-8') as f:
            segmentation_metadata = json.load(f)
        
        # Log input counts
        meta_logger.info(f"Loaded inputs: {len(keyframe_metadata)} keyframes, {len(features_metadata)} features, {len(segmentation_metadata['shots'])} shots, {len(set(s['scene_label'] for s in segmentation_metadata['scenes']))} scenes, {len(segmentation_metadata['events'])} events")
        
        if not (keyframe_metadata and features_metadata and segmentation_metadata):
            raise ValueError("Empty metadata files")
        
        # Extract technical metadata
        technical_metadata = extract_technical_metadata(video_path)
        if not technical_metadata:
            raise ValueError("Failed to extract technical metadata")
        
        # Prepare descriptive metadata
        # Keyframes: Combine Step 1 and Step 2
        keyframe_texts = [kf['audio']['transcript'] for kf in features_metadata]
        keyframe_keywords = extract_keywords(keyframe_texts)
        keyframes = [
            {
                'keyframe_num': kf['keyframe_num'],
                'original_frame_num': kf['original_frame_num'],
                'timestamp': kf['timestamp'],
                'dominant_colors': kf['visual']['dominant_colors'],
                'faces': kf['visual']['faces'],
                'transcript': kf['audio']['transcript'],
                'rms_energy': kf['audio']['rms_energy'],
                'keywords': keyframe_keywords[i] if i < len(keyframe_keywords) else []
            }
            for i, kf in enumerate(features_metadata)
        ]
        
        # Shots: From Step 3
        shots = segmentation_metadata['shots']
        
        # Scenes: From Step 3 with keywords
        scene_texts = [scene['dialogue'] for scene in segmentation_metadata['scenes']]
        scene_keywords = extract_keywords(scene_texts)
        scenes = [
            {
                'scene_label': scene['scene_label'],
                'start_time': scene['start_time'],
                'end_time': scene['end_time'],
                'start_frame': scene['start_frame'],
                'end_frame': scene['end_frame'],
                'keyframe_num': scene['keyframe_num'],
                'original_frame_num': scene['original_frame_num'],
                'dialogue': scene['dialogue'],
                'keywords': scene_keywords[i] if i < len(scene_keywords) else []
            }
            for i, scene in enumerate(segmentation_metadata['scenes'])
        ]
        
        # Events: From Step 3 with keywords
        event_texts = [event['transcript'] for event in segmentation_metadata['events']]
        event_keywords = extract_keywords(event_texts)
        events = [
            {
                'keyframe_num': event['keyframe_num'],
                'original_frame_num': event['original_frame_num'],
                'timestamp': event['timestamp'],
                'event_types': event['event_types'],
                'transcript': event['transcript'],
                'num_faces': event['num_faces'],
                'keywords': event_keywords[i] if i < len(event_keywords) else []
            }
            for i, event in enumerate(segmentation_metadata['events'])
        ]
        
        # Contextual metadata: Sentiment analysis
        keyframe_sentiments = analyze_sentiment(keyframe_texts)
        scene_sentiments = analyze_sentiment(scene_texts)
        event_sentiments = analyze_sentiment(event_texts)
        
        contextual_metadata = {
            'keyframes': [
                {'keyframe_num': kf['keyframe_num'], 'sentiment': keyframe_sentiments[i]}
                for i, kf in enumerate(keyframes)
            ],
            'scenes': [
                {'scene_label': scene['scene_label'], 'sentiment': scene_sentiments[i]}
                for i, scene in enumerate(scenes)
            ],
            'events': [
                {'keyframe_num': event['keyframe_num'], 'sentiment': event_sentiments[i]}
                for i, event in enumerate(events)
            ]
        }
        
        # Combine metadata
        output_metadata = {
            'technical': technical_metadata,
            'descriptive': {
                'keyframes': keyframes,
                'shots': shots,
                'scenes': scenes,
                'events': events
            },
            'contextual': contextual_metadata
        }
        
        # Save metadata
        output_path = 'metadata_index.json'
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_metadata, f, indent=2)
        meta_logger.info(f"Saved metadata to {output_path}")
        
        return output_metadata
    except Exception as e:
        meta_logger.error(f"Metadata generation failed: {str(e)}")
        return {}

if __name__ == "__main__":
    video_path = "howl_scene.mp4"
    keyframe_metadata_path = "keyframes/keyframe_metadata.json"
    features_metadata_path = "features_metadata_vosk.json"
    segmentation_metadata_path = "segmentation_metadata.json"
    
    # Verify inputs
    for path in [video_path, keyframe_metadata_path, features_metadata_path, segmentation_metadata_path]:
        if not os.path.exists(path):
            meta_logger.error(f"Input not found: {path}")
            exit(1)
    
    main(video_path, keyframe_metadata_path, features_metadata_path, segmentation_metadata_path)