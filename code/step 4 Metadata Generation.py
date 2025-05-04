import ffmpeg
import json
import os
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from pathlib import Path
import nltk
from nltk.corpus import stopwords
import re

# Download NLTK stopwords if not already present
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
meta_logger = logging.getLogger('metadata')

def extract_technical_metadata(video_path):
    """Extract technical details from video using ffmpeg."""
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB
        tech_metadata = {
            'format': probe['format']['format_name'],
            'duration': float(probe['format']['duration']),
            'resolution': f"{video_info['width']}x{video_info['height']}",
            'codec': video_info['codec_name'],
            'frame_rate': eval(video_info['r_frame_rate']),
            'file_size_mb': round(file_size, 2)
        }
        meta_logger.info(f"Extracted technical metadata: {tech_metadata}")
        return tech_metadata
    except Exception as e:
        meta_logger.error(f"Failed to extract technical metadata: {str(e)}")
        return {}

def extract_keywords(texts, max_keywords=10):
    """Extract keywords using TF-IDF, excluding stopwords."""
    try:
        stop_words = set(stopwords.words('english')).union({'chunk', 'unintelligible'})
        # Clean text: remove punctuation, lowercase
        cleaned_texts = [re.sub(r'[^\w\s]', '', text.lower()) for text in texts]
        if not cleaned_texts or all(len(text.split()) < 2 for text in cleaned_texts):
            return []
        
        vectorizer = TfidfVectorizer(
            stop_words=list(stop_words),
            max_features=max_keywords,
            token_pattern=r'\b\w+\b'  # Match single words
        )
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
        feature_names = vectorizer.get_feature_names_out()
        keywords = []
        
        # Get top keywords per document
        for doc_idx in range(tfidf_matrix.shape[0]):
            doc_scores = tfidf_matrix[doc_idx].toarray().flatten()
            top_indices = doc_scores.argsort()[-max_keywords:][::-1]
            doc_keywords = [feature_names[idx] for idx in top_indices if doc_scores[idx] > 0]
            keywords.extend(doc_keywords)
        
        # Deduplicate and limit
        keywords = list(dict.fromkeys(keywords))[:max_keywords]
        meta_logger.info(f"Extracted keywords: {keywords}")
        return keywords
    except Exception as e:
        meta_logger.error(f"Failed to extract keywords: {str(e)}")
        return []

def analyze_sentiment(texts):
    """Perform sentiment analysis per text segment."""
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        sentiments = []
        for text in texts:
            if not text.strip() or len(text.split()) < 2:
                sentiments.append({'label': 'NEUTRAL', 'score': 0.0})
                continue
            # Truncate to 512 tokens (BERT limit)
            tokens = text.split()[:512]
            truncated_text = ' '.join(tokens)
            result = sentiment_pipeline(truncated_text)[0]
            sentiments.append({
                'label': result['label'].lower(),
                'score': round(result['score'], 3)
            })
        meta_logger.info(f"Analyzed sentiment for {len(sentiments)} segments")
        return sentiments
    except Exception as e:
        meta_logger.error(f"Failed to analyze sentiment: {str(e)}")
        return [{'label': 'NEUTRAL', 'score': 0.0} for _ in texts]

def generate_metadata(video_path, features_metadata_path, segmentation_metadata_path):
    """Generate technical, descriptive, and contextual metadata for indexing."""
    try:
        # Load Step 2 and Step 3 metadata
        with open(features_metadata_path, 'r') as f:
            features_metadata = json.load(f)
        with open(segmentation_metadata_path, 'r') as f:
            segmentation_metadata = json.load(f)

        # Extract technical metadata
        tech_metadata = extract_technical_metadata(video_path)

        # Prepare texts for keyword extraction and sentiment analysis
        transcripts = [kf['audio']['transcript'] for kf in features_metadata]
        ocr_texts = [kf['text'] for kf in features_metadata]
        all_texts = [f"{t} {o}" for t, o in zip(transcripts, ocr_texts) if t.strip() or o.strip()]

        # Extract global keywords
        global_keywords = extract_keywords(all_texts, max_keywords=10)

        # Analyze sentiment per keyframe
        keyframe_sentiments = analyze_sentiment(all_texts)

        # Generate scene metadata
        scene_metadata = []
        for scene in segmentation_metadata['scenes']:
            start_time = scene['start_time']
            end_time = scene['end_time']
            # Find keyframes in this scene
            scene_keyframes = [
                kf for kf in features_metadata
                if start_time <= kf['timestamp'] < end_time
            ]
            scene_transcripts = [kf['audio']['transcript'] for kf in scene_keyframes]
            scene_ocr = [kf['text'] for kf in scene_keyframes]
            scene_texts = [f"{t} {o}" for t, o in zip(scene_transcripts, scene_ocr) if t.strip() or o.strip()]
            scene_keywords = extract_keywords(scene_texts, max_keywords=5)
            scene_sentiment = analyze_sentiment(scene_texts)[0] if scene_texts else {'label': 'NEUTRAL', 'score': 0.0}
            scene_summary = ' '.join([t for t in scene_transcripts if t.strip()])[:100] or "No dialogue."
            
            scene_metadata.append({
                'scene_label': scene['scene_label'],
                'start_time': start_time,
                'end_time': end_time,
                'keywords': scene_keywords,
                'sentiment': scene_sentiment,
                'summary': scene_summary,
                'keyframe_count': len(scene_keyframes)
            })

        # Generate event metadata
        event_metadata = []
        for event in segmentation_metadata['events']:
            kf = next((kf for kf in features_metadata if kf['keyframe_num'] == event['keyframe_num']), None)
            if not kf:
                continue
            event_text = f"{kf['audio']['transcript']} {kf['text']}".strip()
            event_keywords = extract_keywords([event_text], max_keywords=3) if event_text else []
            event_sentiment = analyze_sentiment([event_text])[0] if event_text else {'label': 'NEUTRAL', 'score': 0.0}
            event_metadata.append({
                'timestamp': event['timestamp'],
                'event_types': event['event_types'],
                'keywords': event_keywords,
                'sentiment': event_sentiment,
                'transcript': event['transcript'],
                'num_faces': event['num_faces']
            })

        # Compile metadata
        metadata = {
            'technical': tech_metadata,
            'descriptive': {
                'title': "Howl's Moving Castle - Scene",
                'global_keywords': global_keywords,
                'genres': ['animation', 'fantasy', 'adventure'],  # Based on movie
                'themes': ['magic', 'love', 'war'],
                'language': 'English'
            },
            'contextual': {
                'scenes': scene_metadata,
                'events': event_metadata,
                'keyframe_sentiments': [
                    {'keyframe_num': kf['keyframe_num'], 'timestamp': kf['timestamp'], 'sentiment': sent}
                    for kf, sent in zip(features_metadata, keyframe_sentiments)
                ]
            }
        }

        # Save metadata
        output_path = 'metadata_index.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        meta_logger.info(f"Saved metadata to {output_path}")

        return metadata
    except Exception as e:
        meta_logger.error(f"Metadata generation failed: {str(e)}")
        return {}

if __name__ == "__main__":
    video_path = "howl_scene.mp4"
    features_metadata_path = "features_metadata_vosk.json"
    segmentation_metadata_path = "segmentation_metadata.json"

    # Verify inputs
    for path in [video_path, features_metadata_path, segmentation_metadata_path]:
        if not os.path.exists(path):
            meta_logger.error(f"Input file not found: {path}")
            exit(1)

    metadata = generate_metadata(video_path, features_metadata_path, segmentation_metadata_path)