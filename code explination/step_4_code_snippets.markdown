# Step 4: Metadata Generation - Code Snippets with Explanations

## Snippet 1: Technical Metadata Extraction
**Code**:
```python
def extract_technical_metadata(video_path):
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
```

**Explanation**:
This code’s like a video librarian! It uses FFmpeg to grab details like duration, resolution, and format, like filling out a catalog card for a movie. It gives us the video’s technical blueprint!

## Snippet 2: Keyword Extraction
**Code**:
```python
def extract_keywords(texts, max_keywords=10):
    valid_texts = [t for t in texts if t.strip()]
    if not valid_texts:
        return [[] for _ in texts]
    vectorizer = TfidfVectorizer(
        max_features=max_keywords,
        stop_words='english',
        token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
    )
    tfidf_matrix = vectorizer.fit_transform(valid_texts)
    feature_names = vectorizer.get_feature_names_out()
    keywords = [[] for _ in texts]
    valid_idx = 0
    for i, text in enumerate(texts):
        if text.strip():
            scores = tfidf_matrix[valid_idx].toarray().flatten()
            top_indices = scores.argsort()[-max_keywords:][::-1]
            keywords[i] = [feature_names[idx] for idx in top_indices if scores[idx] > 0]
            valid_idx += 1
    return keywords
```

**Explanation**:
This code’s like a word picker! It uses TF-IDF to find the top words in transcripts and dialogues, like choosing the best tags for a movie scene. It makes keyframes and scenes searchable by keywords!

## Snippet 3: Sentiment Analysis
**Code**:
```python
def analyze_sentiment(texts):
    valid_texts = [t for t in texts if t.strip()]
    if not valid_texts:
        return [{"label": "NEUTRAL", "score": 0.0} for _ in texts]
    sentiment_analyzer = pipeline(
        'sentiment-analysis',
        model='distilbert-base-uncased-finetuned-sst-2-english',
        tokenizer='distilbert-base-uncased-finetuned-sst-2-english',
        device=-1
    )
    results = sentiment_analyzer([t[:512] for t in valid_texts])
    sentiments = [{"label": "NEUTRAL", "score": 0.0} for _ in texts]
    valid_idx = 0
    for i, text in enumerate(texts):
        if text.strip():
            sentiments[i] = results[valid_idx]
            valid_idx += 1
    return sentiments
```

**Explanation**:
This code’s like a mood reader! It uses DistilBERT to detect if dialogues are positive, negative, or neutral, like sensing the vibe of a movie scene. It adds emotional context for better searches!

## Snippet 4: Metadata Consolidation
**Code**:
```python
def main(video_path, keyframe_metadata_path, features_metadata_path, segmentation_metadata_path):
    with open(keyframe_metadata_path, 'r') as f:
        keyframe_metadata = json.load(f)
    with open(features_metadata_path, 'r') as f:
        features_metadata = json.load(f)
    with open(segmentation_metadata_path, 'r') as f:
        segmentation_metadata = json.load(f)
    technical_metadata = extract_technical_metadata(video_path)
    keyframe_texts = [kf['audio']['transcript'] for kf in features_metadata]
    keyframe_keywords = extract_keywords(keyframe_texts)
    keyframes = [
        {
            'keyframe_num': kf['keyframe_num'],
            'timestamp': kf['timestamp'],
            'keywords': keyframe_keywords[i]
        }
        for i, kf in enumerate(features_metadata)
    ]
    output_metadata = {'technical': technical_metadata, 'descriptive': {'keyframes': keyframes}}
    return output_metadata
```

**Explanation**:
This code’s like a master binder! It combines technical, descriptive, and contextual metadata from Steps 1–3, like organizing a movie’s full story into one neat file. It creates a complete index for searching!