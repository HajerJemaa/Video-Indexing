# Step 4: Metadata Consolidation - Code Snippets with Explanations

## Snippet 1: Technical Metadata Extraction
**Code**:
```python
def extract_technical_metadata(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        file_size = os.path.getsize(video_path) / (1024 * 1024)
        tech_metadata = {
            'format': probe['format']['format_name'],
            'duration': float(probe['format']['duration']),
            'resolution': f"{video_info['width']}x{video_info['height']}",
            'codec': video_info['codec_name'],
            'frame_rate': eval(video_info['r_frame_rate']),
            'file_size_mb': round(file_size, 2)
        }
        return tech_metadata
    except Exception as e:
        return {}
```

**Explanation**:
This code’s like a video’s ID card maker! It uses FFmpeg to grab details like format (MP4), resolution (1920x1080), and size, just like listing someone’s name and height. These facts help us understand the video’s “vitals” for organizing it later.

## Snippet 2: Keyword Extraction
**Code**:
```python
def extract_keywords(texts, max_keywords=10):
    stop_words = set(stopwords.words('english')).union({'chunk', 'unintelligible'})
    cleaned_texts = [re.sub(r'[^\w\s]', '', text.lower()) for text in texts]
    if not cleaned_texts or all(len(text.split()) < 2 for text in cleaned_texts):
        return []
    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        max_features=max_keywords,
        token_pattern=r'\b\w+\b'
    )
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    feature_names = vectorizer.get_feature_names_out()
    keywords = []
    for doc_idx in range(tfidf_matrix.shape[0]):
        doc_scores = tfidf_matrix[doc_idx].toarray().flatten()
        top_indices = doc_scores.argsort()[-max_keywords:][::-1]
        doc_keywords = [feature_names[idx] for idx in top_indices if doc_scores[idx] > 0]
        keywords.extend(doc_keywords)
    keywords = list(dict.fromkeys(keywords))[:max_keywords]
    return keywords
```

**Explanation**:
Think of this code as a word spotlight! It scans transcripts and text using TF-IDF to pick out star words like “dog” or “park,” ignoring boring ones like “the.” It’s like choosing the best hashtags for a video clip to make it searchable.

## Snippet 3: Sentiment Analysis
**Code**:
```python
def analyze_sentiment(texts):
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiments = []
    for text in texts:
        if not text.strip() or len(text.split()) < 2:
            sentiments.append({'label': 'NEUTRAL', 'score': 0.0})
            continue
        tokens = text.split()[:512]
        truncated_text = ' '.join(tokens)
        result = sentiment_pipeline(truncated_text)[0]
        sentiments.append({
            'label': result['label'].lower(),
            'score': round(result['score'], 3)
        })
    return sentiments
```

**Explanation**:
This code’s like a mood detector! It uses a smart model (DistilBERT) to read text and decide if it’s happy, sad, or neutral, like guessing if a scene feels joyful. It helps us tag videos with emotions, so we can find all the cheerful moments.

## Snippet 4: Metadata Consolidation
**Code**:
```python
def generate_metadata(video_path, features_metadata_path, segmentation_metadata_path):
    with open(features_metadata_path, 'r') as f:
        features_metadata = json.load(f)
    with open(segmentation_metadata_path, 'r') as f:
        segmentation_metadata = json.load(f)
    tech_metadata = extract_technical_metadata(video_path)
    transcripts = [kf['audio']['transcript'] for kf in features_metadata]
    ocr_texts = [kf['text'] for kf in features_metadata]
    all_texts = [f"{t} {o}" for t, o in zip(transcripts, ocr_texts) if t.strip() or o.strip()]
    global_keywords = extract_keywords(all_texts, max_keywords=10)
    metadata = {
        'technical': tech_metadata,
        'descriptive': {
            'title': "Howl's Moving Castle - Scene",
            'global_keywords': global_keywords,
            'genres': ['animation', 'fantasy', 'adventure'],
            'themes': ['magic', 'love', 'war'],
            'language': 'English'
        },
        'contextual': {...}  # Scenes, events, sentiments (omitted for brevity)
    }
    output_path = 'metadata_index.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    return metadata
```

**Explanation**:
This code’s like a master librarian! It gathers technical details, keywords, and scene moods, then organizes them into a neat JSON book. It’s our way of creating a video “catalog” that makes finding the best scenes super easy.