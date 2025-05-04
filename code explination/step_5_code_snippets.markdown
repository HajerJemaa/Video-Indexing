# Step 5: Indexing - Code Snippets with Explanations

## Snippet 1: Database Creation
**Code**:
```python
def create_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            video_id INTEGER PRIMARY KEY,
            file_path TEXT,
            format TEXT,
            duration REAL,
            resolution TEXT,
            frame_rate REAL,
            file_size REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS keyframes (
            keyframe_id INTEGER PRIMARY KEY,
            video_id INTEGER,
            keyframe_num INTEGER,
            frame_num INTEGER,
            timestamp REAL,
            keywords TEXT,
            transcript TEXT,
            sentiment TEXT,
            FOREIGN KEY (video_id) REFERENCES videos(video_id)
        )
    ''')
    conn.commit()
    return conn
```

**Explanation**:
This code’s like a librarian setting up shelves! It uses SQLite to create tables for videos and keyframes, organizing details like duration and keywords. It’s our way of building a tidy storage room for all the video’s metadata.

## Snippet 2: Metadata Storage
**Code**:
```python
def store_metadata(conn, metadata):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO videos (file_path, format, duration, resolution, frame_rate, file_size)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        metadata['technical']['file_path'],
        metadata['technical']['format'],
        metadata['technical']['duration'],
        metadata['technical']['resolution'],
        metadata['technical']['frame_rate'],
        metadata['technical']['file_size']
    ))
    video_id = cursor.lastrowid
    for kf in metadata['descriptive']['keyframes']:
        cursor.execute('''
            INSERT INTO keyframes (video_id, keyframe_num, frame_num, timestamp, keywords, transcript, sentiment)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            video_id,
            kf['keyframe_num'],
            kf['frame_num'],
            kf['timestamp'],
            ','.join(kf['keywords']),
            kf['transcript'],
            kf['sentiment']
        ))
    conn.commit()
```

**Explanation**:
This code’s like a librarian filing books! It takes Step 4’s metadata and stores it in SQLite tables, neatly saving video details and keyframe info like keywords and transcripts. It keeps everything organized for quick lookups later.

## Snippet 3: Full-Text Index Creation
**Code**:
```python
def create_whoosh_index(index_dir, metadata):
    schema = Schema(
        keyframe_num=NUMERIC(stored=True),
        timestamp=NUMERIC(stored=True, decimal_places=2),
        keywords=KEYWORD(stored=True, commas=True),
        transcript=TEXT(stored=True),
        scene_summary=TEXT(stored=True),
        sentiment=TEXT(stored=True),
        event_description=TEXT(stored=True)
    )
    Path(index_dir).mkdir(exist_ok=True)
    ix = create_in(index_dir, schema)
    writer = ix.writer()
    for kf in metadata['descriptive']['keyframes']:
        writer.add_document(
            keyframe_num=kf['keyframe_num'],
            timestamp=kf['timestamp'],
            keywords=','.join(kf['keywords']),
            transcript=kf['transcript'],
            scene_summary="",
            sentiment=kf['sentiment'],
            event_description=""
        )
    writer.commit()
    return ix
```

**Explanation**:
This code’s like a search engine builder! It uses Whoosh to create an index of keywords, transcripts, and sentiments, like a super-smart card catalog. It makes finding scenes with “Sophie” or “castle” lightning-fast.

## Snippet 4: Search with Ranking
**Code**:
```python
def search_index(index_dir, query_str, filter_sentiment=None, time_range=None, limit=10):
    ix = open_dir(index_dir)
    with ix.searcher(weighting=scoring.TF_IDF()) as searcher:
        query = MultifieldParser(
            ['keywords', 'transcript', 'scene_summary', 'event_description'],
            schema=ix.schema
        ).parse(query_str)
        results = searcher.search(query, limit=limit)
        filtered_results = []
        for hit in results:
            if filter_sentiment and hit['sentiment'].lower() != filter_sentiment.lower():
                continue
            if time_range and not (time_range[0] <= hit['timestamp'] <= time_range[1]):
                continue
            filtered_results.append({
                'keyframe_num': hit['keyframe_num'],
                'timestamp': hit['timestamp'],
                'keywords': hit['keywords'].split(','),
                'transcript': hit['transcript'],
                'sentiment': hit['sentiment']
            })
        return filtered_results
```

**Explanation**:
This code’s like a treasure hunter! It searches the Whoosh index for queries like “Sophie Howl,” filters by mood or time, and ranks results with TF-IDF to find the best matches. It’s how we dig up the perfect video moments!