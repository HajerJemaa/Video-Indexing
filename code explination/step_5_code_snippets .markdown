# Step 5: Indexing and Storage - Code Snippets with Explanations

## Snippet 1: Create SQLite Database
**Code**:
```python
def create_sqlite_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS keyframes (
            keyframe_num INTEGER PRIMARY KEY,
            original_frame_num INTEGER,
            timestamp REAL,
            dominant_colors TEXT,
            faces TEXT,
            transcript TEXT,
            rms_energy REAL,
            keywords TEXT
        )
    ''')
    conn.commit()
    index_logger.info(f"Created SQLite database: {db_path}")
    return conn
```

**Explanation**:
This code’s like a library builder! It sets up a SQLite database with tables for keyframes, shots, and more, like creating shelves for a movie’s details. It prepares a home for all our metadata!

## Snippet 2: Store Metadata in SQLite
**Code**:
```python
def store_metadata_in_sqlite(conn, metadata):
    cursor = conn.cursor()
    for kf in metadata['descriptive']['keyframes']:
        cursor.execute('''
            INSERT OR REPLACE INTO keyframes (keyframe_num, original_frame_num, timestamp, dominant_colors, faces, transcript, rms_energy, keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            kf['keyframe_num'], kf['original_frame_num'], kf['timestamp'],
            json.dumps(kf['dominant_colors']), json.dumps(kf['faces']),
            kf['transcript'], kf['rms_energy'], json.dumps(kf['keywords'])
        ))
    conn.commit()
    index_logger.info("Stored metadata in SQLite database")
```

**Explanation**:
This code’s like a librarian filing books! It saves keyframe details like transcripts and colors into the SQLite database, like neatly placing movie facts on shelves. It keeps everything organized for later use!

## Snippet 3: Create Whoosh Index
**Code**:
```python
def create_whoosh_index(index_dir, metadata):
    schema = Schema(
        id=ID(stored=True),
        type=TEXT(stored=True),
        content=TEXT(stored=True),
        timestamp=NUMERIC(float, stored=True),
        keyframe_num=NUMERIC(int, stored=True)
    )
    Path(index_dir).mkdir(exist_ok=True)
    ix = create_in(index_dir, schema)
    writer = ix.writer()
    for kf in metadata['descriptive']['keyframes']:
        content = ' '.join([kf['transcript'], ' '.join(kf['keywords'])]).strip()
        writer.add_document(
            id=f"keyframe_{kf['keyframe_num']}",
            type="keyframe",
            content=content,
            timestamp=kf['timestamp'],
            keyframe_num=kf['keyframe_num']
        )
    writer.commit()
    index_logger.info(f"Created Whoosh index in {index_dir}")
```

**Explanation**:
This code’s like a search engine maker! It builds a Whoosh index for keyframes’ transcripts and keywords, like creating a fast lookup for movie dialogue. It makes searching super quick!

## Snippet 4: Orchestrate Indexing
**Code**:
```python
def main(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    db_path = 'howl_metadata.db'
    conn = create_sqlite_db(db_path)
    store_metadata_in_sqlite(conn, metadata)
    conn.close()
    index_dir = 'whoosh_index'
    create_whoosh_index(index_dir, metadata)
    index_logger.info("Indexing and storage completed")
    return True
```

**Explanation**:
This code’s like a project manager! It loads metadata, sets up the SQLite database, stores all details, and builds the Whoosh index, like directing a team to organize a movie’s archive. It ties everything together!