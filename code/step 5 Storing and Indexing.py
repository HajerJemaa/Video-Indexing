import sqlite3
import json
import os
import logging
from pathlib import Path
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, NUMERIC, KEYWORD
from whoosh.qparser import MultifieldParser
from whoosh import scoring

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('indexing.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('indexing')

def create_database(db_path):
    """Create SQLite database and tables."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scenes (
                scene_id INTEGER PRIMARY KEY,
                video_id INTEGER,
                start_time REAL,
                end_time REAL,
                summary TEXT,
                sentiment TEXT,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY,
                video_id INTEGER,
                keyframe_id INTEGER,
                timestamp REAL,
                description TEXT,
                sentiment TEXT,
                FOREIGN KEY (video_id) REFERENCES videos(video_id),
                FOREIGN KEY (keyframe_id) REFERENCES keyframes(keyframe_id)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_video_id ON keyframes(video_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_keyframe_num ON keyframes(keyframe_num)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scene_id ON scenes(scene_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON keyframes(timestamp)')
        
        conn.commit()
        logger.info(f"Created database at {db_path}")
        return conn
    except Exception as e:
        logger.error(f"Failed to create database: {str(e)}")
        raise

def store_metadata(conn, metadata):
    """Store metadata in SQLite database."""
    try:
        cursor = conn.cursor()
        
        # Store video metadata
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
        
        # Store keyframes
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
        
        # Store scenes
        for scene in metadata['contextual']['scenes']:
            cursor.execute('''
                INSERT INTO scenes (video_id, start_time, end_time, summary, sentiment)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                video_id,
                scene['start_time'],
                scene['end_time'],
                scene['summary'],
                scene['sentiment']
            ))
        
        # Store events
        for event in metadata['contextual']['events']:
            cursor.execute('''
                SELECT keyframe_id FROM keyframes WHERE video_id = ? AND keyframe_num = ?
            ''', (video_id, event['keyframe_num']))
            keyframe_id = cursor.fetchone()[0]
            
            cursor.execute('''
                INSERT INTO events (video_id, keyframe_id, timestamp, description, sentiment)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                video_id,
                keyframe_id,
                event['timestamp'],
                event['description'],
                event['sentiment']
            ))
        
        conn.commit()
        logger.info("Stored metadata in database")
    except Exception as e:
        logger.error(f"Failed to store metadata: {str(e)}")
        raise

def create_whoosh_index(index_dir, metadata):
    """Create Whoosh index for full-text search."""
    try:
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
            # Find associated scene
            scene_summary = ""
            for scene in metadata['contextual']['scenes']:
                if scene['start_time'] <= kf['timestamp'] <= scene['end_time']:
                    scene_summary = scene['summary']
                    break
            
            # Find associated event
            event_description = ""
            for event in metadata['contextual']['events']:
                if event['keyframe_num'] == kf['keyframe_num']:
                    event_description = event['description']
                    break
            
            writer.add_document(
                keyframe_num=kf['keyframe_num'],
                timestamp=kf['timestamp'],
                keywords=','.join(kf['keywords']),
                transcript=kf['transcript'],
                scene_summary=scene_summary,
                sentiment=kf['sentiment'],
                event_description=event_description
            )
        
        writer.commit()
        logger.info(f"Created Whoosh index at {index_dir}")
        return ix
    except Exception as e:
        logger.error(f"Failed to create Whoosh index: {str(e)}")
        raise

def search_index(index_dir, query_str, filter_sentiment=None, time_range=None, limit=10):
    """Search Whoosh index with query, filters, and TF-IDF ranking."""
    try:
        ix = open_dir(index_dir)
        with ix.searcher(weighting=scoring.TF_IDF()) as searcher:
            query = MultifieldParser(
                ['keywords', 'transcript', 'scene_summary', 'event_description'],
                schema=ix.schema
            ).parse(query_str)
            
            # Apply filters
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
                    'scene_summary': hit['scene_summary'],
                    'sentiment': hit['sentiment'],
                    'event_description': hit['event_description'],
                    'score': hit.score
                })
            
            logger.info(f"Search query '{query_str}' returned {len(filtered_results)} results")
            return filtered_results
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return []

def main(metadata_path, db_path, index_dir):
    """Main function for storing and indexing metadata."""
    try:
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
        
        # Create and populate database
        conn = create_database(db_path)
        store_metadata(conn, metadata)
        conn.close()
        
        # Create Whoosh index
        create_whoosh_index(index_dir, metadata)
        
        # Example searches
        example_queries = [
            {"query": "Sophie Howl", "sentiment": None, "time_range": None},
            {"query": "castle moving", "sentiment": "positive", "time_range": None},
            {"query": "dialogue", "sentiment": None, "time_range": (0, 100)}
        ]
        
        for q in example_queries:
            results = search_index(
                index_dir,
                q["query"],
                filter_sentiment=q["sentiment"],
                time_range=q["time_range"],
                limit=5
            )
            logger.info(f"\nQuery: {q['query']}")
            if q["sentiment"]:
                logger.info(f"Sentiment filter: {q['sentiment']}")
            if q["time_range"]:
                logger.info(f"Time range: {q['time_range']}")
            for i, res in enumerate(results, 1):
                logger.info(f"Result {i}: Keyframe {res['keyframe_num']} (t={res['timestamp']:.2f}s), "
                           f"Score: {res['score']:.2f}, Keywords: {res['keywords']}, "
                           f"Transcript: {res['transcript'][:50]}..., Sentiment: {res['sentiment']}")
        
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    metadata_path = "metadata_index.json"
    db_path = "howl_metadata.db"
    index_dir = "whoosh_index"
    
    for path in [metadata_path]:
        if not os.path.exists(path):
            logger.error(f"Input file not found: {path}")
            exit(1)
    
    main(metadata_path, db_path, index_dir)