import sqlite3
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID, NUMERIC
import json
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('indexing_storage.log', mode='w'),
        logging.StreamHandler()
    ]
)
index_logger = logging.getLogger('indexing_storage')

def create_sqlite_db(db_path):
    """Create SQLite database with tables."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical (
                id INTEGER PRIMARY KEY,
                path TEXT,
                format TEXT,
                duration REAL,
                resolution TEXT,
                frame_rate REAL,
                codec TEXT,
                file_size_mb REAL
            )
        ''')
        
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shots (
                shot_index INTEGER PRIMARY KEY,
                start_frame INTEGER,
                end_frame INTEGER,
                start_time REAL,
                end_time REAL,
                keyframe_num INTEGER,
                original_frame_num INTEGER,
                FOREIGN KEY (keyframe_num) REFERENCES keyframes(keyframe_num)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scenes (
                scene_label INTEGER,
                start_time REAL,
                end_time REAL,
                start_frame INTEGER,
                end_frame INTEGER,
                keyframe_num INTEGER,
                original_frame_num INTEGER,
                dialogue TEXT,
                keywords TEXT,
                PRIMARY KEY (scene_label, start_time),
                FOREIGN KEY (keyframe_num) REFERENCES keyframes(keyframe_num)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyframe_num INTEGER,
                original_frame_num INTEGER,
                timestamp REAL,
                event_types TEXT,
                transcript TEXT,
                num_faces INTEGER,
                keywords TEXT,
                FOREIGN KEY (keyframe_num) REFERENCES keyframes(keyframe_num)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyframe_num INTEGER,
                scene_label INTEGER,
                type TEXT,
                label TEXT,
                score REAL,
                FOREIGN KEY (keyframe_num) REFERENCES keyframes(keyframe_num),
                FOREIGN KEY (scene_label) REFERENCES scenes(scene_label)
            )
        ''')
        
        conn.commit()
        index_logger.info(f"Created SQLite database: {db_path}")
        return conn
    except Exception as e:
        index_logger.error(f"SQLite database creation failed: {str(e)}")
        conn.close()
        return None

def store_metadata_in_sqlite(conn, metadata):
    """Store metadata in SQLite database."""
    try:
        cursor = conn.cursor()
        
        # Technical metadata
        tech = metadata['technical']
        cursor.execute('''
            INSERT INTO technical (path, format, duration, resolution, frame_rate, codec, file_size_mb)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            tech['path'], tech['format'], tech['duration'], tech['resolution'],
            tech['frame_rate'], tech['codec'], tech['file_size_mb']
        ))
        
        # Keyframes
        for kf in metadata['descriptive']['keyframes']:
            cursor.execute('''
                INSERT OR REPLACE INTO keyframes (keyframe_num, original_frame_num, timestamp, dominant_colors, faces, transcript, rms_energy, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kf['keyframe_num'], kf['original_frame_num'], kf['timestamp'],
                json.dumps(kf['dominant_colors']), json.dumps(kf['faces']),
                kf['transcript'], kf['rms_energy'], json.dumps(kf['keywords'])
            ))
        
        # Shots
        for shot in metadata['descriptive']['shots']:
            cursor.execute('''
                INSERT OR REPLACE INTO shots (shot_index, start_frame, end_frame, start_time, end_time, keyframe_num, original_frame_num)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                shot['shot_index'], shot['start_frame'], shot['end_frame'],
                shot['start_time'], shot['end_time'], shot['keyframe_num'],
                shot['original_frame_num']
            ))
        
        # Scenes
        for scene in metadata['descriptive']['scenes']:
            cursor.execute('''
                INSERT OR REPLACE INTO scenes (scene_label, start_time, end_time, start_frame, end_frame, keyframe_num, original_frame_num, dialogue, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                scene['scene_label'], scene['start_time'], scene['end_time'],
                scene['start_frame'], scene['end_frame'], scene['keyframe_num'],
                scene['original_frame_num'], scene['dialogue'], json.dumps(scene['keywords'])
            ))
        
        # Events
        for event in metadata['descriptive']['events']:
            cursor.execute('''
                INSERT INTO events (keyframe_num, original_frame_num, timestamp, event_types, transcript, num_faces, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                event['keyframe_num'], event['original_frame_num'], event['timestamp'],
                json.dumps(event['event_types']), event['transcript'],
                event['num_faces'], json.dumps(event['keywords'])
            ))
        
        # Sentiments
        for kf_sent in metadata['contextual']['keyframes']:
            cursor.execute('''
                INSERT INTO sentiments (keyframe_num, type, label, score)
                VALUES (?, ?, ?, ?)
            ''', (
                kf_sent['keyframe_num'], 'keyframe',
                kf_sent['sentiment']['label'], kf_sent['sentiment']['score']
            ))
        
        for scene_sent in metadata['contextual']['scenes']:
            cursor.execute('''
                INSERT INTO sentiments (scene_label, type, label, score)
                VALUES (?, ?, ?, ?)
            ''', (
                scene_sent['scene_label'], 'scene',
                scene_sent['sentiment']['label'], scene_sent['sentiment']['score']
            ))
        
        for event_sent in metadata['contextual']['events']:
            cursor.execute('''
                INSERT INTO sentiments (keyframe_num, type, label, score)
                VALUES (?, ?, ?, ?)
            ''', (
                event_sent['keyframe_num'], 'event',
                event_sent['sentiment']['label'], event_sent['sentiment']['score']
            ))
        
        conn.commit()
        index_logger.info("Stored metadata in SQLite database")
    except Exception as e:
        index_logger.error(f"SQLite storage failed: {str(e)}")
        conn.rollback()

def create_whoosh_index(index_dir, metadata):
    """Create Whoosh full-text index."""
    try:
        # Define schema
        schema = Schema(
            id=ID(stored=True),
            type=TEXT(stored=True),
            content=TEXT(stored=True, analyzer=None),  # TF-IDF default
            timestamp=NUMERIC(float, stored=True),
            keyframe_num=NUMERIC(int, stored=True),
            scene_label=NUMERIC(int, stored=True)
        )
        
        # Create index directory
        Path(index_dir).mkdir(exist_ok=True)
        ix = create_in(index_dir, schema)
        
        # Index documents
        writer = ix.writer()
        
        # Keyframes
        for kf in metadata['descriptive']['keyframes']:
            content = ' '.join([
                kf['transcript'],
                ' '.join(kf['keywords'])
            ]).strip()
            writer.add_document(
                id=f"keyframe_{kf['keyframe_num']}",
                type="keyframe",
                content=content,
                timestamp=kf['timestamp'],
                keyframe_num=kf['keyframe_num'],
                scene_label=-1
            )
        
        # Scenes
        for scene in metadata['descriptive']['scenes']:
            content = ' '.join([
                scene['dialogue'],
                ' '.join(scene['keywords'])
            ]).strip()
            writer.add_document(
                id=f"scene_{scene['scene_label']}_{scene['start_time']}",
                type="scene",
                content=content,
                timestamp=scene['start_time'],
                keyframe_num=scene['keyframe_num'],
                scene_label=scene['scene_label']
            )
        
        # Events
        for event in metadata['descriptive']['events']:
            content = ' '.join([
                event['transcript'],
                ' '.join(event['keywords']),
                ' '.join(event['event_types'])
            ]).strip()
            writer.add_document(
                id=f"event_{event['keyframe_num']}",
                type="event",
                content=content,
                timestamp=event['timestamp'],
                keyframe_num=event['keyframe_num'],
                scene_label=-1
            )
        
        writer.commit()
        index_logger.info(f"Created Whoosh index in {index_dir}")
    except Exception as e:
        index_logger.error(f"Whoosh indexing failed: {str(e)}")

def main(metadata_path):
    """Main function for indexing and storage."""
    try:
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Log input counts
        desc = metadata['descriptive']
        num_keyframes = len(desc['keyframes'])
        num_shots = len(desc['shots'])
        num_scenes = len(desc['scenes'])
        num_unique_scenes = len(set(s['scene_label'] for s in desc['scenes']))
        num_events = len(desc['events'])
        non_empty_dialogues = sum(1 for s in desc['scenes'] if s['dialogue'].strip())
        non_empty_transcripts = sum(1 for k in desc['keyframes'] if k['transcript'].strip())
        index_logger.info(
            f"Loaded metadata: {num_keyframes} keyframes ({non_empty_transcripts} non-empty transcripts), "
            f"{num_shots} shots, {num_scenes} scenes ({num_unique_scenes} unique, {non_empty_dialogues} non-empty dialogues), "
            f"{num_events} events"
        )
        
        if not metadata:
            raise ValueError("Empty metadata file")
        
        # Create SQLite database
        db_path = 'howl_metadata.db'
        conn = create_sqlite_db(db_path)
        if not conn:
            raise ValueError("Failed to create SQLite database")
        
        # Store metadata in SQLite
        store_metadata_in_sqlite(conn, metadata)
        conn.close()
        
        # Create Whoosh index
        index_dir = 'whoosh_index'
        create_whoosh_index(index_dir, metadata)
        
        index_logger.info("Indexing and storage completed")
        return True
    except Exception as e:
        index_logger.error(f"Indexing and storage failed: {str(e)}")
        return False

if __name__ == "__main__":
    metadata_path = "metadata_index.json"
    
    # Verify input
    if not os.path.exists(metadata_path):
        index_logger.error(f"Input not found: {metadata_path}")
        exit(1)
    
    main(metadata_path)