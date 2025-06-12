from flask import Flask, render_template, request, jsonify
from whoosh.index import open_dir
from whoosh.qparser import MultifieldParser
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging
from pathlib import Path
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retrieval.log', mode='w'),
        logging.StreamHandler()
    ]
)
retrieval_logger = logging.getLogger('retrieval')

app = Flask(__name__)

def get_db_connection():
    """Connect to SQLite database."""
    try:
        conn = sqlite3.connect('howl_metadata.db')
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        retrieval_logger.error(f"Database connection failed: {str(e)}")
        return None

def load_index_counts():
    """Load counts from SQLite database."""
    try:
        conn = get_db_connection()
        if not conn:
            return 0, 0, 0
        cursor = conn.cursor()
        keyframes = cursor.execute('SELECT COUNT(*) FROM keyframes').fetchone()[0]
        scenes = cursor.execute('SELECT COUNT(*) FROM scenes').fetchone()[0]
        unique_scenes = cursor.execute('SELECT COUNT(DISTINCT scene_label) FROM scenes').fetchone()[0]
        events = cursor.execute('SELECT COUNT(*) FROM events').fetchone()[0]
        non_empty_transcripts = cursor.execute('SELECT COUNT(*) FROM keyframes WHERE transcript <> ""').fetchone()[0]
        non_empty_dialogues = cursor.execute('SELECT COUNT(*) FROM scenes WHERE dialogue <> ""').fetchone()[0]
        conn.close()
        retrieval_logger.info(
            f"Database counts: {keyframes} keyframes ({non_empty_transcripts} non-empty transcripts), "
            f"{scenes} scenes ({unique_scenes} unique, {non_empty_dialogues} non-empty dialogues), "
            f"{events} events"
        )
        return keyframes, scenes, unique_scenes, events, non_empty_transcripts, non_empty_dialogues
    except Exception as e:
        retrieval_logger.error(f"Loading index counts failed: {str(e)}")
        return 0, 0, 0, 0, 0, 0

def search_index(query_str, limit=10):
    """Search Whoosh index for query."""
    try:
        ix = open_dir('whoosh_index')
        with ix.searcher() as searcher:
            query = MultifieldParser(['content'], ix.schema).parse(query_str)
            results = searcher.search(query, limit=limit)
            hits = [
                {
                    'id': hit['id'],
                    'type': hit['type'],
                    'content': hit['content'],
                    'timestamp': hit['timestamp'],
                    'keyframe_num': hit.get('keyframe_num', -1),
                    'scene_label': hit.get('scene_label', -1)
                }
                for hit in results
            ]
            retrieval_logger.info(f"Search query '{query_str}' returned {len(hits)} results")
            return hits
    except Exception as e:
        retrieval_logger.error(f"Search failed: {str(e)}")
        return []

def fetch_metadata(keyframe_num=None, scene_label=None):
    """Fetch metadata from SQLite database."""
    try:
        conn = get_db_connection()
        if not conn:
            return {}
        cursor = conn.cursor()
        
        metadata = {}
        if keyframe_num is not None:
            # Keyframe metadata
            cursor.execute('SELECT * FROM keyframes WHERE keyframe_num = ?', (keyframe_num,))
            kf = cursor.fetchone()
            if kf:
                metadata['keyframe'] = {
                    'keyframe_num': kf['keyframe_num'],
                    'timestamp': kf['timestamp'],
                    'transcript': kf['transcript'],
                    'keywords': json.loads(kf['keywords']) if kf['keywords'] else []
                }
            
            # Sentiment
            cursor.execute('SELECT label, score FROM sentiments WHERE keyframe_num = ? AND type = "keyframe"', (keyframe_num,))
            sent = cursor.fetchone()
            if sent:
                metadata['sentiment'] = {'label': sent['label'], 'score': sent['score']}
        
        if scene_label is not None:
            # Scene metadata
            cursor.execute('SELECT * FROM scenes WHERE scene_label = ?', (scene_label,))
            scene = cursor.fetchone()
            if scene:
                metadata['scene'] = {
                    'scene_label': scene['scene_label'],
                    'start_time': scene['start_time'],
                    'dialogue': scene['dialogue'],
                    'keywords': json.loads(scene['keywords']) if scene['keywords'] else []
                }
            
            # Sentiment
            cursor.execute('SELECT label, score FROM sentiments WHERE scene_label = ? AND type = "scene"', (scene_label,))
            sent = cursor.fetchone()
            if sent:
                metadata['sentiment'] = {'label': sent['label'], 'score': sent['score']}
        
        conn.close()
        return metadata
    except Exception as e:
        retrieval_logger.error(f"Metadata fetch failed: {str(e)}")
        return {}

def get_recommendations(item_id, item_type, limit=3):
    """Get recommendations based on cosine similarity."""
    try:
        conn = get_db_connection()
        if not conn:
            return []
        cursor = conn.cursor()
        
        # Fetch all items
        items = []
        if item_type == 'keyframe':
            cursor.execute('SELECT keyframe_num, transcript, keywords FROM keyframes')
            items = [
                {'id': f"keyframe_{row['keyframe_num']}", 'type': 'keyframe', 'content': ' '.join([row['transcript'], ' '.join(json.loads(row['keywords']))]).strip()}
                for row in cursor.fetchall()
            ]
        elif item_type == 'scene':
            cursor.execute('SELECT scene_label, start_time, dialogue, keywords FROM scenes')
            items = [
                {'id': f"scene_{row['scene_label']}_{row['start_time']}", 'type': 'scene', 'content': ' '.join([row['dialogue'], ' '.join(json.loads(row['keywords']))]).strip()}
                for row in cursor.fetchall()
            ]
        
        if not items:
            return []
        
        # Compute TF-IDF vectors
        texts = [item['content'] for item in items]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Find target item index
        target_idx = next((i for i, item in enumerate(items) if item['id'] == item_id), -1)
        if target_idx == -1:
            return []
        
        # Compute cosine similarity
        similarities = cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix).flatten()
        similar_indices = similarities.argsort()[-limit-1:-1][::-1]  # Exclude self
        recommendations = [
            {
                'id': items[i]['id'],
                'type': items[i]['type'],
                'content': items[i]['content'],
                'timestamp': fetch_metadata(
                    keyframe_num=int(items[i]['id'].split('_')[-1]) if items[i]['type'] == 'keyframe' else None,
                    scene_label=int(items[i]['id'].split('_')[1]) if items[i]['type'] == 'scene' else None
                ).get(items[i]['type'], {}).get('start_time' if items[i]['type'] == 'scene' else 'timestamp', 0.0)
            }
            for i in similar_indices
        ]
        
        conn.close()
        retrieval_logger.info(f"Generated {len(recommendations)} recommendations for {item_type} {item_id}")
        return recommendations
    except Exception as e:
        retrieval_logger.error(f"Recommendations failed: {str(e)}")
        return []

@app.route('/')
def index():
    """Render search page."""
    keyframes, scenes, unique_scenes, events, non_empty_transcripts, non_empty_dialogues = load_index_counts()
    return render_template('index.html', counts={
        'keyframes': keyframes,
        'scenes': scenes,
        'unique_scenes': unique_scenes,
        'events': events,
        'non_empty_transcripts': non_empty_transcripts,
        'non_empty_dialogues': non_empty_dialogues
    })

@app.route('/search', methods=['GET', 'POST'])
def search():
    """Handle search queries."""
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if not query:
            return render_template('results.html', results=[], recommendations={}, query=query)
        
        results = search_index(query)
        results_with_metadata = []
        recommendations = {}
        
        for result in results:
            metadata = fetch_metadata(
                keyframe_num=result['keyframe_num'] if result['type'] in ['keyframe', 'event'] else None,
                scene_label=result['scene_label'] if result['type'] == 'scene' else None
            )
            results_with_metadata.append({
                'id': result['id'],
                'type': result['type'],
                'content': result['content'],
                'timestamp': result['timestamp'],
                'metadata': metadata
            })
            # Get recommendations
            recs = get_recommendations(result['id'], result['type'])
            recommendations[result['id']] = recs
        
        return render_template('results.html', results=results_with_metadata, recommendations=recommendations, query=query)
    
    return render_template('results.html', results=[], recommendations={}, query='')

@app.route('/video/<float:timestamp>')
def video(timestamp):
    """Render video player at timestamp."""
    video_path = 'howl_scene.mp4'
    return render_template('video.html', video_path=video_path, timestamp=timestamp)

if __name__ == "__main__":
    # Verify inputs
    if not os.path.exists('howl_metadata.db'):
        retrieval_logger.error("Database not found: howl_metadata.db")
        exit(1)
    if not os.path.exists('whoosh_index'):
        retrieval_logger.error("Index not found: whoosh_index")
        exit(1)
    if not os.path.exists('howl_scene.mp4'):
        retrieval_logger.error("Video not found: howl_scene.mp4")
        exit(1)
    
    # Create templates directory
    Path('templates').mkdir(exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)