import sqlite3
import json
import os
import logging
from flask import Flask, request, render_template, send_file
from whoosh.index import open_dir
from whoosh.qparser import MultifieldParser
from whoosh import scoring
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retrieval.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('retrieval')

# Flask app
app = Flask(__name__)

# Load metadata
def load_metadata(metadata_path):
    """Load metadata from JSON."""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to load metadata: {str(e)}")
        return None

# Database connection
def get_db_connection(db_path):
    """Connect to SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        return None

# Search Whoosh index
def search_index(index_dir, query_str, filter_sentiment=None, time_range=None, limit=10):
    """Search Whoosh index with query, filters, and TF-IDF ranking."""
    try:
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

# Recommendation system
def get_recommendations(metadata, keyframe_num, top_n=3):
    """Suggest related keyframes/scenes using cosine similarity."""
    try:
        # Prepare documents for TF-IDF
        documents = []
        keyframe_indices = []
        for kf in metadata['descriptive']['keyframes']:
            text = ' '.join(kf['keywords']) + ' ' + kf['transcript']
            documents.append(text)
            keyframe_indices.append(kf['keyframe_num'])
        
        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Find index of target keyframe
        target_idx = keyframe_indices.index(keyframe_num)
        
        # Compute cosine similarity
        similarities = cosine_similarity(tfidf_matrix[target_idx:target_idx+1], tfidf_matrix).flatten()
        
        # Get top N similar keyframes (excluding self)
        indices = np.argsort(similarities)[-top_n-1:-1][::-1]
        recommendations = [
            {
                'keyframe_num': keyframe_indices[i],
                'keywords': metadata['descriptive']['keyframes'][i]['keywords'],
                'transcript': metadata['descriptive']['keyframes'][i]['transcript'][:50] + '...',
                'similarity': similarities[i]
            }
            for i in indices if i != target_idx
        ]
        
        logger.info(f"Generated {len(recommendations)} recommendations for keyframe {keyframe_num}")
        return recommendations
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {str(e)}")
        return []

# Flask routes
@app.route('/')
def index():
    """Home page with search form and chapters."""
    try:
        conn = get_db_connection('howl_metadata.db')
        if not conn:
            return "Database connection failed", 500
        
        # Get chapters (scenes)
        cursor = conn.cursor()
        cursor.execute('SELECT scene_id, start_time, end_time, summary FROM scenes ORDER BY start_time')
        chapters = [
            {
                'scene_id': row['scene_id'],
                'start_time': row['start_time'],
                'end_time': row['end_time'],
                'summary': row['summary']
            }
            for row in cursor.fetchall()
        ]
        conn.close()
        
        return render_template('index.html', chapters=chapters)
    except Exception as e:
        logger.error(f"Failed to load index page: {str(e)}")
        return "Internal server error", 500

@app.route('/search', methods=['GET', 'POST'])
def search():
    """Search results page."""
    if request.method == 'POST':
        query = request.form.get('query', '')
        sentiment = request.form.get('sentiment', '')
        time_start = request.form.get('time_start', '')
        time_end = request.form.get('time_end', '')
        
        # Prepare filters
        filter_sentiment = sentiment if sentiment else None
        time_range = None
        if time_start and time_end:
            try:
                time_range = (float(time_start), float(time_end))
            except ValueError:
                time_range = None
        
        # Search index
        results = search_index(
            'whoosh_index',
            query,
            filter_sentiment=filter_sentiment,
            time_range=time_range,
            limit=10
        )
        
        # Get recommendations for each result
        metadata = load_metadata('metadata_index.json')
        for result in results:
            result['recommendations'] = get_recommendations(metadata, result['keyframe_num'])
            result['thumbnail'] = f"keyframes/keyframe_{result['keyframe_num']}.jpg"
        
        return render_template('search.html', results=results, query=query)
    
    return render_template('search.html', results=[], query='')

@app.route('/thumbnail/<path:filename>')
def serve_thumbnail(filename):
    """Serve keyframe thumbnails."""
    try:
        return send_file(os.path.join('keyframes', filename))
    except Exception as e:
        logger.error(f"Failed to serve thumbnail {filename}: {str(e)}")
        return "Thumbnail not found", 404

# HTML templates
def create_templates():
    """Create HTML templates for Flask."""
    templates_dir = 'templates'
    Path(templates_dir).mkdir(exist_ok=True)
    
    # index.html
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Howl's Moving Castle - Search and Retrieval</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .search-form { margin-bottom: 20px; }
        .chapter { margin: 10px 0; }
        .chapter a { text-decoration: none; color: #007bff; }
    </style>
</head>
<body>
    <h1>Howl's Moving Castle - Search and Retrieval</h1>
    <div class="search-form">
        <form action="/search" method="post">
            <input type="text" name="query" placeholder="Enter keywords (e.g., Sophie Howl)" required>
            <select name="sentiment">
                <option value="">Any Sentiment</option>
                <option value="positive">Positive</option>
                <option value="neutral">Neutral</option>
                <option value="negative">Negative</option>
            </select>
            <input type="number" name="time_start" placeholder="Start time (s)" step="any">
            <input type="number" name="time_end" placeholder="End time (s)" step="any">
            <input type="submit" value="Search">
        </form>
    </div>
    <h2>Chapters</h2>
    {% for chapter in chapters %}
        <div class="chapter">
            <a href="/search?query={{ chapter.summary | urlencode }}&time_start={{ chapter.start_time }}&time_end={{ chapter.end_time }}">
                Scene {{ chapter.scene_id }} ({{ chapter.start_time | round(2) }}s - {{ chapter.end_time | round(2) }}s): {{ chapter.summary }}
            </a>
        </div>
    {% endfor %}
</body>
</html>
        ''')
    
    # search.html
    with open(os.path.join(templates_dir, 'search.html'), 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Search Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .result { margin: 20px 0; border: 1px solid #ccc; padding: 10px; }
        .thumbnail { max-width: 200px; }
        .recommendation { margin-left: 20px; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>Search Results for "{{ query }}"</h1>
    <a href="/">Back to Home</a>
    {% if results %}
        {% for result in results %}
            <div class="result">
                <h3>Keyframe {{ result.keyframe_num }} ({{ result.timestamp | round(2) }}s)</h3>
                <img src="/thumbnail/{{ result.thumbnail | basename }}" class="thumbnail" alt="Thumbnail">
                <p><strong>Keywords:</strong> {{ result.keywords | join(', ') }}</p>
                <p><strong>Transcript:</strong> {{ result.transcript }}</p>
                <p><strong>Scene Summary:</strong> {{ result.scene_summary }}</p>
                <p><strong>Sentiment:</strong> {{ result.sentiment }}</p>
                <p><strong>Event:</strong> {{ result.event_description }}</p>
                <p><strong>Score:</strong> {{ result.score | round(2) }}</p>
                <h4>Recommendations:</h4>
                {% for rec in result.recommendations %}
                    <div class="recommendation">
                        Keyframe {{ rec.keyframe_num }}: {{ rec.transcript }} (Similarity: {{ rec.similarity | round(2) }})
                    </div>
                {% endfor %}
            </div>
        {% endfor %}
    {% else %}
        <p>No results found.</p>
    {% endif %}
</body>
</html>
        ''')
    
    logger.info("Created HTML templates")

def main():
    """Main function to set up and run the Flask app."""
    try:
        # Create templates
        create_templates()
        
        # Verify inputs
        for path in ['metadata_index.json', 'howl_metadata.db', 'whoosh_index', 'keyframes']:
            if not os.path.exists(path):
                logger.error(f"Input not found: {path}")
                exit(1)
        
        # Run Flask app
        logger.info("Starting Flask app at http://localhost:5000")
        app.run(debug=False)
    except Exception as e:
        logger.error(f"Retrieval setup failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()