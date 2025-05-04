# Step 6: Search and Retrieval - Code Snippets with Explanations

## Snippet 1: Web App Setup
**Code**:
```python
app = Flask(__name__)
@app.route('/')
def index():
    conn = get_db_connection('howl_metadata.db')
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
```

**Explanation**:
This code’s like a tour guide setting up a welcome desk! It uses Flask to create a web page showing video chapters (scenes) from the SQLite database. It’s the front door to our video search, inviting users to explore!

## Snippet 2: Metadata Search
**Code**:
```python
def search_index(index_dir, query_str, filter_sentiment=None, time_range=None, limit=10):
    ix = open_dir(index_dir)
    with ix.searcher(weighting=scoring.TF_IDF()) as searcher:
        query = MultifieldParser(
            ['keywords', 'transcript', 'scene_summary', 'event_description'],
            schema=ix.schema
        ).parse(query_str)
        results = searcher.search(query, limit=10)
        filtered_results = []
        for hit in results:
            if filter_sentiment and hit['sentiment'].lower() != filter_sentiment.lower():
                continue
            if time_range and not (time_range[0] <= hit['timestamp'] <= time_range[1]):
                continue
            filtered_results.append({
                'keyframe_num': hit['keyframe_num'],
                'timestamp': hit['timestamp'],
                'keywords': hit['keywords'].split(',')
            })
        return filtered_results
```

**Explanation**:
This code’s like a super-smart librarian! It searches the Whoosh index for queries like “Sophie Howl,” filtering by mood or time, and ranks results with TF-IDF. It finds the best video moments in a snap!

## Snippet 3: Recommendations
**Code**:
```python
def get_recommendations(metadata, keyframe_num, top_n=3):
    documents = []
    keyframe_indices = []
    for kf in metadata['descriptive']['keyframes']:
        text = ' '.join(kf['keywords']) + ' ' + kf['transcript']
        documents.append(text)
        keyframe_indices.append(kf['keyframe_num'])
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    target_idx = keyframe_indices.index(keyframe_num)
    similarities = cosine_similarity(tfidf_matrix[target_idx:target_idx+1], tfidf_matrix).flatten()
    indices = np.argsort(similarities)[-top_n-1:-1][::-1]
    recommendations = [
        {'keyframe_num': keyframe_indices[i]}
        for i in indices if i != target_idx
    ]
    return recommendations
```

**Explanation**:
This code’s like a movie buddy suggesting similar scenes! It uses TF-IDF and cosine similarity to find keyframes close to the one you’re viewing, like recommending more “Howl” dialogues. It’s a discovery booster!

## Snippet 4: HTML Templates
**Code**:
```python
with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Howl's Moving Castle - Search and Retrieval</title>
</head>
<body>
    <h1>Howl's Moving Castle - Search and Retrieval</h1>
    <div class="search-form">
        <form action="/search" method="post">
            <input type="text" name="query" placeholder="Enter keywords (e.g., Sophie Howl)">
            <input type="submit" value="Search">
        </form>
    </div>
    <h2>Chapters</h2>
    {% for chapter in chapters %}
        <div class="chapter">
            <a href="/search?query={{ chapter.summary | urlencode }}">
                Scene {{ chapter.scene_id }}: {{ chapter.summary }}
            </a>
        </div>
    {% endfor %}
</body>
</html>
    ''')
```

**Explanation**:
This code’s like a webpage decorator! It creates an HTML template for the Flask app, showing a search bar and clickable chapters. It’s the pretty face of our app, making video browsing fun and easy!