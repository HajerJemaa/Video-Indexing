# Step 6: Retrieval and Presentation - Code Snippets with Explanations

## Snippet 1: Search Whoosh Index
**Code**:
```python
def search_index(query_str, limit=10):
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
```

**Explanation**:
This code’s like a treasure hunter! It searches the Whoosh index for words in transcripts or keywords, like finding movie scenes matching a query. It digs up the best matches for users!

## Snippet 2: Fetch Metadata from SQLite
**Code**:
```python
def fetch_metadata(keyframe_num=None, scene_label=None):
    conn = get_db_connection()
    if not conn:
        return {}
    cursor = conn.cursor()
    metadata = {}
    if keyframe_num is not None:
        cursor.execute('SELECT * FROM keyframes WHERE keyframe_num = ?', (keyframe_num,))
        kf = cursor.fetchone()
        if kf:
            metadata['keyframe'] = {
                'keyframe_num': kf['keyframe_num'],
                'timestamp': kf['timestamp'],
                'transcript': kf['transcript'],
                'keywords': json.loads(kf['keywords'])
            }
    conn.close()
    return metadata
```

**Explanation**:
This code’s like a librarian! It grabs details like transcripts or keywords from the SQLite database, like pulling a book of movie facts for a specific scene. It fills in the search results with rich info!

## Snippet 3: Generate Recommendations
**Code**:
```python
def get_recommendations(item_id, item_type, limit=3):
    conn = get_db_connection()
    cursor = conn.cursor()
    items = []
    if item_type == 'keyframe':
        cursor.execute('SELECT keyframe_num, transcript, keywords FROM keyframes')
        items = [
            {'id': f"keyframe_{row['keyframe_num']}", 'content': ' '.join([row['transcript'], ' '.join(json.loads(row['keywords']))])}
            for row in cursor.fetchall()
        ]
    texts = [item['content'] for item in items]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    target_idx = next(i for i, item in enumerate(items) if item['id'] == item_id)
    similarities = cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix).flatten()
    similar_indices = similarities.argsort()[-limit-1:-1][::-1]
    recommendations = [{'id': items[i]['id']} for i in similar_indices]
    conn.close()
    return recommendations
```

**Explanation**:
This code’s like a movie matchmaker! It uses TF-IDF and cosine similarity to suggest similar keyframes or scenes, like recommending films based on a favorite scene. It keeps users exploring!

## Snippet 4: Handle Search Requests
**Code**:
```python
@app.route('/search', methods=['GET', 'POST'])
def search():
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
            recommendations[result['id']] = get_recommendations(result['id'], result['type'])
        return render_template('results.html', results=results_with_metadata, recommendations=recommendations, query=query)
    return render_template('results.html', results=[], recommendations={}, query='')
```

**Explanation**:
This code’s like a tour guide! It handles search queries in the Flask app, fetching results, metadata, and recommendations, then displaying them, like guiding users through a movie’s highlights. It makes searching fun and interactive!