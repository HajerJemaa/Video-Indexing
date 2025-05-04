# Presentation: Step 5 - Storing and Indexing

## 1. Introduction
- **What is Video Indexing?**
  - Organizing videos for fast search and retrieval of specific moments.
  - Example: Quickly finding all "dog" scenes in a video.
- **Our Project**:
  - Processed videos, extracted multimodal features (visual, audio, text), segmented, consolidated metadata, and now storing and indexing.
  - Multimodal: Leverages visual (objects, faces), audio (speech, noise), and text (keywords, sentiment).

## 2. Purpose of Step 5: Storing and Indexing
- **Goal**:
  - Store metadata persistently and create an index for efficient retrieval.
  - Index supports queries by keywords, sentiment, or themes.
- **Why Important?**
  - Enables fast search in Step 6 (e.g., “find positive dog scenes”).
  - Organizes multimodal data (Modalité) for unified access.
  - Example: An inverted index maps “dog” to segment IDs for instant lookup.

## 3. Techniques Used
- **1. Metadata Storage**:
  - **Tool**: SQLite database.
  - **How**: Stores metadata in a table with segment IDs, video paths, timestamps, keywords, sentiment, and cluster IDs.
  - **Why**: Provides persistent, queryable storage for indexing and retrieval.
- **2. Bag-of-Words Representation**:
  - **Tool**: `scikit-learn`’s `CountVectorizer`.
  - **How**: Converts keywords into frequency vectors (e.g., “dog: 2, park: 1”).
  - **Why**: Simplifies text features for clustering and indexing (Bag-of-Words).
- **3. Inverted Index Creation**:
  - **Tool**: Whoosh search library.
  - **How**: Maps keywords and sentiment to segment IDs and timestamps (Fichier inverse).
  - **Why**: Enables fast text-based search (e.g., “find ‘dog’ segments”).
- **4. Clustering**:
  - **Tool**: `scikit-learn`’s K-means.
  - **How**: Groups segments by similar keywords into clusters (e.g., “dog-related” cluster).
  - **Why**: Organizes segments for thematic browsing or index efficiency (Clustering).
- **5. Modality Fusion**:
  - **How**: Boosts keywords present in multiple modalities (e.g., text and visual) in the index (Fusion des modalités).
  - **Why**: Improves search accuracy by combining multimodal features.

## 4. Importance of Step 5
- **Enables Fast Retrieval**:
  - Inverted index allows instant lookup of segments by keywords or sentiment.
  - Example: Query “dog” returns segment IDs [1, 3] in milliseconds.
- **Supports Multimodal Search**:
  - Fusion of modalities ensures accurate results (e.g., “dog” in text and visual is prioritized).
- **Organizes Content**:
  - Clustering groups similar segments, aiding browsing (e.g., “show all dog scenes”).
- **Prepares for Step 6**:
  - Index and database support query processing and result ranking.

## 5. Challenges
- **Data Complexity**: Handling multimodal data requires careful schema design.
- **Scalability**: Large datasets need optimized indexing (e.g., Whoosh vs. Elasticsearch).
- **Solution**: Used lightweight tools (SQLite, Whoosh) and simulated modality fusion.

## 6. Demo (Example Output)
- **Input**: Metadata JSON with segments:
  - Segment 1: Keywords: ["dog", "run", "park"], Sentiment: Positive.
  - Segment 2: Keywords: ["sun", "set"], Sentiment: Neutral.
- **Output**:
  - **SQLite Database**: Table with segment metadata and cluster IDs.
  - **Whoosh Index**: Maps “dog” to segment 1, “sun” to segment 2, etc.
  - **Clusters**: Segment 1 in “dog-related” cluster, Segment 2 in “nature” cluster.
- **Example Query**: Search “dog” returns Segment 1 (0.0–5.0s).

## 7. Conclusion
- **Summary**: Step 5 stores metadata and builds an inverted index, enabling fast, multimodal search.
- **Next Steps**: Implement search and retrieval (Step 6) using the index.
- **Key Takeaway**: Indexing organizes multimodal data for efficient, accurate retrieval.

## 8. Questions?
- Ready to explain techniques (e.g., inverted index, K-means) or demo details!