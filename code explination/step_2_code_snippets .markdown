# Step 2: Feature Extraction - Code Snippets with Explanations

## Snippet 1: Dominant Color Extraction
**Code**:
```python
def extract_dominant_colors(image_path, k=3):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(image)
    colors = kmeans.cluster_centers_.astype(int)
    return colors
```

**Explanation**:
This code’s like an artist picking a scene’s top colors! It uses K-means to group pixels in a keyframe into three main colors, like choosing the perfect paint shades for a park scene. It helps us tag scenes as “green” or “brown” for searching later.

## Snippet 2: Motion Detection
**Code**:
```python
cap = cv2.VideoCapture('howl_scene.mp4')
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
lk_params = dict(winSize=(15, 15), maxLevel=2)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(prev_gray, 100, 0.3, 7)
    if corners is not None:
        new_corners, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, corners, None, **lk_params)
        for i in range(len(corners)):
            x, y = corners[i].ravel()
            x2, y2 = new_corners[i].ravel()
            cv2.line(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
    prev_gray = gray
```

**Explanation**:
This code’s like a dance tracker! It uses optical flow to follow moving points (like a dog’s paws) between video frames, drawing green lines to show motion. It’s our way of spotting action, like a chase, to tag exciting scenes.

## Snippet 3: Loudness Estimation
**Code**:
```python
sample_rate, audio_data = wav.read("howl_scene_audio.wav")
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)
frame_size = 1024
rms_energy = [np.sqrt(np.mean(audio_data[i:i+frame_size]**2)) for i in range(0, len(audio_data), frame_size)]
```

**Explanation**:
This code’s like a sound meter at a concert! It measures how loud the audio is by calculating RMS energy in small chunks, catching barks or shouts. It helps us find noisy moments, like a dog’s howl, for indexing.

## Snippet 4: Speech-to-Text
**Code**:
```python
recognizer = sr.Recognizer()
with sr.AudioFile("howl_scene_audio.wav") as source:
    audio = recognizer.record(source)
try:
    text = recognizer.recognize_sphinx(audio)
    print("Extracted Speech:", text)
except sr.UnknownValueError:
    print("Sphinx could not understand the audio")
```

**Explanation**:
This code’s like a super-smart scribe! It uses PocketSphinx to turn audio into text, like writing down “Dog runs in park” from a video’s sound. It gives us words to search, making it easy to find dialogue scenes.