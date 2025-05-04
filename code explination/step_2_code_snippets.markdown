# Step 2: Feature Extraction - Code Snippets with Explanations

## Snippet 1: Object and Face Detection
**Code**:
```python
import cv2
import torch
def detect_objects_faces(image_path):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    img = cv2.imread(image_path)
    results = model(img)
    objects = [label for label, conf in results.xyxy[0][:, -2:].tolist() if conf > 0.5]
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return {"objects": objects, "faces": len(faces)}
```

**Explanation**:
This code is like a super-smart art critic! It uses YOLOv5 to spot objects (like “dog” or “car”) in a picture and OpenCV to find faces, like picking out stars in a crowd. It’s our way of saying, “Hey, there’s a dog and two faces in this scene!” so we can tag them for search later.

## Snippet 2: Color and Movement Detection
**Code**:
```python
import cv2
import numpy as np
def detect_color_movement(curr_frame_path, prev_frame_path=None):
    img = cv2.imread(curr_frame_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [8], [0, 180])
    dominant_color = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown'][np.argmax(hist)]
    movement = 'none'
    if prev_frame_path:
        curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(cv2.imread(prev_frame_path), cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(curr_gray, prev_gray)
        movement = 'high' if np.mean(diff) > 10 else 'low'
    return {"colors": [dominant_color], "movement": movement}
```

**Explanation**:
Think of this code as a painter with a motion sensor! It checks a frame’s main color (like “green” for a park) using HSV histograms and spots movement by comparing it to the previous frame, like catching a dog zooming by. This helps us describe scenes as “green and busy” for indexing.

## Snippet 3: Speech-to-Text
**Code**:
```python
import speech_recognition as sr
def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return {"spoken_text": text}
    except sr.UnknownValueError:
        return {"spoken_text": ""}
```

**Explanation**:
This code is like a super listener! It takes audio and uses Google’s speech-to-text to turn words, like “A dog runs in the park,” into text we can search. If it can’t hear clearly, it just says “nothing here” and moves on, keeping our tags ready for later.

## Snippet 4: Audio Event Detection
**Code**:
```python
import librosa
def detect_audio_event(audio_path):
    y, sr = librosa.load(audio_path)
    rms = librosa.feature.rms(y=y)[0]
    if np.mean(rms) < 0.01:
        return {"event": "silence"}
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    return {"event": "music" if np.mean(spectral_centroid) > 1000 else "noise"}
```

**Explanation**:
This code’s like a DJ analyzing the vibe! It uses Librosa to check if audio is silent, noisy, or musical by measuring energy and sound “brightness.” A loud howl might be “noise,” while a quiet moment is “silence,” helping us tag scenes with the right mood.

## Snippet 5: On-Screen Text Detection
**Code**:
```python
import pytesseract
from PIL import Image
def detect_onscreen_text(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return {"onscreen_text": text.strip() if text.strip() else ""}
```

**Explanation**:
This code’s like a speed-reader for screens! It uses Tesseract to grab text from frames, like “Dog park scene” in subtitles, turning pictures into searchable words. If there’s no text, it gives us a blank page, ready for the next step in our indexing adventure.