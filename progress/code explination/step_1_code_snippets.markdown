# Step 1: Video Processing - Code Snippets with Explanations

## Snippet 1: Video Format Conversion
**Code**:
```python
def check_and_convert_video(input_path, output_path):
    if input_path.endswith('.mp4'):
        return input_path
    try:
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(stream, output_path, vcodec='libx264', acodec='aac', loglevel='error')
        ffmpeg.run(stream)
        step1_logger.info(f"Converted {input_path} to {output_path}")
        return output_path
    except Exception as e:
        step1_logger.error(f"Error converting video {input_path}: {str(e)}")
        return input_path
```

**Explanation**:
This code is like a video translator! It checks if a video is in MP4 format; if not, it uses FFmpeg to convert it to MP4, like turning a foreign book into English so everyone can read it. MP4 is the “universal language” for our tools, ensuring they work smoothly. If something goes wrong, it logs the error and keeps going, like a trusty librarian.

## Snippet 2: Frame Extraction
**Code**:
```python
def extract_frames(video_path, frame_folder, fps_target=1, resize_dim=(640, 360)):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / fps_target)
    Path(frame_folder).mkdir(exist_ok=True)
    frame_metadata = []
    frame_num = 0
    saved_frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)
            frame_path = os.path.join(frame_folder, f"frame_{saved_frame_num}.jpg")
            cv2.imwrite(frame_path, frame)
            timestamp = frame_num / fps
            frame_metadata.append({
                'frame_num': saved_frame_num,
                'timestamp': timestamp,
                'path': frame_path
            })
            saved_frame_num += 1
        frame_num += 1
    metadata_path = os.path.join(frame_folder, 'frame_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(frame_metadata, f, indent=2)
    return frame_metadata
```

**Explanation**:
Think of this code as a photographer snapping pictures from a movie! It grabs one frame every second (1 FPS) using OpenCV, shrinks them to a handy size (640x360), and saves them as JPEGs. It also jots down each frame’s timestamp in a JSON file, like a scrapbook index, so we know exactly when each “photo” was taken for later analysis.

## Snippet 3: Audio Energy Extraction
**Code**:
```python
def extract_audio_energy(audio_path, hop_length=512):
    try:
        y, sr = sf.read(audio_path)
        energy = np.array([
            np.sum(np.abs(y[i:i+hop_length]**2))
            for i in range(0, len(y), hop_length)
        ])
        step1_logger.info(f"Extracted audio energy from {audio_path} with sample rate {sr}")
        return energy
    except Exception as e:
        step1_logger.warning(f"Failed to extract audio from {audio_path}: {str(e)}")
        return None
```

**Explanation**:
This code is like a sound detective! It listens to the video’s audio using `soundfile` and measures how “loud” it is in small chunks (512 samples). Loud moments, like a dog’s howl, get high energy scores, helping us spot important scenes. If the audio’s missing, it shrugs and moves on, keeping our process on track.

## Snippet 4: Keyframe Extraction
**Code**:
```python
def extract_keyframes(video_path, keyframe_folder, audio_path=None, resize_dim=(640, 360)):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    hist_diffs = []
    prev_hist = None
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [64, 64, 64], [0, 180, 0, 256, 0, 256])
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            hist_diffs.append(diff)
        prev_hist = hist
        frame_num += 1
    hist_threshold = np.percentile(hist_diffs, 80) if hist_diffs else 0.3
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    keyframe_metadata = []
    saved_keyframe_num = 0
    prev_hist = None
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [64, 64, 64], [0, 180, 0, 256, 0, 256])
        is_keyframe = False
        if prev_hist is None:
            is_keyframe = True
        else:
            hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if hist_diff > hist_threshold:
                is_keyframe = True
        if is_keyframe:
            frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)
            keyframe_path = os.path.join(keyframe_folder, f"keyframe_{saved_keyframe_num}.jpg")
            cv2.imwrite(keyframe_path, frame)
            timestamp = frame_num / fps
            keyframe_metadata.append({
                'keyframe_num': saved_keyframe_num,
                'frame_num': frame_num,
                'timestamp': timestamp,
                'path': keyframe_path
            })
            saved_keyframe_num += 1
        prev_hist = hist
        frame_num += 1
    metadata_path = os.path.join(keyframe_folder, 'keyframe_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(keyframe_metadata, f, indent=2)
    return keyframe_metadata
```

**Explanation**:
This code is like a movie director picking the best scenes! It uses OpenCV to compare frames’ colors (in HSV) and saves a “keyframe” when the scene changes a lot, like when a dog runs into view. It also checks for loud audio moments (like a howl) to pick keyframes, acting like a smart editor who knows what’s important, and saves everything in a JSON file for later.