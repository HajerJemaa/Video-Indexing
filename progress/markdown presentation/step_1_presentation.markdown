# Presentation: Step 1 - Video Processing

## 1. Introduction
- **What is Video Indexing?**
  - Organizing videos to enable fast search and retrieval of specific moments.
  - Example: Finding all "dog" scenes in a video efficiently.
- **Our Project**:
  - A six-step process: video processing, feature extraction, segmentation, metadata consolidation, indexing, and search/retrieval.
  - Multimodal: Leverages visual (frames, keyframes), audio (sound energy), and text (later steps) data (Modalité).
- **Step 1 Overview**:
  - Prepares the video for analysis by converting formats, extracting frames, and identifying keyframes.

## 2. Purpose of Step 1: Video Processing
- **Goal**:
  - Convert videos to a standard MP4 format.
  - Extract frames for visual analysis.
  - Identify keyframes using visual and audio cues for efficient processing.
- **Why Important?**
  - Ensures compatibility across video formats.
  - Provides raw visual and audio data for feature extraction (Step 2) and segmentation (Step 3).
  - Reduces data volume by selecting keyframes, making downstream processing faster.
  - Example: Keyframes capture significant scene changes (e.g., a dog entering a park).

## 3. Techniques Used
### 3.1 Video Format Conversion
- **Tool**: FFmpeg (`ffmpeg-python`).
- **How**: Converts non-MP4 videos to MP4 using `libx264` (video) and `aac` (audio) codecs.
- **Why**: Standardizes input for consistent processing in later steps.
- **Code Snippet**:
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
- **Explanation**: Checks if the video is MP4; if not, uses FFmpeg to convert it to MP4 with H.264 video and AAC audio codecs. Logs success or errors for debugging. This ensures all videos are in a format compatible with tools like OpenCV.

### 3.2 Frame Extraction
- **Tool**: OpenCV (`cv2`).
- **How**: Extracts frames at 1 FPS, resizes to 640x360, and saves with timestamps in JSON.
- **Why**: Provides visual data for analysis (e.g., object detection in Step 2).
- **Code Snippet**:
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
- **Explanation**: Uses OpenCV to read video frames, extracts one frame per second (1 FPS), resizes for efficiency, and saves as JPEG. Stores frame paths and timestamps in JSON for later use. This creates a manageable set of images for feature extraction.

### 3.3 Audio Energy Extraction
- **Tool**: `soundfile`.
- **How**: Computes short-time energy of audio samples to detect significant changes (e.g., loud sounds).
- **Why**: Supports multimodal keyframe detection by identifying audio events (Modalité).
- **Code Snippet**:
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
- **Explanation**: Loads audio with `soundfile`, calculates energy in 512-sample windows, and returns an array of energy values. This detects audio events (e.g., a howl) to trigger keyframes, enhancing multimodal analysis. Errors are logged to proceed without audio if needed.

### 3.4 Keyframe Extraction
- **Tool**: OpenCV for visual cues, `soundfile` for audio cues.
- **How**:
  - **Visual Cues**: Uses HSV histogram differences (Bhattacharyya distance) to detect scene changes.
  - **Audio Cues**: Uses energy differences to trigger keyframes during significant audio events.
  - Saves keyframes with timestamps in JSON.
- **Why**: Selects representative frames, reducing data while capturing key moments.
- **Code Snippet**:
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
- **Explanation**: Analyzes frames in HSV color space, computes histogram differences to detect scene changes, and selects keyframes when differences exceed the 80th percentile threshold. Optionally uses audio energy (not shown here for brevity) to trigger keyframes. Saves keyframes as JPEGs with JSON metadata. This reduces data for efficient processing.

## 4. Importance of Step 1
- **Enables Downstream Steps**:
  - Frames and keyframes feed into feature extraction (Step 2) and segmentation (Step 3).
  - Example: Keyframes with a dog are analyzed for objects or text.
- **Supports Multimodal Processing**:
  - Combines visual (histograms) and audio (energy) cues for robust keyframe detection (Modalité).
- **Improves Efficiency**:
  - Keyframes reduce the number of frames processed, saving time and resources.
- **Ensures Compatibility**:
  - MP4 conversion guarantees consistent input for tools like OpenCV and FFmpeg.

## 5. Challenges
- **Video Compatibility**: Non-MP4 formats require conversion, which can fail for corrupted files.
- **Parameter Tuning**: Histogram and audio energy thresholds need adjustment for different videos.
- **Solution**: Used robust libraries (FFmpeg, OpenCV) and dynamic thresholding (e.g., 80th percentile for histograms).

## 6. Demo (Example Output)
- **Input**: Video `howl_scene.mp4`.
- **Output**:
  - **Converted Video**: `howl_scene_converted.mp4` (if needed).
  - **Frames**: Folder `frames` with images (e.g., `frame_0.jpg`) and `frame_metadata.json`:
    ```json
    [
      {
        "frame_num": 0,
        "timestamp": 0.0,
        "path": "frames/frame_0.jpg"
      },
      ...
    ]
    ```
  - **Keyframes**: Folder `keyframes` with images (e.g., `keyframe_0.jpg`) and `keyframe_metadata.json`:
    ```json
    [
      {
        "keyframe_num": 0,
        "frame_num": 0,
        "timestamp": 0.0,
        "path": "keyframes/keyframe_0.jpg"
      },
      ...
    ]
    ```
- **Example**: Keyframe at 2.5s captures a scene change (e.g., dog appears) triggered by visual histogram or audio howl.

## 7. Conclusion
- **Summary**: Step 1 processes videos by converting formats, extracting frames, and selecting keyframes using visual and audio cues.
- **Next Steps**: Use frames/keyframes for feature extraction (Step 2) and segmentation (Step 3).
- **Key Takeaway**: Video processing lays the foundation for multimodal indexing by preparing visual and audio data.

## 8. Questions?
- Ready to explain techniques (e.g., HSV histograms, audio energy) or demo outputs!