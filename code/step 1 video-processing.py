import cv2
import os
import numpy as np
import soundfile as sf
import ffmpeg
import json
from pathlib import Path
import logging

# Set up logging with console and file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_indexing.log', mode='w'),
        logging.StreamHandler()
    ]
)

# Custom logger for Step 1
step1_logger = logging.getLogger('step1')

def check_and_convert_video(input_path, output_path):
    """
    Check if video is MP4; convert if not using FFmpeg.
    Returns: Path to MP4 video.
    """
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

def extract_frames(video_path, frame_folder, fps_target=1, resize_dim=(640, 360)):
    """
    Extract frames at a specified FPS and resize them.
    Saves timestamps in a JSON file.
    Returns: List of frame metadata.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        step1_logger.error(f"Cannot open video {video_path}")
        return []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / fps_target)  # Frames to skip for target FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    Path(frame_folder).mkdir(exist_ok=True)
    frame_metadata = []
    frame_num = 0
    saved_frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % frame_interval == 0:
            # Resize frame
            frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)
            # Save frame
            frame_path = os.path.join(frame_folder, f"frame_{saved_frame_num}.jpg")
            cv2.imwrite(frame_path, frame)
            # Store metadata
            timestamp = frame_num / fps
            frame_metadata.append({
                'frame_num': saved_frame_num,
                'original_frame_num': frame_num,
                'timestamp': timestamp,
                'path': frame_path
            })
            saved_frame_num += 1
        
        frame_num += 1
    
    cap.release()
    # Save metadata
    metadata_path = os.path.join(frame_folder, 'frame_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(frame_metadata, f, indent=2)
    
    step1_logger.info(f"Extracted {saved_frame_num} frames at {fps_target} FPS in '{frame_folder}'")
    return frame_metadata

def extract_keyframes_from_frames(frame_folder, keyframe_folder, resize_dim=(640, 360)):
    """
    Extract keyframes from already extracted frames using HSV histogram differences.
    Saves timestamps in a JSON file.
    Returns: List of keyframe metadata.
    """
    # Load frame metadata
    metadata_path = os.path.join(frame_folder, 'frame_metadata.json')
    if not os.path.exists(metadata_path):
        step1_logger.error(f"Frame metadata not found: {metadata_path}")
        return []
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            frame_metadata = json.load(f)
        step1_logger.info(f"Loaded {len(frame_metadata)} frames from {metadata_path}")
    except Exception as e:
        step1_logger.error(f"Failed to load frame metadata: {str(e)}")
        return []
    
    if not frame_metadata:
        step1_logger.error("No frames available for keyframe extraction")
        return []
    
    # Compute histogram differences
    hist_diffs = []
    prev_hist = None
    for frame_info in frame_metadata:
        frame_path = frame_info['path']
        frame = cv2.imread(frame_path)
        if frame is None:
            step1_logger.warning(f"Cannot read frame {frame_path}")
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [64, 64, 64], [0, 180, 0, 256, 0, 256])
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            hist_diffs.append(diff)
        prev_hist = hist
    
    # Set stricter threshold (90th percentile)
    hist_threshold = np.percentile(hist_diffs, 90) if hist_diffs else 0.3
    step1_logger.info(f"Histogram threshold set to {hist_threshold:.4f}")
    
    # Extract keyframes
    Path(keyframe_folder).mkdir(exist_ok=True)
    keyframe_metadata = []
    prev_hist = None
    saved_keyframe_num = 0
    
    for frame_info in frame_metadata:
        frame_path = frame_info['path']
        frame = cv2.imread(frame_path)
        if frame is None:
            step1_logger.warning(f"Cannot read frame {frame_path}")
            continue
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [64, 64, 64], [0, 180, 0, 256, 0, 256])
        
        is_keyframe = False
        if prev_hist is None:  # First frame
            is_keyframe = True
        else:
            hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if hist_diff > hist_threshold:
                is_keyframe = True
        
        if is_keyframe:
            # Copy frame as keyframe
            keyframe_path = os.path.join(keyframe_folder, f"keyframe_{saved_keyframe_num}.jpg")
            cv2.imwrite(keyframe_path, frame)
            # Store metadata
            keyframe_metadata.append({
                'keyframe_num': saved_keyframe_num,
                'frame_num': frame_info['frame_num'],
                'original_frame_num': frame_info['original_frame_num'],
                'timestamp': frame_info['timestamp'],
                'path': keyframe_path
            })
            step1_logger.info(f"Saved keyframe {saved_keyframe_num} from frame {frame_info['frame_num']} (t={frame_info['timestamp']:.2f}s)")
            saved_keyframe_num += 1
        
        prev_hist = hist
    
    # Save metadata
    metadata_path = os.path.join(keyframe_folder, 'keyframe_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(keyframe_metadata, f, indent=2)
    
    step1_logger.info(f"Extracted {saved_keyframe_num} keyframes in '{keyframe_folder}'")
    return keyframe_metadata

if __name__ == "__main__":
    video_path = "howl_scene.mp4"
    audio_path = "howl_scene_audio_mono.wav"
    frame_folder = "frames"
    keyframe_folder = "keyframes"
    
    # Extract audio (mono, 16kHz)
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream.audio, audio_path, ar=16000, ac=1, format='wav', loglevel='error')
        ffmpeg.run(stream)
        step1_logger.info(f"Extracted audio to {audio_path}")
    except Exception as e:
        step1_logger.warning(f"Error extracting audio: {str(e)}")
        audio_path = None
    
    # Check and convert video
    try:
        video_path = check_and_convert_video(video_path, "howl_scene_converted.mp4")
    except Exception as e:
        step1_logger.error(f"Video conversion failed: {str(e)}")
    
    # Extract frames
    try:
        frame_metadata = extract_frames(video_path, frame_folder, fps_target=1)
    except Exception as e:
        step1_logger.error(f"Frame extraction failed: {str(e)}")
        frame_metadata = []
    
    # Extract keyframes from frames
    try:
        keyframe_metadata = extract_keyframes_from_frames(frame_folder, keyframe_folder)
    except Exception as e:
        step1_logger.error(f"Keyframe extraction failed: {str(e)}")
        keyframe_metadata = []