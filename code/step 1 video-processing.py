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

def extract_audio_energy(audio_path, hop_length=512):
    """
    Extract audio energy to detect significant changes using soundfile.
    Returns: List of energy values or None if failed.
    """
    try:
        # Use soundfile to load audio
        y, sr = sf.read(audio_path)
        # Compute short-time energy
        energy = np.array([
            np.sum(np.abs(y[i:i+hop_length]**2))
            for i in range(0, len(y), hop_length)
        ])
        step1_logger.info(f"Extracted audio energy from {audio_path} with sample rate {sr}")
        return energy
    except Exception as e:
        step1_logger.warning(f"Failed to extract audio energy from {audio_path}: {str(e)}. Proceeding without audio cues.")
        return None

def extract_keyframes(video_path, keyframe_folder, audio_path=None, resize_dim=(640, 360)):
    """
    Extract keyframes using visual (HSV histogram) and optional audio (energy) cues.
    Saves timestamps in a JSON file.
    Returns: List of keyframe metadata.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        step1_logger.error(f"Cannot open video {video_path}")
        return []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extract audio energy if provided
    audio_energy = extract_audio_energy(audio_path) if audio_path else None
    energy_per_frame = None
    energy_threshold = 0
    if audio_energy is not None:
        # Normalize energy to match frame rate
        energy_per_frame = np.interp(
            np.arange(0, total_frames) * (len(audio_energy) / total_frames),
            np.arange(len(audio_energy)),
            audio_energy
        )
        # Compute energy differences
        energy_diff = np.diff(energy_per_frame, prepend=energy_per_frame[0])
        energy_threshold = np.percentile(energy_diff, 90)  # Top 10% changes
        step1_logger.info(f"Audio energy threshold set to {energy_threshold:.4f}")
    else:
        step1_logger.info("No audio energy data available; using visual histograms only")
    
    # Compute dynamic histogram threshold
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
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video
    
    hist_threshold = np.percentile(hist_diffs, 80) if hist_diffs else 0.3
    step1_logger.info(f"Histogram threshold set to {hist_threshold:.4f}")
    
    # Extract keyframes
    Path(keyframe_folder).mkdir(exist_ok=True)
    keyframe_metadata = []
    prev_hist = None
    frame_num = 0
    saved_keyframe_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [64, 64, 64], [0, 180, 0, 256, 0, 256])
        
        is_keyframe = False
        if prev_hist is None:  # First frame
            is_keyframe = True
        else:
            hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if hist_diff > hist_threshold:
                is_keyframe = True
            if audio_energy is not None and frame_num < len(energy_diff):
                if energy_diff[frame_num] > energy_threshold:
                    is_keyframe = True
                    step1_logger.info(f"Audio-triggered keyframe at frame {frame_num} (t={frame_num/fps:.2f}s)")
        
        if is_keyframe:
            # Resize frame
            frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)
            # Save keyframe
            keyframe_path = os.path.join(keyframe_folder, f"keyframe_{saved_keyframe_num}.jpg")
            cv2.imwrite(keyframe_path, frame)
            # Store metadata
            timestamp = frame_num / fps
            keyframe_metadata.append({
                'keyframe_num': saved_keyframe_num,
                'frame_num': frame_num,
                'timestamp': timestamp,
                'path': keyframe_path
            })
            step1_logger.info(f"Saved keyframe {saved_keyframe_num} at frame {frame_num} (t={timestamp:.2f}s)")
            saved_keyframe_num += 1
        
        prev_hist = hist
        frame_num += 1
    
    cap.release()
    # Save metadata
    metadata_path = os.path.join(keyframe_folder, 'keyframe_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(keyframe_metadata, f, indent=2)
    
    step1_logger.info(f"Extracted {saved_keyframe_num} keyframes in '{keyframe_folder}'")
    return keyframe_metadata

if __name__ == "__main__":
    video_path = "howl_scene.mp4"
    audio_path = "howl_scene_audio.wav"
    frame_folder = "frames"
    keyframe_folder = "keyframes"
    
    # Extract audio for multimodal keyframe detection
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream.audio, audio_path, loglevel='error')
        ffmpeg.run(stream)
        step1_logger.info(f"Extracted audio to {audio_path}")
    except Exception as e:
        step1_logger.warning(f"Error extracting audio: {str(e)}. Proceeding without audio cues.")
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
    
    # Extract keyframes
    try:
        keyframe_metadata = extract_keyframes(video_path, keyframe_folder, audio_path)
    except Exception as e:
        step1_logger.error(f"Keyframe extraction failed: {str(e)}")
        keyframe_metadata = []