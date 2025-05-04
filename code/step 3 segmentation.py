import cv2
import numpy as np
from sklearn.cluster import KMeans
import json
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('segmentation.log', mode='w'),
        logging.StreamHandler()
    ]
)
seg_logger = logging.getLogger('segmentation')

def detect_shots(video_path, hist_threshold=0.3, motion_threshold=5.0):
    """Detect shot boundaries using histogram differences and motion analysis."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        seg_logger.info(f"Video: {video_path}, FPS: {fps}, Frames: {frame_count}, Size: {frame_width}x{frame_height}")

        prev_hist = None
        prev_frame = None
        shot_boundaries = [0]  # Start at frame 0
        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Histogram-based detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # Motion-based detection
            motion_magnitude = 0.0
            if prev_frame is not None:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
                if corners is not None:
                    lk_params = dict(winSize=(15, 15), maxLevel=2)
                    new_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None, **lk_params)
                    status = status.ravel()
                    motion_vectors = new_corners[status == 1] - corners[status == 1]
                    magnitudes = np.sqrt(np.sum(motion_vectors**2, axis=1))
                    motion_magnitude = float(np.mean(magnitudes)) if len(magnitudes) > 0 else 0.0

            # Detect shot boundary
            hist_diff = 0.0
            if prev_hist is not None:
                hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if hist_diff > hist_threshold or motion_magnitude > motion_threshold:
                    shot_boundaries.append(frame_number)
                    seg_logger.info(f"Shot boundary at frame {frame_number} (hist_diff: {hist_diff:.2f}, motion: {motion_magnitude:.2f})")

            prev_hist = hist
            prev_frame = frame.copy()
            frame_number += 1

        shot_boundaries.append(frame_number)  # End of video
        cap.release()
        return shot_boundaries, fps, frame_width, frame_height
    except Exception as e:
        seg_logger.error(f"Shot detection failed: {str(e)}")
        return [], 0, 0, 0

def group_shots_into_scenes(shot_boundaries, features_metadata, num_scenes=5):
    """Group shots into scenes using K-means clustering on Step 2 features."""
    try:
        if not features_metadata:
            raise ValueError("No features metadata provided")

        # Extract features for clustering (dominant colors, motion, RMS energy)
        features = []
        shot_timestamps = []
        for i in range(len(shot_boundaries) - 1):
            start_frame = shot_boundaries[i]
            end_frame = shot_boundaries[i + 1]
            start_time = start_frame / 25.0  # Assuming 25 FPS (adjust if needed)
            # Find closest keyframe
            closest_keyframe = min(
                features_metadata,
                key=lambda x: abs(x['timestamp'] - start_time),
                default=None
            )
            if closest_keyframe:
                # Use dominant colors (flatten RGB), motion, RMS energy
                colors = np.array(closest_keyframe['visual']['dominant_colors']).flatten()
                motion = closest_keyframe['visual']['motion_magnitude']
                rms = closest_keyframe['audio']['rms_energy']
                features.append(np.concatenate([colors, [motion, rms]]))
                shot_timestamps.append(start_time)
            else:
                features.append(np.zeros(10))  # 3 colors * 3 RGB + motion + RMS
                shot_timestamps.append(start_time)

        if not features:
            raise ValueError("No valid features for clustering")

        # Cluster shots into scenes
        features = np.array(features)
        kmeans = KMeans(n_clusters=min(num_scenes, len(features)), random_state=0, n_init=10)
        labels = kmeans.fit_predict(features)
        scenes = []
        for i, label in enumerate(labels):
            scenes.append({
                'shot_index': i,
                'start_frame': shot_boundaries[i],
                'end_frame': shot_boundaries[i + 1],
                'start_time': shot_timestamps[i],
                'scene_label': int(label)
            })
        seg_logger.info(f"Grouped {len(scenes)} shots into {num_scenes} scenes")
        return scenes
    except Exception as e:
        seg_logger.error(f"Scene grouping failed: {str(e)}")
        return []

def detect_events(features_metadata, motion_threshold=5.0, face_threshold=2):
    """Detect events based on motion, dialogue, and faces."""
    try:
        events = []
        for kf in features_metadata:
            motion = kf['visual']['motion_magnitude']
            transcript = kf['audio']['transcript'].strip()
            num_faces = len(kf['visual']['faces'])
            is_event = False
            event_type = []

            # High motion
            if motion > motion_threshold:
                is_event = True
                event_type.append('action')
            # Significant dialogue
            if transcript and len(transcript.split()) > 3:  # At least 3 words
                is_event = True
                event_type.append('dialogue')
            # Multiple faces
            if num_faces >= face_threshold:
                is_event = True
                event_type.append('social')

            if is_event:
                events.append({
                    'keyframe_num': kf['keyframe_num'],
                    'timestamp': kf['timestamp'],
                    'frame_num': kf['frame_num'],
                    'event_types': event_type,
                    'motion': motion,
                    'transcript': transcript,
                    'num_faces': num_faces
                })
                seg_logger.info(f"Event at t={kf['timestamp']:.2f}s: {event_type}, transcript: {transcript[:50]}...")

        seg_logger.info(f"Detected {len(events)} events")
        return events
    except Exception as e:
        seg_logger.error(f"Event detection failed: {str(e)}")
        return []

def save_segments(video_path, shot_boundaries, fps, frame_width, frame_height):
    """Save video segments for each shot."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {video_path}")

        os.makedirs("segments", exist_ok=True)
        segment_metadata = []

        for i in range(len(shot_boundaries) - 1):
            start_frame = shot_boundaries[i]
            end_frame = shot_boundaries[i + 1]
            output_path = f"segments/shot_{i}.mp4"

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (frame_width, frame_height)
            )

            for f in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            out.release()
            segment_metadata.append({
                'shot_index': i,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_frame / fps,
                'end_time': end_frame / fps,
                'output_path': output_path
            })
            seg_logger.info(f"Saved shot {i} from frame {start_frame} to {end_frame} as {output_path}")

        cap.release()
        return segment_metadata
    except Exception as e:
        seg_logger.error(f"Segment saving failed: {str(e)}")
        return []

def main(video_path, features_metadata_path):
    """Main function for video segmentation: shots, scenes, events."""
    try:
        # Load Step 2 features
        with open(features_metadata_path, 'r') as f:
            features_metadata = json.load(f)
        if not features_metadata:
            raise ValueError("No features found in metadata")

        # Detect shots
        shot_boundaries, fps, frame_width, frame_height = detect_shots(
            video_path, hist_threshold=0.3, motion_threshold=5.0
        )
        if not shot_boundaries:
            raise ValueError("No shots detected")

        # Group shots into scenes
        scenes = group_shots_into_scenes(shot_boundaries, features_metadata, num_scenes=5)

        # Detect events
        events = detect_events(
            features_metadata, motion_threshold=5.0, face_threshold=2
        )

        # Save segments
        segment_metadata = save_segments(
            video_path, shot_boundaries, fps, frame_width, frame_height
        )

        # Combine metadata
        output_metadata = {
            'shots': segment_metadata,
            'scenes': scenes,
            'events': events,
            'video_info': {
                'path': video_path,
                'fps': fps,
                'frame_width': frame_width,
                'frame_height': frame_height,
                'total_frames': shot_boundaries[-1]
            }
        }

        # Save metadata
        output_path = 'segmentation_metadata.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_metadata, f, indent=2)
        seg_logger.info(f"Saved segmentation metadata to {output_path}")

        return output_metadata
    except Exception as e:
        seg_logger.error(f"Segmentation failed: {str(e)}")
        return {}

if __name__ == "__main__":
    video_path = "howl_scene.mp4"
    features_metadata_path = "features_metadata_vosk.json"

    # Verify inputs
    if not os.path.exists(video_path):
        seg_logger.error(f"Video not found: {video_path}")
    elif not os.path.exists(features_metadata_path):
        seg_logger.error(f"Features metadata not found: {features_metadata_path}")
    else:
        main(video_path, features_metadata_path)