from ultralytics import YOLO
import supervision as sv
import cv2
import pickle
import os
import numpy as np

class Tracking:
    def __init__(self, model):
        self.model = model
        self.trac = sv.ByteTrack()

    def detect_frame(self, frame):
        batch_size = 4
        detections = []
        for i in range(0, len(frame), batch_size):
            batch = frame[i:i+batch_size]
            detect_batch = self.model.predict(batch, conf=0.2)
            detections.extend(detect_batch) 
        return detections

    def draw_rect(self, frame, bbox, track_id):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(137, 250, 7), thickness=2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame

    def get_object(self, frame, read_from_stub=False, stub_path=None):
        
        if isinstance(frame, list):
            frame = np.array(frame)  

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frame(frame)
        tracks = {'licence': []}

        for frame_num, detection in enumerate(detections):
            sv_detection = sv.Detections.from_ultralytics(detection)
            detection_track = self.trac.update_with_detections(sv_detection)

            track_data = {}
            if detection_track.xyxy.size > 0:
                for i in range(len(detection_track.xyxy)):
                    bbox = detection_track.xyxy[i].tolist()
                    track_id = int(detection_track.tracker_id[i])

                    
                    x1, y1, x2, y2 = map(int, bbox)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[2], x2)  
                    y2 = min(frame.shape[1], y2)

                    
                    roi = frame[y1:y2, x1:x2]

                    track_data[track_id] = {'bbox': (x1, y1, x2, y2)}
                    print(f'Track ID: {track_id}, BBox (xyxy): {(x1, y1, x2, y2)}')

            tracks['licence'].append(track_data)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks

    def annotion(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            if frame_num < len(tracks['licence']):
                licence_dict = tracks['licence'][frame_num]
                for track_id, licence in licence_dict.items():
                    frame = self.draw_rect(frame, licence['bbox'], track_id)

            output_video_frames.append(frame)
        return output_video_frames

def downscale_frame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def process_video_in_chunks(frames, chunk_size=100):
    chunks = [frames[i:i + chunk_size] for i in range(0, len(frames), chunk_size)]
    return [np.array(chunk) for chunk in chunks]
