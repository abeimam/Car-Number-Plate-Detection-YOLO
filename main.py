from my_tracking import Tracking, downscale_frame, process_video_in_chunks
from ultralytics import YOLO
from utils import read_video, save_video

def main():
    video_path = "/home/mahi/Project/input/car.mp4"
    output_path = "/home/mahi/Project/output/output_video.avi"
    model_path = "/home/mahi/Project/license_plate_detector.pt"  # Path to your YOLO model

    # Initialize the tracking object
    tracker = Tracking(model=YOLO(model_path))

    # Read the video
    frames = read_video(video_path)

    # Process the video in chunks
    chunked_frames = process_video_in_chunks(frames)
    annotated_frames = []

    for chunk in chunked_frames:
        # Get the tracked objects in the chunk
        tracks = tracker.get_object(chunk)

        # Annotate the frames with bounding boxes and IDs
        annotated_chunk = tracker.annotion(chunk, tracks)
        annotated_frames.extend(annotated_chunk)

    # Save the annotated video
    save_video(annotated_frames, output_path, fps=30, size=(640, 480))

if __name__ == "__main__":
    main()