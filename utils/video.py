import cv2

def read_video(path):
    cap = cv2.VideoCapture(path)
    frame = []
    while (cap.isOpened()):
        ret, frames = cap.read()
        if not ret:
            break
        frame.append(frames)
    cap.release()
    return frame

def save_video(frame, save_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width = frame[0].shape[:2]
    out = cv2.VideoWriter(filename=save_path,
                    fourcc=fourcc,
                    fps=24,
                    frameSize=(width, height))
    for frames in frame:
        out.write(frames)
    out.release()
