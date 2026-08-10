import cv2
import os
import numpy as np
from memory_tracker import YoloMemoryTracker

def debug_full():
    tracker = YoloMemoryTracker(yolo_model='yolov8m.pt', reid_threshold=0.68, conf_threshold=0.35)

    print("=== Full Dataset Debug (Frames 1 to 89) ===")
    for idx in range(1, 90):
        fpath = f"frames/frame_{idx:04d}.jpg"
        if not os.path.exists(fpath):
            continue
        frame = cv2.imread(fpath)
        
        # Crop & embedding inspection
        results = tracker.process_frame(frame, idx)
        for r in results:
            if r['id'] == 3:
                print(f"[ID 3 APPEARED] Frame {idx:03d} ({fpath}) -> Box: {r['bbox']} | Conf: {r['conf']:.2f} | ReID Score: {r['reid_score']:.2f}")

if __name__ == '__main__':
    debug_full()
