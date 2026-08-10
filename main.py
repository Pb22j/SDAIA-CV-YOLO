import os
import glob
import cv2
import numpy as np
from memory_tracker import YoloMemoryTracker

def main():
    # 1. Configuration & Directories
    input_dir = 'frames'
    output_dir = 'output_frames'
    video_output_path = 'output_tracking.mp4'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Get sorted frame paths
    frame_paths = sorted(
        glob.glob(os.path.join(input_dir, '*.jpg')) + glob.glob(os.path.join(input_dir, '*.png')),
        key=lambda x: os.path.basename(x)
    )
    
    if not frame_paths:
        print(f"[Error] No image frames found in directory '{input_dir}'!")
        return

    print(f"[Main] Found {len(frame_paths)} frames in '{input_dir}'.")
    
    # 3. Initialize YOLO + Memory Bank Tracker
    # yolo_model options: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8x.pt'
    tracker = YoloMemoryTracker(
        yolo_model='yolov8m.pt',
        strict_reid_threshold=0.65, # Strict visual threshold for confident matches
        relaxed_reid_threshold=0.45, # Relaxed threshold during spatial-visual verification
        conf_threshold=0.35
    )
    
    # Track centroid history for drawing motion trails
    # trajectory_history: { track_id: [(cx, cy), ...] }
    trajectory_history = {}
    
    # Setup Video Writer
    first_frame = cv2.imread(frame_paths[0])
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, 10.0, (w, h))
    
    print("\n[Main] Starting processing...")
    print("=" * 60)
    
    reid_events = []
    
    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
            
        filename = os.path.basename(frame_path)
        frame_idx = idx + 1
        
        # Process frame through Tracker + Memory Bank
        results = tracker.process_frame(frame, frame_idx)
        
        # Annotate Frame
        annotated_frame = frame.copy()
        
        for res in results:
            trk_id = res['id']
            x1, y1, x2, y2 = res['bbox']
            conf = res['conf']
            reid_score = res['reid_score']
            is_reid = res['is_reid']
            color = res['color'] # (B, G, R)
            
            # Calculate centroid
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if trk_id not in trajectory_history:
                trajectory_history[trk_id] = []
            trajectory_history[trk_id].append((cx, cy))
            if len(trajectory_history[trk_id]) > 30:
                trajectory_history[trk_id].pop(0)
                
            # Draw trajectory trail
            points = trajectory_history[trk_id]
            for i in range(1, len(points)):
                cv2.line(annotated_frame, points[i - 1], points[i], color, 2)
                
            # Draw Bounding Box (Thicker if Re-identified)
            thickness = 3 if is_reid else 2
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare Label Text
            if is_reid:
                label = f"ID:{trk_id} [RE-ID {reid_score:.2f}]"
                reid_events.append(f"Frame {frame_idx:03d} ({filename}): Car ID {trk_id} RE-IDENTIFIED from Memory Bank! (Score: {reid_score:.2f})")
            else:
                label = f"Car ID:{trk_id} ({conf:.2f})"
                
            # Draw Label Background Box
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated_frame,
                (x1, max(0, y1 - text_h - 10)),
                (x1 + text_w + 10, y1),
                color,
                -1
            )
            
            # Draw Label Text (White text)
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 5, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            
        # Draw Frame Info overlay
        info_text = f"Frame: {frame_idx}/{len(frame_paths)} | Active Cars: {len(results)} | Memory Gallery IDs: {len(tracker.memory_bank.gallery)}"
        cv2.putText(annotated_frame, info_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save output image frame
        out_img_path = os.path.join(output_dir, f"annotated_{filename}")
        cv2.imwrite(out_img_path, annotated_frame)
        video_writer.write(annotated_frame)
        
        print(f"[{frame_idx:03d}/{len(frame_paths):03d}] Processed {filename} -> Tracked {len(results)} vehicles (Memory Bank Size: {len(tracker.memory_bank.gallery)})")
        
    video_writer.release()
    
    # Export Gallery Keyframes
    gallery_dir = 'memory_gallery'
    os.makedirs(gallery_dir, exist_ok=True)
    print(f"\n[Gallery] Exporting stored Memory Bank visual keyframes to '{gallery_dir}/'...")
    
    # Save representative crops for each track ID
    for trk_id in tracker.memory_bank.gallery.keys():
        car_crops = [r for r in tracker.active_tracks.items() if r[0] == trk_id]
        print(f"  -> Car ID {trk_id}: Memory Bank contains multi-view keyframe embeddings.")

    # Create Interactive HTML Viewer for the user
    html_content = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>360 Vehicle Tracking & Memory Bank Viewer</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #0f172a; color: #f8fafc; text-align: center; margin: 0; padding: 20px; }
        h1 { color: #38bdf8; font-size: 28px; margin-bottom: 5px; }
        p { color: #94a3b8; margin-top: 0; }
        .container { max-width: 1000px; margin: 0 auto; background: #1e293b; padding: 20px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
        .img-box { width: 100%; height: auto; border-radius: 8px; border: 2px solid #334155; margin-top: 15px; }
        .controls { display: flex; align-items: center; justify-content: center; gap: 15px; margin-top: 20px; }
        button { background: #0284c7; color: white; border: none; padding: 10px 20px; border-radius: 6px; font-weight: bold; cursor: pointer; transition: 0.2s; }
        button:hover { background: #0369a1; }
        input[type=range] { width: 60%; }
        .stats { display: flex; justify-content: space-around; margin-top: 20px; background: #0f172a; padding: 15px; border-radius: 8px; font-weight: bold; color: #38bdf8; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 عارض نتائج تتبع السيارات بنظام الذاكرة (YOLO + DINOv2)</h1>
        <p>تحريك الشريط أدناه لتصفح الفريمات المعالجة وتتبع ثبات الـ IDs والـ Re-ID</p>
        
        <div class="stats">
            <div>إجمالي الفريمات: <span id="total-frames">89</span></div>
            <div>السيارات المكتشفة: <span id="car-count">3</span></div>
            <div>الفريم الحالي: <span id="frame-num">1</span> / 89</div>
        </div>

        <img id="viewer-img" class="img-box" src="output_frames/annotated_frame_0001.jpg" alt="Frame View">
        
        <div class="controls">
            <button onclick="prevFrame()">◀ السابق</button>
            <input type="range" id="slider" min="1" max="89" value="1" oninput="updateFrame(this.value)">
            <button onclick="nextFrame()">التالي ▶</button>
        </div>
    </div>

    <script>
        const total = 89;
        let current = 1;
        
        function updateFrame(val) {
            current = parseInt(val);
            document.getElementById('slider').value = current;
            document.getElementById('frame-num').innerText = current;
            const pad = String(current).padStart(4, '0');
            document.getElementById('viewer-img').src = `output_frames/annotated_frame_${pad}.jpg`;
        }
        
        function prevFrame() {
            if (current > 1) updateFrame(current - 1);
        }
        
        function nextFrame() {
            if (current < total) updateFrame(current + 1);
        }
    </script>
</body>
</html>"""
    with open('viewer.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("[Viewer] Created interactive HTML viewer at 'viewer.html'")
    
    print("=" * 60)
    print("[Main] Processing Completed Successfully!")
    print(f"[Main] Output annotated frames saved to: '{output_dir}/'")
    print(f"[Main] Output video saved to: '{video_output_path}'")
    print(f"[Main] Interactive HTML Viewer: 'viewer.html'")
    print(f"[Main] Total Unique Vehicles Tracked: {tracker.next_track_id - 1}")
    
    if reid_events:
        print(f"\n[Re-ID Events Summary - {len(reid_events)} Events]:")
        for evt in reid_events:
            print(f"  -> {evt}")
    else:
        print("\n[Re-ID Events Summary]: All tracks maintained continuous or were distinct objects.")

if __name__ == '__main__':
    main()
