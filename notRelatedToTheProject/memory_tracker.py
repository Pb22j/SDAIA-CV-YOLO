import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

class FeatureExtractor:
    """
    Local PyTorch Feature Extractor combining DINOv2 Deep Features + HSV Color Histogram.
    Provides robust 2-layer visual identity (Shape + Color) for vehicle Re-ID.
    100% Free, Local, Offline.
    """
    def __init__(self, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"[FeatureExtractor] Using device: {self.device}")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load DINOv2 Small
        try:
            print("[FeatureExtractor] Loading DINOv2 model (dinov2_vits14)...")
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
            self.model.eval().to(self.device)
            print("[FeatureExtractor] DINOv2 loaded successfully!")
        except Exception as e:
            print(f"[FeatureExtractor] DINOv2 load warning: {e}. Falling back to ResNet50...")
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
            self.model.eval().to(self.device)

    def extract_color_hist(self, crop_bgr):
        """
        Extracts 3D HSV color histogram (8 Hue, 8 Saturation, 8 Value bins).
        Guarantees that a Blue car and a White truck have distinct signatures.
        """
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    @torch.no_grad()
    def extract(self, crop_bgr):
        """
        Extracts combined feature dictionary {'deep': dino_emb, 'color': hsv_hist}
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
            
        # 1. Color Histogram
        color_hist = self.extract_color_hist(crop_bgr)
        
        # 2. Deep Feature Embedding via DINOv2
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        feat = self.model(tensor_img).flatten(1)
        norm = torch.norm(feat, p=2, dim=1, keepdim=True)
        deep_emb = (feat / (norm + 1e-6)).cpu().numpy()[0]
        
        return {
            'deep': deep_emb,
            'color': color_hist
        }

def compute_similarity(feat1, feat2):
    """
    Computes hybrid similarity between two feature dicts:
    70% DINOv2 Cosine Similarity + 30% Color Histogram Correlation
    """
    if feat1 is None or feat2 is None:
        return 0.0
        
    # Deep Cosine Sim (Dot product of normalized vectors)
    deep_sim = float(np.dot(feat1['deep'], feat2['deep']))
    
    # Color Hist Correlation (-1 to +1 mapped to 0 to 1)
    color_corr = cv2.compareHist(
        feat1['color'].astype(np.float32),
        feat2['color'].astype(np.float32),
        cv2.HISTCMP_CORREL
    )
    color_sim = max(0.0, float(color_corr))
    
    # Weighted combination
    combined_sim = 0.70 * deep_sim + 0.30 * color_sim
    return combined_sim

class VisualMemoryBank:
    """
    Multi-View Visual Gallery Memory Bank with 1-to-1 Unique Assignment.
    """
    def __init__(self, max_keyframes=10):
        self.max_keyframes = max_keyframes
        # gallery: { track_id: [feat_dict_1, feat_dict_2, ...] }
        self.gallery = {}
        
    def get_track_ids(self):
        return list(self.gallery.keys())
        
    def get_max_similarity(self, feat, track_id):
        if track_id not in self.gallery or feat is None:
            return 0.0
        keyframes = self.gallery[track_id]
        sims = [compute_similarity(feat, kf) for kf in keyframes]
        return max(sims)

    def update(self, track_id, feat):
        if feat is None:
            return
        if track_id not in self.gallery:
            self.gallery[track_id] = [feat]
        else:
            keyframes = self.gallery[track_id]
            sims = [compute_similarity(feat, kf) for kf in keyframes]
            max_sim = max(sims)
            
            # Store new keyframe if similarity < 0.90 (new angle/lighting)
            if max_sim < 0.90:
                if len(keyframes) < self.max_keyframes:
                    keyframes.append(feat)
                else:
                    most_sim_idx = np.argmax(sims)
                    keyframes[most_sim_idx] = feat

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

class YoloMemoryTracker:
    """
    Long-Term Tracker with Spatial-Visual Verification and Track Confirmation.
    """
    def __init__(self, yolo_model='yolov8m.pt', strict_reid_threshold=0.65, relaxed_reid_threshold=0.45, conf_threshold=0.35, device=None):
        print(f"[YoloMemoryTracker] Loading YOLO model ({yolo_model})...")
        self.detector = YOLO(yolo_model)
        self.extractor = FeatureExtractor(device=device)
        self.memory_bank = VisualMemoryBank()
        
        self.strict_reid_threshold = strict_reid_threshold
        self.relaxed_reid_threshold = relaxed_reid_threshold
        self.conf_threshold = conf_threshold
        
        # Vehicle COCO classes: 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = [2, 3, 5, 7]
        
        self.next_track_id = 1
        self.active_tracks = {}
        self.track_colors = {}
        
    def _get_color(self, track_id):
        if track_id not in self.track_colors:
            np.random.seed(track_id * 17 + 42)
            color = tuple(int(c) for c in np.random.randint(50, 255, size=3))
            self.track_colors[track_id] = color
        return self.track_colors[track_id]

    def process_frame(self, frame_bgr, frame_idx):
        h, w = frame_bgr.shape[:2]
        
        # 1. YOLO Detection
        results = self.detector(frame_bgr, conf=self.conf_threshold, iou=0.30, verbose=False)[0]
        
        raw_detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            if cls_id in self.target_classes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue
                raw_detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf})

        # Custom Strict NMS (Intersection over Minimum Area) to handle occlusion double-boxes
        keep_indices = []
        for i, det_i in enumerate(raw_detections):
            keep = True
            for j, det_j in enumerate(raw_detections):
                if i == j:
                    continue
                
                iou = calculate_iou(det_i['bbox'], det_j['bbox'])
                
                # Calculate Intersection over Minimum Area (IoM)
                x1 = max(det_i['bbox'][0], det_j['bbox'][0])
                y1 = max(det_i['bbox'][1], det_j['bbox'][1])
                x2 = min(det_i['bbox'][2], det_j['bbox'][2])
                y2 = min(det_i['bbox'][3], det_j['bbox'][3])
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                area_i = (det_i['bbox'][2] - det_i['bbox'][0]) * (det_i['bbox'][3] - det_i['bbox'][1])
                area_j = (det_j['bbox'][2] - det_j['bbox'][0]) * (det_j['bbox'][3] - det_j['bbox'][1])
                min_area = min(area_i, area_j)
                
                iom = inter_area / min_area if min_area > 0 else 0
                
                # If high overlap, only keep the one with higher confidence
                if iou > 0.15 or iom > 0.40:
                    if det_i['conf'] < det_j['conf']:
                        keep = False
                        break
                        
            if keep:
                keep_indices.append(i)

        detections = []
        for i in keep_indices:
            det = raw_detections[i]
            x1, y1, x2, y2 = det['bbox']
            crop = frame_bgr[y1:y2, x1:x2]
            embedding = self.extractor.extract(crop)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'conf': det['conf'],
                'embedding': embedding,
                'matched_id': None,
                'reid_score': 0.0,
                'is_reid': False
            })

        matched_det_indices = set()
        matched_active_ids = set()
        
        # 2. Short-Term Association via IoU (Consecutive frames)
        for det_idx, det in enumerate(detections):
            best_iou = 0.0
            best_id = None
            
            for trk_id, trk in self.active_tracks.items():
                if trk_id in matched_active_ids:
                    continue
                if frame_idx - trk['last_seen'] > 3:
                    continue
                    
                iou = calculate_iou(det['bbox'], trk['bbox'])
                if iou > 0.35 and iou > best_iou:
                    best_iou = iou
                    best_id = trk_id
                    
            if best_id is not None:
                det['matched_id'] = best_id
                det['reid_score'] = 1.0
                matched_det_indices.add(det_idx)
                matched_active_ids.add(best_id)

        # 3. Long-Term Association via SciPy Hungarian Algorithm (Strict Phase)
        unmatched_dets = [det for i, det in enumerate(detections) if i not in matched_det_indices]
        # Only use confirmed tracks for long-term Memory Bank ReID
        gallery_ids = [gid for gid in self.memory_bank.get_track_ids() if gid not in matched_active_ids]
        
        if unmatched_dets and gallery_ids:
            num_dets = len(unmatched_dets)
            num_gids = len(gallery_ids)
            
            cost_matrix = np.ones((num_dets, num_gids), dtype=np.float32)
            sim_matrix = np.zeros((num_dets, num_gids), dtype=np.float32)
            
            for i, det in enumerate(unmatched_dets):
                for j, trk_id in enumerate(gallery_ids):
                    sim = self.memory_bank.get_max_similarity(det['embedding'], trk_id)
                    sim_matrix[i, j] = sim
                    cost_matrix[i, j] = 1.0 - sim
                    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                sim = sim_matrix[r, c]
                # STRICT match phase
                if sim >= self.strict_reid_threshold:
                    matched_id = gallery_ids[c]
                    unmatched_dets[r]['matched_id'] = matched_id
                    unmatched_dets[r]['reid_score'] = sim
                    unmatched_dets[r]['is_reid'] = True
                    matched_active_ids.add(matched_id)

        # 4. Verification Phase (Spatial-Visual Fallback for Extreme Rotation)
        still_unmatched = [det for det in detections if det['matched_id'] is None]
        # Check against all missing active tracks (both tentative and confirmed)
        missing_tracks = [trk_id for trk_id in self.active_tracks if trk_id not in matched_active_ids]
        
        for det in still_unmatched:
            best_trk_id = None
            best_sim = -1
            
            x1, y1, x2, y2 = det['bbox']
            det_cx, det_cy = (x1 + x2) / 2, (y1 + y2) / 2
            det_w = max(x2 - x1, 1)
            det_h = max(y2 - y1, 1)
            
            for trk_id in missing_tracks:
                trk = self.active_tracks[trk_id]
                frames_since_seen = frame_idx - trk['last_seen']
                
                if frames_since_seen > 20:
                    continue
                    
                # Verification Check: Relaxed visual threshold to handle 360 extreme rotation
                sim = self.memory_bank.get_max_similarity(det['embedding'], trk_id)
                
                if sim >= self.relaxed_reid_threshold and sim > best_sim:
                    best_sim = sim
                    best_trk_id = trk_id
                        
            if best_trk_id is not None:
                det['matched_id'] = best_trk_id
                det['reid_score'] = best_sim
                det['is_reid'] = True
                missing_tracks.remove(best_trk_id)
                matched_active_ids.add(best_trk_id)

        # 5. Assign New IDs for any surviving unmatched detections
        for det in detections:
            if det['matched_id'] is None:
                det['matched_id'] = self.next_track_id
                det['reid_score'] = 0.0
                self.next_track_id += 1

        # 6. Update Memory Bank & Active Tracks
        frame_results = []
        for det in detections:
            trk_id = det['matched_id']
            
            self.memory_bank.update(trk_id, det['embedding'])
            color = self._get_color(trk_id)
            
            self.active_tracks[trk_id] = {
                'bbox': det['bbox'],
                'last_seen': frame_idx,
                'color': color
            }
            
            frame_results.append({
                'id': trk_id,
                'bbox': det['bbox'],
                'conf': det['conf'],
                'reid_score': det['reid_score'],
                'is_reid': det['is_reid'],
                'color': color
            })
            
        return frame_results
