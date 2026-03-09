import torch
from traffic_analyzer.utils.config_loader import load_config
from traffic_analyzer.core.detector import YOLODetector
from traffic_analyzer.core.tracker import ByteTracker
from traffic_analyzer.core.analyzer import Analyzer

def main():
    cfg    = load_config('config.json')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    detector = YOLODetector(
        model_path=cfg.model.yolo_path,
        device=device,
        conf=cfg.model.conf_threshold,
        imgsz=cfg.model.imgsz,
        half=cfg.model.half,
        iou=cfg.model.iou,
        agnostic_nms=cfg.model.agnostic_nms,
        target_classes=[2, 3, 5, 7],
    )

    tracker = ByteTracker(
        lost_track_buffer=150,
        min_matching_threshold=0.8,
        min_consecutive_frames=3,
    )

    Analyzer(cfg, detector, tracker).run()


if __name__ == "__main__":
    main()
