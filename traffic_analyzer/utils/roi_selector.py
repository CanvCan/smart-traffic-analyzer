import cv2
import json
from traffic_analyzer.utils.config_loader import load_config

SCALING_FACTOR = 0.5

selected_rois = []
current_roi = None
is_selecting = False


def mouse_callback(event, x, y, flags, param):
    global current_roi, is_selecting, selected_rois

    orig_x = int(x / SCALING_FACTOR)
    orig_y = int(y / SCALING_FACTOR)

    if event == cv2.EVENT_LBUTTONDOWN:
        is_selecting = True
        current_roi = [x, y, x, y, orig_x, orig_y, orig_x, orig_y]

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_selecting:
            current_roi[2], current_roi[3] = x, y
            current_roi[6], current_roi[7] = orig_x, orig_y

    elif event == cv2.EVENT_LBUTTONUP:
        is_selecting = False
        x1, x2 = sorted([current_roi[4], current_roi[6]])
        y1, y2 = sorted([current_roi[5], current_roi[7]])
        selected_rois.append([x1, y1, x2, y2])
        print(f"Region {len(selected_rois)} saved: {[x1, y1, x2, y2]}")


def run(config_path: str = 'config.json'):
    global selected_rois, current_roi, is_selecting
    selected_rois = []
    current_roi = None
    is_selecting = False

    # Read video path from project config
    cfg = load_config(config_path)
    video_path = cfg.camera.video_path

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    display_w = int(frame.shape[1] * SCALING_FACTOR)
    display_h = int(frame.shape[0] * SCALING_FACTOR)
    resized_frame = cv2.resize(frame, (display_w, display_h))

    cv2.namedWindow("ROI Selector")
    cv2.setMouseCallback("ROI Selector", mouse_callback)

    print(f"[ROI Selector] Video: {video_path} | Scale: {SCALING_FACTOR * 100}%")
    print("Draw areas in order: Left_Lane → Right_Lane → Top_Horizon")
    print("  'r' → reset   |   'q' → save & exit")

    while True:
        temp = resized_frame.copy()

        if is_selecting and current_roi:
            cv2.rectangle(temp,
                          (current_roi[0], current_roi[1]),
                          (current_roi[2], current_roi[3]),
                          (0, 255, 255), 2)

        for i, roi in enumerate(selected_rois):
            sx1 = int(roi[0] * SCALING_FACTOR)
            sy1 = int(roi[1] * SCALING_FACTOR)
            sx2 = int(roi[2] * SCALING_FACTOR)
            sy2 = int(roi[3] * SCALING_FACTOR)
            cv2.rectangle(temp, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
            cv2.putText(temp, f"Area {i + 1}", (sx1, sy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("ROI Selector", temp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            selected_rois = []
            print("[ROI Selector] Reset.")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    _print_config(selected_rois)


def _print_config(rois: list) -> None:
    lane_names = ["Left_Lane", "Right_Lane", "Top_Horizon"]
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    output = {}

    for i, roi in enumerate(rois):
        if i < len(lane_names):
            output[lane_names[i]] = {
                "roi": roi,
                "color": colors[i]
            }

    print("\n--- COPY INTO config.json → lanes ---")
    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    run()
