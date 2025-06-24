from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8-compatible model
model = YOLO('models/yolov7_custom.pt')

# Initialize Deep SORT
tracker = DeepSort(max_age=30)

# Load video
cap = cv2.VideoCapture('videos/15sec_input_720p.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_tracked.mp4', fourcc, 30.0, (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model.predict(frame, conf=0.4, classes=[0])  # class 0 = person
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'player'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f'Player {track_id}', (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow('Player Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
