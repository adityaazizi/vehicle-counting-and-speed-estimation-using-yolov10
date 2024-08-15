import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict, deque


CURRENT_DIRECTORY = os.getcwd()
SOURCE_VIDEO_PATH = os.path.join(
    CURRENT_DIRECTORY,
    'sample/input/video.mp4',
)  # add your video path
TARGET_VIDEO_PATH = os.path.join(
    CURRENT_DIRECTORY,
    'sample/output/video.mp4',
)  # add your video path
MODEL = os.path.join(
    CURRENT_DIRECTORY,
    'weight/yolov10n.pt',
)
CONFIDENCE_THRESHOLD = 0.25
MATCH_THRESHOLD = 0.8
IOU_THRESHOLD = 0.5
MODEL_RESOLUTION = 1280

SOURCE = np.array([
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159]
])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

LINE_START = sv.Point(0, 1250)
LINE_END = sv.Point(3840, 1250)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


model = YOLO(MODEL)
model.fuse()

video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
byte_track = sv.ByteTrack(
    frame_rate=video_info.fps,
    track_activation_threshold=CONFIDENCE_THRESHOLD,
    minimum_matching_threshold=MATCH_THRESHOLD,
    lost_track_buffer=video_info.fps
)

polygon_zone = sv.PolygonZone(
    polygon=SOURCE,
)

line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

thickness = sv.calculate_optimal_line_thickness(
    resolution_wh=video_info.resolution_wh
)
text_scale = sv.calculate_optimal_text_scale(
    resolution_wh=video_info.resolution_wh
)
box_annotator = sv.BoxAnnotator(
    thickness=thickness,
    color_lookup=sv.ColorLookup.TRACK
)
label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=thickness,
    color_lookup=sv.ColorLookup.TRACK
)
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=thickness,
    text_thickness=thickness,
    text_scale=text_scale,
)

view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

fourcc = cv2.VideoWriter_fourcc(*'avc1')
vid_writer = cv2.VideoWriter(
    TARGET_VIDEO_PATH,
    fourcc, video_info.fps,
    (video_info.width, video_info.height))
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

for _ in tqdm(range(video_info.total_frames), desc="Rendering videos with Bounding Box: "):
    ret, frame = cap.read()
    if not ret:
        break

    result = model(
        frame,
        imgsz=MODEL_RESOLUTION,
        verbose=False,
        device='mps'
    )[0]

    detections = sv.Detections.from_ultralytics(result)
    detections = detections[polygon_zone.trigger(detections)]
    detections = detections.with_nms(IOU_THRESHOLD)
    detections = byte_track.update_with_detections(detections)

    line_zone.trigger(detections)

    points = detections.get_anchors_coordinates(
        anchor=sv.Position.BOTTOM_CENTER
    )
    points = view_transformer.transform_points(points=points).astype(int)

    for tracker_id, [_, y] in zip(detections.tracker_id, points):
        coordinates[tracker_id].append(y)

    labels = []

    for tracker_id in detections.tracker_id:
        coordinate_start = coordinates[tracker_id][-1]
        coordinate_end = coordinates[tracker_id][0]
        distance = abs(coordinate_start - coordinate_end)
        time = len(coordinates[tracker_id]) / video_info.fps
        speed = distance / time * 3.6
        labels.append(f"{int(speed)} km/h")

    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )
    annotated_frame = line_zone_annotator.annotate(
        annotated_frame,
        line_counter=line_zone
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    vid_writer.write(annotated_frame)

cap.release()
vid_writer.release()
