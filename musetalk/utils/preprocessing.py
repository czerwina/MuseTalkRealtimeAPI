from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm


coord_placeholder = (0.0, 0.0, 0.0, 0.0)
FACE_LANDMARKER_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"
)

# MediaPipe Face Landmarker points selected in the conventional 68-point order.
MEDIAPIPE_68_INDICES = np.asarray(
    [
        127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400,
        378, 379, 365,
        70, 63, 105, 66, 107,
        336, 296, 334, 293, 300,
        168, 6, 197, 195,
        5, 4, 19, 94, 2,
        33, 160, 159, 158, 133, 153,
        362, 385, 386, 387, 263, 373,
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
        78, 95, 88, 178, 87, 14, 317, 402,
    ],
    dtype=np.int32,
)

_face_landmarker = None
_last_bbox_diagnostics = None


class MuseTalkFaceLandmarker:
    def __init__(self, model_path=FACE_LANDMARKER_PATH):
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"MediaPipe Face Landmarker model not found: {model_path}. "
                "Run download_weights.bat or download_weights.sh first."
            )

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image_rgb):
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape (height, width, 3)")

        height, width = image_rgb.shape[:2]
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(image_rgb),
        )
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None

        points = result.face_landmarks[0]
        all_landmarks = np.asarray(
            [(point.x * width, point.y * height) for point in points],
            dtype=np.float32,
        )
        if len(all_landmarks) <= int(MEDIAPIPE_68_INDICES.max()):
            raise RuntimeError(
                "MediaPipe Face Landmarker returned fewer landmarks than expected"
            )
        return np.rint(all_landmarks[MEDIAPIPE_68_INDICES]).astype(np.int32)

    def close(self):
        self._landmarker.close()


def get_face_landmarks(image_rgb):
    global _face_landmarker
    if _face_landmarker is None:
        _face_landmarker = MuseTalkFaceLandmarker()
    return _face_landmarker.detect(image_rgb)


def bbox_from_landmarks(landmarks, image_shape, bbox_shift=0):
    height, width = image_shape[:2]
    face_landmarks = np.asarray(landmarks, dtype=np.int32)
    if face_landmarks.shape != (68, 2):
        raise ValueError(
            f"Expected 68 two-dimensional face landmarks, got {face_landmarks.shape}"
        )

    half_face_coord = face_landmarks[29].copy()
    half_face_coord[1] += int(bbox_shift)
    half_face_dist = int(np.max(face_landmarks[:, 1]) - half_face_coord[1])
    upper_bound = max(0, int(half_face_coord[1]) - half_face_dist)
    x1 = max(0, int(np.min(face_landmarks[:, 0])))
    y1 = upper_bound
    x2 = min(width, int(np.max(face_landmarks[:, 0])))
    y2 = min(height, int(np.max(face_landmarks[:, 1])))

    if x2 <= x1 or y2 <= y1:
        return coord_placeholder
    return x1, y1, x2, y2


def read_imgs(img_list):
    frames = []
    print("Reading images...")
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        if frame is None:
            raise ValueError(f"Could not read image: {img_path}")
        frames.append(frame)
    return frames


def get_bbox_range(img_list, upperbondrange=0):
    global _last_bbox_diagnostics
    img_list = list(img_list)
    image_paths = tuple(img_list)
    if (
        _last_bbox_diagnostics is None
        or _last_bbox_diagnostics[0] != image_paths
        or _last_bbox_diagnostics[1] != upperbondrange
    ):
        get_landmark_and_bbox(img_list, upperbondrange)
    _, _, avg_minus, avg_plus = _last_bbox_diagnostics
    return (
        f"Total frames: {len(img_list)}. Manual adjustment range: "
        f"[-{avg_minus}..{avg_plus}], current value: {upperbondrange}"
    )


def get_landmark_and_bbox(img_list, upperbondrange=0):
    global _last_bbox_diagnostics
    img_list = list(img_list)
    frames = read_imgs(img_list)
    coords_list = []
    minus_ranges = []
    plus_ranges = []
    if upperbondrange:
        print(f"Detecting face landmarks with bbox_shift={upperbondrange}")
    else:
        print("Detecting face landmarks")

    for frame in tqdm(frames):
        landmarks = get_face_landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if landmarks is None:
            coords_list.append(coord_placeholder)
            continue
        minus_ranges.append(abs(landmarks[30, 1] - landmarks[29, 1]))
        plus_ranges.append(abs(landmarks[29, 1] - landmarks[28, 1]))
        coords_list.append(
            bbox_from_landmarks(landmarks, frame.shape, upperbondrange)
        )

    detected = len(coords_list) - coords_list.count(coord_placeholder)
    avg_minus = int(np.mean(minus_ranges)) if minus_ranges else 0
    avg_plus = int(np.mean(plus_ranges)) if plus_ranges else 0
    _last_bbox_diagnostics = (
        tuple(img_list),
        upperbondrange,
        avg_minus,
        avg_plus,
    )
    print(f"Detected usable face boxes in {detected} of {len(frames)} frames")
    return coords_list, frames
