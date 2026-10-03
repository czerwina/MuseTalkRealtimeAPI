from pathlib import Path
import unittest

import numpy as np

from musetalk.utils.preprocessing import (
    FACE_LANDMARKER_PATH,
    MEDIAPIPE_68_INDICES,
    MuseTalkFaceLandmarker,
    bbox_from_landmarks,
    get_bbox_range,
    get_landmark_and_bbox,
)


class PreprocessingTests(unittest.TestCase):
    def test_mediapipe_mapping_has_68_landmarks(self):
        self.assertEqual(MEDIAPIPE_68_INDICES.shape, (68,))
        self.assertGreaterEqual(MEDIAPIPE_68_INDICES.min(), 0)
        self.assertLess(MEDIAPIPE_68_INDICES.max(), 478)

    def test_bbox_from_landmarks_applies_shift_and_clamps_to_frame(self):
        landmarks = np.full((68, 2), [20, 30], dtype=np.int32)
        landmarks[0] = [-5, 50]
        landmarks[29] = [40, 45]
        landmarks[30] = [45, 50]
        landmarks[67] = [80, 90]

        bbox = bbox_from_landmarks(landmarks, (100, 100, 3), bbox_shift=5)

        self.assertEqual(bbox, (0, 10, 80, 90))

    def test_bbox_from_landmarks_rejects_non_68_point_input(self):
        with self.assertRaisesRegex(ValueError, "Expected 68"):
            bbox_from_landmarks(np.zeros((478, 2)), (100, 100, 3))

    @unittest.skipUnless(
        Path(FACE_LANDMARKER_PATH).is_file(),
        "Run download_weights.bat to download the MediaPipe task model",
    )
    def test_face_landmarker_detects_sample_face(self):
        import cv2

        sample_path = Path("assets/demo/musk/musk.png")
        image_bgr = cv2.imread(str(sample_path))
        self.assertIsNotNone(image_bgr)

        detector = MuseTalkFaceLandmarker()
        try:
            landmarks = detector.detect(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        finally:
            detector.close()

        self.assertIsNotNone(landmarks)
        self.assertEqual(landmarks.shape, (68, 2))
        bbox = bbox_from_landmarks(landmarks, image_bgr.shape)
        self.assertNotEqual(bbox, (0.0, 0.0, 0.0, 0.0))

        coords, frames = get_landmark_and_bbox([str(sample_path)])
        self.assertEqual(len(coords), 1)
        self.assertEqual(len(frames), 1)
        self.assertNotEqual(coords[0], (0.0, 0.0, 0.0, 0.0))
        self.assertIn("Total frames: 1", get_bbox_range([str(sample_path)]))


if __name__ == "__main__":
    unittest.main()
