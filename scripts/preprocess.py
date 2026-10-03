import os
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from typing import List, Tuple
import json
import cv2
from musetalk.utils.preprocessing import bbox_from_landmarks, get_face_landmarks


class AnalyzeFace:
    def __call__(
        self, im: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        if im.ndim == 4:
            if im.shape[0] != 1:
                raise ValueError("AnalyzeFace accepts a single frame at a time")
            im = im[0]
        if im.ndim != 3 or im.shape[2] != 3:
            raise ValueError("Input frame must have shape (height, width, 3)")

        landmarks = get_face_landmarks(im)
        if landmarks is None:
            return np.array([]), []
        bbox = bbox_from_landmarks(landmarks, im.shape)
        if bbox == (0.0, 0.0, 0.0, 0.0):
            return np.array([]), []
        return landmarks, [bbox]

def convert_video(org_path: str, dst_path: str, vid_list: List[str]) -> None:

    """
    Convert video files to a specified format and save them to the destination path.

    Parameters:
    org_path (str): The directory containing the original video files.
    dst_path (str): The directory where the converted video files will be saved.
    vid_list (List[str]): A list of video file names to process.

    Returns:
    None
    """
    for idx, vid in enumerate(vid_list):
        if vid.endswith('.mp4'):
            org_vid_path = os.path.join(org_path, vid)
            dst_vid_path = os.path.join(dst_path, vid)
                
            if org_vid_path != dst_vid_path:
                cmd = [
                    "ffmpeg", "-hide_banner", "-y", "-i", org_vid_path, 
                    "-r", "25", "-crf", "15", "-c:v", "libx264", 
                    "-pix_fmt", "yuv420p", dst_vid_path
                ]
                subprocess.run(cmd, check=True)

            if idx % 1000 == 0:
                print(f"### {idx} videos converted ###")

def segment_video(org_path: str, dst_path: str, vid_list: List[str], segment_duration: int = 30) -> None:
    """
    Segment video files into smaller clips of specified duration.

    Parameters:
    org_path (str): The directory containing the original video files.
    dst_path (str): The directory where the segmented video files will be saved.
    vid_list (List[str]): A list of video file names to process.
    segment_duration (int): The duration of each segment in seconds. Default is 30 seconds.

    Returns:
    None
    """
    for idx, vid in enumerate(vid_list):
        if vid.endswith('.mp4'):
            input_file = os.path.join(org_path, vid)
            original_filename = os.path.basename(input_file)

            command = [
                'ffmpeg', '-i', input_file, '-c', 'copy', '-map', '0',
                '-segment_time', str(segment_duration), '-f', 'segment',
                '-reset_timestamps', '1',
                os.path.join(dst_path, f'clip%03d_{original_filename}')
            ]

            subprocess.run(command, check=True)

def extract_audio(org_path: str, dst_path: str, vid_list: List[str]) -> None:
    """
    Extract audio from video files and save as WAV format.

    Parameters:
    org_path (str): The directory containing the original video files.
    dst_path (str): The directory where the extracted audio files will be saved.
    vid_list (List[str]): A list of video file names to process.

    Returns:
    None
    """
    for idx, vid in enumerate(vid_list):
        if vid.endswith('.mp4'):
            video_path = os.path.join(org_path, vid)
            audio_output_path = os.path.join(dst_path, os.path.splitext(vid)[0] + ".wav")
            try:
                command = [
                    'ffmpeg', '-hide_banner', '-y', '-i', video_path,
                    '-vn', '-acodec', 'pcm_s16le', '-f', 'wav',
                    '-ar', '16000', '-ac', '1', audio_output_path,
                ]
                
                subprocess.run(command, check=True)
                print(f"Audio saved to: {audio_output_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error extracting audio from {vid}: {e}")

def split_data(video_files: List[str], val_list_hdtf: List[str]) -> (List[str], List[str]):
    """
    Split video files into training and validation sets based on val_list_hdtf.

    Parameters:
    video_files (List[str]): A list of video file names.
    val_list_hdtf (List[str]): A list of validation file identifiers.

    Returns:
    (List[str], List[str]): A tuple containing the training and validation file lists.
    """
    val_files = [f for f in video_files if any(val_id in f for val_id in val_list_hdtf)]
    train_files = [f for f in video_files if f not in val_files]
    return train_files, val_files

def save_list_to_file(file_path: str, data_list: List[str]) -> None:
    """
    Save a list of strings to a file, each string on a new line.

    Parameters:
    file_path (str): The path to the file where the list will be saved.
    data_list (List[str]): The list of strings to save.

    Returns:
    None
    """
    with open(file_path, 'w') as file:
        for item in data_list:
            file.write(f"{item}\n")

def generate_train_list(cfg):
    train_file_path = cfg.video_clip_file_list_train
    val_file_path = cfg.video_clip_file_list_val
    val_list_hdtf = cfg.val_list_hdtf

    meta_list = os.listdir(cfg.meta_root)

    sorted_meta_list = sorted(meta_list)
    train_files, val_files = split_data(meta_list, val_list_hdtf)

    save_list_to_file(train_file_path, train_files)
    save_list_to_file(val_file_path, val_files)

    print(val_list_hdtf)    

def analyze_video(org_path: str, dst_path: str, vid_list: List[str]) -> None:
    """
    Convert video files to a specified format and save them to the destination path.

    Parameters:
    org_path (str): The directory containing the original video files.
    dst_path (str): The directory where the meta json will be saved.
    vid_list (List[str]): A list of video file names to process.

    Returns:
    None
    """
    analyze_face = AnalyzeFace()
    
    for vid in tqdm(vid_list, desc="Processing videos"):
        #vid = "clip005_WDA_BernieSanders_000.mp4"
        #print(vid)
        if vid.endswith('.mp4'):
            vid_path = os.path.join(org_path, vid)
            wav_path = vid_path.replace(".mp4",".wav")
            vid_meta = os.path.join(dst_path, os.path.splitext(vid)[0] + ".json")
            if os.path.exists(vid_meta):
                continue
            print('process video {}'.format(vid))

            total_bbox_list = []
            total_pts_list = []
            isvalid = True

            # process
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                print(f"Could not open video: {vid_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame_idx in range(total_frames):
                ret, frame_bgr = cap.read()
                if not ret:
                    isvalid = False
                    print(f"Could not read frame {frame_idx} of {vid}")
                    break
                if frame_idx==0:
                    video_height,video_width,_ = frame_bgr.shape
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pts_list, bbox_list = analyze_face(frame_rgb)

                if len(bbox_list)>0 and None not in bbox_list:
                    bbox = bbox_list[0]
                else:
                    isvalid = False
                    bbox = []
                    print(f"set isvalid to False as broken img in {frame_idx} of {vid}")
                    break

                #print(pts_list)
                if len(pts_list)>0 and pts_list is not None:
                    pts = pts_list.tolist()
                else:
                    isvalid = False
                    pts = []
                    break

                if frame_idx==0:
                    x1,y1,x2,y2 = bbox 
                    face_height, face_width = y2-y1,x2-x1

                total_pts_list.append(pts)
                total_bbox_list.append(bbox)

            cap.release()
            if not total_bbox_list:
                continue

            meta_data = {
                    "mp4_path": vid_path,
                     "wav_path": wav_path,
                     "video_size": [video_height, video_width],
                     "face_size": [face_height, face_width],
                     "frames": total_frames,
                     "face_list": total_bbox_list,
                     "landmark_list": total_pts_list,
                     "isvalid":isvalid,
            }        
            with open(vid_meta, 'w') as f:
                json.dump(meta_data, f, indent=4)
            


def main(cfg):
    # Ensure all necessary directories exist
    os.makedirs(cfg.video_root_25fps, exist_ok=True)
    os.makedirs(cfg.video_audio_clip_root, exist_ok=True)
    os.makedirs(cfg.meta_root, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.video_file_list), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.video_clip_file_list_train), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.video_clip_file_list_val), exist_ok=True)

    vid_list = os.listdir(cfg.video_root_raw)
    sorted_vid_list = sorted(vid_list)
 
    # Save video file list
    with open(cfg.video_file_list, 'w') as file:
        for vid in sorted_vid_list:
            file.write(vid + '\n')

    # 1. Convert videos to 25 FPS
    convert_video(cfg.video_root_raw, cfg.video_root_25fps, sorted_vid_list)
    
    # 2. Segment videos into 30-second clips
    segment_video(cfg.video_root_25fps, cfg.video_audio_clip_root, vid_list, segment_duration=cfg.clip_len_second)
    
    # 3. Extract audio
    clip_vid_list = os.listdir(cfg.video_audio_clip_root)
    extract_audio(cfg.video_audio_clip_root, cfg.video_audio_clip_root, clip_vid_list)
    
    # 4. Generate video metadata
    analyze_video(cfg.video_audio_clip_root, cfg.meta_root, clip_vid_list)
    
    # 5. Generate training and validation set lists
    generate_train_list(cfg)
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/preprocess.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    main(config)
    
