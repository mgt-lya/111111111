import cv2
import numpy as np
import os
from tqdm import tqdm

def create_video_from_sequence(input_path, output_path, sequence):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input directory {input_path} not found")
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, "output_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    first_frame_path = os.path.join(input_path, sequence[0])
    if not os.path.exists(first_frame_path):
        raise FileNotFoundError(f"First frame {first_frame_path} not found")

    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        raise ValueError(f"Failed to read first frame {first_frame_path}")


    height, width = first_frame.shape[:2]

    video_writer = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))

    for filename in tqdm(sequence, desc="Processing frames"):
        file_path = os.path.join(input_path, filename)

        if not os.path.exists(file_path):
            print(f"Warning: File {filename} not found, skipping")
            continue

        frame = cv2.imread(file_path)
        if frame is None:
            print(f"Warning: Failed to read {filename}, skipping")
            continue

        try:
            video_writer.write(frame)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    video_writer.release()
    print(f"\nVideo successfully created at: {output_file}")


if __name__ == "__main__":
    INPUT_PATH = "/home/ubuntu-user/Downloads/ITI-GERD-main/Images"
    OUTPUT_PATH = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/dataUnet/temp/"

    SEQUENCE = [
        "232.png", "267.png", "113.png", "52.png", "438.png",
        "657.png", "106.png", "128.png", "632.png", "509.png"
    ]

    create_video_from_sequence(INPUT_PATH, OUTPUT_PATH, SEQUENCE)