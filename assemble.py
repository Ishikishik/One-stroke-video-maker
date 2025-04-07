import cv2
import os

def disassemble(video_path: str, output_dir: str):
    os.makedirs(output_dir,exist_ok=True)
    cap =cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('エラー:ビデオ開けず')
        exit()
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 画像ファイルの保存（連番で保存）
        filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(filename, frame)

        frame_count += 1

    # 後処理
    cap.release()
