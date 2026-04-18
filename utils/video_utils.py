import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(ouput_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()

def save_frames_to_folder(output_video_frames, output_folder):
    # Crea la cartella se non esiste già
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i, frame in enumerate(output_video_frames):
        # Genera un nome file progressivo (es: frame_0001.jpg)
        file_path = os.path.join(output_folder, f"frame_{i:04d}.jpg")
        
        # Salva il frame
        cv2.imwrite(file_path, frame)
    
    print(f"Salvataggio completato: {len(output_video_frames)} frame salvati in '{output_folder}'")