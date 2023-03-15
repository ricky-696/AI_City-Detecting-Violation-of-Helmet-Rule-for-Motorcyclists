from ultralytics import YOLO
import cv2
import os
import glob


video_path = '/mnt/Nami/2023_AI_City_challenge_datasets/Track_5/videos'
model = YOLO('Yolov8_person.pt')

video_name = glob.glob(os.path.join(video_path, "*.mp4"))

for name in video_name:
    res = model.predict(source = name, conf = 0.25) #confidence
    name_split = name.split('/')
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(os.path.join('Train_res', name_split[len(name_split) - 1]), 
                            fourcc, 5, (1920, 1080))

    for r in res:
        res_plotted = r.plot()
        video.write(res_plotted)

    video.release()
    cv2.destroyAllWindows()
