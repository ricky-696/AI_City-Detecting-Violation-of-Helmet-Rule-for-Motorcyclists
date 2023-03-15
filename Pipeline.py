from ultralytics import YOLO
import cv2
import glob
import os
import torch


def Driver_Passenger(tracker):
    # DP_dir: 'none' if no dir
    DP_dir = get_dir(tracker)
                   
    # determine Driver, Passenger1, Passenger2
    DP_ID = {}
    frame_n = 0
    for t in tracker: # one frame
        frame_n += 1
        if t.boxes.is_track:
            
            # create person and motor dict
            person, motor = {}, {}
            for b in t.boxes.data:
                x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                id, conf, class_ = str(int(b[4])), float(b[5]), int(b[6])
                dir = DP_dir[id]
                
                if dir != 'none':
                    if id == '10' and class_ == 1:
                        print(frame_n)
                    if class_ == 0:
                        motor[id] = [id, x1, y1, x2, y2, dir]
                    else:
                        person[id] = [id, x1, y1, x2, y2, dir]
            
            # find person on motor        
            motor_P = {} # person on motor 
            for p in person:
                for m in motor:
                    if aabb_collision(person[p], motor[m]):
                        if m in motor_P:
                            motor_P[m].append(person[p])
                        else:
                            motor_P[m] = [person[p]]

            # determine Driver or Passenger
            label = ['D', 'P1', 'P2']            
            for idx in motor_P:
                motor_P[idx] = sorted(motor_P[idx], key = lambda item: compare(item, DP_dir[idx]))

                label_idx = 0
                for p in motor_P[idx]:
                    track_id, x1, y1, x2, y2, p_dir = p
                    if not(track_id in DP_ID):
                        DP_ID[track_id] = label[label_idx]
                    label_idx += 1

              
    return DP_ID 
                

def compare(item, dir):
    id, x1, y1, x2, y2, p_dir = item
    if dir == 'up':
        return x1
    elif dir == 'down':
        return -x1
    elif dir == 'left':
        return y1
    elif dir == 'right':
        return -y1
    else:
        raise ValueError('Invalid direction: %s' % dir)


def get_dir(tracker):
    # Tracker.boxes.data: x, y, x, y, track_id, conf, class
    DP_dir = {} # dict
    for t in tracker:
        if t.boxes.is_track:
            for b in t.boxes.data:
                track_id = str(int(b[4]))
                if track_id in DP_dir:
                    if DP_dir[track_id].size(0) == 7:
                        x1, y1 = DP_dir[track_id][0], DP_dir[track_id][1]
                        x2, y2 = b[0], b[1]
                        dx, dy = (x2 - x1), (y2 - y1)

                        DP_dir[track_id] = det_dir(b, dx, dy)
                else:
                    DP_dir[track_id] = b
                                            
    dir = ['up', 'left', 'down', 'right']
    for i in DP_dir:
        if DP_dir[i].size(0) == 8:
            DP_dir[i] = dir[int(DP_dir[i][7])]
        else:
            DP_dir[i] = 'none'
                  
    return DP_dir


def det_dir(track_tensor, dx, dy):
    # dir: 0 = up, 1 = left, 2 = down, 3 = right
    if abs(dx) > abs(dy):
        if dx > 0: 
            return torch.cat((track_tensor, torch.tensor([3])), 0)
        else:
            return torch.cat((track_tensor, torch.tensor([1])), 0)
    elif abs(dx) < abs(dy):
        if dy > 0:
            return torch.cat((track_tensor, torch.tensor([2])), 0)
        else:
            return torch.cat((track_tensor, torch.tensor([0])), 0)
    else:
        if dy > 0: 
            return torch.cat((track_tensor, torch.tensor([2])), 0)
        else:
            return torch.cat((track_tensor, torch.tensor([0])), 0)


def aabb_collision(p, m):
    # [id, x1, y1, x2, y2, dir]
    p_x1, p_y1, p_x2, p_y2 = p[1], p[2], p[3], p[4]
    m_x1, m_y1, m_x2, m_y2 = m[1], m[2], m[3], m[4]
    
    if p_y2 > m_y1 and m_y2 > p_y1:
        if p_x2 > m_x1 and m_x2 > p_x1:
            return True
    
    return False


def out_video():
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter('DP_' + video_name, 
                            fourcc, 5, (1920, 1080))
    for t in tracker:
        if t.boxes.is_track:
            img = t.orig_img
            frame = draw_bbox(img, t, DP_ID)
            video.write(frame)
                    
    video.release()
    cv2.destroyAllWindows()


def draw_bbox(img, t, DP_ID):
    for b in t.boxes.data:
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        class_idx = int(b[max(b.size()) - 1])
        track_id = str(int(b[4]))
        if  track_id in DP_ID:
            label = '#' + track_id + '_' + DP_ID[track_id]
        else:
            label = '#' + track_id + '_' + class_[class_idx]
        
        cv2.rectangle(img, (x1, y1), (x2, y2),
                      bbox_color[class_idx], 3)
        cv2.putText(img, label, (x1 - 1, y1 - 1), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0))
    return img


if __name__ == '__main__':
    video_path = '/mnt/Nami/2023_AI_City_challenge_datasets/Track_5/videos'
    gt_path = '/mnt/Nami/2023_AI_City_challenge_datasets/Track_5/gt.txt'
    video_name = '099.mp4'
    bbox_color = [(255, 0, 0), (0, 255, 0)]
    class_ = ['motorbike', 'Person']
    model = YOLO('Yolov8_person.pt')

    # get track_id
    tracker = model.track(source = os.path.join(video_path, video_name))
    
    # For every track_id, determine D, P1, P2
    DP_ID = Driver_Passenger(tracker)

    #out_video()
                
