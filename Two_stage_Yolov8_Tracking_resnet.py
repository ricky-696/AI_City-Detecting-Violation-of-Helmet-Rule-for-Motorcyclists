from ultralytics import YOLO
import resnet_helmet.load_data as load_data
from resnet_helmet.Resnet_NF import resnet_152, resnet_101
from arg import arg


import cv2
import glob
import os
import torch
import torchvision


def debug_draw(img, b, filename):
    x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
    cv2.rectangle(img, (x1, y1), (x2, y2),(0, 0, 255), 3)
    cv2.imwrite(filename, img)


def get_video_id(video_name):
    id = video_name.split('.')
    res = ''
    start = False
    for s in id[0]:
        if start:
            res = res + s
        else:
            if s != '0': 
                start = True
                res = res + s
            
    return res


def Driver_Passenger(DPM, dir, track_id):
    # determine DP, and motor
    motor, person = [], []
    for b in DPM.boxes: # group's bbox
        axis = b.data[0]
        class_ = int(b.cls)
        conf = round(float(b.conf), 2)
        x1, y1, x2, y2 = float(axis[0]), float(axis[1]), float(axis[2]), float(axis[3])
        
        if class_ == 1:
            person.append([x1, y1, x2, y2, conf, track_id])
        else:
            motor.append([x1, y1, x2, y2, conf, track_id, 'motorbike'])

    # determine Driver or Passenger
    label = ['D', 'P1', 'P2']            
    person = sorted(person, key = lambda item: compare(item, dir))

    idx = 0
    if len(person) > 3:
        for b in DPM.boxes.data:
            debug_draw(DPM.orig_img, b, 'test.jpg')

    for p in person:
        p.append(label[idx])
        idx += 1

    return person, motor 
                

def compare(item, dir):
    x1, y1, x2, y2 = item[0], item[1], item[2], item[3]

    if dir == 'up':
        return y1
    elif dir == 'down':
        return -y1
    elif dir == 'left':
        return x1
    elif dir == 'right':
        return -x1
    else:
        raise ValueError('Invalid direction')


def get_dir(tracker):
    # Tracker.boxes.data: x, y, x, y, track_id, conf, class
    DP_dir = {} # dict
    No_id = 0
    for t in tracker:
        img = t.orig_img
        if t.boxes.is_track:
            for b in t.boxes.data:
                track_id = str(int(b[4]))
                if track_id in DP_dir:
                    x1, y1 = to_mid_axis(DP_dir[track_id][0], DP_dir[track_id][1], DP_dir[track_id][2], DP_dir[track_id][3])
                    x2, y2 = to_mid_axis(b[0], b[1], b[2], b[3])
                    dx, dy = (x2 - x1), (y2 - y1)
                    
                    first_axis = DP_dir[track_id]
                    if DP_dir[track_id].size(0) > 7:
                        first_axis = DP_dir[track_id][ : -1]
                        
                    DP_dir[track_id] = det_dir(first_axis, dx, dy)
                else:                                         
                    DP_dir[track_id] = b
        else:
            for b in t.boxes.data:
                DP_dir['N_' + str(No_id)] = b
                No_id += 1
                                            
    dir = ['up', 'left', 'down', 'right']
    for i in DP_dir:
        if DP_dir[i].size(0) != 8: # if no detect dir, used mid_axis to detect
            x1, y1, x2, y2, track_id = DP_dir[i][0], DP_dir[i][1], DP_dir[i][2], DP_dir[i][3], DP_dir[i][4] 
            x1, y1 = to_mid_axis(x1, y1, x2, y2)
            mid_x, mid_y = video_w / 2, video_h / 2
            dx, dy = mid_x - x1, mid_y - y1
            DP_dir[i] = det_dir(DP_dir[i], dx, dy)
        
        DP_dir[i] = dir[int(DP_dir[i][-1])]
                  
    return DP_dir


def to_mid_axis(x1, y1, x2, y2):
    w, h = (x2 - x1), (y2 - y1)
    return (x1 + w), (y1 + h)


def det_dir(track_tensor, dx, dy):
    # dir: 0 = up, 1 = left, 2 = down, 3 = right
    if abs(dx) > abs(dy):
        if dx > 0: 
            return torch.cat((track_tensor.to('cuda:' + str(device_num)), torch.tensor([3]).to('cuda:' + str(device_num))), 0)
        else:
            return torch.cat((track_tensor.to('cuda:' + str(device_num)), torch.tensor([1]).to('cuda:' + str(device_num))), 0)
    elif abs(dx) < abs(dy):
        if dy > 0:
            return torch.cat((track_tensor.to('cuda:' + str(device_num)), torch.tensor([2]).to('cuda:' + str(device_num))), 0)
        else:
            return torch.cat((track_tensor.to('cuda:' + str(device_num)), torch.tensor([0]).to('cuda:' + str(device_num))), 0)
    else:
        if dy > 0: 
            return torch.cat((track_tensor.to('cuda:' + str(device_num)), torch.tensor([2]).to('cuda:' + str(device_num))), 0)
        else:
            return torch.cat((track_tensor.to('cuda:' + str(device_num)), torch.tensor([0]).to('cuda:' + str(device_num))), 0)


def collision(p, m):
    # [id, x1, y1, x2, y2, dir]
    p_x1, p_y1, p_x2, p_y2 = p[1], p[2], p[3], p[4]
    m_x1, m_y1, m_x2, m_y2 = m[1], m[2], m[3], m[4]

    # if person under motor
    if p_y1 > m_y1: return False
    
    # aabb
    if p_y2 > m_y1 and m_y2 > p_y1:
        if p_x2 > m_x1 and m_x2 > p_x1:
            return True
    
    return False


def out_frame(frame, person, motor):
    # x1, y1, x2, y2, conf, track_id, (D or P1, P2)
    for p in person:
        x1, y1, x2, y2 = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        track_id, _class = str(p[5]), str(p[6])

        label = '#_' + track_id + '_' + _class

        cv2.rectangle(frame, (x1, y1), (x2, y2),
                        bbox_color[0], 3)
        cv2.putText(frame, label, (x1 - 1, y1 - 1), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))
    
    for m in motor:
        x1, y1, x2, y2 = int(m[0]), int(m[1]), int(m[2]), int(m[3])
        track_id, _class = str(m[5]), str(m[6])

        label = '#_' + track_id + '_' + _class

        cv2.rectangle(frame, (x1, y1), (x2, y2),
                        bbox_color[1], 3)
        cv2.putText(frame, label, (x1 - 1, y1 - 1), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))


def get_helmet(group_img, person):
    h_class = ['Helmet', 'NoHelmet']
       
    # Using Resnet to classify final label
    with torch.cuda.device(device_num):
        with torch.no_grad():
            # p: x1, y1, x2, y2, conf, track_id, (D or P1, P2)
            for p in person:
                x1, y1, x2, y2 = float(p[0]), float(p[1]), float(p[2]), float(p[3])
                x, y, w, h = x1, y1, (x2 - x1), (y2 - y1)

                crop_img = group_img[int(y) : int(y) + int(h), int(x) : int(x) + int(w)]
                crop_img = cv2.resize(crop_img, (224, 224))

                blur_img = cv2.GaussianBlur(crop_img, (0, 0), 100)
                crop_img = cv2.addWeighted(crop_img, 1.5, blur_img, -0.5, 0)

                dataset = load_data.TestDataset(crop_img)
                loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
                
                for data in loader:
                    data = data.to(device_num)
                    res = classifer(data)
                    pred = res.argmax(dim = 1, keepdim = True)

                    # concat (D, P1, P2) and (Helmet, NoHelmet)
                    p[-1] += h_class[int(pred)]
                                

def output_res(video_id, frame_id, person, motor):
    class_id = {'motorbike': '1',
                'DHelmet': '2',
                'DNoHelmet': '3',
                'P1Helmet': '4',
                'P1NoHelmet': '5',
                'P2Helmet': '6',
                'P2NoHelmet': '7'
                }
    # format: 〈video_id〉, 〈frame〉, 〈bb_left〉, 〈bb_top〉, 〈bb_width〉, 〈bb_height〉, 〈class_id〉, 〈confidence〉
    with open(res_file_name, 'a') as f:
        # x1, y1, x2, y2, conf, track_id, (D or P1, P2)
        for p in person:
            x1, y1, x2, y2 = int(p[0]), int(p[1]), int(p[2]), int(p[3])
            conf, track_id, _class = float(p[4]), str(p[5]), str(p[6])
            x1, y1 = out_of_range(x1, y1)
            x2, y2 = out_of_range(x2, y2)
            w, h = (x2 - x1), (y2 - y1)

            res = str(video_id) + ',' + str(frame_id) + ',' + str(x1) + ',' + str(y1) + ',' + str(w) + ',' + str(h)+ ',' + class_id[_class] + ',' + str(conf)
            f.write(res + "\n")
            
        for m in motor:
            x1, y1, x2, y2 = int(m[0]), int(m[1]), int(m[2]), int(m[3])
            conf, track_id, _class = float(m[4]), str(m[5]), str(m[6])
            x1, y1 = out_of_range(x1, y1)
            x2, y2 = out_of_range(x2, y2)
            w, h = (x2 - x1), (y2 - y1)

            res = str(video_id) + ',' + str(frame_id) + ',' + str(x1) + ',' + str(y1) + ',' + str(w) + ',' + str(h)+ ',' + class_id[_class] + ',' + str(conf)
            f.write(res + "\n")
                

def old_pipeline():
    all_video_path = sorted(glob.glob(os.path.join(video_path, "*.mp4")))
    for v_p in all_video_path:

        v_p_split = v_p.split('/')
        video_name = v_p_split[len(v_p_split) - 1]
        video_id = get_video_id(video_name)

        # get group's track_id
        tracker = yolo1.track(source = v_p,
                              device = device_num, 
                              conf = 0.01)
        
        group_dir = get_dir(tracker)

        orig_crop_img_wh = {}

        # cut_group dict format: 
            # cut_group[group_id][0] = group_img 
            # cut_group[group_id][1] = track_id

        cut_group = get_group_img(tracker, orig_crop_img_wh)
                
        DP = {}
        for i in cut_group:
            pred_res = yolo2.predict(source = cut_group[i][0], conf = 0.15)       
            DP[i] = [pred_res, cut_group[i][1]]

        # For every person, determine D, P1, P2
        # DP_ID[group_id]
        DP_ID, M_ID = Driver_Passenger(DP, group_dir)
        
        # Assign DHemlet, DNoHelmet... in DP_ID
        get_helmet(cut_group, DP_ID)

        # assign DP's bbox to original frame
        DP_to_frame(tracker, cut_group, DP_ID, M_ID, orig_crop_img_wh)
        
        filename = 'res_TYTR_test.txt'
        output_res(tracker, video_id, filename, DP_ID, M_ID)

        path = 'Test_res_TYTR'
        if int(video_id) <= 10:
            out_frame(tracker, path, video_name, DP_ID, M_ID)


def get_group_img(frame, g):
    x1, y1, x2, y2 = int(g[0]), int(g[1]), int(g[2]), int(g[3])
    x1, y1 = out_of_range(x1, y1)
    x2, y2 = out_of_range(x2, y2)
    w, h = (x2 - x1), (y2 - y1)
    
    crop_img = frame[int(y1) : int(y1) + int(h), int(x1) : int(x1) + int(w)]
    orig_crop_img_wh = [w, h]
    crop_img = cv2.resize(crop_img, (256, 256), interpolation = cv2.INTER_CUBIC)

    return crop_img, orig_crop_img_wh


def out_of_range(x, y):
    if x < 0: x = 0
    if x > 1920: x = 1920
    if y < 0: y = 0
    if y > 1080: y = 1080
    return x, y


def DP_to_frame(g_axis, person, motor, orig_group_img_wh):
    f_x1, f_y1, f_x2, f_y2 = int(g_axis[0]), int(g_axis[1]), int(g_axis[2]), int(g_axis[3])
    f_x1, f_y1 = out_of_range(f_x1, f_y1)
    f_x2, f_y2 = out_of_range(f_x2, f_y2)

    # get person, motor axis on frame
    g_w, g_h = orig_group_img_wh[0], orig_group_img_wh[1]

    for m in motor:
        m[0], m[1], m[2], m[3] = norm_axis(256, 256, m[0], m[1], m[2], m[3])
        m[0], m[1] = frame_axis(g_w, g_h, f_x1, f_y1, m[0], m[1])
        m[2], m[3] = frame_axis(g_w, g_h, f_x1, f_y1, m[2], m[3])

    for p in person:
        p[0], p[1], p[2], p[3] = norm_axis(256, 256, p[0], p[1], p[2], p[3])
        p[0], p[1] = frame_axis(g_w, g_h, f_x1, f_y1, p[0], p[1])
        p[2], p[3] = frame_axis(g_w, g_h, f_x1, f_y1, p[2], p[3])
            
    
def norm_axis(w, h, x1, y1, x2, y2):
    x1, x2 = x1 / w, x2 / w
    y1, y2 = y1 / h, y2 / h

    return x1, y1, x2, y2


def frame_axis(w, h, fx, fy, bx, by):
    x = fx + (w * bx)
    y = fy + (h * by)

    return x, y


def one_video_pipeline(video_id, video_name):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(os.path.join('/home/Ricky/AI_City_Track5/Test_res_TYTR_DA_all', 'Two_stage_' + video_name), 
                            fourcc, 5, (1920, 1080))
    
    tracker = yolo1.track(source = os.path.join(video_path, video_name),
                              device = device_num, 
                              conf = 0.5)
    
    group_dir = get_dir(tracker)

    frame_id, no_id = 1, 0
    g_id = 0
    for t in tracker: # every frame
        frame = t.orig_img
        track_id = ''
        for g in t.boxes.data: # every group

            # determine dir
            dir = ''
            if t.boxes.is_track:
                track_id = str(int(g[4]))
                dir = group_dir[track_id]
            else:
                track_id = 'N_' + str(no_id)
                dir = group_dir[track_id]
                no_id += 1

            group_img, orig_group_img_wh = get_group_img(frame, g)

            DPM = yolo2.predict(source = group_img, conf = 0.5)

            person, motor = Driver_Passenger(DPM[0], dir, track_id)
            
            # assign final class to person
            get_helmet(group_img, person)

            # assign person, motor's axis into orig_frame
            DP_to_frame(g, person, motor, orig_group_img_wh)

            out_frame(frame, person, motor)

            output_res(video_id, frame_id, person, motor)

        
        video.write(frame)
        frame_id += 1

    video.release()
    cv2.destroyAllWindows()


def all_result(start, end):
    all_video_path = sorted(glob.glob(os.path.join(video_path, "*.mp4")))
    for v_p in all_video_path:

        v_p_split = v_p.split('/')
        video_name = v_p_split[len(v_p_split) - 1]
        video_id = get_video_id(video_name)

        if int(video_id) >= start and int(video_id) <= end:
            one_video_pipeline(video_id, video_name)


if __name__ == '__main__':
    arg = arg()
    video_w, video_h = 1920, 1080
    device_num = 1
    video_path = '/mnt/Nami/2023_AI_City_challenge_datasets/Track_5/sharp/vid'
    res_file_name = 'res_TYTR_DA_all_5_5.txt'
    bbox_color = [(255, 0, 0), (0, 255, 0)]
    class_ = ['motorbike', 'Person']
    yolo1 = YOLO('weights/FHD_stage1_best.pt') # detect Group (motor and person)
    yolo2 = YOLO('weights/stage2_best.pt') # detect motor and person from group
    classifer = resnet_101(num_classes = 2).cuda('cuda:' + str(device_num))
    classifer.load_state_dict(torch.load('weights/resnet101_DA.pt'))
    classifer.eval()
    

    all_result(arg.start, arg.end)
    # one_video_pipeline('85', '085.mp4')