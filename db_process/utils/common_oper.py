import os
import time


def re_print(info):
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {info}....................')



def parse_data(info_iter,save_intercls,save_dir_path,curr_defect_cam_num,class_ignore,score_ignore,curr_schema):
    defect_infos = [[] for i in range(len(curr_defect_cam_num))]
    for infos in info_iter:
        img_path, img_roi, class_name, internal_no, external_no, score = infos
        _, steel_no, cam_no, img_no, left_edge, right_edge, \
            roi_x1, roi_x2, roi_y1, roi_y2, \
            img_x1, img_x2, img_y1, img_y2, \
            steel_x1, steel_x2, steel_y1, steel_y2, _ = os.path.basename(img_path).split('_')
        fx = abs(int(steel_x2)-int(steel_x1))/abs(int(img_x2)-int(img_x1))
        fy = abs(int(steel_y2)-int(steel_y1))/abs(int(img_y2)-int(img_y1))
        # 保存内部编号的缺陷
        if int(internal_no) in save_intercls:
            img_roi_pil = Image.fromarray(img_roi)
            savedir_ = os.path.join(save_dir_path, steel_no)
            savedir_intercls = os.path.join(savedir_, class_name)
            if not os.path.exists(savedir_intercls):
                while 1:
                    try:
                        os.makedirs(savedir_intercls)
                        break
                    except Exception as E:
                        re_print(E)
                        time.sleep(0.1)
            img_name = f'score({score})_{os.path.basename(img_path)}'
            new_path = os.path.join(savedir_intercls, img_name)
            img_roi_pil.save(new_path)
        # 需要写入数据库的类别
        if int(external_no) != class_ignore and float(score) > float(score_ignore):
            defect_id = curr_defect_cam_num[cam_no] + 1
            leftToEdge = int(abs(int(img_x1)-int(left_edge)) * fx)
            rightToEdge = int(abs(int(right_edge)-int(img_x2)) * fx)
            area = (int(steel_x2)-int(steel_x1))*(int(steel_y2)-int(steel_y1))
            grade, cycle = 0, 0
            try:
                # 存数据
                if curr_schema['flag'] == 0:
                    img_roi_pil = Image.fromarray(img_roi)
                    dir_path = curr_schema[f'camera{cam_no}']
                    save_dir_ = os.path.join(dir_path,steel_no)
                    if not os.path.exists(save_dir_):
                        while 1:
                            try:
                                os.makedirs(save_dir_)
                                break
                            except Exception as E:
                                re_print(E)
                                time.sleep(0.1)
                    img_path = os.path.join(save_dir_, f'{defect_id}.bmp')
                    img_roi_pil.save(img_path)
                # 整理待写缺陷
                curr_roi_info = (int(defect_id),
                                 int(cam_no),
                                 int(steel_no),
                                 int(img_no),
                                 int(external_no),
                                 int(roi_x1),
                                 int(roi_x2),
                                 int(roi_y1),
                                 int(roi_y2),
                                 int(img_x1),
                                 int(img_x2),
                                 int(img_y1),
                                 int(img_y2),
                                 int(steel_x1),
                                 int(steel_x2),
                                 int(steel_y1),
                                 int(steel_y2),
                                 int(grade),
                                 int(area),
                                 int(leftToEdge),
                                 int(rightToEdge),
                                 int(cycle)
                                 )
                defect_infos[int(cam_no)-1].append(curr_roi_info)
            except Exception as E:
                re_print(E)
            else:
                curr_defect_cam_num[cam_no] = defect_id
    defect_infos = [tuple(ls) for ls in defect_infos]
    return curr_defect_cam_num,tuple(defect_infos)










