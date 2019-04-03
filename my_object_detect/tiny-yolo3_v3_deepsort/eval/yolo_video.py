from yolo3.yolo import YOLO

import tools.developer_kit as dk
import sys
sys.path.append('../')
import config.cfg_tiny_yolo3 as cfg
###################   hand tiny yolo路径设置   ####################################
project_root = cfg.project_root
output_root = cfg.output_root
model_path = cfg.model_path
anchors_path = cfg.anchors_path
classes_path = cfg.classes_path
# 图片和video路径改这里
#jpg path
# 测试某文件夹所有图片
data_set_root = cfg.data_set_root
img_path = cfg.img_path
# 测试mp4
video_path = cfg.video_path
rot_number = cfg.rot_number
# 测试单张jpg
single_img_path = cfg.single_img_path
deepsort_model_filename = cfg.deepsort_model_filename
save_video_name = cfg.save_video_name
#######################     end    ################################################

setting_dict = {'model_path':model_path,
'anchors_path':anchors_path,
 'classes_path':classes_path
 }

if __name__ == '__main__':
    choice = 4
    if choice == 1:  #一张图片
        dk.detect_img(YOLO(setting_dict),single_img_path)
    if choice == 2:  # 图片list
        dk.detect_img_list(YOLO(setting_dict),img_path=img_path)
    elif choice == 3: # 视频
        dk.detect_video(YOLO(setting_dict), video_path, rot_number,output_path = "")
    elif choice == 4:  # yolo视频流+目标跟踪deepsort
        dk.detect_video_with_deepsort(YOLO(setting_dict), video_path, rot_number,
            output_path=save_video_name,deepsort_model_filename=deepsort_model_filename)
    else:
        print('done')


