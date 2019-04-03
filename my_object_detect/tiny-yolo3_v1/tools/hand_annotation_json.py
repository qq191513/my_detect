import glob
import os
import json
#把labelme所有标注生成的json取出图片路径和boxes存到某个txt
#####################  train  ############################
#jpg path
data_set_root = '/home/mo/work/data_set'
jpg_path = os.path.join(data_set_root,'my_detect_hand_merged_v1/*.jpg')
jpg_list = glob.glob(jpg_path)

#json path
json_path = os.path.join(data_set_root,'my_detect_hand_merged_v1')

#save path
project_root = '/home/mo/work/yolo/keras-yolo3-master'
save_path = os.path.join(project_root,'generate')

#save txt
save_txt = os.path.join(save_path,'hand_train.txt')
#######################################################

# #####################  test  ############################
# #jpg path
# data_set_root = '/home/mo/work/data_set'
# jpg_path = os.path.join(data_set_root,'hand_detect/test/hand/*.jpg')
# jpg_list = glob.glob(jpg_path)
#
# #xml path
# xml_path = os.path.join(data_set_root,'hand_detect/test/hand')
#
# #save path
# project_root = '/home/mo/work/yolo/keras-yolo3-master'
# save_path = os.path.join(project_root,'generate')
#
# #save txt
# save_txt = os.path.join(save_path,'hand_test.txt')
# #######################################################
cls_id = 0
def get_joson_boxes(jpg):
    jpg_name = jpg.split('/')[-1].split('.')[0]
    json_name = jpg_name + '.json'
    json_file = os.path.join(json_path, json_name)
    try:
        print(json_file)
        in_file = json.load(open(json_file))
        shapes = in_file['shapes'][0]
        points = shapes['points']
        xmin = str(int(points[0][0]))
        ymin = str(int(points[0][1]))
        xmax = str(int(points[1][0]))
        ymax = str(int(points[1][1]))
        cls = str(cls_id)
        boxes = ' ' +xmin+','+ymin+','+xmax+','+ymax+','+cls
    except:
        print('找不到这个图片的json文件！')
    return boxes

with open(save_txt, 'w') as f:
    for jpg in jpg_list:
        #写jpg路径
        f.write(jpg)
        #写box参数
        boxes = get_joson_boxes(jpg)
        f.write(boxes)
        #写新行
        f.write('\n')
