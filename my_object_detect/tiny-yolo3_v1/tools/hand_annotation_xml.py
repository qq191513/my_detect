import glob
import os
import xml.etree.ElementTree as ET
#把LabelImg所有标注生成的xml取出图片路径和boxes存到某个txt
#####################  train  ############################
#jpg path
data_set_root = '/home/mo/work/data_set'
jpg_path = os.path.join(data_set_root,'detect_merge_224_224/train/hand/*.jpg')
jpg_list = glob.glob(jpg_path)

#xml path
xml_path = os.path.join(data_set_root,'detect_merge_224_224/train/hand')

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
def get_xml_boxes(jpg):
    jpg_name = jpg.split('/')[-1].split('.')[0]
    xml_name = jpg_name + '.xml'
    xml_file = os.path.join(xml_path, xml_name)
    boxes_list=''
    try:
        in_file = open(xml_file)
        tree=ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            boxes_list=" " + ",".join([str(a) for a in b]) + ',' + str(cls_id)
            print(boxes_list)
    except:
        print(jpg)
        print('找不到这个图片的XML！')
    return boxes_list

with open(save_txt, 'w') as f:
    for jpg in jpg_list:
        #写jpg路径
        f.write(jpg)

        #写box参数
        boxes = get_xml_boxes(jpg)
        f.write(boxes)
        #写新行
        f.write('\n')
