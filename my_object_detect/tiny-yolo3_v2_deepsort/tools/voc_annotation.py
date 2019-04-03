import xml.etree.ElementTree as ET
import os
sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

classes = ["person"]

#save path
project_root = '/home/mo/work/yolo/keras-yolo3-master'
save_path = 'generate'

# dataset path
data_set_root = '/home/mo/work/data_set'

def convert_annotation(year, image_id, list_file):
    data_set_file = os.path.join(data_set_root,'VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    in_file = open(data_set_file)
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        d=" " + ",".join([str(a) for a in b]) + ',' + str(cls_id)
        list_file.write(d)

for year, image_set in sets:

    # dataset path
    VOC_id_dataset = os.path.join(data_set_root, 'VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set))
    print('id_txt: ',VOC_id_dataset)

    # save path
    save_txt = '%s_%s.txt' % (year, image_set)
    save_file = os.path.join(project_root, save_path, save_txt)
    print('save_txt: ',save_file)

    image_ids = open(VOC_id_dataset).read().strip().split()
    list_file = open(save_file, 'w')
    for image_id in image_ids:
        VOC_jpg_dataset = '%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (data_set_root, year, image_id)
        list_file.write(VOC_jpg_dataset)
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()

