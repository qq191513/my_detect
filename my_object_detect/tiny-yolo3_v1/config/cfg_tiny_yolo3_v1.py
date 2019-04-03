import os


project_root = '/home/mo/work/yolo/my_object_detect/'
branch_name = 'tiny-yolo3_v1'
output_root = os.path.join('/home/mo/work/output/hand_tiny_yolo3/',branch_name)

# ################################   tiny yolo3路径设置 训练hand  #######################################
annotation_path = os.path.join(project_root,branch_name, 'generate/hand_train.txt')
classes_path = os.path.join(project_root,branch_name, 'generate/hand_classes.txt')
anchors_path = os.path.join(project_root,branch_name, 'generate/hand_anchor_tiny_yolo.txt')
load_pretrained =True
restore_model_path = os.path.join(output_root, 'trained_weights_stage_1.h5')
final_fine_tune_save_weights_path = os.path.join(output_root, 'trained_weights_stage_1.h5')
checkpoint = os.path.join(output_root, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
log_dir = os.path.join(output_root, 'logs')
freeze_body = 3
epochs = 300
batch_size =32
# #############################################################################################


###################   hand tiny yolo3路径设置   测试hand  #############################

model_path = os.path.join(output_root, 'trained_weights_stage_1.h5')
anchors_path = os.path.join(project_root,branch_name, 'generate/hand_anchor_tiny_yolo.txt')
classes_path = os.path.join(project_root,branch_name, 'generate/hand_classes.txt')
# 图片和video路径改这里
#jpg path
# 测试某文件夹所有图片
data_set_root = '/home/mo/work/data_set'
img_path = os.path.join(data_set_root,'my_detect_hand_merged_v1/*.jpg')
# 测试mp4
# video_path = os.path.join('/home/mo/work/data_set/my_detect_hand_merged_v1_video','video_20190301_201739.mp4')
video_path = os.path.join('/home/mo/work/data_set/my_detect_hand_merged_v1_video','video_20190301_201915.mp4')
rot_number = 0
# 测试单张jpg
single_img_path = os.path.join(data_set_root,'my_detect_hand_merged_v1/9.jpg')
font_path = os.path.join(project_root,branch_name,'tools/font/FiraMono-Medium.otf')
##########################      end       ###################################


