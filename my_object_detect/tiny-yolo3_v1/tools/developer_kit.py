
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from timeit import default_timer as timer
from PIL import Image
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from tools.utils import get_random_data
import tensorflow as tf
import os
import time
import glob
import cv2
def detect_img(yolo,img):

    image = Image.open(img)
    r_image = yolo.detect_image(image)
    r_image.show()
    time.sleep(10)
    yolo.close_session()

def detect_img_list(yolo,img_path):
    img_list = glob.glob(img_path)
    for img in img_list:
        image = Image.open(img)
        r_image = yolo.detect_image(image)
        # r_image.show()
        r_image = np.array(r_image)
        r_image = cv2.cvtColor(r_image,cv2.COLOR_BGR2RGB)
        cv2.imshow('fuck',r_image)
        cv2.waitKey(1000)
        # r_image.destroyWindow('fuck')
    yolo.close_session()

def detect_video(yolo, video_path,rot_number, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps  = vid.get(cv2.CAP_PROP_FPS)
    video_size  = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        # 1、opencv是以brg方式打开的，所以要转换成rbg才能识别
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2、图片旋转
        frame = np.rot90(frame, rot_number)

        # 3、检测
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)


        # 4、Image类型转化成ndarry型
        result = np.asarray(image)

        # 5、还要还原回BGR格式
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    # 输入口
    image_input = Input(shape=(None, None, 3))  # 图片输入格式

    # 初始化变量并打印
    h, w = input_shape  # 尺寸：416*416
    num_anchors = len(anchors)  # anchor数量
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    # 标签张量
    y_true = []
    for l in range(3):
        d = {0: 32, 1: 16, 2: 8}[l]
        k = h //d
        j = w // {0: 32, 1: 16, 2: 8}[l]
        b = Input(shape=(k, j,num_anchors // 3, num_classes + 5))
        y_true.append(b)

    # yolo架构初始化
    model_body = yolo_body(image_input, num_anchors // 3, num_classes)  # model

    # 加载预训练模型
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)  # 加载参数，跳过错误
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            # 冻结层(即固定某层参数在训练的时候不变)因为下载那些大佬训练得非常好的model，前面不需要训练了，固定就行了，只要训练后面的几层进行分类
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False  # 将其他层的训练关闭
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    # yolo loss函数初始化
    model_loss = Lambda(yolo_loss,
                        output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors,
                                   'num_classes': num_classes,
                                   'ignore_thresh': 0.5})(model_body.output + y_true)  # 后面是输入，前面是输出

    model = Model([model_body.input] + y_true, model_loss)  # 模型，inputs和outputs

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    # 输入口
    image_input = Input(shape=(None, None, 3))

    # 初始化变量并打印
    h, w = input_shape
    num_anchors = len(anchors)

    # 标签张量
    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    # tiny yolo架构初始化
    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    # 加载预训练模型
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    # tiny yolo loss函数初始化
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

def set_gpu():
    # 1、设置GPU模式
    session_config = tf.ConfigProto(
        device_count={'GPU': 0},
        gpu_options={'allow_growth': 1,
                     # 'per_process_gpu_memory_fraction': 0.1,
                     'visible_device_list': '0'},
        allow_soft_placement=True)
    return  session_config