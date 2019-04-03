# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
from timeit import default_timer as timer
import tools.developer_kit as dk
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from tools.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

import sys
sys.path.append('../')
import config.cfg_tiny_yolo3 as cfg
###################   hand tiny yolo路径设置   #############################
project_root = cfg.project_root
font_path = cfg.font_path
##########################      end       #################################


class YOLO(object):

    def __init__(self,_obj):
        self.__dict__.update(_obj)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.input_shape = (224,224) # multiple of 32, hw
        self.gpu_num = 1
        self.score = 0.3
        self.iou = 0.45
        self.model_image_size =(224, 224)
        self.gpu_num = 1
        self.sess = tf.Session(config=dk.set_gpu())
        K.set_session(self.sess)
        self.boxes, self.scores, self.classes = self.predict()




    def _get_class(self):
        # classes_path = os.path.expanduser(classes_path)
        with open(self.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        # anchors_path = os.path.expanduser(self.anchors_path)
        with open(self.anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def predict(self):
        # 初始化变量并打印
        assert self.model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        num_classes = len(self.class_names)
        num_anchors = len(self.anchors)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        ### 直接加载模型参数，不用再重新定义网络
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(self.model_path, compile=False)
        except:
            if is_tiny_version :
                self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes)
            else:
                self.yolo_model =self.yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match

        #依据最后的输出端口判读是否加载正确
        gg = self.yolo_model.layers[-1].output_shape[-1]
        nn = num_anchors / len(self.yolo_model.output) * (num_classes + 5)
        assert gg ==nn , 'Mismatch between model and given anchor and class sizes'

        #生成画框颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        #随机种子
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # 多GOU模式
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)

        # 网络输出预测结果
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    # def detect_image(self, image):
    #     start = timer()
    #
    #     if self.model_image_size != (None, None):
    #         assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
    #         assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
    #         boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
    #     else:
    #         new_image_size = (image.width - (image.width % 32),
    #                           image.height - (image.height % 32))
    #         boxed_image = letterbox_image(image, new_image_size)
    #     image_data = np.array(boxed_image, dtype='float32')
    #
    #     print(image_data.shape)
    #     image_data /= 255.
    #     image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    #
    #     out_boxes, out_scores, out_classes = self.sess.run(
    #         [self.boxes, self.scores, self.classes],
    #         feed_dict={
    #             self.yolo_model.input: image_data,
    #             self.input_image_shape: [image.size[1], image.size[0]],
    #             K.learning_phase(): 0
    #         })
    #
    #     print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
    #
    #     font = ImageFont.truetype(font=font_path,
    #                 size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    #     thickness = (image.size[0] + image.size[1]) // 300
    #
    #     for i, c in reversed(list(enumerate(out_classes))):
    #         predicted_class = self.class_names[c]
    #         box = out_boxes[i]
    #         score = out_scores[i]
    #
    #         label = '{} {:.2f}'.format(predicted_class, score)
    #         draw = ImageDraw.Draw(image)
    #         label_size = draw.textsize(label, font)
    #
    #         top, left, bottom, right = box
    #         top = max(0, np.floor(top + 0.5).astype('int32'))
    #         left = max(0, np.floor(left + 0.5).astype('int32'))
    #         bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    #         right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    #         print(label, (left, top), (right, bottom))
    #
    #         if top - label_size[1] >= 0:
    #             text_origin = np.array([left, top - label_size[1]])
    #         else:
    #             text_origin = np.array([left, top + 1])
    #
    #         # My kingdom for a good redistributable image drawing library.
    #         for i in range(thickness):
    #             draw.rectangle(
    #                 [left + i, top + i, right - i, bottom - i],
    #                 outline=self.colors[c])
    #         draw.rectangle(
    #             [tuple(text_origin), tuple(text_origin + label_size)],
    #             fill=self.colors[c])
    #         draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    #         del draw
    #
    #     end = timer()
    #     print(end - start)
    #     return image

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print('image_data.shape: ',image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font=font_path,
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()



