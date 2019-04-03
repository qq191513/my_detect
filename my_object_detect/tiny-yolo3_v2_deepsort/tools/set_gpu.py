#########################   使用GPU  动态申请显存占用 ####################
# 1、使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放内存，所以会导致碎片
# 2、visible_device_list指定使用的GPU设备号；
# 3、allow_soft_placement如果指定的设备不存在，允许TF自动分配设备（这个设置必须有，否则无论如何都会报cudnn不匹配的错误）
# 4、per_process_gpu_memory_fraction  指定每个可用GPU上的显存分配比
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
session_config = tf.ConfigProto(
            device_count={'GPU': 0},  #不能写成小写的gpu，否则无效
            gpu_options={'allow_growth': 1,
                # 'per_process_gpu_memory_fraction': 0.1,
                'visible_device_list': '0'},
                allow_soft_placement=True) #这个设置必须有，否则无论如何都会报cudnn不匹配的错误

sess = tf.Session(config=session_config)
KTF.set_session(sess)


#########################   END   ####################################