import os
import sys
import tools.developer_kit as dk
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
from tools.set_gpu import *
import sys
sys.path.append('../')
import config.cfg_tiny_yolo3 as cfg


# ################################   tiny yolo路径设置 训练hand  #######################################
annotation_path = cfg.annotation_path
classes_path = cfg.classes_path
anchors_path = cfg.anchors_path
output_root = cfg.output_root
load_pretrained = cfg.load_pretrained
restore_model_path = cfg.restore_model_path
final_fine_tune_save_weights_path = cfg.final_fine_tune_save_weights_path
checkpoint = cfg.checkpoint
log_dir = cfg.log_dir
freeze_body = cfg.freeze_body
epochs = cfg.epochs
batch_size = cfg.batch_size
# #############################################################################################

os.makedirs(output_root,exist_ok=True)

def _main():
    ### 训练参数
    class_names = dk.get_classes(classes_path)
    num_classes = len(class_names)
    anchors = dk.get_anchors(anchors_path)
    # batch_size = batch_size # note that more GPU memory is required after unfreezing the body（意思是全部重新训练要多张GPU）
    input_shape = (224,224) # multiple of 32, hw

    ### 创建模型
    #freeze_body=1训练大部分权重，为了学习新的类别，freeze_body=2冻结大部分层，只更新最后的类别个数
    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = dk.create_tiny_model(input_shape, anchors, num_classes,load_pretrained=load_pretrained,
            freeze_body=freeze_body, weights_path=restore_model_path)
    else:
        model = dk.create_model(input_shape, anchors, num_classes,load_pretrained=load_pretrained,
            freeze_body=freeze_body, weights_path=restore_model_path) # make sure you know what you freeze

    ### 训练日记
    logging = TensorBoard(log_dir=log_dir)
    ### 恢复模型
    ckt = ModelCheckpoint(checkpoint,monitor='val_loss', save_weights_only=True, save_best_only=True, period=5)

    ### 划分数据
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    ### 训练
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(dk.data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=dk.data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=epochs,
                initial_epoch=0,
                callbacks=[logging, ckt])

    ### 微调
    ### 学习调度
    # 当参考的评价指标停止改进时, 降低学习率, factor为每次下降的比例, 训练过程中,
    # 当指标连续patience次数还没有改进时, 降低学习率;
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    ### 训练终止方式
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    fine_tune_epoch= 100
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(dk.data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=dk.data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=epochs+fine_tune_epoch,
            initial_epoch=epochs+1,
            callbacks=[logging, ckt, reduce_lr, early_stopping])
        model.save_weights(final_fine_tune_save_weights_path)



if __name__ == '__main__':
    _main()
