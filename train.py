#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train YOLO model for your own dataset.
"""
import os, time, random, argparse
import pathlib
import numpy as np
import tensorflow.keras.backend as K
import tensorflow_model_optimization as tfmot
#from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, TerminateOnNaN, LambdaCallback
from tensorflow_model_optimization.sparsity import keras as sparsity

from yolo5.model import get_yolo5_train_model
from yolo5.data import yolo5_data_generator_wrapper, Yolo5DataGenerator
from yolo3.model import get_yolo3_train_model
from yolo3.data import yolo3_data_generator_wrapper, Yolo3DataGenerator
from yolo2.model import get_yolo2_train_model
from yolo2.data import yolo2_data_generator_wrapper, Yolo2DataGenerator
from common.utils import get_classes, get_anchors, get_dataset, optimize_tf_gpu
from common.model_utils import get_optimizer
from common.callbacks import EvalCallBack, CheckpointCleanCallBack, DatasetShuffleCallBack

# Try to enable Auto Mixed Precision on TF 2.0
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
optimize_tf_gpu(tf, K)


def change_input_size(model,h,w,ch=3):
    with tfmot.quantization.keras.quantize_scope():
        # loaded_model = tf.keras.models.load_model(keras_quantized_model_file)
        model.layers[0]._batch_input_shape = (None,h,w,ch)
        new_model = tf.keras.models.model_from_json(model.to_json())
        for layer,new_layer in zip(model.layers,new_model.layers):
            new_layer.set_weights(layer.get_weights())
    return new_model


def save_quantized_model(model, log_dir, epoch=1):
    os.makedirs(log_dir, exist_ok=True)    

    model2save = tf.keras.Model(
        model.get_layer('image_input').input,
        [model.get_layer('quant_predict_conv_1').output, 
            model.get_layer('quant_predict_conv_2').output])

    model2save = change_input_size(model2save, *(args.model_input_shape))

    tf.keras.save_model(model2save, os.path.join(log_dir, f'trained_{epoch}.h5'))
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model2save)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    quantized_tflite_model = converter.convert()
    quantized_tflite_file = pathlib.Path(os.path.join(log_dir, f'trained_{epoch}_uint8.tflite'))
    quantized_tflite_file.write_bytes(quantized_tflite_model)


class QuantizeModelCheckpoint(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        
        log_dir = 'logs/quant'

        save_quantized_model(self.model, log_dir, epoch=epoch)


def main(args):
    log_dir = os.path.join('logs', '000')
    class_names = get_classes(args.classes_path)
    num_classes = len(class_names)

    anchors = get_anchors(args.anchors_path)
    num_anchors = len(anchors)

    # get freeze level according to CLI option
    if args.weights_path:
        freeze_level = 0
    else:
        freeze_level = 1

    if args.freeze_level is not None:
        freeze_level = args.freeze_level

    # callbacks for training process
    logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=1)
    if args.quantize_aware_training:
        checkpoint_q = QuantizeModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-quant.h5'),
            monitor='val_loss',
            mode='min',
            verbose=1,
            save_weights_only=False,
            save_best_only=True,
            period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, mode='min', patience=10, verbose=1, cooldown=0, min_lr=1e-10)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='min')
    checkpoint_clean = CheckpointCleanCallBack(log_dir, max_val_keep=5, max_eval_keep=2)
    terminate_on_nan = TerminateOnNaN()

    callbacks = [logging, checkpoint, reduce_lr, early_stopping, terminate_on_nan, checkpoint_clean]
    
    if args.quantize_aware_training:
        callbacks.append(checkpoint_q)

    # get train&val dataset
    dataset = get_dataset(args.annotation_file)
    if args.val_annotation_file:
        val_dataset = get_dataset(args.val_annotation_file)
        num_train = len(dataset)
        num_val = len(val_dataset)
        dataset.extend(val_dataset)
    else:
        val_split = args.val_split
        num_val = int(len(dataset)*val_split)
        num_train = len(dataset) - num_val

    # assign multiscale interval
    if args.multiscale:
        rescale_interval = args.rescale_interval
    else:
        rescale_interval = -1  #Doesn't rescale

    # model input shape check
    input_shape = args.model_input_shape
    assert (input_shape[0]%32 == 0 and input_shape[1]%32 == 0), 'model_input_shape should be multiples of 32'

    # get different model type & train&val data generator
    if args.model_type.startswith('scaled_yolo4_') or args.model_type.startswith('yolo5_'):
        # Scaled-YOLOv4 & YOLOv5 entrance, use yolo5 submodule but now still yolo3 data generator
        # TODO: create new yolo5 data generator to apply YOLOv5 anchor assignment
        get_train_model = get_yolo5_train_model
        data_generator = yolo5_data_generator_wrapper

        # tf.keras.Sequence style data generator
        #train_data_generator = Yolo5DataGenerator(dataset[:num_train], args.batch_size, input_shape, anchors, num_classes, args.enhance_augment, rescale_interval, args.multi_anchor_assign)
        #val_data_generator = Yolo5DataGenerator(dataset[num_train:], args.batch_size, input_shape, anchors, num_classes, multi_anchor_assign=args.multi_anchor_assign)

        tiny_version = False
    elif args.model_type.startswith('yolo3_') or args.model_type.startswith('yolo4_'):
    #if num_anchors == 9:
        # YOLOv3 & v4 entrance, use 9 anchors
        get_train_model = get_yolo3_train_model
        data_generator = yolo3_data_generator_wrapper

        # tf.keras.Sequence style data generator
        #train_data_generator = Yolo3DataGenerator(dataset[:num_train], args.batch_size, input_shape, anchors, num_classes, args.enhance_augment, rescale_interval, args.multi_anchor_assign)
        #val_data_generator = Yolo3DataGenerator(dataset[num_train:], args.batch_size, input_shape, anchors, num_classes, multi_anchor_assign=args.multi_anchor_assign)

        tiny_version = False
    elif args.model_type.startswith('tiny_yolo3_') or args.model_type.startswith('tiny_yolo4_'):
    #elif num_anchors == 6:
        # Tiny YOLOv3 & v4 entrance, use 6 anchors
        get_train_model = get_yolo3_train_model
        data_generator = yolo3_data_generator_wrapper

        # tf.keras.Sequence style data generator
        #train_data_generator = Yolo3DataGenerator(dataset[:num_train], args.batch_size, input_shape, anchors, num_classes, args.enhance_augment, rescale_interval, args.multi_anchor_assign)
        #val_data_generator = Yolo3DataGenerator(dataset[num_train:], args.batch_size, input_shape, anchors, num_classes, multi_anchor_assign=args.multi_anchor_assign)

        tiny_version = True
    elif args.model_type.startswith('yolo2_') or args.model_type.startswith('tiny_yolo2_'):
    #elif num_anchors == 5:
        # YOLOv2 & Tiny YOLOv2 use 5 anchors
        get_train_model = get_yolo2_train_model
        data_generator = yolo2_data_generator_wrapper

        # tf.keras.Sequence style data generator
        #train_data_generator = Yolo2DataGenerator(dataset[:num_train], args.batch_size, input_shape, anchors, num_classes, args.enhance_augment, rescale_interval)
        #val_data_generator = Yolo2DataGenerator(dataset[num_train:], args.batch_size, input_shape, anchors, num_classes)

        tiny_version = False
    else:
        raise ValueError('Unsupported model type')

    # prepare online evaluation callback
    if args.eval_online:
        eval_callback = EvalCallBack(args.model_type, dataset[num_train:], anchors, class_names, args.model_input_shape, args.model_pruning, log_dir, eval_epoch_interval=args.eval_epoch_interval, save_eval_checkpoint=args.save_eval_checkpoint, elim_grid_sense=args.elim_grid_sense)
        callbacks.insert(-1, eval_callback) # add before checkpoint clean

    # prepare train/val data shuffle callback
    if args.data_shuffle:
        shuffle_callback = DatasetShuffleCallBack(dataset)
        callbacks.append(shuffle_callback)

    # prepare model pruning config
    pruning_end_step = np.ceil(1.0 * num_train / args.batch_size).astype(np.int32) * args.total_epoch
    if args.model_pruning:
        pruning_callbacks = [sparsity.UpdatePruningStep(), sparsity.PruningSummaries(log_dir=log_dir, profile_batch=0)]
        callbacks = callbacks + pruning_callbacks

    # prepare optimizer
    optimizer = get_optimizer(args.optimizer, args.learning_rate, average_type=None, decay_type=None)

    # support multi-gpu training
    if args.gpu_num >= 2:
        # devices_list=["/gpu:0", "/gpu:1"]
        devices_list=["/gpu:{}".format(n) for n in range(args.gpu_num)]
        strategy = tf.distribute.MirroredStrategy(devices=devices_list)
        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            # get multi-gpu train model
            model = get_train_model(args.model_type, anchors, num_classes, args.quantize_aware_training, weights_path=args.weights_path, freeze_level=freeze_level, optimizer=optimizer, label_smoothing=args.label_smoothing, elim_grid_sense=args.elim_grid_sense, model_pruning=args.model_pruning, pruning_end_step=pruning_end_step)

    else:
        # get normal train model
        model = get_train_model(args.model_type, anchors, num_classes, args.quantize_aware_training, weights_path=args.weights_path, freeze_level=freeze_level, optimizer=optimizer, label_smoothing=args.label_smoothing, elim_grid_sense=args.elim_grid_sense, model_pruning=args.model_pruning, pruning_end_step=pruning_end_step)

    # model.summary()

    # Transfer training some epochs with frozen layers first if needed, to get a stable loss.
    initial_epoch = args.init_epoch
    epochs = initial_epoch + args.transfer_epoch
    print("Transfer training stage")
    print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(num_train, num_val, args.batch_size, input_shape))
    #model.fit_generator(train_data_generator,
    model.fit_generator(data_generator(dataset[:num_train], args.batch_size, input_shape, anchors, num_classes, args.enhance_augment, rescale_interval, multi_anchor_assign=args.multi_anchor_assign),
            steps_per_epoch=max(1, num_train//args.batch_size),
            #validation_data=val_data_generator,
            validation_data=data_generator(dataset[num_train:], args.batch_size, input_shape, anchors, num_classes, multi_anchor_assign=args.multi_anchor_assign),
            validation_steps=max(1, num_val//args.batch_size),
            epochs=epochs,
            initial_epoch=initial_epoch,
            #verbose=1,
            workers=1,
            use_multiprocessing=False,
            max_queue_size=10,
            callbacks=callbacks)

    # Wait 2 seconds for next stage
    time.sleep(2)

    if args.decay_type or args.average_type:
        # rebuild optimizer to apply learning rate decay or weights averager,
        # only after unfreeze all layers
        if args.decay_type:
            callbacks.remove(reduce_lr)

        if args.average_type == 'ema' or args.average_type == 'swa':
            # weights averager need tensorflow-addons,
            # which request TF 2.x and have version compatibility
            import tensorflow_addons as tfa
            callbacks.remove(checkpoint)
            avg_checkpoint = tfa.callbacks.AverageModelCheckpoint(filepath=os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                                                  update_weights=True,
                                                                  monitor='val_loss',
                                                                  mode='min',
                                                                  verbose=1,
                                                                  save_weights_only=False,
                                                                  save_best_only=True,
                                                                  period=1)
            callbacks.insert(-1, avg_checkpoint) # add before checkpoint clean

        steps_per_epoch = max(1, num_train//args.batch_size)
        decay_steps = steps_per_epoch * (args.total_epoch - args.init_epoch - args.transfer_epoch)
        optimizer = get_optimizer(args.optimizer, args.learning_rate, average_type=args.average_type, decay_type=args.decay_type, decay_steps=decay_steps)

    # Unfreeze the whole network for further tuning
    # NOTE: more GPU memory is required after unfreezing the body
    print("Unfreeze and continue training, to fine-tune.")

    if args.gpu_num >= 2:
        with strategy.scope():
            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change

    else:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change

    print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(num_train, num_val, args.batch_size, input_shape))
    #model.fit_generator(train_data_generator,
    model.fit_generator(data_generator(dataset[:num_train], args.batch_size, input_shape, anchors, num_classes, args.enhance_augment, rescale_interval, multi_anchor_assign=args.multi_anchor_assign),
        steps_per_epoch=max(1, num_train//args.batch_size),
        #validation_data=val_data_generator,
        validation_data=data_generator(dataset[num_train:], args.batch_size, input_shape, anchors, num_classes, multi_anchor_assign=args.multi_anchor_assign),
        validation_steps=max(1, num_val//args.batch_size),
        epochs=args.total_epoch,
        initial_epoch=epochs,
        #verbose=1,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
        callbacks=callbacks)

    # Finally store model
    if args.model_pruning:
        model = sparsity.strip_pruning(model)
        
    if args.quantize_aware_training:
        save_quantized_model(model, log_dir, epoch='last')
    else:
        model.save(os.path.join(log_dir, 'trained_final.h5'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model definition options
    parser.add_argument('--model_type', type=str, required=False, default='yolo3_mobilenet_lite',
        help='YOLO model type: yolo3_mobilenet_lite/tiny_yolo3_mobilenet/yolo3_darknet/..., default=%(default)s')
    parser.add_argument('--anchors_path', type=str, required=False, default=os.path.join('configs', 'yolo3_anchors.txt'),
        help='path to anchor definitions, default=%(default)s')
    parser.add_argument('--model_input_shape', type=str, required=False, default='416x416',
        help = "Initial model image input shape as <height>x<width>, default=%(default)s")
    parser.add_argument('--weights_path', type=str, required=False, default=None,
        help = "Pretrained model/weights file for fine tune")

    # Data options
    parser.add_argument('--annotation_file', type=str, required=False, default='trainval.txt',
        help='train annotation txt file, default=%(default)s')
    parser.add_argument('--val_annotation_file', type=str, required=False, default=None,
        help='val annotation txt file, default=%(default)s')
    parser.add_argument('--val_split', type=float, required=False, default=0.1,
        help = "validation data persentage in dataset if no val dataset provide, default=%(default)s")
    parser.add_argument('--classes_path', type=str, required=False, default=os.path.join('configs', 'voc_classes.txt'),
        help='path to class definitions, default=%(default)s')

    # Training options
    parser.add_argument('--batch_size', type=int, required=False, default=16,
        help = "Batch size for train, default=%(default)s")
    parser.add_argument('--optimizer', type=str, required=False, default='adam', choices=['adam', 'rmsprop', 'sgd'],
        help = "optimizer for training (adam/rmsprop/sgd), default=%(default)s")
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-3,
        help = "Initial learning rate, default=%(default)s")
    parser.add_argument('--average_type', type=str, required=False, default=None, choices=[None, 'ema', 'swa', 'lookahead'],
        help = "weights average type, default=%(default)s")
    parser.add_argument('--decay_type', type=str, required=False, default=None, choices=[None, 'cosine', 'exponential', 'polynomial', 'piecewise_constant'],
        help = "Learning rate decay type, default=%(default)s")
    parser.add_argument('--transfer_epoch', type=int, required=False, default=20,
        help = "Transfer training (from Imagenet) stage epochs, default=%(default)s")
    parser.add_argument('--freeze_level', type=int,required=False, default=None, choices=[None, 0, 1, 2],
        help = "Freeze level of the model in transfer training stage. 0:NA/1:backbone/2:only open prediction layer")
    parser.add_argument('--init_epoch', type=int,required=False, default=0,
        help = "Initial training epochs for fine tune training, default=%(default)s")
    parser.add_argument('--total_epoch', type=int,required=False, default=250,
        help = "Total training epochs, default=%(default)s")
    parser.add_argument('--multiscale', default=False, action="store_true",
        help='Whether to use multiscale training')
    parser.add_argument('--rescale_interval', type=int, required=False, default=10,
        help = "Number of iteration(batches) interval to rescale input size, default=%(default)s")
    parser.add_argument('--enhance_augment', type=str, required=False, default=None, choices=[None, 'mosaic'],
        help = "enhance data augmentation type (None/mosaic), default=%(default)s")
    parser.add_argument('--label_smoothing', type=float, required=False, default=0,
        help = "Label smoothing factor (between 0 and 1) for classification loss, default=%(default)s")
    parser.add_argument('--multi_anchor_assign', default=False, action="store_true",
        help = "Assign multiple anchors to single ground truth")
    parser.add_argument('--elim_grid_sense', default=False, action="store_true",
        help = "Eliminate grid sensitivity")
    parser.add_argument('--data_shuffle', default=False, action="store_true",
        help='Whether to shuffle train/val data for cross-validation')
    parser.add_argument('--gpu_num', type=int, required=False, default=1,
        help='Number of GPU to use, default=%(default)s')
    parser.add_argument('--model_pruning', default=False, action="store_true",
        help='Use model pruning for optimization, only for TF 1.x')
    parser.add_argument('--quantize_aware_training', default=False, action="store_true",
        help='Use quantization aware training')


    # Evaluation options
    parser.add_argument('--eval_online', default=False, action="store_true",
        help='Whether to do evaluation on validation dataset during training')
    parser.add_argument('--eval_epoch_interval', type=int, required=False, default=10,
        help = "Number of iteration(epochs) interval to do evaluation, default=%(default)s")
    parser.add_argument('--save_eval_checkpoint', default=False, action="store_true",
        help='Whether to save checkpoint with best evaluation result')

    args = parser.parse_args()
    height, width = args.model_input_shape.split('x')
    args.model_input_shape = (int(height), int(width))

    main(args)
