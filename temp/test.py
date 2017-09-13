# coding:utf-8
#
# import numpy as np
# from params import *
#
# def train_data_generator(purpose):    """Training data generator
#     :return: (x[BATCH, IMG_ROWS, IMG_COLS, NUM_CH], y)
#    """
#     _x = np.zeros((FLAGS.batch_size, FLAGS.img_w, FLAGS.img_h, FLAGS.img_c), dtype=np.float)
#     _y = np.zeros(FLAGS.batch_size, dtype=np.float)
#     out_idx = 0
#     ## Preconditions
#     # purpose = 'train'
#     assert len(imgs[purpose]) == len(wheels[purpose])
#     n_purpose = len(imgs[purpose])    assert n_purpose > 0
#     while 1:        """Loading random frame of the video repeatly
#         """
#         ## Get a random line and get the steering angle
#         frame_idx = np.random.randint(n_purpose)        ## Find angle
#         angle = wheels[purpose][frame_idx]        ## Find frame
#         img = imgs[purpose][frame_idx]        ## Implement data augmentation
#         # img, angle = data_augment_pipeline(img, angle)
#
#         # Check if we've got valid values
#         if img is not None:
#             _x[out_idx] = img
#             _y[out_idx] = angle
#             out_idx += 1
#         # Check if we've enough values to yield
#         if out_idx >= FLAGS.batch_size:            yield _x, _y            # Reset the values back
#             _x = np.zeros((FLAGS.batch_size, FLAGS.img_w, FLAGS.img_h, FLAGS.img_c), dtype=np.float)
#             _y = np.zeros(FLAGS.batch_size, dtype=np.float)
#             out_idx = 0
import h5py
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
% matplotlib inline

h5_path = 'deep_tesla_test.hdf5'
if os.path.exists(h5_path):
    os.remove(h5_path)

f = h5py.File(h5_path, "w")
origin_length = 5
train_origin_length = int(origin_length * 0.8)
train_length = 2 * train_origin_length
val_length = origin_length - train_origin_length

images = f.create_dataset('images', (origin_length, 720, 1280, 3), dtype='uint8')
labels = f.create_dataset('labels', (origin_length,), dtype='float32')
tmp_images = f.create_dataset('tmp_images', (train_length, 215, 640, 3), dtype='uint8')
tmp_labels = f.create_dataset('tmp_labels', (train_length,), dtype='float32')
train_images = f.create_dataset('train_images', (train_length, 215, 640, 3), dtype='uint8')
train_labels = f.create_dataset('train_labels', (train_length,), dtype='float32')
val_images = f.create_dataset('val_images', (val_length, 215, 640, 3), dtype='uint8')
val_labels = f.create_dataset('val_labels', (val_length,), dtype='float32')

x_train = []
y_train = []

# 读取所有原始数据
image_index = 0
landmark_index = 0
for i in range(1, 2):
    # 读取转向角度
    y_train_original = []
    pathY = '../epochs/epoch0' + str(i) + '_steering.csv'
    wheel_sig = pd.read_csv(pathY)
    y_train_original.extend(wheel_sig['wheel'].values)
    length = len(y_train_original)
    labels[landmark_index:landmark_index + length] = y_train_original[0:origin_length]
    landmark_index += len(y_train_original)

    # 读取图片
    pathX = '../epochs/epoch0' + str(i) + '_front.mkv'
    cap = cv2.VideoCapture(pathX)

    while True:
        ret, img = cap.read()
        if (ret and image_index < origin_length):
            f['image'][image_index] = img
            image_index += 1
        else:
            break
    cap.release()

# 设置临时数据和校验数据
index = range(origin_length)
random.shuffle(index)
for i in range(val_length):
    val_images[i] = img_pre_process(images[index[i]])
    val_labels[i] = labels[index[i]]

for i in range(val_length, origin_length):
    img = img_pre_process(images[index[i]])
    tmp_images[i - val_length] = img
    tmp_labels[i - val_length] = labels[index[i]]

    flip_img = img_horizontal_flip(img)
    tmp_images[i - val_length + train_origin_length] = flip_img
    tmp_labels[i - val_length + train_origin_length] = -labels[index[i]]

index = range(train_length)
random.shuffle(index)
for i in range(train_length):
    train_images[i] = tmp_images[index[i]]
    train_labels[i] = tmp_labels[index[i]]

# 打乱临时数据生成训练数据

f.close()
