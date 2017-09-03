# coding:utf-8
import cv2
import pandas as pd
import numpy as np
import params
import random


# 读取所有的数据
def load_data():
    # 读取训练集
    y_train_original = []
    x_train = []
    y_train = []


    for i in range(1, 10):
        # 读取转向角度
        pathY = './epochs/epoch0' + str(i) + '_steering.csv'
        wheel_sig = pd.read_csv(pathY)
        y_train_original.extend(wheel_sig['wheel'].values)

        # 读取图片
        pathX = './epochs/epoch0' + str(i) + '_front.mkv'
        cap = cv2.VideoCapture(pathX)

        i = 0
        while True:
            ret, img = cap.read()
            if(ret):
                img_mb = img_change_brightness(img)
                img_ohf = img_horizontal_flip(img)
                img_mbhf = img_horizontal_flip(img_mb)
                x_train.append(img_pre_process(img))
                x_train.append(img_pre_process(img_mb))
                x_train.append(img_pre_process(img_ohf))
                x_train.append(img_pre_process(img_mbhf))
                y_train.append(y_train_original[i])
                y_train.append(y_train_original[i])
                y_train.append(-y_train_original[i])
                y_train.append(-y_train_original[i])
                i += 1
            else:
                break
        cap.release()

    # 打乱顺序
    index = [i for i in range(len(x_train))]
    random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # 读取测试集
    x_test = []
    y_test = []
    pathX = './epochs/epoch10_front.mkv'
    cap = cv2.VideoCapture(pathX)
    while True:
        ret, img = cap.read()
        if (ret):
            x_test.append(img_pre_process(img))
        else:
            break
    cap.release()
    pathY = './epochs/epoch10_steering.csv'
    wheel_sig = pd.read_csv(pathY)
    y_test.append(wheel_sig['wheel'])

    x_test = np.array(x_test)
    y_test = np.array(y_test).reshape((len(y_test[0])))

    return (x_train,y_train),(x_test,y_test)

def img_pre_process(img):
    """
    Processes the image and returns it
    :param img: The image to be processed
    :return: Returns the processed image
    """
    ## Chop off 1/3 from the top and cut bottom 150px(which contains the head of car)
    shape = img.shape
    img = img[int(shape[0]/3):shape[0]-150, 0:shape[1]]
    ## Resize the image
    img = cv2.resize(img, (params.FLAGS.img_w, params.FLAGS.img_h), interpolation=cv2.INTER_AREA)
    ## Return the image sized as a 4D array
    return np.resize(img, (params.FLAGS.img_w, params.FLAGS.img_h, params.FLAGS.img_c))

def img_change_brightness(img):
    """ Changing brightness of img to simulate day and night conditions
    :param img: The image to be processed
    :return: Returns the processed image
   """
    # Convert the image to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Compute a random brightness value and apply to the image
    brightness = np.random.uniform(0.5,1.5)
    img[:, :, 2] = img[:, :, 2] * brightness
    # Convert back to RGB and return
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

def img_horizontal_flip(img):
    img = img[:,::-1,:]
    return img
