# coding:utf-8
import cv2
import pandas as pd
import numpy as np
import params


# 读取所有的数据
def load_data():
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    cap = cv2.VideoCapture('./epochs/epoch01_front.mkv')
    ret, img = cap.read()

    for i in range(1, 10):
        # 读取图片
        pathX = './epochs/epoch0' + str(i) + '_front.mkv'
        cap = cv2.VideoCapture(pathX)

        while True:
            ret, img = cap.read()
            if(ret):
                x_train.append((img_pre_process(img)))
            else:
                break
        cap.release()

        # 读取转向角度
        pathY = './epochs/epoch0' + str(i) + '_steering.csv'
        wheel_sig = pd.read_csv(pathY)
        y_train.extend(wheel_sig['wheel'].values)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

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

(X_train, y_train), (X_test, y_test) = load_data()
n_train, n_test = X_train.shape[0], X_test.shape[0]