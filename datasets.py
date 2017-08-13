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

    for i in range(1, 2):
        # 读取图片
        pathX = './epochs/epoch0' + str(i) + '_front.mkv'
        cap = cv2.VideoCapture(pathX)
        while True:
            ret, img = cap.read()
            if(ret):
                x_train.append(img)
            else:
                break
        cap.release()

        # for i in range(10):
        #     ret, img = cap.read()
        #     if (ret):
        #         x_train.append(img)
        #     else:
        #         break
        # cap.release()

        # 读取转向角度
        pathY = './epochs/epoch0' + str(i) + '_steering.csv'
        wheel_sig = pd.read_csv(pathY)
        y_train.append(wheel_sig['wheel'])

    x_train = np.array(x_train)
    print(len(y_train[0]))
    y_train = np.array(y_train).reshape((len(y_train[0])))
    print(y_train.shape)

    pathX = './epochs/epoch10_front.mkv'
    cap = cv2.VideoCapture(pathX)
    while True:
        ret, img = cap.read()
        if (ret):
            x_test.append(img)
        else:
            break
    cap.release()
    pathY = './epochs/epoch10_steering.csv'
    wheel_sig = pd.read_csv(pathY)
    y_test.append(wheel_sig['wheel'])

    x_test = np.array(x_test)
    y_test = np.array(y_test).reshape((len(y_test[0])))

    return (x_train,y_train),(x_test,y_test)

(X_raw, y_raw), (X_raw_test, y_raw_test) = load_data()