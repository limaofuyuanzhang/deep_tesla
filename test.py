# coding:utf-8

import datasets
import utils
import params
# 加载数据
import datasets
(X_train, y_train), (X_test, y_test) = datasets.load_data()
n_train, n_test = X_train.shape[0], X_test.shape[0]


from keras.layers import *
from keras.models import Model
from keras import optimizers
import keras

# inputs = Input(shape=(params.FLAGS.img_h, params.FLAGS.img_w, params.FLAGS.img_c))
# x=Lambda(lambda x:x/255.0-0.5)(inputs)
# x=Conv2D(24, (5, 5), activation="relu", strides=(2, 2), padding="valid")(x)
# x=Conv2D(36, (5, 5), activation="relu", strides=(2, 2), padding="valid")(x)
# x=Conv2D(48, (5, 5), activation="relu", strides=(2, 2), padding="valid")(x)
# x=Conv2D(64, (3, 3), activation="relu", strides=(1, 1), padding="valid")(x)
# x=Conv2D(64, (3, 3), activation="relu", strides=(1, 1), padding="valid")(x)
# x=Flatten()(x)
# x=Dense(1164, activation='relu')(x)
# x=Dense(100, activation='relu')(x)
# x=Dense(50, activation='relu')(x)
# x=Dense(10, activation='relu')(x)
# outputs=Dense(1, activation='tanh')(x)
# model = Model(inputs=inputs,outputs=outputs)
# model.compile(optimizer=optimizers.Adadelta(),
#               loss='mse',
#               metrics=['accuracy'])
# model.fit(x=X_train,y=y_train,epochs=10,validation_split=0.2)

inputs = Input(shape=(params.FLAGS.img_h, params.FLAGS.img_w, params.FLAGS.img_c))
x=Lambda(lambda x:x/255.0-0.5)(inputs)
x=Conv2D(16, (3, 3), activation="relu")(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Conv2D(32, (3, 3), activation="relu")(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Conv2D(64, (3, 3), activation="relu")(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Flatten()(x)
x=Dense(500, activation='relu')(x)
x=Dense(100, activation='relu')(x)
x=Dense(20, activation='relu')(x)
outputs=Dense(1)(x)
model = Model(inputs=inputs,outputs=outputs)
model.compile(optimizer=optimizers.Adadelta(),
              loss='mse',
              metrics=['accuracy'])
model.fit(x=X_train,y=y_train,epochs=10,validation_split=0.2)


model.save_weights('simple-model.h5')
with open('simple-model.json', 'w') as f:
    f.write(model.to_json())
test_loss= model.evaluate(X_test, y_test,show_accuracy=True)
print('Test loss is:{}'.format(test_loss))
