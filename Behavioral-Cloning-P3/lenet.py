from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import data as DT

batch_size = 32
train_generator = DT.generator(DT.train_samples, batch_size)
validation_generator = DT.generator(DT.validation_samples, batch_size)
nb_train = len(DT.train_samples) * 2
nb_valid = len(DT.validation_samples) * 2

# try to visualize the cropping images/reverse image
model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)),
                     input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 127.5 - 1.))
model.add(Convolution2D(6, 5, 5))
# model.add(Convolution2D(6, 5, 5, input_shape=(160, 320, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # default (2, 2)

model.add(Convolution2D(6, 5, 5,))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # default (2, 2)

model.add(Convolution2D(6, 5, 5,))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # default (2, 2)

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(120, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(84, activation="relu"))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=nb_train,
                    validation_data=validation_generator,
                    nb_val_samples=nb_valid,
                    # validation_split=0.2,
                    # shuffle=True,
                    # batch_size=batch_size,
                    nb_epoch=2)

model.save('model.h5')
