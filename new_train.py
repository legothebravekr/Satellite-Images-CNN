# ref: https://github.com/kss0222/CNN-for-regression/

from PIL import Image
import os, glob, sys, numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers import Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import losses
from keras import backend as K 
import matplotlib.pyplot as plt
import math
from keras.optimizers import SGD, Adam
from keras import metrics
from keras import models, layers, optimizers  
from datetime import datetime

#data reading section
img_dir = 'img_new'

image_w = 64
image_h = 64

pixel = image_h * image_w * 3

X = []
y = []
filenames = []

files = glob.glob(img_dir+"/"+ "*.png")

read = False

if read:
	for i, f in enumerate(files):
		try:
			img = Image.open(f)
			img = img.convert("RGB")
			img = img.resize((image_w, image_h))
			data = np.asarray(img)
			
			filenames.append(f)
			population=filenames[i][:-3].split(",")[3][:-1]   
			n = float(population)

			X.append(data)
			y.append(n)
			
			if i % 1000 == 0:
				print(" :\t", filenames[i]+ "  \t", y[i])
				
		except:
			print("error occured in " + str(i))
				
	X = np.array(X)
	Y = np.array(y, dtype=np.float32) 

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

	xy = (X_train, X_test, Y_train, Y_test)
	np.save("./binary_image_data.npy", xy)

X_train, X_test, y_train, y_test = np.load('./binary_image_data.npy', allow_pickle=True)

X_train = X_train.astype('float32')  
X_train = (X_train - np.mean(X_train))/np.std(X_train)
X_test = X_test.astype('float32')
X_test = (X_test - np.mean(X_test))/np.std(X_test)

print(X_train.shape)
print(X_train.shape[0])
print(np.mean(y_train))
print(np.mean(y_test))


y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


# model definition
droprate=0.2

model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=(64,64,3), activation="relu")) 
model.add(BatchNormalization())
model.add(Dropout(droprate))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(droprate))

model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(droprate))

model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))  
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1))
 
def mse(y_test, y_pred):
    return K.abs(K.sqrt(K.mean(K.square(y_pred - y_test))))

model.compile(loss= 'mean_squared_error', optimizer='adam', metrics=[mse])

model_dir = './model'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_path = model_dir + "/cnn_regression_classify.model"
    
# training settings

checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.summary()

tensorcallback = TensorBoard(log_dir='./logs',
                            histogram_freq=0,
                            write_graph=True,
                            write_images=False)

history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2, callbacks=[checkpoint, early_stopping, tensorcallback])

model.save(model_path)

test_loss = model.evaluate(X_test, y_test, verbose=0)
print('ground truth mean: ' + str(np.mean(y_test)))
print('ground truth std: ' + str(np.std(y_test)))
print('predicted mean: ' + str(K.mean(model(X_test))))
print('predicted std: ' + str(K.std(model(X_test))))
print('test loss:', test_loss[0])

print(history.history.keys())

# plotting
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('data_' + datetime.now().strftime('%y%m%d_%H%M%S') +'.png', dpi=300)
