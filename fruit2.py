import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from visualkeras import layered_view



path = "Data2/Training"
pathTest = "Data2/Test"
myList = os.listdir(path)
myTestList = os.listdir(pathTest)

noOfClasses = len(myList)
numOfTests = len(myTestList)
print("Label(sınıf)", noOfClasses)


images = []
imagesTest = []
labels = []  
testLabels = []

for i in range(noOfClasses):
    myImageList = os.listdir(path + "\\" + myList[i])
    for j in myImageList:
        img = cv2.imread(path + "\\" + myList[i] + "\\" + j)
        img = cv2.resize(img, (100,100))
        images.append(img)
        labels.append(i)

for i in range(numOfTests):
    myImageList = os.listdir(pathTest + "\\" + myTestList[i])
    for j in myImageList:
        img = cv2.imread(pathTest + "\\" + myTestList[i] + "\\" + j)
        img = cv2.resize(img, (100,100))
        imagesTest.append(img)
        testLabels.append(i)
        
print(len(images))

print("---------------------")

print(len(imagesTest))        

images = np.array(images)
imagesTest = np.array(imagesTest)
labels = np.array(labels)
testLabels = np.array(testLabels)

print(images.shape)
print(imagesTest.shape)

x_train, y_train = images, labels
x_test, y_test = imagesTest, testLabels

# test_ratio hesaplanması
test_ratio = len(x_test) / (len(x_train) + len(x_test))

# Eğitim ve test verilerinin ayrılması
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=test_ratio, random_state=42)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print(images.shape)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    return img


x_train = np.array(list(map(preProcess, x_train)))
x_test = np.array(list(map(preProcess, x_test)))
x_validation = np.array(list(map(preProcess, x_validation)))

x_train = x_train.reshape(-1, 100, 100, 3)
print(x_train.shape)
x_test = x_test.reshape(-1, 100, 100, 3)
x_validation = x_validation.reshape(-1, 100, 100, 3)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                              height_shift_range=0.1,
                              zoom_range=0.1,
                              rotation_range=10)


dataGen.fit(x_train)
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)




model = Sequential()
model.add(Conv2D(input_shape=(100, 100, 3), filters=32, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(input_shape=(100, 100, 3), filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(input_shape=(100, 100, 3), filters=128, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(input_shape=(100, 100, 3), filters=256, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(input_shape=(100, 100, 3), filters=256, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(input_shape=(100, 100, 3), filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dense(units=1024, activation="relu"))
model.add(Dropout(0.35))
model.add(Dense(units=noOfClasses, activation="softmax"))


model.compile(loss="categorical_crossentropy", optimizer=("adam"), metrics=["accuracy"])


hist = model.fit_generator(dataGen.flow(x_train, y_train, batch_size=256),
                            validation_data=(x_validation, y_validation),
                            epochs=32, steps_per_epoch=x_train.shape[0] // 256, shuffle=1 )
 
fig, ax = plt.subplots()
ax.set_title("Nöral Ağ Mimarisi")
layered_view(model, to_file="model.png").show()


model.save("model_trained_2.h5")

hist.history.keys()
plt.figure()
plt.plot(hist.history["loss"], label="Eğitim Loss")
plt.plot(hist.history["val_loss"], label="Val Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"], label="Eğitim Accuracy")
plt.plot(hist.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.show()

score = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", score[0])
print("Test acc:", score[1])

y_pred = model.predict(x_validation)
y_pred_class = np.argmax(y_pred, axis=1)
Y_true = np.argmax(y_validation, axis=1)

cm = confusion_matrix(Y_true, y_pred_class)
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cm, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt=".1f", ax=ax)
plt.xlabel("predicted")
plt.ylabel("true")
plt.title("cm")
plt.show()



















