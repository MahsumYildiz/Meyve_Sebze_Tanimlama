import cv2
import numpy as np
from keras.models import load_model

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    return img

# Modeli yükleme
model = load_model("model_trained_2.h5")

# Tanım yapılacak fotoğrafın dosya yolunu buraya yazıcaz
image_to_predict = "domates2.jpg"

img = cv2.imread(image_to_predict)
img = cv2.resize(img, (100, 100))
img = preProcess(img)
img = img.reshape(1, 100, 100, 3)

predictions = model.predict(img)
classIndex = np.argmax(predictions)
probVal = np.amax(predictions)

class_names = [
    'Apple Braeburn',    
    'Apple Golden 2',
    'Banana',     
    'Orange',    
    'Tomato 4',
    'Tomato not Ripened'
]


class_name = class_names[classIndex]

print(f"Fotoğraf tahmini sınıfı: {class_name}, Olasılık Değeri: {probVal}")
