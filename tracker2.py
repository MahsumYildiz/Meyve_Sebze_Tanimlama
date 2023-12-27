import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.equalizeHist(img)
    img = img / 255
    return img

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 480)

class_names = [
  'Apple Braeburn',    
  'Apple Golden 2',
  'Banana',      
  'Orange',    
  'Tomato 4',
  'Tomato not Ripened'
]


model = load_model("model_trained_2.h5")

tracker = cv2.TrackerCSRT_create()


# İlk karede nesneyi seçin
ret, frame = cap.read()
bbox = cv2.selectROI("Select Object", frame, False)
tracker.init(frame, bbox)

while True:
    success, frame = cap.read()
    img = np.asarray(frame)
    img = cv2.resize(img, (100, 100))
    img = preProcess(img)
    img = img.reshape(1, 100, 100, 3)

    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probVal = np.amax(predictions)
    class_name = class_names[classIndex]
    print(class_name, probVal)

    success, bbox = tracker.update(frame)

    # Başarıyla takip edildiyse, çerçeve içine alICAZ
    if success:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    if probVal > 0.5:
       
        cv2.putText(frame, class_name + " " + str(probVal), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Fruit sınıflandırma", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()