from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading
import sys

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(-10, 0, 761, 521))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("Meyve-Sebze.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setEnabled(True)
        self.pushButton.setGeometry(QtCore.QRect(150, 180, 421, 191))
        font = QtGui.QFont()
        font.setPointSize(30)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton.setFont(font)
        self.pushButton.setMouseTracking(False)
        self.pushButton.setTabletTracking(False)
        self.pushButton.setToolTipDuration(-1)
        self.pushButton.setAutoFillBackground(False)
        self.pushButton.setStyleSheet("background-color: rgba(52, 152, 219, 0.2);\n"
                                      "border: 5px solid #000000; \n"
                                      "border-radius: 10px;")
        self.pushButton.setIconSize(QtCore.QSize(16, 16))
        self.pushButton.setAutoDefault(False)
        self.pushButton.setDefault(False)
        self.pushButton.setFlat(False)
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 749, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Uygulamayı Başlat"))

    def pre_process(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        return img

    def run_second_code(self):
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

        while True:
            success, frame = cap.read()
            img = np.asarray(frame)
            img = cv2.resize(img, (100, 100))
            img = self.pre_process(img)
            img = img.reshape(1, 100, 100, 3)

            predictions = model.predict(img)
            class_index = np.argmax(predictions)
            prob_val = np.amax(predictions)
            class_name = class_names[class_index]
            print(class_name, prob_val)

            if prob_val > 0.5:
                cv2.putText(frame, class_name + " " + str(prob_val), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Meyve Sınıflandırma", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

class MyMainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.setupUi(self)

        # Tıklanma olayını işleyen fonksiyonu bağla
        self.pushButton.clicked.connect(self.start_application)

    def start_application(self):
        # Yeni bir thread başlat
        thread = threading.Thread(target=self.run_second_code)
        thread.start()
        QtCore.QCoreApplication.instance().quit()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MyMainWindow()
    MainWindow.show()
    sys.exit(app.exec_())
