import sys
from os import path

import cv2
import numpy as np
import torch
import time

from PyQt5 import QtCore, QtGui, QtWidgets

from model import BiSeNet
from test import evaluate
import source


class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        read, data = self.camera.read()
        if read:
            self.image_data.emit(data)


class FaceDetectionWidget(QtWidgets.QWidget):
    def __init__(self, haar_cascade_filepath, net, parent=None):
        super().__init__(parent)
        self.classifier = cv2.CascadeClassifier(haar_cascade_filepath)
        self.net = net
        self.image = QtGui.QImage()
        self._width = 2
        self._min_size = (30, 30)
        self.timer = 0
    
    def initUI(self, main_widget):
        self.main_widget = main_widget
        self.face = main_widget.face
        self.pink = main_widget.pink
        self.cosmetic = main_widget.cosmetic
        self.main_window = main_widget.main_window
        self.barcode = main_widget.barcode


    def detect_faces(self, image: np.ndarray):
        # haarclassifiers work better in black and white
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image)

        faces = self.classifier.detectMultiScale(gray_image,
                                                 scaleFactor=1.3,
                                                 minNeighbors=4,
                                                 flags=cv2.CASCADE_SCALE_IMAGE,
                                                 minSize=self._min_size)

        return faces

    def makeup(self, image, parsing, color):
        b, g, r = color

        filled = image.copy()
        tar_color = np.zeros_like(image)
        tar_color[:, :, 0] = b
        tar_color[:, :, 1] = g
        tar_color[:, :, 2] = r
        
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)
        
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
        
        masked = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

        filled[parsing == 12] = masked[parsing == 12]
        filled[parsing == 13] = masked[parsing == 13]
        blured = cv2.GaussianBlur(filled, (5, 5), 0)
        image[parsing == 12] = blured[parsing == 12]
        image[parsing == 13] = blured[parsing == 13]
        
        return image

    def image_data_slot(self, image_data):
        start = time.time()
        parsing = evaluate(image_data, self.net)
        self.isDetected = len(image_data[parsing==12]) > 0
        if(self.isDetected):
            self.timer = 0
            self.face.setStyleSheet("image: url(:/newPrefix/person_1.png);")
            self.pink.setStyleSheet("")
        else:
            if self.timer == 0:
                self.timer = time.time() + 3
            elif self.timer <= time.time():
                self.main_window.code = ''
            self.face.setStyleSheet("image: url(:/newPrefix/person_0.png);")
            self.pink.setStyleSheet("image: url(:/newPrefix/pink.png);")
                
        if(self.main_window.code == ''):
            face = image_data
            self.cosmetic.setStyleSheet("image: url(:/newPrefix/cosmetic_0.png);")
            self.barcode.setStyleSheet("image: url(:/newPrefix/none.png);")
        else:
            print(self.main_window.code)
            color = [125, 93, 253]
            face = self.makeup(image_data, parsing, color)
            self.cosmetic.setStyleSheet("image: url(:/newPrefix/cosmetic_1.png);")
            self.barcode.setStyleSheet(f"image: url(:/newPrefix/{self.main_window.code}.png);")


        self.image = self.get_qimage(face)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())
        print("Time Required : ", time.time() - start)
        
        self.update()

    def get_qimage(self, image: np.ndarray):
        image = cv2.resize(image, (1440, 1080))
        image = cv2.flip(image, 1)
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()


class MainWidget(QtWidgets.QWidget):
    def __init__(self, haarcascade_filepath, state_dict, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        fp = haarcascade_filepath
        self.face_detection_widget = FaceDetectionWidget(fp, state_dict)

        self.initUI(main_window)
        self.face_detection_widget.initUI(self)


        # TODO: set video port
        self.record_video = RecordVideo()
        self.record_video.start_recording()

        image_data_slot = self.face_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)

        
        
        # layout = QtWidgets.QVBoxLayout()

        # layout.addWidget(self.face_detection_widget)

        # self.setLayout(layout)

    def initUI(self, MainWindow):

        self.setObjectName("centralwidget")

        self.layout = QtWidgets.QHBoxLayout()  
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.addWidget(self.face_detection_widget)  
        self.layout.addStretch(1)
        self.setLayout(self.layout) 

        self.back_1 = QtWidgets.QLabel(self)
        self.back_1.setGeometry(QtCore.QRect(40, 861, 180, 180))
        self.back_1.setStyleSheet("image: url(:/newPrefix/back.png);")
        self.back_1.setText("")
        self.back_1.setObjectName("back_1")

        self.back_2 = QtWidgets.QLabel(self)
        self.back_2.setGeometry(QtCore.QRect(250, 861, 180, 180))
        self.back_2.setStyleSheet("image: url(:/newPrefix/back.png);")
        self.back_2.setText("")
        self.back_2.setObjectName("back_2")

        self.face = QtWidgets.QLabel(self)
        self.face.setGeometry(QtCore.QRect(40, 861, 180, 180))
        self.face.setStyleSheet("image: url(:/newPrefix/person_0.png);")
        self.face.setText("")
        self.face.setObjectName("face")

        self.cosmetic = QtWidgets.QLabel(self)
        self.cosmetic.setGeometry(QtCore.QRect(250, 861, 180, 180))
        self.cosmetic.setStyleSheet("image: url(:/newPrefix/cosmetic_0.png);")
        self.cosmetic.setText("")
        self.cosmetic.setObjectName("cosmetic")

        self.pink = QtWidgets.QLabel(self)
        self.pink.setGeometry(QtCore.QRect(0, 0, 1440, 1080))
        self.pink.setStyleSheet("image: url(:/newPrefix/pink.png);")
        self.pink.setText("")
        self.pink.setObjectName("pink")

        self.barcode = QtWidgets.QLabel(self)
        self.barcode.setGeometry(QtCore.QRect(1440, 0, 480, 1080))
        self.barcode.setStyleSheet("image: url(:/newPrefix/none.png);")
        self.barcode.setText("")
        self.barcode.setObjectName("barcode")
        
        #self.retranslateUi(MainWindow)
        #QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent=parent)
        self.initWindow()
        self.reading = False
        self.code = ''
        

    def initWindow(self):
        self.setObjectName("MainWindow")
        self.setEnabled(True)
        self.resize(1920, 1080)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())

        self.setSizePolicy(sizePolicy)
        self.setMaximumSize(QtCore.QSize(1920, 1080))
        
    def keyPressEvent(self, e):
        if self.reading == False:
            self.code = ''
        if e.key() == QtCore.Qt.Key_Return:
            self.reading = False
        else:
            self.code += chr(e.key())
            self.reading = True


def main(haar_cascade_filepath, state_dict):
    app = QtWidgets.QApplication(sys.argv)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes) # BiSeNet으로 net이리는 인스턴스 생성됨. 인자로 19 넣어서 만듦.
    net.cuda() # Tensor들을 GPU로 보내기
    net.load_state_dict(state_dict)
    net.eval()

    main_window = MainWindow()

    main_widget = MainWidget(haar_cascade_filepath, net, main_window)
    main_window.setCentralWidget(main_widget)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    script_dir = path.dirname(path.realpath(__file__))
    cp = path.join(script_dir, 'cp', '79999_iter.pth')
    state_dict = torch.load(cp)
    cascade_filepath = path.join(script_dir,
                                 'data',
                                 'haarcascade_frontalface_default.xml')

    cascade_filepath = path.abspath(cascade_filepath)
    main(cascade_filepath, state_dict)
