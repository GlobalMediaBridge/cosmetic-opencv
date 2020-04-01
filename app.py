import sys
from os import path

import cv2
import numpy as np
import torch
import time

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui

from model import BiSeNet
from test import evaluate


class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)

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
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)

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

        tar_color = np.zeros_like(image)
        tar_color[:, :, 0] = b
        tar_color[:, :, 1] = g
        tar_color[:, :, 2] = r
        
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)
        
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
        
        masked = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

        image[parsing == 12] = masked[parsing == 12]
        image[parsing == 13] = masked[parsing == 13]
        return image

    def image_data_slot(self, image_data):
        start = time.time()
        parsing = evaluate(image_data, self.net)

        color = [0, 0, 255]
        face = self.makeup(image_data, parsing, color)
        self.image = self.get_qimage(face)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())
        print("Time Required : ", time.time() - start)
        

        self.update()

    def get_qimage(self, image: np.ndarray):
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
    def __init__(self, haarcascade_filepath, state_dict, parent=None):
        super().__init__(parent)
        self.reading = False

        self.initUI()
        fp = haarcascade_filepath
        self.face_detection_widget = FaceDetectionWidget(fp, state_dict)

        # TODO: set video port
        self.record_video = RecordVideo()

        image_data_slot = self.face_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.face_detection_widget)
        self.run_button = QtWidgets.QPushButton('Start')
        layout.addWidget(self.run_button)

        self.run_button.clicked.connect(self.record_video.start_recording)
        self.setLayout(layout)

    def initUI(self):
        self.setWindowTitle('My First Application')
        self.move(300, 300)
        self.resize(400, 200)

    def keyPressEvent(self, e):
        if self.reading == False:
            self.code = ''
        if e.key() == QtCore.Qt.Key_Return:
            self.reading = False
        else:
            self.code += chr(e.key())
            self.reading = True
        print(self.code)


def main(haar_cascade_filepath, state_dict):
    app = QtWidgets.QApplication(sys.argv)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes) # BiSeNet으로 net이리는 인스턴스 생성됨. 인자로 19 넣어서 만듦.
    net.cuda() # Tensor들을 GPU로 보내기
    net.load_state_dict(state_dict)
    net.eval()

    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget(haar_cascade_filepath, net)
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
