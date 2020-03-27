import numpy as np
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

cur_char = -1
prev_char = -1

reading = False
code = ''

while(True):
    ret, frame = cap.read()

    # 5px 짜리 직선그리기 (시작점),(끝점),(색상),크기
    frame = cv2.line(frame, (0, 0), (511, 511), (255, 0, 0), 5)
    # 3px 짜리 사각형 그리기 (시작점),(다음점),(색상),크기
    frame = cv2.rectangle(frame, (384, 0), (510, 128), (0, 255, 0), 3)
    # 꽉찬 원그리기 (중심점),반지름(색상),채우냐 마냐 -1 = 채우기 1 = 채우기x
    frame = cv2.circle(frame, (447, 63), 63, (0, 0, 255), -1)  # 원그리기
    # 글자쓰기
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'OpenCV', (10, 500), font,
                4, (255, 255, 255), 2, cv2.LINE_AA)
    ''' 전체화면
    cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        "frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    '''

    c = cv2.waitKey(1)
    if c == 27:
        break
    if c > -1:
        if reading == False:
            code = ''
        if c == 13:
            reading = False
        else:
            reading = True
            code += chr(c)
            cur_char = c

    prev_char = c

    if cur_char == ord('g'):
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif cur_char == ord('y'):
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    elif cur_char == ord('h'):
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    else:
        output = frame

    if reading == False:
        cv2.putText(frame, code, (10, 20), font,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', output)


cap.release()
cv2.destroyAllWindows()
