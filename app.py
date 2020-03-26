import numpy as np
import cv2

cap = cv2.VideoCapture(0)

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
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):  # q버튼을 누르면 영상 꺼짐
        break

cap.release()
cv2.destroyAllWindows()
