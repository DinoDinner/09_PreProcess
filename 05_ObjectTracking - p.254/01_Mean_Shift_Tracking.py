import cv2

# 평균 이동 추적을 위한 초기 사각형 설정
track_window = None  # 객체의 위치 정보 저장할 변수
roi_hist = None  # 히스토그램을 저장할 변수
trem_crit = (cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)  # 10회 반복 수행 후 수렴

# 동영상 파일 열기
cap = cv2.VideoCapture(
    "/Users/park.s.w/Documents/GitHub/09_PreProcess/99_Studyfile/02_Sample/slow_traffic_small.mp4"
)

# 첫 프레임에서 추적할 객체 선택
ret, frame = cap.read()  # ret : retval(반환)
# print(ret, frame)
x, y, w, h = cv2.selectROI("Select Object", frame, False, False)
# ROI : Region of Interest(관심영역)

# 추적할 객체의 초기 히스토그램 계산
roi = frame[y : y + h, x : x + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 100])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# 추적할 객체의 초기 윈도우 설정
track_window = (x, y, w, h)

# cv2.imshow('roi test')

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 추적할 객체의 히스토그램 역투명 계산 : 객체의 색상 분포를 파악. 객체 색상과 유사도 판단
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # 평균 이동 알고리즘을 총해 객체 위치 추정
    _, track_window = cv2.meanShift(dst, track_window, trem_crit)

    # 이전 것과 비교해서 동일하면 계속 추적하는 형태
    # 계산된 역투명 히스토그램은 추적하려는 객체의 위치를 보다 정확하게 표시합니다.
    # 평균 이동 알고리즘 등의 방법을 사용하여 이 역투명 히스토그램을 이용해 객체의 위치를 추정하고
    # 업데이트하는 것이 일반적인 객체 추적 방법입니다.

    # 추적 결과를 사각형으로 표시
    x, y, w, h = track_window
    print("추적 결과 좌표", x, y, w, h)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 프레임 출력
    cv2.imshow("Mean Shift Tracking >>", frame)

    # 'q'키를 누르면 좋료
    if cv2.waitKey(30) & 0xFF == ord("q"):
        exit()

# 자원 해제
cap.release()
cv2.destroyAllWindows()
