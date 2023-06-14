# 동영상 파일 열기
import cv2

cap = cv2.VideoCapture("../99_Studyfile/02_Sample/slow_traffic_small.mp4")

# SIFT 객체 생성
sift = cv2.SIFT_create()

while True:
    # 프레임읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 특징점 검출
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    # print(keypoints, descriptors)

    # 특징점 그리기
    frame = cv2.drawKeypoints(
        frame, keypoints, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )

    # 프레임 출력
    cv2.imshow("SIFT", frame)

    # q키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
