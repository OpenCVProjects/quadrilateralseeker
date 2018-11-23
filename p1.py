import cv2

first_frame = None

video = cv2.VideoCapture(0)

while True:

    check, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    # if first_frame is None:
    #     first_frame = gray
    #     continue

    # delta_frame = cv2.absdiff(first_frame, gray)
    # tresh_frame = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    #
    # tresh_frame = cv2.dilate(tresh_frame, None, iterations = 2)

    (_,cnts,_) = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea < 10000:
            continue

        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3 )

    cv2.imshow("video", gray)
    # cv2.imshow("Delta Frame", delta_frame)
    # cv2.imshow("Tresh Binary", tresh_frame)
    cv2.imshow("rectangle", frame)

    key = cv2.waitKey(1)

    if key==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
