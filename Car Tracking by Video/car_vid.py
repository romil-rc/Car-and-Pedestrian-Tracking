import cv2

video = cv2.VideoCapture('carv.mp4')

classifier_file = 'car_detector.xml'

car_tracker = cv2.CascadeClassifier(classifier_file)

while True:

    (read_successful, frame) = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    cars = car_tracker.detectMultiScale(grayscaled_frame)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # print(cars)

    cv2.imshow('Video car detector',  frame)

    cv2.waitKey(1)

print('Code Completed')