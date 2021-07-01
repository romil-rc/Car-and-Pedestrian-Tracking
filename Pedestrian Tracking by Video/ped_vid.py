import cv2

video = cv2.VideoCapture('people.mp4')

classifier_file = 'pedestrian_detector.xml'

pedestrian_tracker = cv2.CascadeClassifier(classifier_file)

while True:
    (read_successful, frame) = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    for(x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow("Pedestrian Tracker", frame)

    cv2.waitKey(1)

    