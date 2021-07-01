import cv2

video = cv2.VideoCapture('Add video here')

car_classifier_file = 'car_detector.xml'
pedestrian_classifier_file = 'pedestrian_detector.xml'

car_tracker = cv2.CascadeClassifier(car_classifier_file)
ped_tracker = cv2.CascadeClassifier(pedestrian_classifier_file)

while True:

    (read_successful, frame) = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = ped_tracker.detectMultiScale(grayscaled_frame)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame,(x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow('Car & Pedestrian Detector', frame)

    key = cv2.waitKey(1)

    # Q or q will close the screen
    if key==81 or key==113:
        break

video.release()