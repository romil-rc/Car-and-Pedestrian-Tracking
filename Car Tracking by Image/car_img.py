import cv2

img_file='car4.jpg'

classifier_file = 'car_detector.xml'

img = cv2.imread(img_file)

black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

car_tracker = cv2.CascadeClassifier(classifier_file)

cars = car_tracker.detectMultiScale(black_n_white)
#print(cars)

# car1 = cars[0]
# (x, y, w, h) = car1
#print(car1)
for(x, y, w, h) in cars:
    cv2.rectangle(img,(x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow('This is a car detector', img)

cv2.waitKey()

print("Code Completed")