import cv2
import numpy as numpy

cap = cv2.VideoCapture('throww.mpg')

ret, frame1 = cap.read()
ret, frame2 = cap.read()

font = cv2.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
	hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
	hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
	l_b = numpy.array([109, 0 , 14])
	u_b = numpy.array([179, 255, 255])
	mask1 = cv2.inRange(hsv1, l_b, u_b)
	mask2 = cv2.inRange(hsv2, l_b, u_b)
	res1 = cv2.bitwise_and(frame1, frame1, mask=mask1)
	res2 = cv2.bitwise_and(frame2, frame2, mask=mask2)

	diff = cv2.absdiff(res1, res2)
	gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	_, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
	dilated = cv2.dilate(thresh, None, iterations=3)
	contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	for contour in contours:
		(x, y, w, h) = cv2.boundingRect(contour)

		if cv2.contourArea(contour) < 900:
			continue
		cv2.putText(frame1, 'Team1', (x-2, y-2), font, 0.8, (152,61,42), 2, cv2.LINE_AA)
		cv2.rectangle(frame1, (x, y), (x+w, y+h), (152,61,42), 2)
		#cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3)

	hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
	hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
	l_b = numpy.array([0, 11 , 220])
	u_b = numpy.array([19, 44, 255])
	mask1 = cv2.inRange(hsv1, l_b, u_b)
	mask2 = cv2.inRange(hsv2, l_b, u_b)
	res1 = cv2.bitwise_and(frame1, frame1, mask=mask1)
	res2 = cv2.bitwise_and(frame2, frame2, mask=mask2)

	diff = cv2.absdiff(res1, res2)
	gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	_, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
	dilated = cv2.dilate(thresh, None, iterations=3)
	contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	for contour in contours:
		(x, y, w, h) = cv2.boundingRect(contour)

		if cv2.contourArea(contour) < 900:
			continue
		cv2.putText(frame1, 'Team2', (x-2, y-2), font, 0.8, (14,60,59), 2, cv2.LINE_AA)
		cv2.rectangle(frame1, (x, y), (x+w, y+h), (14,60,59), 2)



	#cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
	cv2.imshow("feed",frame1)
	frame1=frame2
	ret, frame2 = cap.read()

	if cv2.waitKey(40) == 27:
		break

cv2.destroyAllWindows
cap.release()