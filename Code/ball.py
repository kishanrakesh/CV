from collections import deque
import numpy as np
import argparse
import cv2
import imutils
import time
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())
greenLower = (162, 42, 0)
greenUpper = (255, 255, 111)
pts = deque(maxlen=args["buffer"])
vs = cv2.VideoCapture(args["video"])
time.sleep(2.0)
prev_x=None 
prev_y=None
def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2
	
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)
while True:
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame
	if frame is None:
		break
	frame = imutils.resize(frame, width=1280)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		if prev_x !=None and prev_y !=None:
			if abs(prev_x - x) > 10 and abs(prev_y - y) > 10:
				__draw_label(frame, 'Jitter', (20,20), (100,255,255))
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		if radius > 10:
			cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
			prev_x=int(x)
			prev_y=int(y)
	pts.appendleft(center)
	for i in range(1, len(pts)):
		if pts[i - 1] is None or pts[i] is None:
			continue
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(50) & 0xFF
	if key == ord("q"):
		break
vs.release()
# close all windows
cv2.destroyAllWindows()