import cv2
import streamlit as st 
import os

st.title("Cat face detector")
img_file = st.selectbox('Select a file', os.listdir('.'))

if st.button("Detect cat faces"):
	image = cv2.imread(img_file)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	detector = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
	rects = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(75, 75))

	# loop over the cat faces and draw a rectangle surrounding each
	for (i, (x, y, w, h)) in enumerate(rects):
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
		cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

	# show the detected cat faces
	cv2.imshow("Cat Faces", image)
	cv2.waitKey(0)