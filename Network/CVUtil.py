import cv2 as cv
import numpy as np
class CVUtil:
	def __init__(self):
		self.emotion = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

	def video_capture(self,model):
		image_capture = cv.VideoCapture(0)
		flag = True
		while flag:
			retval, image = image_capture.read()
			if retval is None:
				break
			cascade = cv.CascadeClassifier('haarcascade.xml')
			gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
			#gray_image = cv.equalizeHist(gray_image)

			faces = cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

			for (x_axis, y_axis, width, height) in faces:
				pt1 = (x_axis, y_axis - 50)
				thickness = 2
				pt2 =  (x_axis + width, y_axis + height + 10)
				color = (11,255,1)
				cv.rectangle(image, pt1, pt2, color,thickness)
				roi_gray = gray_image[y_axis:y_axis + height, x_axis:x_axis + width]

				cropped_img = np.expand_dims(np.expand_dims(cv.resize(roi_gray, (48, 48)), -1), 0)
				prediction = model.predict(cropped_img)
				max_ = int(np.argmax(prediction))

				#put text parameters
				text = self.emotion[max_]
				org = (x_axis + 20, y_axis - 60)
				color = (11, 255, 1)
				font = cv.FONT_HERSHEY_PLAIN
				thickness = 2
				line_type = cv.LINE_AA
				# Putting Text on the Window
				cv.putText(image, text, org, font, 1, color,thickness, line_type)

				image_capture.set(cv.CAP_PROP_FPS, 30)

			src_image = image
			dst_image = (1600, 900)
			interpolation = cv.INTER_CUBIC
			image_show = cv.resize(src_image, (1600, 900), interpolation= interpolation)
			cv.imshow('Video', image_show)
			k = cv.waitKey(1) & 0xFF
			if k == ord('q'):
				break
		image_capture.release()
		cv.destroyAllWindows()