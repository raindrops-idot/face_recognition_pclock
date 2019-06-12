import numpy as np
import os
import cv2


class FaceRecongize(object):
	"""
	1、加载训练数据并进行初步处理
	2、使用训练数据进行人脸识别
	3、训练文件夹的名字以训练数据的名字命名
	"""
	
	def __init__(self, data_path, cascade_classifier):
		"""
		:param data_path: 训练数据的存储位置
		:param cascade_classifier 分类器位置
		"""
		self.data_path = data_path
		self.cascade_classifier = cascade_classifier

	def loagd_train_data(self):
		"""
		1、加载训练数据并进行初步处理
		:return:
			images:[m,height,width]  m为样本数,height为高,width为宽
			names：名字的集合
			labels：标签
		"""
		images = []
		labels = []
		names = []
	
		label = 0
		# 过滤所有的文件夹
		for subDirname in os.listdir(self.data_path):
			subject_path = os.path.join(self.data_path, subDirname)
			if os.path.isdir(subject_path):
				# 每一个文件夹下存放着一个人的照片
				names.append(subDirname)
				for fileName in os.listdir(subject_path):
					img_path = os.path.join(subject_path, fileName)
					img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
					images.append(img)
					labels.append(label)
				label += 1
		images = np.asarray(images)
		labels = np.asarray(labels)
		return images, labels, names
	
	def face_rec(self):
		"""
		1、加载数据后进行人脸识别
		:return:
			result[0] 标签
			result[1] 置信度
		"""
		# 加载训练数据
		x, y, names = self.loagd_train_data()
	
		model = cv2.face.EigenFaceRecognizer_create()
		model.train(x, y)
		
		# 创建一个级联分类器 加载一个 .xml 分类器文件. 它既可以是Haar特征也可以是LBP特征的分类器.
		face_cascade = cv2.CascadeClassifier(self.cascade_classifier)
		
		# 打开摄像头
		camera = cv2.VideoCapture(0)
		result = []
		while True:
			# 读取一帧图像
			ret, frame = camera.read()
			# 判断图片读取成功
			if ret:
				gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				# 人脸检测
				faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
				for (x, y, w, h) in faces:
					# 在原图像上绘制矩形
					frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
					roi_gray = gray_img[y:y+h, x:x+w]
					
					# 宽92 高112
					roi_gray = cv2.resize(roi_gray, (92, 112), interpolation=cv2.INTER_LINEAR)
					result = model.predict(roi_gray)
					print('Label:%s,confidence:%.2f' % (result[0], result[1]))
					cv2.putText(frame, names[result[0]], (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
				
				cv2.imshow('recongizing', frame)
				# 如果按下q键则退出
				if cv2.waitKey(100) & 0xff == ord('q'):
					break
		camera.release()
		cv2.destroyAllWindows()
		return result
	