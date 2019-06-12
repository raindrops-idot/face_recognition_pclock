import os
import cv2
import shutil


class GatherFaceImage(object):
	"""
	1、收集人脸识别数据
	2、存储规则：以姓名作为文件夹名字，文件夹下存储人脸图图像
	"""
	def __init__(self, face_path, cascade_classifier):
		"""
		:param face_path: 人脸数据的存储位置
		:param cascade_classifier: 分类器位置
		"""
		self.face_path = face_path
		self.cascade_classifier = cascade_classifier
		
	def gather_face_image(self):
		"""
		生成自己的人脸识别数据 ,若数据存在则删除重新收集，不能出现同名
		生成的图片满足以下条件
			1、图像是灰度格式，后缀为.pgm
			2、图像大小要一样
		打开摄像头，读取帧，检测帧中的人脸，并剪切，缩放
		"""
		name = input('输入英文名字或名字首字母缩写:')
		# 如果路径存在则删除
		path = os.path.join(self.face_path, name)
		if os.path.isdir(path):
			# os.remove(path)   #删除文件
			# os.removedirs(path)   #删除空文件夹
			shutil.rmtree(path)    # 递归删除文件夹
		
		# 创建文件夹
		os.mkdir(path)
		
		# 创建一个级联分类器 加载一个 .xml 分类器文件. 它既可以是Haar特征也可以是LBP特征的分类器.
		face_cascade = cv2.CascadeClassifier(self.cascade_classifier)
		
		# 打开摄像头
		camera = cv2.VideoCapture(0)
		# 计数
		count = 1
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
					cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
					# 调整图像大小 和ORL人脸库图像一样大小
					f = cv2.resize(frame[y:y+h, x:x+w], (92, 112))
					# 保存人脸
					cv2.imwrite('%s\\%s.png' % (path, str(count)), f)
					count += 1
				cv2.imshow('dealing', frame)
				# 如果按下q键则退出
				if cv2.waitKey(100) & 0xff == ord('q'):
					break
		camera.release()
		cv2.destroyAllWindows()
		
		