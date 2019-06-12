# utf-8
import FaceRecongize as faceRecongize
import  GatherFaceImage as gatherFaceImage

if __name__ == '__main__':
	cascade_classifier_path = "E:/zhaoying/face_recognition_pclock/haarcascade_frontalface_default.xml"
	# 训练数据存储路径
	data_train_path = 'E:\zhaoying\\face\\'
	# 生成自己的人脸数据
	# gatherFaceImage.GatherFaceImage(data_train_path, cascade_classifier_path).gather_face_image()
	
	# generator(data_train, cascade_classifier)
	faceRecongize.FaceRecongize(data_train_path, cascade_classifier_path).face_rec()
	