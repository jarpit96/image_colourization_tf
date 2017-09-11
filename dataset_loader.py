import cv2
import os
import numpy as np
from skimage import color
from random import shuffle

class dataset():

	def __init__(self, batch_size,test_percentage = 20, validation_percentage =20, path = "ILSVRCData/input.txt"):
		self.batch_size = batch_size
		self.image_names = self.input_array(path)
		
		shuffle(self.image_names)
		print test_percentage

		self.n_train_records = int(len(self.image_names)*((100.0-int(test_percentage)-int(validation_percentage))/100.0))
		self.n_test_records = int(len(self.image_names)-self.n_train_records)*(test_percentage/float(test_percentage+validation_percentage))
		self.n_validation_records = len(self.image_names)-self.n_test_records-self.n_train_records

		self.currentBatch = 0

		self.n_batches = int(self.n_train_records/batch_size)

	def getNextBatch(self):
		self.currentBatch+=1
		images =[]
		print("Start",(self.currentBatch-1)*(self.batch_size), "\tEnd:", min(self.currentBatch*self.batch_size, self.n_train_records))
		for image_index in xrange((self.currentBatch-1)*(self.batch_size),min(self.currentBatch*self.batch_size, self.n_train_records)):
			images.append(self.getImage256(self.image_names[image_index]))

		print np.array(images).shape
		return self.rgb2lab(np.array(images))

	def getTestData(self):
		image_urls = self.image_names[int(self.n_train_records):int(self.n_test_records)+int(self.n_train_records)]
		images = []
		for image in image_urls:
			images.append(self.getImage256(image))
		return self.rgb2lab(np.array(images))

	def getValidationData(self):

		image_urls = self.image_names[int(self.n_test_records)+int(self.n_train_records):len(self.image_names)]
		images = []
		for image in image_urls:
			images.append(self.getImage256(image))
		return self.rgb2lab(np.array(images))


	def input_array(self,path):
		lines = []
		filePath = os.path.join(os.getcwd(), path)
		with open(filePath) as f:
			lines = f.readlines()
			# you may also want to remove whitespace characters like `\n` at the end of each line
		lines = [x.strip() for x in lines] 
		# print lines
		return lines

	def getImage256(self,name, path ="ILSVRCData/data256x256/", fileExtension = ".JPEG"):
		# print(os.path.join(os.getcwd(), path+name))
		# print os.path.isfile(os.path.join(os.getcwd(), path+name+fileExtension))
		return cv2.imread(os.path.join(os.getcwd(), path+name+fileExtension))

	def display_image(self,img):
		screen_res = 1280, 720
		scale_width = screen_res[0] / img.shape[1]
		scale_height = screen_res[1] / img.shape[0]
		scale = min(scale_width, scale_height)

		window_width = int(img.shape[1] * scale)
		window_height = int(img.shape[0] * scale)

		cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('dst_rt', window_width, window_height)

		cv2.imshow('dst_rt', img)
		cv2.imwrite('new_image.jpg',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def rgb2lab(self,data):

		'''Preprocess
		Args: 
		data: RGB batch (N * H * W * 3)
		Return:
		data_l: L channel batch (N * H * W * 1)
		data_ab: ab channel batch (N*H*W*1)
		'''
		print data.shape
		N = data.shape[0]
		H = data.shape[1]
		W = data.shape[2]

		#rgb2lab
		img_lab = color.rgb2lab(data)
		print img_lab.shape
		# display_image(img_lab)

		#slice
		#l: [0, 100]
		img_l = img_lab[:, :, :, 0:1]
			#ab: [-110, 110]
		data_ab = img_lab[:, :, :, 1:]

		#scale img_l to [-50, 50]
		data_l = img_l - 50

		print data_l.shape
		print data_ab.shape
		print data.shape
		return data_l, data_ab, data 

# obj = dataset(batch_size = 100)
# obj.getNextBatch()
# # print "Starting Next Batch"
# # obj.getNextBatch()

# print("Test Data")
# obj.getTestData()

# print ("Validation Data")
# obj.getValidationData()
	# image =  getImage("ILSVRC2016_test_00000001")
	# display_image(image)
	# rgb2lab(image)

# org = np.float32(cv2.imread(os.path.join(os.getcwd(), "ILSVRCData/data256x256/"+"ILSVRC2016_test_00000001"+".JPEG"), -1))/255.
# lab_image = cv2.cvtColor(org, cv2.COLOR_BGR2LAB)
# print image.shape

# display_image(image)
# display_image(lab_image)
