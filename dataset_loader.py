import cv2
import os
import numpy as np
import h5py
from skimage import color
from random import shuffle

class dataset():

	def __init__(self, batch_size=100,test_percentage = 20, validation_percentage =20, path = "ILSVRCData/input.txt"):
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
		'''
		Get the Next Training Batch
		Return:
		data_l(NxHxWX1), data_lab(NxHxWX3)retuned for batch Size
		'''
		self.currentBatch+=1
		images =[]
		print("Start",(self.currentBatch-1)*(self.batch_size), "\tEnd:", min(self.currentBatch*self.batch_size, self.n_train_records))
		for image_index in xrange((self.currentBatch-1)*(self.batch_size),min(self.currentBatch*self.batch_size, self.n_train_records)):
			images.append(self.getImage256(self.image_names[image_index]))

		print np.array(images).shape
		return self.rgb2lab(np.array(images, dtype=np.float32))

	def getTestData(self):
		'''
		Get the Whole test Data
		Return:
		data_l(NxHxWX1), data_lab(NxHxWX3) retuned for test Data Division
		'''
		image_urls = self.image_names[int(self.n_train_records):int(self.n_test_records)+int(self.n_train_records)]
		images = []
		for image in image_urls:
			images.append(self.getImage256(image))
		return self.rgb2lab(np.array(images, dtype=np.float32))

	def getValidationData(self):
		'''
		Get the Whole Validation Data
		Return:
		data_l(NxHxWX1), data_lab(NxHxWX3) retuned for Validation Data Division
		'''

		image_urls = self.image_names[int(self.n_test_records)+int(self.n_train_records):len(self.image_names)]
		images = []
		for image in image_urls:
			images.append(self.getImage256(image))
		return self.rgb2lab(np.array(images, dtype=np.float32))


	def input_array(self,path):
		'''
		Read The text File containing image Name
		Args:
		path:path to image_name file
		Return:
		lines: List of image_names
		'''
		lines = []
		filePath = os.path.join(os.getcwd(), path)
		with open(filePath) as f:
			lines = f.readlines()
			# you may also want to remove whitespace characters like `\n` at the end of each line
		lines = [x.strip() for x in lines] 
		# print lines
		return lines

	def getImage256(self,name, path ="ILSVRCData/data256x256/", fileExtension = ".JPEG"):
		'''
		Get Image from 256x256 folder with name
		Args:
		name:Image Name
		path=Path to Image Data Folder
		fileExtension = image file extension
		Return:
		image data in RGB channels
		'''
		# print(os.path.join(os.getcwd(), path+name))
		# print os.path.isfile(os.path.join(os.getcwd(), path+name+fileExtension))
		return cv2.imread(os.path.join(os.getcwd(), path+name+fileExtension))

	def display_image(self,img):
		'''
		Display Image using cv
		Args:
		img: Image Data
		'''
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

	def storeDataLab(self):
		images = []
		for name in self.image_names:
			images.append(self.getImage256(name))
		print np.array(images).shape
		
		l, _ = self.rgb2lab(np.array(images))
		print l.shape
		del images[:]
		h5f = h5py.File('lData.h5', 'w')
		h5f.create_dataset('dataset_1', data=l)
		h5f.close()

	def rgb2lab(self,data):

		'''RGB 2 Lab Color Space COnversion
		Args: 
		data: RGB batch (N * H * W * 3)
		Return:
		data_l: L channel batch (N * H * W * 1)
		data_lab: lab channel batch (N*H*W*3)
		'''
		# print data.shape
		N = data.shape[0]
		H = data.shape[1]
		W = data.shape[2]

		#rgb2lab
		img_lab = color.rgb2lab(data)
		# print img_lab.shape
		# display_image(img_lab)

		#slice
		#l: [0, 100]
		img_l = img_lab[:, :, :, 0:1]
			#ab: [-110, 110]
		data_ab = img_lab[:, :, :, 1:]

		#scale img_l to [-50, 50]
		data_l = img_l - 50

		# print data_l.shape
		# print data_ab.shape
		# print data.shape
		return data_l, img_lab

# obj = dataset()
# obj.storeDataLab()

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

