import cv2
import numpy as np
import os

def reduceSize(image, n_width, n_height):
	h = image.shape[0]
	w = image.shape[1]

	if w > h:
	  image = cv2.resize(image, (int(n_width * w / h), n_width))

	  mirror = np.random.randint(0, 2)
	  if mirror:
	    image = np.fliplr(image)
	  crop_start = np.random.randint(0, int(n_width * w / h) - n_width + 1)
	  image = image[:, crop_start:crop_start + n_width, :]
	else:
	  image = cv2.resize(image, (n_width, int(n_width * h / w)))
	  mirror = np.random.randint(0, 2)
	  if mirror:
	    image = np.fliplr(image)
	  crop_start = np.random.randint(0, int(n_width * h / w) - n_width + 1)
	  image = image[crop_start:crop_start + n_width, :, :]
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	print('img\t','height:', image.shape[0],'width:', image.shape[1] )
	return image


def resampleImage(name):

	print name
	print(os.getcwd())
	img = cv2.imread(name)
	# print(img)

	img  = reduceSize(img, 256, 256)
	return img
	# screen_res = 1280, 720
	# scale_width = screen_res[0] / img.shape[1]
	# scale_height = screen_res[1] / img.shape[0]
	# scale = min(scale_width, scale_height)

	# window_width = int(img.shape[1] * scale)
	# window_height = int(img.shape[0] * scale)

	# cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('dst_rt', window_width, window_height)

	# cv2.imwrite('image.jpg',img)

	# cv2.imshow('dst_rt', img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

def readAndWriteImage(dir_path):

	file_path = dir_path + 'test.txt'
	input_file = open(file_path, 'r')

	fileName = []

	for line in input_file:
		line = line.strip().split()[0]
		if(os.path.isfile(dir_path+'/data/'+line+'.JPEG')):
			image = resampleAndShowImage(dir_path+'/data/'+line+'.JPEG')
			cv2.imwrite(dir_path+'data256x256/'+line+'.JPEG',image)
			fileName.append(line)
	thefile = open('/Users/newuser/Documents/Major1/image_colourization_tf/ILSVRCData/input.txt', 'w')
	for item in fileName:
  		thefile.write("%s\n" % item)


readAndWriteImage('/Users/newuser/Documents/Major1/image_colourization_tf/ILSVRCData/')










