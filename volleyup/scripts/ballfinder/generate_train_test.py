import cv2
import os
from random import randint
import random

data_dir = "../../data"
training_dir = os.path.join(data_dir, "training_data")
annotation_dir = os.path.join(training_dir, "annotations")

def sample(image, size):
	"""
	Samples a square patch from the given image 
		size: edge length of square sample patch
	"""
	height, width, *_ = image.shape
	i, j = randint(0, height - size -1 ), randint(0, width - size -1)
	return (image[i:i+size, j:j+size], i, j)

class Subsampler:
	def __init__(self, image, square_size, x0_t, x1_t, y0_t, y1_t):
		yi, xi, *_ = image.shape
		x0, x1 = 50-square_size//2 -x0_t, 50 + square_size
		y0, y1 = 50-square_size//2 -y0_t, 50 + square_size
		# grab the pixels that are not in the ballbox
		self.ss_pixels = []
		for i in range(yi):
			for j in range(xi):
				if y0 <= i <= y1:
					continue
				if x0 <= j <= x1:
					continue
				self.ss_pixels.append(image[i][j])
		self.ss_size = len(self.ss_pixels)
	def get_subsample(self):
		return self.ss_pixels[randint(0, self.ss_size -1)]
	
# # Generate positive samples
for file_name in os.listdir(annotation_dir):
	file_path = os.path.join(annotation_dir, file_name)
	dataset = os.path.splitext(file_name)[0]
	image_folder = os.path.join(data_dir, dataset)
	#get frame annotations (x, y, rectangle size)
	annotations = [i.strip().split("\t") for i in open(file_path) if i != "\n"]
	print(annotations)
	#create positive training images
	for num, ann in enumerate(annotations):
		frame, x, y, square_size = ann[0], int(ann[-3]), int(ann[-2]), int(ann[-1])
		im_path = os.path.join(data_dir, dataset, frame + ".jpg")
		print(im_path)
		image = cv2.imread(im_path)
		try:
			yi, xi, *_ = image.shape
		except:
			print(im_path)
			raise
		sub_raw = image[y:y + square_size, x: x+square_size]
		cv2.imwrite(os.path.join(training_dir, "data", "_".join(["pos", dataset, str(num), "frame", frame]) + ".png"), sub_raw)
		xc, yc, = x + square_size//2, y+ square_size//2
		x0, x1, y0, y1 = xc-50, xc + 50, yc -50, yc + 50
		# handle the case of literal edge cases (negative/overly positive boundaries)
		# populate otherwise empty areas randomly from non-ball areas
		print(x0, y0, x1, y1)
		print(0, 0, xi, yi)
		if any([x0 < 0, y0 < 0, x1 >= xi, y1 >= yi]):
			sub_100 = image[0:100, 0:100].copy()
			#figure out which of the 9 potential positobns it can be in
			x0_t = max(0, -x0)
			x0 = max(x0, 0)
			y0_t = max(0, -y0)
			y0 = max(y0, 0)
			x1_t = max(0, x1-xi)
			x1 = min(x1, xi)
			y1_t = max(0, y1-yi)
			y1 = min(y1, y1)
			print("ball_coords: ", x0, x1, y0, y1)
			print("100x100 coords: ", x0_t, 100-x1_t, y0_t, 100 - y1_t)
			# image of the ball without portions exceeeding boundary
			ball_image = image[y0:y1, x0:x1]
			print(ball_image.shape)
			print(sub_100[y0_t:100 - y1_t, x0_t:100-x1_t ].shape)
			sub_100[y0_t:100 - y1_t, x0_t:100-x1_t] = ball_image

			# Populate areas exceeding boundary with random pixels outside the ballbox
			ss = Subsampler(ball_image, square_size, x0_t, x1_t, y0_t, y1_t)
			for j in range(0, x0_t):
				for i in range(100):
					sub_100[i][j] = ss.get_subsample()
			for j in range(100-x1_t, 100):
				for i in range(100):
					sub_100[i][j] = ss.get_subsample()
			print(x0_t, x1-x0)
			for j in range(x0_t, x1-x0):
				for i in range(0, y0_t):
					print(i, j)
					sub_100[i][j] = ss.get_subsample()
			for j in range(x0, x1):
				for i in range(100-y1_t, 100):
					sub_100[i][j] = ss.get_subsample()


		# if not in the center of the image
		else:
			# ball in the center of 100x100 image
			sub_100 = image[y0:y1, x0:x1]
		
		cv2.imwrite(os.path.join(training_dir, "data_100", "_".join(["pos", dataset, str(num), "frame", frame]) + ".png"), sub_100)

#Generate negative samples
# full_train_dir = os.path.join(data_dir, "beachVolleyballFull")
# #number of training datasets to make
# train_count = 1000
# sample_size = 100
# file_paths = [os.path.join(full_train_dir, i) for i in os.listdir(full_train_dir)]
# target_files = sorted(random.sample(file_paths, train_count))
# for path in target_files:
# 	frame = os.path.split(path)[1][:-4]
# 	im = cv2.imread(path)
# 	sampled_image, i, j = sample(im, sample_size)
# 	out_path = os.path.join(training_dir, "data_100", "_".join(["neg", "beachVolleyballFull", str(i), str(j), frame]) + ".png")
# 	cv2.imwrite(out_path, sampled_image)





