# Takes in images from folder 'example_images' and an occluder image in 'occluders'
# Adds the occluder to the object(s) in the example_image and saves into 'occluded_images' 
# Average IOU is calculated using masks saved in occluded_masks 
# Adjustable parameters: occlusion_ratio (which is squared) 
# Demo at bottom 


import cv2
import numpy as np
import os

# Assumes annotations are in the same folder alongside images, and of the same name with txt 
# Returns a list of all of the annotations as a list for a single image 
def get_annotations(img_path):
	annotations_path = img_path[:-3]+'txt'
	f = open(annotations_path, "r")
	annotations_list = []
	for l in f: 
		a = [round(float(i),3) for i in l.split(" ")[1:]]
		annotations_list.append(a)
	return annotations_list 

# Returns list of images corresponding to paths; the dataset to be occluded 
def get_images(): 
	images = ['example_images/'+file_name for file_name in os.listdir('example_images') if file_name.endswith('jpg')]
	return images


# Returns masks saved during make_occ_data function
# used for IOU stats 
def get_occluded_masks(): 
	images = ['occluded_masks/'+file_name for file_name in os.listdir('occluded_masks')]
	return images

# Returns the ROI params based on the object's annotation w.r.t the global image 	
# annotations are in yolo format x,y,w,h 
# 	where (x,y) are the center point of the bb, w,h are the width and height of the bb, 
# 	both are given as ratio of w.r.t. the global image size 
def get_roi_params(g_img, obj_annotations):
    x,y,w,h= obj_annotations
    gh, gw = g_img.shape[:2]

    x_min = int((gw*x)-((w*gw)/2))
    x_max = int(x_min+(w*gw))
    
    y_min = int((gh*y)-((gh*h)/2))
    y_max = int(y_min+(h*gh))
    return [x_min, x_max, y_min, y_max]

# Returns average IOU of data 
# Inputs are occlusion masks are saved during make_occ_dataset
#  (These show the coverage of the ROI (currently based on the bounding box coordinates))
# 	Call using images saved in 'occluded_masks' 
def IOU(images):
	scores = []
	for i in images:
		image = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
		intersection = np.sum(image == 255)
		union = np.sum(image != 255)+intersection
		IOU = intersection/union 
		scores.append(IOU)
	average_IOU = sum(scores)/len(scores)
	return average_IOU

# Helper function for isolating occluder from background 
def threshold_non_white(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds for non-white colors
    # Adjust the upper bound for Value (V) to exclude bright whites
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([179, 255, 230])
  
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)
    return result
    
   
# Given list of image paths, and assuming the annotations exist in the same folder 
# and a path of occluder with white background 
# Saves new dataset with occluder inserted 
def make_occ_dataset(images, occ_img, occ_ratio = 0.5):
	for i in range(0,len(images)):
		# g_img, the background image where the object is defined in a bb by annotations
		img_path = images[i]
		g_img = cv2.imread(img_path)
		obj_annotations = get_annotations(img_path)
		masks = []
		for j in range(0,len(obj_annotations)):
			# locate roi in background image where the object is 
			x_min, x_max, y_min, y_max = get_roi_params(g_img, obj_annotations[j])
			ROI = g_img[y_min:y_max, x_min:x_max]

			# the image to be placed on the ROI, resize w.r.t. defined ratio and ROI 
			roi_width, roi_height = x_max-x_min, y_max-y_min

			# Resize the occ_img by the ROI, then shrinks by the occlusion ratio 
			occ_img_resized = cv2.resize(occ_img, (round(roi_width*occ_ratio), round(roi_height*occ_ratio)), interpolation=cv2.INTER_AREA)
			occ_img_frame = np.zeros_like(ROI)
			th, tw  = occ_img_resized.shape[:2]
			occ_img_frame[0:th, 0:tw] = occ_img_resized 
			occ_img_resized = occ_img_frame

			img2gray = cv2.cvtColor(occ_img_resized,cv2.COLOR_BGR2GRAY)
			ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
			mask_inv = cv2.bitwise_not(mask)
			

			img1_bg = cv2.bitwise_and(ROI, ROI ,mask = mask_inv)
			img2_fg = cv2.bitwise_and(occ_img_resized, occ_img_resized ,mask = mask)

			
			# Combine the two masks and put into the ROI section of the global image 
			dst = cv2.add(img1_bg,img2_fg)
			g_img[y_min:y_max, x_min:x_max] = dst
			

		outstring = img_path.split('/')[1]
		cv2.imwrite('occluded_images/'+outstring, g_img) #save image
		
		# This part is for IOU stats 
		cv2.imwrite('occluded_masks/'+outstring, mask) #save image


if __name__ == '__main__':
	# gets images from 'example_images' 
	images = get_images()
	
	# Choose occluder 
	occ_img = cv2.imread('occluders/twig.png')
	occ_img = threshold_non_white(occ_img) # blacks out the background 
	
	# saves dataset in occluded_images, and masks in occluded_masks 
	make_occ_dataset(images, occ_img)
	
	# Calculate IOU from masks 
	# occluded_masks = get_occluded_masks()
	# print('Occlusion IOU:' , occluded_masks) 
	
	
	
# Resources 
# https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html
# https://docs.opencv.org/4.x/d3/df2/tutorial_py_basic_ops.html
# https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html

