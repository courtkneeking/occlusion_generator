## Custom Partial Occlusion Dataset Generator
Adds custom partial occlusions to objects in images for comprehensive occlusion-handling experimentation

### Set up 
1. Place occluder objects into folder called occlusions. They can be of any size, shape, but must be on isolated white background. 
2. Place dataset into folder named example_images. There should also be a corresponding annotation file to each image, defining the bounding box coordinates for each object. 
3. Adjust the desired parameters in the add_occluders file. 

### Generate Dataset
4. Run add_occluders.py
5. The new dataset with the custom occlusions will saved in the occluded_images folder. 


