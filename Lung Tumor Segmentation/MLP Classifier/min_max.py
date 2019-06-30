###### Calculates min and max coordinates/ corners of cube enclosing the tumour region in each image

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import skimage.io
import os
import numpy

####### Parameters

weeks=7

####### Files

script_dir=os.path.dirname(os.path.abspath(__file__))

input_path=os.path.join(script_dir, 'input/patientList.txt')
output_path=os.path.join(script_dir, 'input/min_max.txt')
data_path=os.path.join(script_dir, 'input/data/')

f_patientList = open(input_path,'r')
f_min_max = open(output_path, 'w')

###### Code

patientList=f_patientList.read().split('\n')

print('Calculating min and max coordinates for each image:')
print('Press ctrl+c to exit')

for patient_ID in patientList:

	try:
	#print(patient_ID)
		for week in range(0,weeks):
			print(str(patient_ID)+'_'+str(week))
			image_path=data_path+str(patient_ID)+"/BIN"+str(week)+"_resized.tiff"
			if(os.path.exists(image_path)):
				img = skimage.io.imread(image_path, plugin='tifffile') # Binary image
				# print(str(img.shape))
				slices=img.shape[0];
				rows=img.shape[1]; 
				cols=img.shape[2]; 
				max_z=0;max_y=0;max_x=0;
				min_z=slices;min_y=rows;min_x=cols;
				for z in range(0,slices):
					for y in range(0,rows):
						for x in range(0,cols):
							if(img[z][y][x]==255):
								if(z<min_z):
									min_z=z
								if(x<min_x):
									min_x=x
								if(y<min_y):
									min_y=y
								if(z>max_z):
									max_z=z
								if(x>max_x):
									max_x=x
								if(y>max_y):
									max_y=y		
				f_min_max.write(str(patient_ID)+"_"+str(week))
				minPt=str(min_z)+" "+str(min_y)+" "+str(min_x)
				maxPt=str(max_z)+" "+str(max_y)+" "+str(max_x)
				f_min_max.write(" "+minPt+" "+maxPt+"\n")
				print(str(minPt)+' , '+str(maxPt))				

	except(KeyboardInterrupt, SystemExit):
		print('Keyboard interrupt caught')
		raise

#####	

f_min_max.close();
f_patientList.close();