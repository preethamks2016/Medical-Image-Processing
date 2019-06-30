
# Calculates Features for all patients given in patientsList.txt

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import os
from numpy import *
import numpy as np
import math
import skimage.io
from skimage.feature import greycomatrix, greycoprops
from skimage import data
import argparse

####### Parameters

parser = argparse.ArgumentParser("Window and Greylevel Parameters")

parser.add_argument("p", type=int, help="window length")
parser.add_argument("q", type=int, help="window height")
parser.add_argument("r", type=int, help="window depth")
parser.add_argument("G", type=int, help="Greylevels")
args = parser.parse_args()

p=args.p
q=args.q
r=args.r
G=args.G
parameters=str(p)+'_'+str(q)+'_'+str(r)+'_'+str(G)

weeks=7 # maximum number of weeks possible for a patient

# additional margin for box enclosing the min_max box
non_cancer_margin=20

# glcm parameters 
radial_distances=[2,4] # list of radial distances
angles=[0, np.pi/4, np.pi/2, 3*np.pi/4] # list of angles
M=len(radial_distances) # number of radial distances
N=len(angles) # number of angles

####### Files

script_dir = '.'
patientList_path=os.path.join(script_dir, 'input/patientList.txt')
f_patientList = open(patientList_path,'r')
patientList=f_patientList.read().split('\n')
f_patientList.close()

min_max_path=os.path.join(script_dir, 'input/min_max.txt')
f_min_max = open(min_max_path,'r')
min_max_list=f_min_max.read().split('\n')
f_min_max.close()

data_path=os.path.join(script_dir, 'input/data/')

features_path=os.path.join(script_dir, 'output/Features/'+parameters)
if not os.path.exists(features_path):
		os.makedirs(features_path)

###### Code

#### I  Iterating over weekly images of every patient:

print('Iterating over weekly images of every patient: <patient_ID_week>')
print('Press <ctrl+c> to exit')

iter_=0
for patient_ID in patientList:

	try:
		for week in range(0,weeks):

			print(str(patient_ID)+'_'+str(week))
			BIN_image_path=data_path+str(patient_ID)+"/BIN"+str(week)+"_resized.tiff"
			CBCT_image_path=data_path+str(patient_ID)+"/CBCT"+str(week)+"_Orig_usi_resized.tiff"
		
			if(os.path.exists(BIN_image_path)):

				img0 = skimage.io.imread(BIN_image_path, plugin='tifffile') # Binary image
				img1 = skimage.io.imread(CBCT_image_path, plugin='tifffile') # CBCT image

				slices=img0.shape[0]; # 64
				rows=img0.shape[1]; # 245
				cols=img0.shape[2]; # 384

				# Normalizing the intensity values to 0 : G-1
				max_greylevel = int(np.amax(img1)); # varies with image
				img1 = np.floor(img1*(G-1)/max_greylevel)

				min_max_Pt=min_max_list[iter_].split(' ')
				iter_=iter_+1
				min_max_Pt = list(map(int, min_max_Pt[1:]))
				# print(min_max_Pt)
				min_z=min_max_Pt[0]; min_y=min_max_Pt[1]; min_x=min_max_Pt[2];
				max_z=min_max_Pt[3]; max_y=min_max_Pt[4]; max_x=min_max_Pt[5];
				# print(str(min_z)+','+str(min_y)+','+str(min_x))
				# print(str(max_z)+','+str(max_y)+','+str(max_x))
				
				#### Iterating over voxels enclosed by min_max_box

				zlim0=max(math.ceil(r/2),min_z)
				zlim1=min(slices-math.ceil(r/2),max_z+1)
				ylim0=max(math.ceil(q/2),min_y-non_cancer_margin)
				ylim1=min(rows-math.ceil(q/2),max_y+1+non_cancer_margin)
				xlim0=max(math.ceil(p/2),min_x-non_cancer_margin)
				xlim1=min(cols-math.ceil(p/2),max_x+1+non_cancer_margin)

				# Features per weekly image 
				FeatureList=[]
				for z in range(zlim0,zlim1):
					for y in range(ylim0,ylim1):
						for x in range(xlim0,xlim1):

							# Taking a r x q x p volume from 3 D image centred at (z,y,x) 
							S = img1[z+int(-1*math.floor(r/2.0)):z+int(math.ceil(r/2.0)),y+int(-1*math.floor(q/2.0)):y+int(math.ceil(q/2.0)),x+int(-1*math.floor(p/2.0)):x+int(math.ceil(p/2.0))]

							##### Feature Extraction per voxel
							f=0
							F= [0]*(M*N*r*6 + 4) # M x N x 8 features of each of the 'r' sub-slices in slice(chunk)
							if(img0[z][y][x]==255):
								F[f]=1.0;f=f+1;
							else:
								F[f]=0.0;f=f+1;

							F[f]=z;f=f+1;
							F[f]=y;f=f+1;
							F[f]=x;f=f+1;

							for count in range(0,r):

								matrix=S[count,:,:].astype(int)
								P = greycomatrix(matrix, radial_distances, angles, levels=G, symmetric=False, normed=True)

								# Features
								# 1 Contrast
								tmp = greycoprops(P,prop='contrast').ravel()
								F[f:f+tmp.size] = tmp
								f+=tmp.size
								# 2 Dissimilarity
								tmp = greycoprops(P,prop='dissimilarity').ravel()
								F[f:f+tmp.size] = tmp
								f+=tmp.size
								# 3 homogeneity
								tmp = greycoprops(P,prop='homogeneity').ravel()
								F[f:f+tmp.size] = tmp
								f+=tmp.size
								# 4 ASM
								tmp = greycoprops(P,prop='ASM').ravel()
								F[f:f+tmp.size] = tmp
								f+=tmp.size
								# 5 energy
								tmp = greycoprops(P,prop='energy').ravel()
								F[f:f+tmp.size] = tmp
								f+=tmp.size
								# 6 correlation
								tmp = greycoprops(P,prop='correlation').ravel()
								F[f:f+tmp.size] = tmp
								f+=tmp.size

							FeatureList.append(F)
				
				featureFile_path=features_path+'/'+str(patient_ID)+'_'+str(week)+'.dat'
				FeatureList=np.array(FeatureList)
				FeatureList.dump(featureFile_path)

	except(KeyboardInterrupt, SystemExit):
		print('Keyboard interrupt caught')
		featureFile_path=features_path+'/'+str(patient_ID)+'_'+str(week)+'.dat'
		FeatureList=np.array(FeatureList)
		FeatureList.dump(featureFile_path)
		raise


		