import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import sys
import glob
import errno
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from sklearn.externals import joblib
import skimage.io
from skimage import color


## Testing / Visualization 

scaler=joblib.load('scaler.save')
clf=joblib.load('model.pkl')

img = skimage.io.imread('CBCT0_Orig_usi_resized.tiff', plugin='tifffile') # CBCT image
img_out = color.gray2rgb(img)

name='./output/Features/25_25_9_32/13-1071_0.dat'

try:
    with open(name) as f: # No need to specify 'r': this is the default.
        array=np.load(f)
        Y_t=array[:,0]
        X_t=array[:,4:]
        loc=array[:,1:4]
        X_test = scaler.transform(X_t)
        pred = clf.predict(X_test)
        #print(type(pred[0]))
        for i in range(0,len(pred)):
	        if(pred[i]):
	        	img_out[int(loc[i][0]),int(loc[i][1]),int(loc[i][2]),1]=255

except IOError as exc:
    if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
        raise

skimage.io.imsave("13-1071_0_seg.tiff",img_out, plugin='tifffile') # CBCT image

# print('Testing:')
# test_path = './output/Features/'+parameters +'/*_1.dat'   # Training on 0th week
# test_files = glob.glob(test_path)

# Y_test=np.array([])
# predictions=np.array([])

# for i in range(train_count,len(files)):
#     name=files[i]
#     # print(name)
#     try:
#         with open(name) as f: # No need to specify 'r': this is the default.
#             array=np.load(f)
#             Y_t=array[:,0]
#             X_t=array[:,4:]

#             rus = RandomUnderSampler(random_state=0)
#             X_t, Y_t = rus.fit_sample(X_t, Y_t);
#             Y_test=np.concatenate([Y_test,Y_t]) 

#             X_test = scaler.transform(X_t)
#             pred = clf.predict(X_test)
#             predictions=np.concatenate([predictions,pred])

#     except IOError as exc:
#         if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
#             raise

# #print(clf.score(X_test,Y_test))
# print(Y_test.shape)
# print(confusion_matrix(Y_test,predictions));
# print(classification_report(Y_test,predictions));
