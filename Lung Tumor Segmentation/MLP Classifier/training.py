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

p=25
q=25
r=9
G=32
#train_fraction=0.8

parameters=str(p)+'_'+str(q)+'_'+str(r)+'_'+str(G)

train_path = './output/Features/'+parameters +'/*_0.dat'   # Training on 0th week
files = glob.glob(train_path)

print('Number of files=:'+str(len(files)))
#train_count=int(math.ceil(train_fraction*(len(files))))

clf = MLPClassifier(hidden_layer_sizes=(432,256,64), activation='logistic', solver='sgd', learning_rate='invscaling',
    batch_size=400, shuffle=True, random_state=0, verbose=False, early_stopping=False, warm_start=True, max_iter=300 )

scaler = StandardScaler()

for i in range(0,len(files)): # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
    name=files[i]
    # print(name)
    try:
        with open(name) as f: # No need to specify 'r': this is the default.
            array=np.load(f)
            X_train=array[:,4:]

            # # Fit only to the training data
            scaler.partial_fit(X_train)
    
    except IOError as exc:
        if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
            raise

count=0
for j in range(0,1):
    for i in range(0,len(files)): # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
        name=files[i]
        # print(name)
        try:
            with open(name) as f: # No need to specify 'r': this is the default.
                print(count)
                array=np.load(f)
                X_train=array[:,4:]
                Y_train=array[:,0]
                # Random Sampler
                rus = RandomUnderSampler(random_state=0)
                X_train, Y_train = rus.fit_sample(X_train, Y_train);
                # print(len(X_train))
                # X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2);
                # # Now apply the transformations to the data:
                X_train = scaler.transform(X_train)
                clf.fit(X_train,Y_train)
                count+=1

        except IOError as exc:
            if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
                raise

joblib.dump(scaler,'scaler.save')
joblib.dump(clf,'model.pkl')

## Training accuracy

Y_test=np.array([])
predictions=np.array([])

for i in range(0,len(files)):
    name=files[i]
    # print(name)
    try:
        with open(name) as f: # No need to specify 'r': this is the default.
            array=np.load(f)
            Y_t=array[:,0]
            X_t=array[:,4:]

            rus = RandomUnderSampler(random_state=0)
            X_t, Y_t = rus.fit_sample(X_t, Y_t);
            Y_test=np.concatenate([Y_test,Y_t]) 

            X_test = scaler.transform(X_t)
            pred = clf.predict(X_test)
            predictions=np.concatenate([predictions,pred])

    except IOError as exc:
        if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
            raise

#print(clf.score(X_test,Y_test))
# print(Y_test.shape)
print('Training Results:');
print(confusion_matrix(Y_test,predictions));
print(classification_report(Y_test,predictions));
