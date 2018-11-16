import numpy as np
from sklearn.linear_model import LogisticRegression
from struct import unpack
import gzip
import sys
from numpy import zeros, uint8, float32
from pylab import imshow, show, cm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#Function used for reading the dataset
def get_labeled_data(imagefile, labelfile):
   
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')
    images.read(4) 
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    labels.read(4) 
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

 
    x = zeros((N, rows, cols), dtype=float32)
    y = zeros((N, 1), dtype=uint8)
    for i in range(N):
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (x, y)

#Function used for splitting the dataset	
def split_training_data(data, label):
        datalen = 0.8 * len(data)
        intdatalen = int(datalen)
        int_train = data[:intdatalen]
        int_dev = data[intdatalen:]
        int_train_label = label[:intdatalen]
        int_dev_label = label[intdatalen:]
        return (int_train, int_dev, int_train_label, int_dev_label)
		
		
def main():

	imagefilename = sys.argv[1]
	labelfilename = sys.argv[2]
	trainData,trainLabel = get_labeled_data(imagefilename,labelfilename)
    
	int_train, int_dev, int_train_label, int_dev_label = split_training_data(trainData,trainLabel)
	
	resultArray = []
	for i in range(len(int_train)):
			resultArray.append(int_train[i].flatten())
	int_train = np.asarray(resultArray)

	std_scaler=StandardScaler()
	trainData_scaled=std_scaler.fit_transform(int_train.astype(np.float32))
    
	#Using scikit learn's library for training the data and predicting over the test data
	lrClassifier = LogisticRegression(penalty = 'l2', random_state=42, multi_class='multinomial', solver = 'lbfgs', fit_intercept = True, intercept_scaling=1)
	lrClassifier.fit(int_train, int_train_label)
	
	checkArray = []
	for i in range(len(int_dev)):
			checkArray.append(int_dev[i].flatten())
	int_dev_np = np.asarray(checkArray)
	
	predict = lrClassifier.predict(int_dev_np)
	accuracy = accuracy_score(int_dev_label, predict, normalize=True)
	print("the accuracy is")
	print(accuracy*100)
	
if __name__ == "__main__":
    main()
