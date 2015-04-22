import numpy as np,sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcess
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import cross_validation
import ensemble
import pyGPs
from pyGPs.Validation import valid
from scipy.io import loadmat

numClass = 10
numSamples = 250
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Poly SVM","Sigmoid","Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes","LogisticRegression"]

classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025),
    SVC(kernel ="rbf", gamma=2, C=1),
    SVC(kernel="poly",C = 0.2 , gamma = 0.2),
    SVC(kernel="sigmoid"),
    DecisionTreeClassifier(max_depth=7),
    RandomForestClassifier(max_depth=7, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LogisticRegression(),
	]

ensemble_classifier = ensemble.EnsembleClassifier(classifiers,voting = 'hard')
#--------------------------------------------------
# Test using various sample sizes
#--------------------------------------------------

def printStatistics(name,prediction,y_test):
	print "--------------",name,"---------------"
	print (classification_report(prediction,y_test))
	print 
	cm = confusion_matrix(prediction,y_test)
	print cm
	print("CORRECT PREDICTIONS:")
	correct = np.trace(cm)
	print(correct)
	print("TOTAL PREDICTIONS:")
	total = np.sum(cm)
	print(total)
	print("ACCURACY:")
	print(float(correct)/float(total))
	print 
	#--------------------------------------------------------
	# Uncomment if we want to generate graphs
	#--------------------------------------------------------
	# plt.matshow(cm)
	# plt.title('Confusion matrix for: '+str(name))
	# plt.colorbar()
	# plt.ylabel('True label')
	# plt.xlabel('Predicted label')
	# plt.savefig('Images/'+str(name)+'.png')
	# plt.show()

print 
print "-------------------USPS Handwritten Digits Recognition---------------------"
print "------------Comparative Results among different classifiers----------------"
print 

print "Loading USPS handwritten digits dataset............."
data = loadmat('usps_resampled.mat')
x_train = data['train_patterns'].T   # train patterns
y_train = data['train_labels'].T     # train labels
x_test = data['test_patterns'].T   # test patterns
y_test = data['test_labels'].T     # test labels  
y_train = np.argmax(y_train, axis=1)
y_train = np.reshape(y_train, (y_train.shape[0],1))
y_test = np.argmax(y_test, axis=1)
y_test = np.reshape(y_test, (y_test.shape[0],1))
print "Dataset loaded............."

print 
print "Reducing training Data to only "+str(numSamples)+" samples...."
x_train  = x_train[:numSamples,:]
y_train = y_train[:numSamples,:]
print "Done....."

#---------------------------------------------------------------------------------

print "-----------------Iterating over standard classifiers-----------------------"
x_train_stclass = StandardScaler().fit_transform(x_train)
y_train_stclass = y_train.ravel()
for name, clf in zip(names,classifiers):
	clf.fit(x_train_stclass,y_train_stclass)
	score = clf.score(x_test, y_test)
	prediction = clf.predict(x_test)
	printStatistics(name,prediction,y_test)

#---------------------------------------------------------------------------------
print "-----------------Using an ensemble of above methods------------------------"
ensemble_classifier.fit(x_train_stclass,y_train_stclass)
prediction = ensemble_classifier.predict(x_test)
printStatistics("EnsembleClassifier",prediction,y_test)

#---------------------------------------------------------------------------------
print 
print "----------------Modelling a gaussian process to fit data-------------------"
print 
gp_classifier = pyGPs.GPMC(numClass)
m = pyGPs.mean.Zero()
k = pyGPs.cov.RBF()
gp_classifier.setPrior(mean=m,kernel=k)
# gp_classifier.useInference("Laplace")
gp_classifier.setData(x_train,y_train)
predictive_vote = gp_classifier.optimizeAndPredict(x_test)
predictive_class = np.argmax(predictive_vote, axis=1)
prediction = np.reshape(predictive_class, (predictive_class.shape[0],1))
printStatistics("Gaussian Process- Multi Class",prediction,y_test)






	





