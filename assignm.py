



def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
#    return [ (1234567, 'Ada', 'Lovelace'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos') ]
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    ##         "INSERT YOUR CODE HERE"   
    import numpy as np
    import pandas as pd
    path=dataset_path
    df=pd.read_csv(path) # read a comma separated text file
    df1=df.fillna(df)   # filter out rows which has blank values
    dataset1 =df.values[:,1:2]
    dataset2 =df.values[:, 2:]
    y=np.array(dataset1) # set numpy array for y
    X=np.array(dataset2) # set numpy array for X
    return X, y
    #print(X)
    #raise NotImplementedError()

def build_tain_data(X_data, y_data):
    from sklearn.model_selection import train_test_split
    X =X_data
    y =y_data
    X_train , X_test , y_train , y_test = train_test_split(X,y , random_state=4)
    return X_train , X_test , y_train , y_test
	

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(X_training, y_training):
    '''  
    Build a Naive Bayes classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    from sklearn.naive_bayes import GaussianNB  
    X = X_training
    y = y_training
    clf = GaussianNB()
    clf.fit(X, y)  
    return clf 
    #raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    from sklearn import tree
    X = X_training
    y = y_training
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)  
    return clf    
    #raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    from sklearn.neighbors import NearestNeighbors
    X = X_training
    y = y_training
    clf = NearestNeighbors()
    clf.fit(X, y)  
    return clf
    #raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    from sklearn import svm
    X = X_training
    y = y_training
    clf = svm.SVC()
    clf.fit(X, y)  
    return clf    
    #raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass
    # call your functions here
    
    X_data_set , y_data_set = prepare_dataset("medical_records.data")
    X_training, X_testing, y_training , y_testing = build_tain_data(X_data_set, y_data_set)
    build_NB_classifier(X_training, y_training)
    build_DT_classifier(X_training, y_training)
    build_NN_classifier(X_training, y_training)
    build_SVM_classifier(X_training, y_training)

