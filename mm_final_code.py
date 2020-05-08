# Importing all the required libraries
import pandas as pd                                                         # For loading the DataFrame
from sklearn import preprocessing                                           # For normalising the data
from sklearn.model_selection import train_test_split, cross_val_score       # For splitting the data into train and test data sets
from sklearn.tree import DecisionTreeClassifier                             # Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier                         # Random Forest Classifier
from sklearn import svm                                                     # Support Vector Machine Classifier
from sklearn import neighbors                                               # K Nearest Neighbours Classifier
from sklearn.naive_bayes import MultinomialNB                               # Naive Bayes Classifier
from sklearn.preprocessing import MinMaxScaler                              # Used to transform and normalize the input features
from sklearn.linear_model import LogisticRegression                         # Logistic Regression Classifier
import numpy as np                                                          # For mathematical operations

# Importing required libraries for the Neural Network
from keras.layers import Dense 
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

# To ignore any warnings that might pop up
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Loading our data set and printing first 5 rows
df = pd.read_csv("D://VS_code//Udemy_project//mammographic_masses.data.txt")
print(df.head())

# Procesing the data set and adding column names
masses_data = df
masses_data = pd.read_csv("mammographic_masses.data.txt", na_values=['?'], names = ['BI_RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity'])
masses_data.head()

# Checking if data is randomly missing and then dropping missing values
print(masses_data.loc[(masses_data['Age'].isnull()) | (masses_data['Shape'].isnull()) | (masses_data['Margin'].isnull()) |(masses_data['Density'].isnull())])
masses_data.dropna(inplace=True)
print(masses_data.describe())

# Converting the columns into features and labels for our classification
all_features = masses_data[['Age','Shape','Margin','Density']].values
all_classes = masses_data[['Severity']].values
feature_names = ['Age','Shape','Margin','Density']

# Scaling and normalizing the features 
scaler = preprocessing.StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)

# Converting features and labels into train and test set of a ratio 75%:25% 
np.random.seed(1234)
X_train, X_test, Y_train, Y_test = train_test_split(all_features_scaled, all_classes, train_size=0.75, random_state=15)

# Decision Tree Classifier with accuracy score
clf = DecisionTreeClassifier(random_state=5)
clf.fit(X_train,Y_train)
cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=86)
cv_scores.mean()*100

# Random Forest Classifier with accuracy score
clf = RandomForestClassifier(n_estimators=17, random_state=3)
clf.fit(X_train, Y_train)
cv_scores = cross_val_score(clf,all_features_scaled,all_classes, cv = 86)
cv_scores.mean()*100

# SVM with 3 types of kernels (linear, rbf, poly)
C = 2.0
svc = svm.SVC(kernel='linear',C=C)
cv_scores=cross_val_score(svc,all_features_scaled,all_classes,cv=15)
cv_scores.mean()*100

svc = svm.SVC(kernel='rbf',C=1.0)
cv_scores=cross_val_score(svc,all_features_scaled,all_classes,cv=15)
cv_scores.mean()*100

svc = svm.SVC(kernel='poly',C=1.0)
cv_scores=cross_val_score(svc,all_features_scaled,all_classes,cv=15)
cv_scores.mean()*100

# KNN classifier with values of k from 1 to 50 to decide which is the best value
li = []
for i in range(1,50):
    clf = neighbors.KNeighborsClassifier(n_neighbors=i)
    cv_score=cross_val_score(clf,all_features_scaled,all_classes,cv=18)
    li.append(cv_score.mean())
knn = max(li)*100
k = li.index(knn/100)+1

# Naive Bayes Classifier with accuracy score
scaler = MinMaxScaler()
all_features_minmax = scaler.fit_transform(all_features)
clf = MultinomialNB()
cv_score = cross_val_score(clf, all_features_minmax, all_classes, cv=20)
cv_score.mean()*100

# Logistic Regression with accuracy score
clf = LogisticRegression()
cv_score = cross_val_score(clf, all_features_minmax,all_classes, cv = 18)
cv_score.mean()*100

# Creating a neural network to predict the severity of the masses
def create_model():
    model = Sequential()
    model.add(Dense(6,activation='relu',kernel_initializer='normal',input_dim=4))
    model.add(Dense(1,activation='sigmoid',kernel_initializer='normal'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

est = KerasClassifier(build_fn=create_model, epochs=10, verbose=0)
cv_score = cross_val_score(est, all_features_scaled, all_classes, cv = 10)
cv_score_1 = cv_score.mean()*100