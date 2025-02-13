# Section1: Search Algorithms

## 2.1.Dreath-First Search (DFS)

The algorithm starts at the root node (selecting some arbitrary node as the root node in the case of a graph) and explores as far as possible along each branch before backtracking.

In this example, you will find a graph solved with DFS. Link of Description: Graph Example - DFSLinks to an external site.

```python
# Using a Python dictionary to act as an adjacency list
graph = {
'5' : ['3','7'],
'3' : ['2', '4'],
'7' : ['8'],
'2' : [],
'4' : ['8'],
'8' : []
}

visited = set() # Set to keep track of visited nodes of graph.

def dfs(visited, graph, node):  #function for dfs 
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

# Driver Code
print("Following is the Depth-First Search")
dfs(visited, graph, '5')
```

## 2.1.Breath-First Search (BFS)

BFS algorithm starts at the tree root and explores all nodes at the present depth prior to moving on to the nodes at the next depth level. Extra memory, usually a queue, is needed to keep track of the child nodes that were encountered but not yet explored. 

In this example, you will find a graph solved with BFS. Link of Description: Graph Example - BFSLinks to an external site.

```python
graph = {
'5' : ['3','7'],
'3' : ['2', '4'],
'7' : ['8'],
'2' : [],
'4' : ['8'],
'8' : []
}

visited = [] # List for visited nodes.
queue = []     #Initialize a queue

def bfs(visited, graph, node): #function for BFS
visited.append(node)
queue.append(node)

while queue:          # Creating loop to visit each node
    m = queue.pop(0) 
    print (m, end = " ") 

    for neighbour in graph[m]:
    if neighbour not in visited:
        visited.append(neighbour)
        queue.append(neighbour)

# Driver Code
print("Following is the Breadth-First Search")
bfs(visited, graph, '5')    # function calling
```

# Section2: Introduction to Supervised and Unsupervised Learning 

There are several library options to develop machine learning projects including Pytorch and Tensorflow, also (others like scikit-learn, Tensorforce, and coach-RL depending on the application). The choice between these often depends on individual preferences, specific project requirements, and the development paradigm favored by practitioners. In this course, we will be focusing on Pytorch:

## 1.1.Pytorch (Library)
PyTorch, a powerful open-source machine learning library, stands at the forefront of deep learning frameworks. Developed by Facebook's AI Research lab (FAIR), PyTorch seamlessly combines flexibility and efficiency, making it a preferred choice for researchers and developers alike.  From prototyping to production deployment, PyTorch empowers practitioners to tackle diverse machine learning tasks(It uses more resources than TF but provides more speed when coupled with GPU).

Some of the most famous real-world applications done by Pytorch:

* OpenAI's GPT-2 and GPT-3: OpenAI used PyTorch to implement its state-of-the-art language models
* Facebook Translation: PyTorch is utilized in the development of Facebook's machine translation systems.
* Tesla Autopilot: Tesla has been reported to use PyTorch in its Autopilot system, which is responsible for various aspects of autonomous driving.
* AI for Healthcare at Mount Sinai Hospital: Mount Sinai Hospital in New York has employed PyTorch in various projects related to healthcare and medical imaging.

### To use Pytorch we use these import functions
```python
#To load the whole library for making datasets, torch optimizer, and other functions
import torch

#to use torch functions related to building neural networks, fitting data, loss functions and other functions
import torch.nn as nn
```
## 1.2. Building dataset/importing in Pytorch
### You can build a tensor dataset in Pytorch or import it from Pandas dataframe as we mentioned in the previous lab.
```python
#making tensor input and output data
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
print(x_data)
print(y_data)
```
### load the dataset from pandas data frame - X and Y are pandas data frames
```python
data = {'Height': [165.4, 175.9, 125.2, 189.5], 'Age': [25, 30, 15, 40]}
df = pd.DataFrame(data)
X = df['Height']
Y=df['Age']
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int)
print(X)
print(Y)

#Reshape function used to transpose or lower the dimension of the data- if your y function consists of 1 column of data in the shape of (1,X), you can reduce the number of dimensions to one by the below command.
Y = torch.tensor(Y, dtype=torch.int).reshape(-1, 1)
print(Y)
```
### For saving and loading your trained model in Pytorch the below codes are used. "model" is your trained model:
```python
#Showing learn parameters during the training stage

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

#Saving your model

torch.save(model.state_dict(), PATH)

#Loading your model

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```


# Section3: Supervised Machine Learning

## 3.0. Getting Started

In this section, we will develop basic ML algorithms to map inputs and outputs.

Every machine learning code has 5 main sections: I. importing preprocessed datasets (refer to Lab#1), II. defining the algorithm, III. training the model, IV. evaluating the model, V. visualizing outputs (optional). It is important to check in each ML code whether it includes all of these sections. As a suggestion, it is a good practice to specify each of these sections with comments (done with # for single-line comments and '' open close'' for multi-line comments) which will make debugging and code readability much easier. As you become more skilled in coding you might need less or no commenting. The 5-section commenting is done in the example below. 

Now let us build some basic ML algorithms!

## 3.1.Decision Tree 

Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. Since implementing the decision tree is much easier with the Scikit-learn library, for this particular algorithm we will use it instead of Pytorch.

Some advantages of decision trees are:

* Simple to understand and to interpret.
* Trees can be visualized.
* Requires little data preparation.

The disadvantages of decision trees include:

* Decision-tree learners can create over-complex trees that do not generalize the data well,
* Overfitting
* Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.

Let's find how the decision tree works with an example. Download the heart study dataset below. It is a collected dataset from more than 4000 patients (age, smoking habit, BMI, blood pressure medicine, history of stroke etc.) to predict high or low risk of heart attack in adults.

Link to dataset: [Framingham heart study dataset](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset/)
#### Importing the libraries that are being used
```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split #preprocessing: Train/Test split
from sklearn.model_selection import cross_val_score #Scoring the output result
import matplotlib.pyplot as plt #Ploting
from sklearn.tree import DecisionTreeClassifier #Sci-kit learn function - DT
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
```
### Section I-importing data load the dataset, split into input (X) and output (y) variables.
#### Inputing the dataset and get an overview
```python
dframe=pd.read_csv(r"framingham.csv")
```
#### Preprocessing data
```python
df=dframe.copy()
df.head()
```
#### IT SEEMS EDUCATION HAS NO REALTION WITH HEART DISEASE SO LETS DROP IT
```python
df.drop(["education"],axis=1,inplace=True)
df.rename(columns={'TenYearCHD':'target'},inplace=True) #change the name of "TenYearCHD" to "target"
```
#### good! now lets see all attributes are numerical attributes in one place
```python
df
print(df.shape)
df.info()
```
#### Finidng the number of missing data
```python
df.isnull().sum()
```
#### Handling missing data: here, filling all null values in glucose attribute with the mean value
```python
df['glucose'].fillna(df['glucose'].mean(),inplace=True)
df.dropna(inplace=True)
```
#### validation in preprocessing: Now lets visualize the data. First, Lets see the number of 0 and 1 in in the "Outcome" which is the number of diabetes patients
```python
catag=[i for i in df.columns if len(df[i].unique())<4]
random=[i for i in df.columns if len(df[i].unique())>=4]
catag.remove("target")
plt.figure(figsize=(25,20))
for n,column in enumerate(catag):
    plot=plt.subplot(3,2,n+1)
    plt.xlabel(column,fontsize=20)
    plt.ylabel("COUNT",fontsize=20)
    sns.countplot(x=df[column],color='yellow',data=df)
    plt.title(f'{column.title()}',weight='bold',fontsize=30)
    plt.tight_layout()
```
#### Now, Lets see how many patients in each group are dianetes (1) or not (0) (finding the most relevant data to diabetes risk)
```python
plt.figure(figsize=(25,20))
for n,column in enumerate(catag):
    plot=plt.subplot(3,2,n+1)
    plt.xlabel(column,fontsize=20)
    plt.ylabel("count",fontsize=20)
    sns.countplot(x=df[column],hue=df["target"],data=df)
    plt.title(f'{column.title()}',weight='bold',fontsize=30)
    plt.tight_layout()
```
#### Saving X and Y data in separate variables and splitting train/validation samples
```python
x=df.drop(["target"],axis=1)
y=df["target"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.22,random_state=40)
xtrain.shape,xtest.shape,ytrain.shape,ytest.shape
```

### Section II-defining the model: Decision tree algorithm- Applying the decision tree
#### Defining a decision tree with a maximum branch number of 3. More branches can lead tomore accurate results, but might lead to overfitting as well
```python
dtree=DecisionTreeClassifier(criterion="entropy", max_depth=3) 
```

### Secion III-training the model
```python
dtree.fit(xtrain,ytrain)
```
#### test for validation dataset
```python
ypred_dtree=dtree.predict(xtest)
dtree_model=accuracy_score(ytest,ypred_dtree)
```
#### test for train data
```python
ypred_dtree2=dtree.predict(xtrain)
```

### Section IV-evaluating the model
```python
dtree_model2=accuracy_score(ytrain,ypred_dtree2)
print(f"accuracy for validation set :{dtree_model}\naccuracy for train set :{dtree_model2}")
score2 = cross_val_score(dtree, xtrain, ytrain, cv = 30)
print(f"\nafter cross validation the accuracy is {round(score2.mean(),2)}")
#Precesion, F1-score, Accuracy
print(classification_report(ytest,ypred_dtree))
```

### Section V-visualizing the outputs
#### Text format visualization 
```python
from sklearn import tree
text_representation = tree.export_text(dtree)
print(text_representation)
```

#### Visualizing the decision tree in boxes
```python
fig = plt.figure(figsize=(25,20))
feature_names = df.columns.tolist() #saving the features in a list
feature_names.remove("target") #Removing the target (0,1) from the features
class_names = ["0", "1"] #Manually adding the target classes, 0 means patient does not have diabetes, 1 means they have
print (feature_names, class_names)
_ = tree.plot_tree(dtree, feature_names= feature_names, class_names=class_names)
```

For more information refer to scikit-learn documentation about the decision tree and the command: [Decision Tree Fundamentals](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)., External Link-DecisionTreeClassifier

## 3.2. Random Forest (Supplementary)

This is a similar classifying algorithm to the decision tree. A random forest is a meta-estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

For more information and to learn about the Random Forest algorithm follow this link: External Link - Sci-kit Learn: Random ForestLinks to an external site.

 

## 3.3.linear regression


Here you get the basics of linear regression with an example. Before we delve into the linear regression mode, it is worth noting that in the text above we mentioned that each ML code usually consists of 5 general parts. In the "II. defining the algorithm" section in Pytorch, there are two structures that can be used (as in Tensorflow). These structures are shown below. Here, we will focus on the "A. Easy structure". Still, it is necessary for you to familiarize yourself with both structures in case you see them in other developers' codes. The below example is a simple 4-layer model written in two different ways which consisted of X inputs, 2 hidden layers, and Y outputs. We will discuss each of these functions later on. So it is OK if you do not understand the layers of the code at this point.

### A. Easy structure
```python
#II. defining an algorithm
#examples of A. easy structure model definition
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(X, x1),
    nn.ReLU(),
    nn.Linear(x1, x2),
    nn.ReLU(),
    nn.Linear(x2, Y),
    nn.Softmax()
)
```
### B. Verbose structure
```python
#II. defining an algorithm

#examples of A. easy structure model definition
import torch.nn as nn

class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(X, x1)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(x1, x2)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(x2, Y)
        self.act_output = nn.Softmax()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x
```
 

Now let's train a regression model with the easy structure mentioned above. Download and place the dataset below in your Python code folder. It is data gathered from 768 women and recorded their health measures and if they had diabetes or not. The inputs and outputs are:

* 8 Inputs (X): Number of times pregnant, Plasma glucose concentration at 2 hours in an oral glucose tolerance test, Diastolic blood pressure (mm Hg), Triceps skin fold thickness (mm), 2-hour serum insulin (μIU/ml), Body mass index (weight in kg/(height in m)2), Diabetes pedigree function, Age (years)
* 1 Output(Y): Class label (0 or 1) - if they  have diabetes
Dataset link: External Link - [Pima indans diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

In the below code, you will build a linear regression model to predict based on X if a patient has a high risk of diabetes. The "Easy Structure" is used here (described above). Pay attention to the 5 sections of the model described above. Also, as another way of loading the dataset, the NumPy library is used here instead of Pandas. Run the code and see the results:

#### importing libraries
```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

### Section I-importing data load the dataset, split into input (X) and output (y) variables. 
```python
dataset = np.genfromtxt('diabetes.csv', delimiter=',', usecols=range(9))      #Loading dataset Also np.loadtxt('diabetes.csv', delimiter=',')
X = dataset[1:,0:8] #getting the 8 first columns as input.The reason for "1"instead of "0" is todiscard the headings
y = dataset[1:,8] #getting the last column as output. The reason for "1"instead of "0" is todiscard the headings
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
#Splitting into train and validation datasets with 67% trainset and 33% validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```
### Section II-defining the model:  linear regression model with 2 layers, input and output
```python
model = nn.Sequential(
  nn.Linear(8, 1),
  nn.Sigmoid()
)
#priniting model layers and specifications
print(model)
```
### Secion III-training the model
```python
n_epochs = 100
batch_size = 8
history = []
# define loss function
loss_fn = nn.MSELoss()
# define optimizer with a spicific learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(n_epochs):
  for i in range(0, len(X_train), batch_size):
        # take a batch
        Xbatch = X_train[i:i+batch_size]
        ybatch = y_train[i:i+batch_size]
        # forward pass
        y_pred =model(Xbatch) #  torch.max(model(Xbatch), 1)
        loss = loss_fn(y_pred, ybatch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()

  model.eval()
  y_pred = model(X_test)
  mse = loss_fn(y_pred, y_test)
  #mse_train = loss_fn(y_, y_test)
  mse = float(mse)
  history.append(mse)
  print(f'Finished epoch {epoch}, latest MSE {mse}')
```
### Section IV-evaluating the model
```python
# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")
```
### Section V-visualizing the outputs
```python
plt.plot(history)
plt.title('Mean Square Error')
plt.xlabel("Epoch")
plt.ylabel("")
plt.show()
```

## 3.4.Support Vector Machine (SVM)

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression, and detection of outliers. Since implementing the decision tree is much easier with the Scikit-learn library, for this particular algorithm we will use it instead of Pytorch. It is mostly used for classification problems. The advantages of SVM are:

* Effective in high-dimensional spaces.
* Versatile
* Memory Efficient: Uses a subset of training points in the decision function (called support vectors).
However, the disadvantages of SVM are:

* The risk of overfitting is great. If the number of features is much greater than the number of samples, one should avoid over-fitting by choosing the Kernel functionsLinks to an external site. and regularization terms are crucial.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation

**C function**: To avoid your model from overfitting, there is an inherent function in SVM called the C function. The more the amount of C function, the harsher the model will be punished when overfitted to the data. (if you are not familiar with overfitting, it is alright, as it will be explained later in the course.)

**Kernel Function**: To make things more interesting, the SVM algorithm has a Kernel function that determines the complexity of classification. This kernel should be chosen based on the nature of the problem. Here. we will only work with the "Linear" kernel. Read more in the documentation link, provided at the end of this section.

Now, let's see a classification example solved with SVM:

### Example 1: classifying with SVM - 2 classes - Breast Cancer Data
#### in this built-in example of the Sci-kit learn library, we try to predict breast cancer type based on input data. 0 means the cancer is benign type, 1 means the cancer is malignant
```python

from sklearn import datasets # Import train_test_split function
from sklearn.model_selection import train_test_split  #Import scikit-learn dataset library
from sklearn import svm #Import svm model
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

#Load dataset 
cancer = datasets.load_breast_cancer()

# print the names of the 13 features 
print("Features: ", cancer.feature_names) 

# print the label type of cancer('malignant' 'benign') 
print("Labels: ", cancer.target_names)

# print data(feature)shape 
cancer.data.shape
# print the cancer data features (top 5 records) 
print(cancer.data[0:5])
# print the cancer labels (0:malignant, 1:benign) 
print(cancer.target)

# Split dataset into training set and test set - 70% training and 30% test 
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) 

#Create a svm Classifier with Linear Kernel
clf = svm.SVC(kernel='linear') 
#Train the model using the training sets
clf.fit(X_train, y_train) 
#Predict the response for test dataset 
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision and Recall: what percentage of positive tuples are labeled as such? 
print("Precision:",metrics.precision_score(y_test, y_pred)) 
print("Recall:",metrics.recall_score(y_test, y_pred))
```
### Examples 2: classifying with SVM - 3 classes in XY plane
#### This example classifies 3 classes of data scattered in the X -Y plane - There is a visualization at the end that shows how accurate the model functions:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=1,gamma=0).fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()
```
 

For more info please refer to the official documentation of the scikit-learn library: [External Link - SVM](https://scikit-learn.org/stable/modules/svm.html)


# Section4: Unsupervised Machine Learning

**Unsupervised machine learning** is the process of inferring underlying hidden patterns from historical data. 

-An Example:

Picture a toddler. The child knows what the family cat looks like (provided they have one) but has no idea that there are a lot of other cats in the world that are all different. The thing is, if the kid sees another cat, he or she will still be able to recognize it as a cat through a set of features such as two ears, four legs, a tail, fur, whiskers, etc.

In machine learning, this kind of prediction is called unsupervised learning. But when parents tell the child that the new animal is a cat – drumroll – that’s considered supervised learning.

## 4.1. Clustering (K-Means)

K-means clustering is an unsupervised algorithm that groups unlabelled data into different clusters. The K in its title represents the number of clusters that will be created. This is something that should be known prior to the model training. For example, if K=4 then 4 clusters would be created, and if K=7 then 7 clusters would be created. The k-means algorithm is used in fraud detection, error detection, and confirming existing clusters in the real world.

Conventional k-means require only a few steps. The first step is to randomly select k centroids, where k is equal to the number of clusters you choose. Centroids are data points representing the center of a cluster.

The quality of the cluster assignments is determined by computing the sum of the squared error (SSE)Links to an external site. after the centroids converge, or match the previous iteration’s assignment. The SSE is defined as the sum of the squared Euclidean distances of each point to its closest centroid. Since this is a measure of error, the objective of k-means is to try to minimize this value. Figure 6 shows the centroids and SSE updating through the first five iterations from two different runs of the k-means algorithm on the same dataset.

Let's show how it works with a small example. The problem is clustering a simple, unlabeled dataset on the X-Y plane. A method called the "Elbow method" is used to find the optimal number of clusters in the code below. Then the centroids are found through iterations.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#defined dataset on the X-Y plane
x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

plt.scatter(x, y)
plt.show()

#making tuples of (X,Y) with the zip function
#the inertia is the same as the SSE (sum of squared error) that was defined above
data = list(zip(x, y))
inertias = []

#testing the data on 1 to 11 clusters and calculating the inertia
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#You can see on the plot that 2 is an elbow point. 2 is a good value for K, so we retrain and visualize the result.
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
plt.scatter(x, y, c=kmeans.labels_)
plt.show()

#Now lets add the centroids of each cluster to the plot

centroids  = kmeans.cluster_centers_ 
plt.scatter(x, y, c=kmeans.labels_)
plt.scatter(centroids[:,0], centroids[:,1], c="red")
plt.show()
```
For more information and other clustering methods refer to the documentation: [External Link - Sci-kit Learn: Clustering](https://scikit-learn.org/stable/modules/clustering.html)

 

## 4.2. Anomaly detection (Supplementary)

For more information refer to the documentation: External Link: [Sci-kit learn - Outliers](https://scikit-learn.org/stable/modules/outlier_detection.html)