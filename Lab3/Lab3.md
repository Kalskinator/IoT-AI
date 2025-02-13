# Section5: Assignment (30 pts - 15Mandatory/15 Optional)
## Task A.3.1: Handwriting Recognition (8 points- Mandatory)

The MNIST dataset: MNIST is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. This dataset contains 6000 images for training and 10000 images for testing the out-of-sample performance. Here, let's use the simple algorithms in this lab to build a handwriting model!

Go to the following link, describing how to import the MNIST  dataset and code a logistic regression algorithm for handwriting recognition.  External Link - MNIST dataset LOGISTIC REGRESSIONLinks to an external site.

Import the MNIST dataset.

### I-Use linear regression and SVM (with Linear kernel) and Random Forest(with a maximum depth of your choice) algorithms to classify the hand-written numbers in 10 output classes (0-9) (5 pts-Mandatory)

### II-Visualize the MSE error against Epoch for 3 algorithms in one line plot, with different colors for each algorithm. A legend should be on the top corner ("SVM", "LR", "RF") (3 pts-Mandatory)

## Task A.3.2: Predict the Rain!  - IOT DATA (12 pts - Mandatory/Optional)

thumbnail_ad3ca9f4548a22ac1efd98d2-702555879.jpg

Fig.9: How AI predicts the weather!

In this task, you have given the weather conditions of Seattle, Washington State, US. Given the assumption that the input data is a prediction of the next day's weather, you should predict the output, weather condition, of tomorrow

The input is the min./max. temperature, precipitation, and wind. Your task is to find out how the weather is going to be based on these parameters. There are 5 output classes: (1)drizzle, (2)rain, (3)sun, (4)snow, (5)fog 

Download the dataset from the Kaggle website-  External Link: Seattle Weather Dataset Download External Link: Seattle Weather Dataset(approx. 1460Rows)

 

Import the dataset.

I-Use Linear regression, SVM (with Linear kernel), and Random Forest(with a maximum depth of less than 10) algorithms to classify the weather data in 5 output classes: "drizzle", "rain", "sun", "snow", "fog"  (5 pts-Mandatory)

II-Visualize the MSE error against Epoch for 3 algorithms in one line plot, with different colors for each algorithm. A legend should be on the top corner ("SVM", "LR", "RF") (2 pts-Mandatory)

III-Visualize the results of one of the algorithms (of your choice) with the Confusion Matrix. The matrix should be 5x5. You can read more about it in This LinkLinks to an external site.. (5pts-Optional)

 

 

Task A.3.3: Guess where did I GO! (Arduino Inertia) (5 pts - Optional)

motion-detection-004.gif

In this task you will train a machine learning model to predict which direction the Arduino moves toward. This is just an example (and impractical!) way of predicting the direction of the Arduino, as it can be understood by looking at ACC_X, ACC_Y, and ACC_Z immediately. But lets do the hard work and do it with a simple ML algorithm!

I-Collect 40 data records with low frequency (100Hz) and duration of 1 second each, in which in 20 of them you move the Arduino Right ->, and in another 20 you move the Arduino Left <-. The process can be done with an Arduino IDE script or through the EdgeImpulse website. (2pts Optional)

Export the data as JSON. Use the script here to transform them to CSV, or import them to Python with a script like below. Note that each data is a time series array and not a single variable. 

 

def load_json_data(json_files):
    data = []
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Assuming JSON data is in the form of a time-series array
            data.append(json_data)
    return data
II- Write a linear regression ML code with the five steps described in the instruction above, specified, to guess the label of the data. "left" or "right". (3pts Optional)

Note #1: As a reminder, the accelerometer detects the acceleration or in other words changes in the speed. therefore your move should be fast and jerky to activate the sensor. 

Note #2: As the population of the dataset is very small, PAY ATTENTION to labeling the data "left" or "right" correctly. Double-check your labeling as one wrong label can result in a malfunction of your ML algorithm.

Submit your data collected in a zip file and your Python code.

 

 

Task A.3.4: Classify the Pinguins (Unsupervised) (5 pts - Optional)

 

orZWHly.png

This dataset is the classification of 3 types of penguins based on the length of their bill (or beak).  Here, you should build a K-means clustering model and evaluate your model in terms of accuracy.

Load the dataset of Pinguins: penguins.csv Download penguins.csv. We only need 3 columns of the dataset: "species", "bill_length_mm", and "bill_depth_mm". the data distribution is shown in the below figure.
Build a K-means clustering model to cluster the penguins' types based on "bill_length_mm" and "bill_depth_mm". Visualize the clusters in an XY plane, like the figure below but with the result of your mode. Put the "centroids" of each cluster in the figure. 
Evaluate the model and find the accuracy of your model
 

output_55_0.bmp