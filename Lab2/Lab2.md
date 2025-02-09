# Lab 2: Assignments (20 pts- 15 Mandatory/5 Optional)

## Task A2.1: Temperature Logging and Collection (3 points - Mandatory)

In this task, you should record the temperature data from Arduino Nano RP2040. The frequency of the record is to be at least 1 Hz (1 second). Record the data for at least 15 minutes (900 data rows or more). To make the temperature vary a bit, do several "hot blow"s like "huh"s or "blows" on the board or simply put your finger on the sensor while recording the data. Save them into a CSV. Remember to Submit both CSV and your Python Code.

Hint: To turn your logging data into CSV file, there are some third-party software online that you can use or IDE script which does that.

### I- Collect the data in a CSV file and submit it with the rest of your results (2pts-Mandatory)

### II- Visualize the data with a line graph with two axes: time & temperature with criteria below (1pt-Mandatory)

* The color of the line should be orange
* Add labels for each axes (Temperature (degrees Celsius), Time(seconds)),
* Turn on the grids 
* Add legend on the top right corner - temperature
 
## Task A2.2: Motion Logging - Acc/Gyr  and Collection (5 points - Mandatory)

In this task, you should record the motion data from Arduino Nano RP2040. The frequency of the record is to be at least 2 Hz (0.5 second). Record the data for at least 10 minutes (1200 rows or more). You should collect both accelerometer and gyroscope data. Wiggle the Arduino slightly and abruptly (!) in different directions and angles (Do both angles and directions!) while collecting the motion, to have some peaks in your data. Save them into a CSV. Do the wiggles in certain intervals so you get something like the figure below. 

Hint: To record date and time you can use RTC module and library, but for the sake of simplicity, you don't need to use that for this task. You can define the starting date and time on top of your sketch and use the "Delay" command to define the data logging intervals.

### I- Collect the data in a CSV file and submit it with the rest of your results (2pts-Mandatory)

### II- Visualize the data with a line graph with two axes with criteria below (1pt-Mandatory)

* plot 6 axis of data
* Add labels for each axes (Acceleration (m.sq/s.sq), Time(seconds)),
* Turn on the grids 
* Add legend on the top right corner - the name of your plots should be Ax, Ay, Az, Gx, Gy, Gz
* Remember to Submit both CSV and your Python Code.

### III- (Manipulate the data) With one of the the functions mentioned in the instruction above, write a code which deletes rows of data from your dataset that the acceleration is 0 or close to 0. This will compress your signals to look something like the right graphs below in Fig. 9. Visualize the data again with the above criteria in a line graph. Pay attention that you need to look at the data and define the threshold accordingly for deleting stationary data. The threshold might be different for different axes  (2pt-Mandatory)

## Task A2.3: Frozen! (5 points- Mandatory)

This is a dataset of collecting in/out temperature data over 5 months in (appx. 97000 data rows): Temperature Readings : IOT DevicesLinks to an external site.

### I- Visualize the indoor and outdoor temperature in one plot with different colors of your choice for the last week (strat from the top 02-12-2018 to 08-12-2018). (2 pts-Mandatory)

### II- Do these modifications on the dataframe made from the CSV dataset: (3pts Mandatory)

* Change the "In" and "Out" text of the "Out\In" column to 1 and 0 respectively.
* Separate the date and time in the "noted_date" column, into two separate columns.
* Keep only the data of the last day 08-12-2018, and remove the rest of the rows with the appropriate function
* Submit the modified CSV and your code together.

## Task A2.4: Digital Health IoT dataset (7 points - Mandatory/Optional)

Dataset #1: A sample of 49 participants using an Apple Watch and a "FitBit" app for 65 minutes for 46 participants. This data was collected as part of a Harvard dataset (6265 rows): Apple Watch and Fitbit dataLinks to an external site., there are three files inside archive. We will be focusing on "aw_fb_data"

### I- Based on the instruction on the distribution transformation, transform the "calories" column to take the shape of a distribution close to normal distribution. The current distribution looks something like the below figure. Experiment with different transforms (log, cube, etc.) to find the right one. Use a transform to make the data distribution more consistent, meaning there are values on each column (1pts - Mandatory)

### II- As mentioned before, the data reflects 49 participants. Make a copy of the original dataframe and Find a way to keep one sample from each participant.  Therefore, the new dataframe should have 49 rows. You should use a specific function or a mix of functions in the instruction. Afterward, visualize the "age", "height", and "weight" of the participants on each subplot (stacked plot). Grids should be on, Legends should be on top, and The color of the line plot for each subplot should be different. (2pts - Optional)

### III- Visualize "steps", "heart_rate", and "calories" of the first three participants in three plots with subplots (stacked plot), in a way that the steps of each three participants are depicted with different colored lines, the same for other two datasets. The legends should be on the top corner of each plot (participant #1, participant #2, participant#3) (2pts - Mandatory)

### IV- Normalize the "age", "height", and "weight", and Standardize "steps" and "heart rate" columns in a separate column at the end of the dataframe (1pts - Mandatory)

### V- Split the dataset into three categories with the following distribution: Train (70%), Validation (15%), and Test (15%) (1pts - Mandatory)

Submit both the CSV file and your code.

## Task A2.5: Gone with the Wind!  (3pts - Optional)

You are presented with a dataset of wind speed and the wind angle of the wind from a meteorological site. The problem is that some data while being so similar have values very differently. Angles are not ideal as model inputs since 360° and 0° should be in close proximity, smoothly transitioning. The direction becomes irrelevant when there is no wind blowing. (for example 0.1m/s at 359° is not represented well or the similarity of 10 m/s at 0.1° and 359.9°). The model will find it more straightforward to interpret if you transform the columns for wind direction and velocity into a wind vector (X and Y).

Download the weather dataset from here: Climate2016.csvDownload Climate2016.csv

We only focus on the columns "windvelo m/s" and "winddeg deg" which represent wind velocity and wind direction. See the data summary with the functions in the text to get the overview of the data

Use mathematical functions to convert the Wind "speed&velocity" vector into two separate X & Y vectors. Add two more columns to your CSV file as windveloX and windveloY, and save your CSV file.
Use the normalize function to normalize the data
Use the "Hist2d" function to visualize the data before and after changes. It should look something like the two below figures 

Hint: Use bins parameter equal to (50, 50) and vmax equal to 400 for the "hist2d" function to get the better results.

Submit both the CSV file and your code.