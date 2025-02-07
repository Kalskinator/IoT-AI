import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Put it on top of your code. It will Make numpy values easier to read. supress avoids 0.0e0 where not necessary, precession is decimal digits
np.set_printoptions(precision=3, suppress=True)

class NumpyArray:
    def creating_arrays():
        # From a list 
        a = np.array([1, 2, 3]) 
        print("Array from list:", a) 

        # All zeros 
        b = np.zeros((2, 2)) 
        print("All zeros:", b) 

        # All ones 
        c = np.ones((3, 3)) 
        print("All ones:", c) 

        # Identity matrix 
        d = np.eye(4) 
        print("Identity matrix:", d) 

        # Range with step 
        e = np.arange(0, 10, 2) 
        print("Array from range with step:", e) 

        # Random values 
        f = np.random.rand(2, 2) 
        print("Random array:", f)

        # Random integers 
        g = np.random.randint(1, 10, (3, 3)) 
        print("Random integer array:", g) 

        # Evenly spaced values 
        h = np.linspace(0, 1, 5) 
        print("Evenly spaced array:", h)

    def creating_2d_arrays():
        # Create a 2x3 array 
        array_2d = np.array([[1, 2, 3], [4, 5, 6]]) 
        print("2D array:") 
        print(array_2d)

    def array_operations():
        a = np.array([1, 2, 3]) 
        b = np.array([4, 5, 6]) 
        # Element-wise addition 
        c = a + b 
        # Output: [5 7 9] 
        print("Element-wise addition:", c)

        # Element-wise subtraction 
        d = a - b 
        # Output: [-3 -3 -3] 
        print("Element-wise subtraction:", d)

        # Element-wise multiplication 
        e = a * b 
        # Output: [4 10 18] 
        print("Element-wise multiplication:", e)

        # Element-wise division 
        f = a / b 
        # Output: [0.25 0.4 0.5] 
        print("Element-wise division:", f)

        # Element-wise exponentiation 
        g = a ** 2 
        # Output: [1 4 9]
        print("Element-wise exponentiation:", g)

    def indexing_and_slicing():
        # 1D array 
        arr_1d = np.array([0, 1, 2, 3, 4, 5]) 
        # Indexing in 1D array 
        print(arr_1d[2]) 
        # Output: 2 

        # Slicing in 1D array 
        print(arr_1d[1:4]) 
        # Output: [1 2 3] 

        # 2D array 
        arr_2d = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]) 
        # Indexing in 2D array 
        print(arr_2d[1, 2]) 
        # Output: 5 

        # Slicing rows in 2D array 
        print(arr_2d[0:2]) 
        # Output: [[0, 1, 2], [3, 4, 5]] 

        # Slicing specific columns for the first two rows in 2D array 
        print(arr_2d[0:2, 1]) 
        # Output: [1 4] 

        # Combining indexing and slicing in 2D array 
        print(arr_2d[1, 0:2]) 
        # Output: [3 4]

        # Reshaping a 1D array to a 2D array 
        a = np.array([1, 2, 3, 4, 5, 6]) 
        reshaped_a = a.reshape(2, 3) 
        print("Reshaped array:\n", reshaped_a) 

        # Output: # [[1 2 3] # [4 5 6]] 
        # Reshaping a 2D array to a 1D array 
        flattened_a = reshaped_a.reshape(-1) 
        print("Flattened array:", flattened_a) 
        # Output: [1 2 3 4 5 6] 

        # Reshaping with unknown dimension 
        unknown_dim_a = a.reshape(2, -1) 

        # -1 is automatically inferred to be 3 
        print("Reshaped with unknown dimension:\n", unknown_dim_a) 
        # Output: # [[1 2 3]

    def universal_functions():
        # Define example arrays 
        arr1 = np.array([1, 4, 9, 16]) 
        arr = np.array([1, 2, 3, 4]) 
        # Square root 
        sqrt_arr = np.sqrt(arr1) 
        print("Square root:", sqrt_arr) 

        # Rounding to the nearest integer 
        rounded_arr = np.round([1.3, 2.7, 4.1]) 
        print("Rounded:", rounded_arr) 

        # Absolute value 
        abs_arr = np.abs([-1, -2, -3])
        print("Absolute value:", abs_arr) 

        # Trigonometric functions 
        sin_arr = np.sin(arr1) 
        print("Sine values:", sin_arr) 
        cos_arr = np.cos(arr1) 
        print("Cosine values:", cos_arr) 

        # Mean 
        mean_value = np.mean(arr) 
        print("Mean value:", mean_value) 

        # Standard deviation 
        std_deviation = np.std(arr) 
        print("Standard deviation:", std_deviation) 

        # Sum of all elements 
        total_sum = np.sum(arr) 
        print("Total sum:", total_sum) 

        # Minimum and Maximum values 
        min_value = np.min(arr) 
        print("Minimum value:", min_value) 
        max_value = np.max(arr) 
        print("Maximum value:", max_value)

    def masking_and_boolean_indexing():
        # Create an array 
        arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 

        # Create a mask 
        mask = arr > 5 

        # Use boolean indexing to create a filtered array 
        filtered_arr = arr[mask] 
        print("Filtered array:", filtered_arr)
        
        # Combine masking and boolean indexing in one line 
        filtered_arr_one_line = arr[arr > 5] 
        print("Filtered array in one line:", filtered_arr_one_line)

    def broadcasting_and_reshaping():
        # Broadcasting with a scalar 
        a = np.array([1, 2, 3]) 
        result = a * 2 
        print("Broadcasting with a scalar:", result) 
        # Output: [2 4 6]

        # Broadcasting with different shaped arrays 
        b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) 
        c = np.array([1, 2, 3]) 
        result = b + c 
        print("Broadcasting with different shaped arrays:\n", result) 
        # Output: 
        # [[ 2 4 6] 
        # [ 5 7 9] 
        # [ 8 10 12]] 

        # Broadcasting a column vector across a 2D array 
        d = np.array([[1], [2], [3]]) 
        result = b + d 
        print("Broadcasting a column vector across a 2D array:\n", result) 
        # Output: 
        # [[ 2 3 4] 
        # [ 6 7 8] 
        # [10 11 12]]

        # Reshaping a 1D array to a 2D array 
        a = np.array([1, 2, 3, 4, 5, 6]) 
        reshaped_a = a.reshape(2, 3) 
        print("Reshaped array:\n", reshaped_a) 
        # Output: 
        # [[1 2 3] 
        # [4 5 6]] 
        # Reshaping a 2D array to a 1D array 
        flattened_a = reshaped_a.reshape(-1) 
        print("Flattened array:\n", flattened_a) 
        # Output: [1 2 3 4 5 6] 
        # Reshaping with unknown dimension 
        unknown_dim_a = a.reshape(2, -1) 
        # -1 is automatically inferred to be 3 
        print("Reshaped with unknown dimension:\n", unknown_dim_a) 

        # Output: 
        # [[1 2 3]
        # [4 5 6]]

class Panda:
    def import_export_create_dataframe():
        data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'Occupation': ['Engineer', 'Doctor', 'Artist', 'Teacher']
        }
        df = pd.DataFrame(data)
        # print(df)

        #Importing data from txt file. the data in the text is separated with ",'
        # dataset = np.loadtxt('example.txt', delimiter=',')

        # Import a DataFrame from file
        actors_df = pd.read_csv('lab2/Data_processing/actors.csv')
        # Display the loaded DataFrame
        print(actors_df)

        # if your database is in .txt format, you still can import the data to a pandas data frame:
        # data = pd.read_csv('output_list.txt', header = None)

        # Save the DataFrame to a CSV file
        actors_df.to_csv('lab2/Data_processing/actors.csv', index=False) 
        # Reload the DataFrame from the CSV file
        reloaded_df = pd.read_csv('lab2/Data_processing/actors.csv')
        # Display the reloaded DataFrame to verify it matches the original
        print(reloaded_df)

    def data_selection_iloc():
        mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
        {'a': 100, 'b': 200, 'c': 300, 'd': 400},
        {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }] 
        df = pd.DataFrame(mydict)


        print("Type",type(df.iloc[0]))
        print("iloc1", df.iloc[[0]])
        print("iloc2", df.iloc[[0, 1]])

        actors_df = pd.read_csv('lab2/Data_processing/actors.csv')
        #OR 
        #to select input, output, or print a specific row or column use "iloc" examples:
        print(actors_df.iloc[7:8,:])
        #X_l = reloaded_df.iloc[:, 1:-1].values # features set
        #y_p = reloaded_df.iloc[:, -1].values # set of study variable 

        #for selecting every 5 row:rows devidable on 5 
        print(actors_df.iloc[lambda x: x.index % 5 == 0])

        # or use Slice [start:stop:step], starting from index 5 showing every 6th row in the dataset -> which will be (5,11,17,23,....)
        print(actors_df[5::6])

class Matplotlibrary:
    def basic_plots():
        data = {'apple': 10, 'orange': 15, 'lemon': 5, 'lime': 20}
        names = list(data.keys())
        values = list(data.values())

        fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
        axs[0].bar(names, values)
        axs[1].scatter(names, values)
        # axs[2].plot(names, values)
        fig.suptitle('Categorical Plotting')
        
        axs[2].fill_between(names, values, alpha=0.7)
        for ax in axs[0],axs[1],axs[2]:
            ax.grid(True)
        plt.show()
        
    def colored_bar_chart():
        pfig, ax = plt.subplots()

        fruits = ['apple', 'blueberry', 'cherry', 'orange']
        counts = [40, 100, 30, 55]
        bar_labels = ['red', 'blue', '_red', 'orange']
        bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

        ax.bar(fruits, counts, label=bar_labels, color=bar_colors)
        ax.set_ylabel('fruit supply')
        ax.set_title('Fruit supply by kind and color')
        ax.legend(title='Fruit color')

        plt.show()

    def stack_area_plot():
        # https://population.un.org/wpp/, license: CC BY 3.0 IGO
        year = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2018]
        population_by_continent = {
            'africa': [228, 284, 365, 477, 631, 814, 1044, 1275],
            'americas': [340, 425, 519, 619, 727, 840, 943, 1006],
            'asia': [1394, 1686, 2120, 2625, 3202, 3714, 4169, 4560],
            'europe': [220, 253, 276, 295, 310, 303, 294, 293],
            'oceania': [12, 15, 19, 22, 26, 31, 36, 39],
        }

        fig, ax = plt.subplots()
        ax.stackplot(year, population_by_continent.values(),
                    labels=population_by_continent.keys(), alpha=0.8)
        ax.legend(loc='upper left', reverse=True)
        ax.set_title('World population')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of people (millions)')

        plt.show()

if __name__ == "__main__":
    # NumpyArray.creating_arrays()
    # NumpyArray.creating_2d_arrays()
    # NumpyArray.array_operations()
    # NumpyArray.indexing_and_slicing()
    # NumpyArray.universal_functions()
    # NumpyArray.masking_and_boolean_indexing()
    # NumpyArray.broadcasting_and_reshaping()

    # Panda.import_export_create_dataframe()
    Panda.data_selection_iloc()
    
    # Matplotlibrary.basic_plots()
    # Matplotlibrary.colored_bar_chart()
    # Matplotlibrary.stack_area_plot()

    # pass
