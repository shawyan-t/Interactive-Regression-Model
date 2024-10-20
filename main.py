# shbang
# Judah Tanninen
# Shawyan Tabari
# Elesey Razumovskiy

# Term 483 Project
# Kaggle: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

# Imports
import sys
import time
import threading
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import plotly.express as py
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Note: Threading implementation inspired by https://www.geeksforgeeks.org/multithreading-python-set-1/#
# Threading and Sleep help for output inspired by https://realpython.com/python-sleep/

class ProgressIndicator(threading.Thread): # Create a thread just for the class that displays progress
    def __init__(self, interval=1):
        super().__init__()
        self.interval = interval
        self.running = True

    def run(self):
        while self.running:
            for i in range(4):            # Loop to create 3 dots for a loading animation
                if not self.running:
                    break
                print('.' * i, end='\r')  # Print Ellipses and move to beginning of line
                time.sleep(self.interval)
            if self.running:
                print('   ', end='\r')    # Clear the line

    def stop(self):
        self.running = False

# STEP 1 (SECURE THE KEYS!)
# Read all the relevant csvs into the dataframes.
print('READING CSVS')
holidayDf = pd.read_csv('holidays_events.csv')
oilDf = pd.read_csv('oil.csv')
trainDf = pd.read_csv('train.csv')
testDf = pd.read_csv('test.csv')
transactionsDf = pd.read_csv('transactions.csv')

# Before merging, limit the training set to 1 year, set below.
while True:
    validYears = [2013, 2014, 2015, 2016, 2017]
    print("\n" *100)
    inputIndex = input("Enter the year you'd like to perform the fit on\n1. 2013\n2. 2014\n3. 2015\n4. 2016\n5. 2017\n\nYear: ")
    inputIndex = int(inputIndex) - 1
    if (0 <= inputIndex < len(validYears)):
        userin = validYears[inputIndex]
        break

print(f"Year Selected: {userin}")
trainDf['datetime'] = pd.to_datetime(trainDf['date'])
trainDf = trainDf[trainDf['datetime'].dt.year == int(userin)]

# STEP 2 (ASCEND FROM DARKNESS!)
# Merge the holiday and oil data onto the training and test frames
print('Merging additional data frames into train/test frames')
trainDf = pd.merge(trainDf, oilDf, on="date", how="left") # Cute little left join
trainDf = pd.merge(trainDf, holidayDf, on="date", how="left")
testDf = pd.merge(testDf, oilDf, on="date", how="left") # Cute little left join
testDf = pd.merge(testDf, holidayDf, on="date", how="left")


# STEP 3 (RAIN FIRE)
# Prepare the dataset, convert values
print('Preparing data, converting values etc.')
# Convert onpromotion to a boolean value
trainDf['onpromotion'] = (trainDf['onpromotion'] > 0).astype(int)
testDf['onpromotion'] = (testDf['onpromotion'] > 0).astype(int)

# Change all the holiday junk to a simple is_holiday numerical column.
trainDf['is_holiday'] = (~trainDf['type'].isna()).astype(int)
testDf['is_holiday'] = (~testDf['type'].isna()).astype(int)

# Convert the dates to a month-date combo that we can then encode to correlate with date (the year doesn't matter that much)
trainDf['monthday'] = trainDf['date'].str[5:] # Get just month and day
testDf['monthday'] = testDf['date'].str[5:] # Get just month and day

# Create a list of the numerical and categorical columns that we want
desiredNumerical = ['store_nbr', 'onpromotion', 'dcoilwtico', 'is_holiday']
desiredCategorical = ['family']

# STEP 4 (Unleash the horde)
# Graphing of the transactions data. 
print("Generating transactions graph")

# Convert 'date' to datetime and sort
transactionsDf['date'] = pd.to_datetime(transactionsDf['date'])
transactionsDf.sort_values('date', inplace=True)

"""
fig = py.scatter_3d(
transactionsDf,
'store_nbr',
'date',
'transactions',
color = 'transactions',
size_max = 15,
size = 'transactions'
)

fig.show()
"""

# Filter the DataFrame for the specified date range
transactionsDf['datetime'] = pd.to_datetime(transactionsDf['date'])
filtered_transactions = transactionsDf[transactionsDf['datetime'].dt.year == int(userin)]

# Aggregate transactions by date
daily_transactions = filtered_transactions.groupby('date')['transactions'].sum().reset_index()

daily_transactions['7_day_MA'] = daily_transactions['transactions'].rolling(window=7).mean()
daily_transactions['14_day_MA'] = daily_transactions['transactions'].rolling(window=14).mean()
daily_transactions['30_day_MA'] = daily_transactions['transactions'].rolling(window=30).mean()
daily_transactions['60_day_MA'] = daily_transactions['transactions'].rolling(window=60).mean()

plt.figure(figsize=(12, 6))
plt.plot(daily_transactions['date'], daily_transactions['transactions'], label='Total Daily Transactions')
plt.plot(daily_transactions['date'], daily_transactions['7_day_MA'], label='7-Day MA')
plt.plot(daily_transactions['date'], daily_transactions['14_day_MA'], label='14-Day MA')
plt.plot(daily_transactions['date'], daily_transactions['30_day_MA'], label='30-Day MA')
plt.plot(daily_transactions['date'], daily_transactions['60_day_MA'], label='60-Day MA')
plt.title('Total Transactions Over Time with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Transactions')
plt.legend()

while True:
    graphNow = input("Graph Generated. Press 1 to show or 2 to continue: ")
    if int(graphNow) == 1:
        plt.show()
        break
    elif int(graphNow) == 2:
        break

# Transactions DF 3D plot
while True:
    wantsTransact = input("3D Plot available for transactions dataframe. Generate Plot?\n1. plot\n2. continue\n\nInput: ")
    if wantsTransact == "1":
        validList = ['date','store_nbr','transactions','datetime']
        while True:
            x = input("Choose the X axis from: 'date','store_nbr','transactions','datetime' ")
            if x in validList:
                transactionsDf[x].fillna(0,inplace=True)
                break
        while True:
            y = input("Choose the Y axis from: 'date','store_nbr','transactions','datetime' ")
            if y in validList:
                transactionsDf[y].fillna(0,inplace=True)
                break
        while True:
            z = input("Choose the Z axis from: 'date','store_nbr','transactions','datetime' ")
            if z in validList:
                transactionsDf[z].fillna(0,inplace=True)
                break
        fig = py.scatter_3d(
            transactionsDf,
            x,
            y,
            z,
            color = z,
            size_max = 15,
            size = z
            )
        fig.show()
    elif wantsTransact == "2":
        break


# Daily Transactions 3D plot
while True:
    wantsTransact = input("3D Plot available for daily transactions dataframe. Generate Plot?\n1. plot\n2. continue\n\nInput: ")
    if wantsTransact == "1":
        validList = ['date','transactions','7_day_MA','14_day_MA','30_day_MA','60_day_MA']
        while True:
            x = input("Choose the X axis from: 'date','transactions','7_day_MA','14_day_MA','30_day_MA','60_day_MA' ")
            if x in validList:
                daily_transactions[x].fillna(0,inplace=True)
                break
        while True:
            y = input("Choose the Y axis from:'date','transactions','7_day_MA','14_day_MA','30_day_MA','60_day_MA' ")
            if y in validList:
                daily_transactions[y].fillna(0,inplace=True)
                break
        while True:
            z = input("Choose the Z axis from: 'date','transactions','7_day_MA','14_day_MA','30_day_MA','60_day_MA' ")
            if z in validList:
                daily_transactions[z].fillna(0,inplace=True)
                break
        fig = py.scatter_3d(
            daily_transactions,
            x,
            y,
            z,
            color = z,
            size_max = 15,
            size = z
            )
        fig.show()
    elif wantsTransact == "2":
        break


# STEP 5 (Skewer the winged beast)
# PREPROCESSING
# Converting categorical columns into numerical ones, replace empty values, scali
print("Preprocessing...")

#preprocessing pipelines
while True:
    validStrats = ['mean', 'most_frequent', 'constant']
    print("\n" *100)
    inputIndex = input("Enter the Strategy you'd like to use for the Numpipe Imputer\n1. Mean\n2. Most Frequent\n3. Constant\n\nStrategy: ")
    inputIndex = int(inputIndex) - 1
    if (0 <= inputIndex < len(validStrats)):
        userinImp = validStrats[inputIndex]
        break

print(f"Strategy Chosen: {userinImp}")
numPipe = Pipeline([
    ('imputer', SimpleImputer(strategy=userinImp)),
    ('scaler', MinMaxScaler())
])


while True:
    validStrats = ['most_frequent', 'constant']
    print("\n" *100)
    inputIndex = input("Enter the Strategy you'd like to use for the Catpipe Imputer\n1. Most Frequent\n2. Constant\n\nStrategy: ")
    inputIndex = int(inputIndex) - 1
    if (0 <= inputIndex < len(validStrats)):
        userinImp2 = validStrats[inputIndex]
        break

print(f"Strategy Chosen: {userinImp2}")
catPipe = Pipeline([
    ('imputer', SimpleImputer(strategy=userinImp2)),
    ('labele', OneHotEncoder())
])

preprocessor = ColumnTransformer([
        ('num', numPipe, desiredNumerical),
        ('cat', catPipe, desiredCategorical)
    ])

# STEP 6 (Wield a fist of iron)
# Here, we create a pipeline that takes in hyper params and creates and finds the best model for us to use.

while True:
    opts = ['gradient','randomforest','decision','linear']
    print("===== Options =====\n\n'gradient' for Gradient Boosting Regressor\n'randomforest' for the Random Forest Tree Regression Model\n'decision' for the Decision Tree Regression Model\n'linear' for the Linear Regression Model\n")
    getModel = input("Enter Chosen Regression Model: ")
    
    if getModel == 'decision':
        print('Beginning model creating...')
        # Hyper params

        # Initial Parameter Lists
        depthParam = [None]
        featParam = ['sqrt']
        critParam = ['friedman_mse']
        purityParam = [0.0]
        splitParam = [2]
        leafParam = [1]
        weightParam = [0.0]
        splitterParam = ["best"]

        # Function to add or remove items from a list
        def update_list(parameter_list, param_name):
            while True:
                # Display current list
                print(f"\nCurrent {param_name}: {parameter_list}")
                action = input(f"Enter 'add' or 'remove' to modify {param_name} (or 'exit' to exit): ").lower()

                # Exit condition
                if action == 'exit':
                    break

                # Adding a new value
                elif action == 'add':
                    new_value = input(f"Enter a new value to add to {param_name}: ")

                    if new_value == 'None':
                        new_value = None

                    # Check constraints
                    if param_name == 'featParam' and new_value not in ['sqrt', 'log2', 'int', 'float', None]:
                        print("Invalid input. Valid params are 'sqrt', 'log2', 'int', 'float', or 'None'.")
                    elif param_name == 'depthParam' and int(new_value) < 0:
                        print("Invalid input. Value must be greater than or equal to 0.")
                    elif param_name == 'critParam' and new_value not in ['friedman_mse', 'squared_error','absolute_error','poisson']:
                        print("Invalid input. Valid params are 'friedman_mse', 'squared_error'.")
                    elif param_name == 'purityParam' and float(new_value) < 0:
                        print("Invalid input. Value must be greater than or equal to 0.")
                    elif param_name == 'splitParam' and int(new_value) < 2:
                        print("Invalid input. Value must be greater than or equal to 2.")
                    elif param_name == 'leafParam' and int(new_value) < 1:
                        print("Invalid input. Value must be greater than or equal to 1.")
                    elif param_name == 'weightParam' and float(new_value) < 0:
                        print("Invalid input. Value must be greater than or equal to 0.")
                    elif param_name == 'splitterParam' and new_value not in ['random','best']:
                        print("Invalid input. Value must be 'random' or 'best")
                    else:
                        intLists = ['leafParam','depthParam','splitParam']
                        floatLists = ['purityParam','weightParam']
                        
                        if param_name in intLists:
                            new_value = int(new_value)
                        elif param_name in floatLists:
                            new_value = float(new_value)

                        if new_value in parameter_list:
                            print("Value already in list")
                        else:
                            parameter_list.append(new_value)

                # Removing a value
                elif action == 'remove':
                    if len(parameter_list) > 1:
                        value_to_remove = input(f"Enter the value you want to remove from {param_name}: ")
                        if value_to_remove in parameter_list:
                            parameter_list.remove(value_to_remove)
                        else:
                            print("Value not in the list.")
                    else:
                        print("Can't remove value. There must be at least one value in the list.")

                else:
                    print("Invalid input. Please enter 'add', 'remove', or 'exit'.")

        # Main loop for updating parameters
        while True:
            print("\nDecision Tree Parameters:")
            print(f"1. Depth Parameters: {depthParam}")
            print(f"2. Feature Parameters: {featParam}")
            print(f"3. Criterion Parameters: {critParam}")
            print(f"4. Purity Parameters: {purityParam}")
            print(f"5. Split Parameters: {splitParam}")
            print(f"6. Leaf Parameters: {leafParam}")
            print(f"7. Weight Parameters: {weightParam}")
            print(f"8. Splitter Parameters: {splitterParam}")

            param_choice = input("\nChoose a parameter to update (1-8) or 'run' to run the fit: ")

            if param_choice == 'run':
                break
            elif param_choice == '1':
                update_list(depthParam, 'depthParam')
            elif param_choice == '2':
                update_list(featParam, 'featParam')
            elif param_choice == '3':
                update_list(critParam, 'critParam')
            elif param_choice == '4':
                update_list(purityParam, 'purityParam')
            elif param_choice == '5':
                update_list(splitParam, 'splitParam')
            elif param_choice == '6':
                update_list(leafParam, 'leafParam')
            elif param_choice == '7':
                update_list(weightParam, 'weightParam')
            elif param_choice == '8':
                update_list(splitterParam, 'splitterParam')
            else:
                print("Invalid choice. Please enter a number between 1-8 or 'run'.")


        params = {
            'model__max_depth': depthParam,  
            'model__max_features': featParam,  
            'model__criterion': critParam,
            'model__min_impurity_decrease': purityParam,
            'model__min_samples_split': splitParam,
            'model__min_samples_leaf': leafParam,
            'model__min_weight_fraction_leaf': weightParam,
            'model__splitter': splitterParam
        }

        # Creates the xs and ys

        xs = trainDf[desiredCategorical + desiredNumerical]
        ys = np.maximum(trainDf['sales'], 0)
        print(ys)


        # Create the pipeline
        model = DecisionTreeRegressor()
        pipeline = Pipeline([ # Technically only two steps, but the preprocessor contains multiple steps
            ('preprocessor', preprocessor), 
            ('model', model)
        ])

        # Create the grid search
        # Note: Could add user input to choose scoring here, for options that dont crash later
        search = GridSearchCV(pipeline, params, scoring="r2", n_jobs=-1, error_score='raise')

        #fitting
        progress_indicator = ProgressIndicator()
        progress_indicator.start()
        search.fit(xs, ys)
        progress_indicator.stop()
        progress_indicator.join()
        bestModel = search.best_estimator_

        print("Fit Complete")

        while True:
            final = input("Choose Data to Display\n\n1. Scores\n2. 3D Visualization\n3. Transaction Graph\n4. CSV\n5. Bar Chart\n6. Line Chart\n7. Pie Chart\n8. Histogram\n9. Box Plot\n10. Exit Program\n\nInput: ")
            if final == "1":
                print("Best hyperparameters:")
                print(search.best_params_)
                print("Best score")
                print(search.best_score_)
            elif final == "2":

                while True:
                    validList = ['family', 'store_nbr', 'onpromotion', 'is_holiday', 'dcoilwtico']
                    while True:
                        x = input("Choose the X axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                        if x in validList:
                            xs[x].fillna(0,inplace=True)
                            break
                    while True:
                        y = input("Choose the Y axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                        if y in validList:
                            xs[y].fillna(0,inplace=True)
                            break
                    while True:
                        z = input("Choose the Z axis from: 'store_nbr', 'onpromotion', 'is_holiday' ")
                        if z in validList:
                            if z == 'family':
                                pass
                            else:
                                xs[z].fillna(0,inplace=True)
                                break
                    fig = py.scatter_3d(
                    xs,
                    x,
                    y,
                    z,
                    color = z,
                    size_max = 15,
                    size = z
                    )
                    fig.show()
                    
                    new = input("Create new plot or exit?\n\n1.new plot\n2.exit\n\nInput: ")
                    if new == "1":
                        pass
                    elif new == "2":
                        break
                    else:
                        print("Invalid Input. Exiting...")
                        break
                
            elif final == "3":
                opt = input(f"Show graph for {userin} or for new year?\n\n1. {userin}\n2. new year\n3. all years\n4. exit\n\nInput: ")
                if opt == "1":
                    start_date2 = f'{userin}-01-01'
                    temp = int(userin)
                    temp = temp+1
                    temp = str(temp)
                    end_date2 = f'{temp}-12-31'
                    newFiltered = transactionsDf[(transactionsDf['date'] >= start_date2) & (transactionsDf['date'] <= end_date2)]

                    # Aggregate transactions by date
                    newDaily = newFiltered.groupby('date')['transactions'].sum().reset_index()

                    newDaily['7_day_MA'] = newDaily['transactions'].rolling(window=7).mean()
                    newDaily['14_day_MA'] = newDaily['transactions'].rolling(window=14).mean()
                    newDaily['30_day_MA'] = newDaily['transactions'].rolling(window=30).mean()
                    newDaily['60_day_MA'] = newDaily['transactions'].rolling(window=60).mean()

                    plt.figure(figsize=(12, 6))
                    plt.plot(newDaily['date'], newDaily['transactions'], label='Total Daily Transactions')
                    plt.plot(newDaily['date'], newDaily['7_day_MA'], label='7-Day MA')
                    plt.plot(newDaily['date'], newDaily['14_day_MA'], label='14-Day MA')
                    plt.plot(newDaily['date'], newDaily['30_day_MA'], label='30-Day MA')
                    plt.plot(newDaily['date'], newDaily['60_day_MA'], label='60-Day MA')
                    plt.title('Total Transactions Over Time with Moving Averages')
                    plt.xlabel('Date')
                    plt.ylabel('Transactions')
                    plt.legend()
                    plt.show()
                elif opt == "2":
                    while True:
                        validYears = ["1","2","3","4","5"]
                        print("\n" *100)
                        newuserin = input("Enter the year you'd like to see the graph on\n1. 2013\n2. 2014\n3. 2015\n4. 2016\n5. 2017\n\nYear: ")
                        if newuserin in validYears:
                            if newuserin == "1":
                                newuserin = 2013
                            elif newuserin == "2":
                                newuserin = 2014
                            elif newuserin == "3":
                                newuserin = 2015
                            elif newuserin == "4":
                                newuserin = 2016
                            elif newuserin == "5":
                                newuserin = 2017
                            break

                    if newuserin == 2013:
                        start_date2 = '2013-01-01'
                        end_date2 = '2014-12-31'
                    elif newuserin == 2014:
                        start_date2 = '2014-01-01'
                        end_date2 = '2015-12-31'
                    elif newuserin == 2015:
                        start_date2 = '2015-01-01'
                        end_date2 = '2016-12-31'
                    elif newuserin == 2016:
                        start_date2 = '2016-01-01'
                        end_date2 = '2017-12-31'
                    elif newuserin == 2017:
                        start_date2 = '2017-01-01'
                        end_date2 = '2018-12-31'
                    
                    newFiltered = transactionsDf[(transactionsDf['date'] >= start_date2) & (transactionsDf['date'] <= end_date2)]

                    # Aggregate transactions by date
                    newDaily = newFiltered.groupby('date')['transactions'].sum().reset_index()

                    newDaily['7_day_MA'] = newDaily['transactions'].rolling(window=7).mean()
                    newDaily['14_day_MA'] = newDaily['transactions'].rolling(window=14).mean()
                    newDaily['30_day_MA'] = newDaily['transactions'].rolling(window=30).mean()
                    newDaily['60_day_MA'] = newDaily['transactions'].rolling(window=60).mean()

                    plt.figure(figsize=(12, 6))
                    plt.plot(newDaily['date'], newDaily['transactions'], label='Total Daily Transactions')
                    plt.plot(newDaily['date'], newDaily['7_day_MA'], label='7-Day MA')
                    plt.plot(newDaily['date'], newDaily['14_day_MA'], label='14-Day MA')
                    plt.plot(newDaily['date'], newDaily['30_day_MA'], label='30-Day MA')
                    plt.plot(newDaily['date'], newDaily['60_day_MA'], label='60-Day MA')
                    plt.title('Total Transactions Over Time with Moving Averages')
                    plt.xlabel('Date')
                    plt.ylabel('Transactions')
                    plt.legend()
                    plt.show()
                
                elif opt == "3":
                    start_date2 = '2013-01-01'
                    end_date2 = '2018-12-31'
                    newFiltered = transactionsDf[(transactionsDf['date'] >= start_date2) & (transactionsDf['date'] <= end_date2)]

                    # Aggregate transactions by date
                    newDaily = newFiltered.groupby('date')['transactions'].sum().reset_index()

                    newDaily['7_day_MA'] = newDaily['transactions'].rolling(window=7).mean()
                    newDaily['14_day_MA'] = newDaily['transactions'].rolling(window=14).mean()
                    newDaily['30_day_MA'] = newDaily['transactions'].rolling(window=30).mean()
                    newDaily['60_day_MA'] = newDaily['transactions'].rolling(window=60).mean()

                    plt.figure(figsize=(12, 6))
                    plt.plot(newDaily['date'], newDaily['transactions'], label='Total Daily Transactions')
                    plt.plot(newDaily['date'], newDaily['7_day_MA'], label='7-Day MA')
                    plt.plot(newDaily['date'], newDaily['14_day_MA'], label='14-Day MA')
                    plt.plot(newDaily['date'], newDaily['30_day_MA'], label='30-Day MA')
                    plt.plot(newDaily['date'], newDaily['60_day_MA'], label='60-Day MA')
                    plt.title('Total Transactions Over Time with Moving Averages')
                    plt.xlabel('Date')
                    plt.ylabel('Transactions')
                    plt.legend()
                    plt.show()

            elif final == "4":
                # Predict the data for the testing data.
                x_test = testDf[desiredCategorical + desiredNumerical]
                y_pred = bestModel.predict(x_test)
                testDf['sales'] = y_pred
                output = testDf[['id', 'sales']]
                output.to_csv('predictions.csv', index=False)
            
            elif final == "5":  # Bar Chart
                x = input("Choose the X axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                y = input("Choose the Y axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                xs[x].fillna(0, inplace=True)
                xs[y].fillna(0, inplace=True)
                fig = py.bar(xs, x=x, y=y)
                fig.show()

            elif final == "6":  # Line Chart
                x = input("Choose the X axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                y = input("Choose the Y axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                xs[x].fillna(0, inplace=True)
                xs[y].fillna(0, inplace=True)
                fig = py.line(xs, x=x, y=y)
                fig.show()

            elif final == "7":  # Pie Chart
                column = input("Choose the column for pie chart from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                values = xs[column].value_counts().reset_index()
                values.columns = ['category', 'count']
                fig = py.pie(values, names='category', values='count')
                fig.show()

            elif final == "8":  # Histogram
                column = input("Choose the column for histogram from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                xs[column].fillna(0, inplace=True)
                fig = py.histogram(xs, x=column)
                fig.show()

            elif final == "9":  # Box Plot
                y = input("Choose the Y axis for box plot from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                xs[y].fillna(0, inplace=True)
                fig = py.box(xs, y=y)
                fig.show()
            elif final == "10":
                break


        py.scatter_3d
        exit(0)

    elif getModel == 'linear':

        print('Beginning model creating...')
        # Hyper params

        # Initial Parameter Lists
        fitParam = [True]
        copyParam = [True]
        jobsParam = [5]


        # Function to add or remove items from a list
        def update_list(parameter_list, param_name):
            while True:
                # Display current list
                print(f"\nCurrent {param_name}: {parameter_list}")
                action = input(f"Enter 'add' or 'remove' to modify {param_name} (or 'exit' to exit): ").lower()

                # Exit condition
                if action == 'exit':
                    break

                # Adding a new value
                elif action == 'add':
                    new_value = input(f"Enter a new value to add to {param_name}: ")

                    if new_value == 'None':
                        new_value = None
                    
                    if new_value == 'True':
                        new_value = True
                    
                    if new_value == 'False':
                        new_value = False

                    # Check constraints
                    if param_name in ['fitParam', 'copyParam'] and new_value not in [True,False]:
                        print("Invalid input. Valid params are 'True' or 'False'")
                    elif param_name == 'jobsParam' and int(new_value) < 0:
                        print("Invalid input. Value must be greater than or equal to 1")
                    else:
                        intLists = ['jobsParam']
                        authLists = ['fitParam','copyParam']
                        
                        if param_name in intLists:
                            new_value = int(new_value)
                        elif param_name in authLists:
                            pass

                        if new_value in parameter_list:
                            print("Value already in list")
                        else:
                            parameter_list.append(new_value)

                # Removing a value
                elif action == 'remove':
                    if len(parameter_list) > 1:
                        value_to_remove = input(f"Enter the value you want to remove from {param_name}: ")
                        if value_to_remove in parameter_list:
                            parameter_list.remove(value_to_remove)
                        else:
                            print("Value not in the list.")
                    else:
                        print("Can't remove value. There must be at least one value in the list.")

                else:
                    print("Invalid input. Please enter 'add', 'remove', or 'exit'.")

        # Main loop for updating parameters
        while True:
            print("\nLinear Regression Parameters: ")
            print(f"1. Intercept Parameters: {fitParam}")
            print(f"2. Copy Parameters: {copyParam}")
            print(f"3. Number of Jobs Parameters: {jobsParam}")

            param_choice = input("\nChoose a parameter to update (1-3) or 'run' to run the fit: ")

            if param_choice == 'run':
                break
            elif param_choice == '1':
                update_list(fitParam, 'fitParam')
            elif param_choice == '2':
                update_list(copyParam, 'copyParam')
            elif param_choice == '3':
                update_list(jobsParam, 'jobsParam')
            else:
                print("Invalid choice. Please enter a number between 1-3 or 'run'.")


        params = {
            'model__fit_intercept': fitParam,
            'model__copy_X' : copyParam,  
            'model__n_jobs': jobsParam,  
        }

        # Creates the xs and ys

        xs = trainDf[desiredCategorical + desiredNumerical]
        ys = np.maximum(trainDf['sales'], 0)
        print(ys)


        # Create the pipeline
        model = LinearRegression()
        pipeline = Pipeline([ # Technically only two steps, but the preprocessor contains multiple steps
            ('preprocessor', preprocessor), 
            ('model', model)
        ])

        # Create the grid search
        # Note: Could add user input to choose scoring here, for options that dont crash later
        search = GridSearchCV(pipeline, params, scoring="r2", n_jobs=-1, error_score='raise')

        #fitting
        progress_indicator = ProgressIndicator()
        progress_indicator.start()
        search.fit(xs, ys)
        progress_indicator.stop()
        progress_indicator.join()
        bestModel = search.best_estimator_

        print("Fit Complete")

        while True:
            final = input("Choose Data to Display\n\n1. Scores\n2. 3D Visualization\n3. Transaction Graph\n4. CSV\n5. Bar Chart\n6. Line Chart\n7. Pie Chart\n8. Histogram\n9. Box Plot\n10. Exit Program\n\nInput: ")
            if final == "1":
                print("Best hyperparameters:")
                print(search.best_params_)
                print("Best score")
                print(search.best_score_)
            elif final == "2":

                while True:
                    validList = ['family', 'store_nbr', 'onpromotion', 'is_holiday', 'dcoilwtico']
                    while True:
                        x = input("Choose the X axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                        if x in validList:
                            xs[x].fillna(0,inplace=True)
                            break
                    while True:
                        y = input("Choose the Y axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                        if y in validList:
                            xs[y].fillna(0,inplace=True)
                            break
                    while True:
                        z = input("Choose the Z axis from: 'store_nbr', 'onpromotion', 'is_holiday' ")
                        if z in validList:
                            if z == 'family':
                                pass
                            else:
                                xs[z].fillna(0,inplace=True)
                                break
                    fig = py.scatter_3d(
                    xs,
                    x,
                    y,
                    z,
                    color = z,
                    size_max = 15,
                    size = z
                    )
                    fig.show()
                    
                    new = input("Create new plot or exit?\n\n1.new plot\n2.exit\n\nInput: ")
                    if new == "1":
                        pass
                    elif new == "2":
                        break
                    else:
                        print("Invalid Input. Exiting...")
                        break
                
            elif final == "3":
                opt = input(f"Show graph for {userin} or for new year?\n\n1. {userin}\n2. new year\n3. all years\n4. exit\n\nInput: ")
                if opt == "1":
                    start_date2 = f'{userin}-01-01'
                    temp = int(userin)
                    temp = temp+1
                    temp = str(temp)
                    end_date2 = f'{temp}-12-31'
                    newFiltered = transactionsDf[(transactionsDf['date'] >= start_date2) & (transactionsDf['date'] <= end_date2)]

                    # Aggregate transactions by date
                    newDaily = newFiltered.groupby('date')['transactions'].sum().reset_index()

                    newDaily['7_day_MA'] = newDaily['transactions'].rolling(window=7).mean()
                    newDaily['14_day_MA'] = newDaily['transactions'].rolling(window=14).mean()
                    newDaily['30_day_MA'] = newDaily['transactions'].rolling(window=30).mean()
                    newDaily['60_day_MA'] = newDaily['transactions'].rolling(window=60).mean()

                    plt.figure(figsize=(12, 6))
                    plt.plot(newDaily['date'], newDaily['transactions'], label='Total Daily Transactions')
                    plt.plot(newDaily['date'], newDaily['7_day_MA'], label='7-Day MA')
                    plt.plot(newDaily['date'], newDaily['14_day_MA'], label='14-Day MA')
                    plt.plot(newDaily['date'], newDaily['30_day_MA'], label='30-Day MA')
                    plt.plot(newDaily['date'], newDaily['60_day_MA'], label='60-Day MA')
                    plt.title('Total Transactions Over Time with Moving Averages')
                    plt.xlabel('Date')
                    plt.ylabel('Transactions')
                    plt.legend()
                    plt.show()
                elif opt == "2":
                    while True:
                        validYears = ["1","2","3","4","5"]
                        print("\n" *100)
                        newuserin = input("Enter the year you'd like to see the graph on\n1. 2013\n2. 2014\n3. 2015\n4. 2016\n5. 2017\n\nYear: ")
                        if newuserin in validYears:
                            if newuserin == "1":
                                newuserin = 2013
                            elif newuserin == "2":
                                newuserin = 2014
                            elif newuserin == "3":
                                newuserin = 2015
                            elif newuserin == "4":
                                newuserin = 2016
                            elif newuserin == "5":
                                newuserin = 2017
                            break

                    if newuserin == 2013:
                        start_date2 = '2013-01-01'
                        end_date2 = '2014-12-31'
                    elif newuserin == 2014:
                        start_date2 = '2014-01-01'
                        end_date2 = '2015-12-31'
                    elif newuserin == 2015:
                        start_date2 = '2015-01-01'
                        end_date2 = '2016-12-31'
                    elif newuserin == 2016:
                        start_date2 = '2016-01-01'
                        end_date2 = '2017-12-31'
                    elif newuserin == 2017:
                        start_date2 = '2017-01-01'
                        end_date2 = '2018-12-31'
                    
                    newFiltered = transactionsDf[(transactionsDf['date'] >= start_date2) & (transactionsDf['date'] <= end_date2)]

                    # Aggregate transactions by date
                    newDaily = newFiltered.groupby('date')['transactions'].sum().reset_index()

                    newDaily['7_day_MA'] = newDaily['transactions'].rolling(window=7).mean()
                    newDaily['14_day_MA'] = newDaily['transactions'].rolling(window=14).mean()
                    newDaily['30_day_MA'] = newDaily['transactions'].rolling(window=30).mean()
                    newDaily['60_day_MA'] = newDaily['transactions'].rolling(window=60).mean()

                    plt.figure(figsize=(12, 6))
                    plt.plot(newDaily['date'], newDaily['transactions'], label='Total Daily Transactions')
                    plt.plot(newDaily['date'], newDaily['7_day_MA'], label='7-Day MA')
                    plt.plot(newDaily['date'], newDaily['14_day_MA'], label='14-Day MA')
                    plt.plot(newDaily['date'], newDaily['30_day_MA'], label='30-Day MA')
                    plt.plot(newDaily['date'], newDaily['60_day_MA'], label='60-Day MA')
                    plt.title('Total Transactions Over Time with Moving Averages')
                    plt.xlabel('Date')
                    plt.ylabel('Transactions')
                    plt.legend()
                    plt.show()
                
                elif opt == "3":
                    start_date2 = '2013-01-01'
                    end_date2 = '2018-12-31'
                    newFiltered = transactionsDf[(transactionsDf['date'] >= start_date2) & (transactionsDf['date'] <= end_date2)]

                    # Aggregate transactions by date
                    newDaily = newFiltered.groupby('date')['transactions'].sum().reset_index()

                    newDaily['7_day_MA'] = newDaily['transactions'].rolling(window=7).mean()
                    newDaily['14_day_MA'] = newDaily['transactions'].rolling(window=14).mean()
                    newDaily['30_day_MA'] = newDaily['transactions'].rolling(window=30).mean()
                    newDaily['60_day_MA'] = newDaily['transactions'].rolling(window=60).mean()

                    plt.figure(figsize=(12, 6))
                    plt.plot(newDaily['date'], newDaily['transactions'], label='Total Daily Transactions')
                    plt.plot(newDaily['date'], newDaily['7_day_MA'], label='7-Day MA')
                    plt.plot(newDaily['date'], newDaily['14_day_MA'], label='14-Day MA')
                    plt.plot(newDaily['date'], newDaily['30_day_MA'], label='30-Day MA')
                    plt.plot(newDaily['date'], newDaily['60_day_MA'], label='60-Day MA')
                    plt.title('Total Transactions Over Time with Moving Averages')
                    plt.xlabel('Date')
                    plt.ylabel('Transactions')
                    plt.legend()
                    plt.show()


            elif final == "4":
                # Predict the data for the testing data.
                x_test = testDf[desiredCategorical + desiredNumerical]
                y_pred = bestModel.predict(x_test)
                testDf['sales'] = y_pred
                output = testDf[['id', 'sales']]
                output.to_csv('predictions.csv', index=False)

            elif final == "5":  # Bar Chart
                x = input("Choose the X axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                y = input("Choose the Y axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                xs[x].fillna(0, inplace=True)
                xs[y].fillna(0, inplace=True)
                fig = py.bar(xs, x=x, y=y)
                fig.show()

            elif final == "6":  # Line Chart
                x = input("Choose the X axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                y = input("Choose the Y axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                xs[x].fillna(0, inplace=True)
                xs[y].fillna(0, inplace=True)
                fig = py.line(xs, x=x, y=y)
                fig.show()

            elif final == "7":  # Pie Chart
                column = input("Choose the column for pie chart from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                values = xs[column].value_counts().reset_index()
                values.columns = ['category', 'count']
                fig = py.pie(values, names='category', values='count')
                fig.show()

            elif final == "8":  # Histogram
                column = input("Choose the column for histogram from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                xs[column].fillna(0, inplace=True)
                fig = py.histogram(xs, x=column)
                fig.show()

            elif final == "9":  # Box Plot
                y = input("Choose the Y axis for box plot from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                xs[y].fillna(0, inplace=True)
                fig = py.box(xs, y=y)
                fig.show()
            elif final == "10":
                break


        py.scatter_3d
        exit(0)

    elif getModel == 'randomforest':

        print('Beginning model creating...')
        # Hyper params

        # Initial Parameter Lists
        estParam = [100]
        depthParam = [2]
        featParam = ['sqrt']
        critParam = ['friedman_mse']
        purityParam = [0.0]
        splitParam = [2]
        leafParam = [1]

        # Function to add or remove items from a list
        def update_list(parameter_list, param_name):
            while True:
                # Display current list
                print(f"\nCurrent {param_name}: {parameter_list}")
                action = input(f"Enter 'add' or 'remove' to modify {param_name} (or 'exit' to exit): ").lower()

                # Exit condition
                if action == 'exit':
                    break

                # Adding a new value
                elif action == 'add':
                    new_value = input(f"Enter a new value to add to {param_name}: ")

                    if new_value == 'None':
                        new_value = None

                    # Check constraints
                    if param_name == 'featParam' and new_value not in ['sqrt', 'log2', 'int', 'float', None]:
                        print("Invalid input. Valid params are 'sqrt', 'log2', 'int', 'float', or 'None'.")
                    elif param_name in ['estParam', 'depthParam'] and int(new_value) < 1:
                        print("Invalid input. Value must be greater than or equal to 1.")
                    elif param_name == 'critParam' and new_value not in ['friedman_mse', 'squared_error', 'absolute_error', 'poisson']:
                        print("Invalid input. Valid params are 'friedman_mse', 'squared_error'.")
                    elif param_name == 'purityParam' and float(new_value) < 0:
                        print("Invalid input. Value must be greater than or equal to 0.")
                    elif param_name == 'splitParam' and int(new_value) < 2:
                        print("Invalid input. Value must be greater than or equal to 2.")
                    elif param_name == 'leafParam' and int(new_value) < 0:
                        print("Invalid input. Value must be greater than or equal to 0.")
                    else:
                        intLists = ['estParam','depthParam','splitParam','leafParam']
                        floatLists = ['learnParam','purityParam']
                        
                        if param_name in intLists:
                            new_value = int(new_value)
                        elif param_name in floatLists:
                            new_value = float(new_value)

                        if new_value in parameter_list:
                            print("Value already in list")
                        else:
                            parameter_list.append(new_value)

                # Removing a value
                elif action == 'remove':
                    if len(parameter_list) > 1:
                        value_to_remove = input(f"Enter the value you want to remove from {param_name}: ")
                        if value_to_remove in parameter_list:
                            parameter_list.remove(value_to_remove)
                        else:
                            print("Value not in the list.")
                    else:
                        print("Can't remove value. There must be at least one value in the list.")

                else:
                    print("Invalid input. Please enter 'add', 'remove', or 'exit'.")

        # Main loop for updating parameters
        while True:
            print("\nRandom Forest Boosting Parameters:")
            print(f"1. Estimator Parameters: {estParam}")
            print(f"2. Depth Parameters: {depthParam}")
            print(f"3. Feature Parameters: {featParam}")
            print(f"4. Criterion Parameters: {critParam}")
            print(f"5. Purity Parameters: {purityParam}")
            print(f"6. Split Parameters: {splitParam}")
            print(f"7. Leaf Parameters: {leafParam}")

            param_choice = input("\nChoose a parameter to update (1-7) or 'run' to run the fit: ")

            if param_choice == 'run':
                break
            elif param_choice == '1':
                update_list(estParam, 'estParam')
            elif param_choice == '2':
                update_list(depthParam, 'depthParam')
            elif param_choice == '3':
                update_list(featParam, 'featParam')
            elif param_choice == '4':
                update_list(critParam, 'critParam')
            elif param_choice == '5':
                update_list(purityParam, 'purityParam')
            elif param_choice == '6':
                update_list(splitParam, 'splitParam')
            elif param_choice == '7':
                update_list(leafParam, 'leafParam')
            else:
                print("Invalid choice. Please enter a number between 1-7 or 'run'.")


        params = {
            'model__n_estimators': estParam, #made this large since data set is in the millions of rows
            'model__max_depth': depthParam,  
            'model__max_features': featParam,  
            'model__criterion': critParam,
            'model__min_impurity_decrease': purityParam,
            'model__min_samples_split': splitParam,
            'model__min_samples_leaf': leafParam,
        }

        # Creates the xs and ys

        xs = trainDf[desiredCategorical + desiredNumerical]
        ys = np.maximum(trainDf['sales'], 0)
        print(ys)


        # Create the pipeline
        model = RandomForestRegressor()
        pipeline = Pipeline([ # Technically only two steps, but the preprocessor contains multiple steps
            ('preprocessor', preprocessor), 
            ('model', model)
        ])

        # Create the grid search
        # Note: Could add user input to choose scoring here, for options that dont crash later
        search = GridSearchCV(pipeline, params, scoring="r2", n_jobs=-1, error_score='raise')

        #fitting
        progress_indicator = ProgressIndicator()
        progress_indicator.start()
        search.fit(xs, ys)
        progress_indicator.stop()
        progress_indicator.join()
        bestModel = search.best_estimator_

        print("Fit Complete")

        while True:
            final = input("Choose Data to Display\n\n1. Scores\n2. 3D Visualization\n3. Transaction Graph\n4. CSV\n5. Bar Chart\n6. Line Chart\n7. Pie Chart\n8. Histogram\n9. Box Plot\n10. Exit Program\n\nInput: ")
            if final == "1":
                print("Best hyperparameters:")
                print(search.best_params_)
                print("Best score")
                print(search.best_score_)
            elif final == "2":

                while True:
                    validList = ['family', 'store_nbr', 'onpromotion', 'is_holiday', 'dcoilwtico']
                    while True:
                        x = input("Choose the X axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                        if x in validList:
                            xs[x].fillna(0,inplace=True)
                            break
                    while True:
                        y = input("Choose the Y axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                        if y in validList:
                            xs[y].fillna(0,inplace=True)
                            break
                    while True:
                        z = input("Choose the Z axis from: 'store_nbr', 'onpromotion', 'is_holiday' ")
                        if z in validList:
                            if z == 'family':
                                pass
                            else:
                                xs[z].fillna(0,inplace=True)
                                break
                    fig = py.scatter_3d(
                    xs,
                    x,
                    y,
                    z,
                    color = z,
                    size_max = 15,
                    size = z
                    )
                    fig.show()
                    
                    new = input("Create new plot or exit?\n\n1.new plot\n2.exit\n\nInput: ")
                    if new == "1":
                        pass
                    elif new == "2":
                        break
                    else:
                        print("Invalid Input. Exiting...")
                        break
                
            elif final == "3":
                opt = input(f"Show graph for {userin} or for new year?\n\n1. {userin}\n2. new year\n3. all years\n4. exit\n\nInput: ")
                if opt == "1":
                    start_date2 = f'{userin}-01-01'
                    temp = int(userin)
                    temp = temp+1
                    temp = str(temp)
                    end_date2 = f'{temp}-12-31'
                    newFiltered = transactionsDf[(transactionsDf['date'] >= start_date2) & (transactionsDf['date'] <= end_date2)]

                    # Aggregate transactions by date
                    newDaily = newFiltered.groupby('date')['transactions'].sum().reset_index()

                    newDaily['7_day_MA'] = newDaily['transactions'].rolling(window=7).mean()
                    newDaily['14_day_MA'] = newDaily['transactions'].rolling(window=14).mean()
                    newDaily['30_day_MA'] = newDaily['transactions'].rolling(window=30).mean()
                    newDaily['60_day_MA'] = newDaily['transactions'].rolling(window=60).mean()

                    plt.figure(figsize=(12, 6))
                    plt.plot(newDaily['date'], newDaily['transactions'], label='Total Daily Transactions')
                    plt.plot(newDaily['date'], newDaily['7_day_MA'], label='7-Day MA')
                    plt.plot(newDaily['date'], newDaily['14_day_MA'], label='14-Day MA')
                    plt.plot(newDaily['date'], newDaily['30_day_MA'], label='30-Day MA')
                    plt.plot(newDaily['date'], newDaily['60_day_MA'], label='60-Day MA')
                    plt.title('Total Transactions Over Time with Moving Averages')
                    plt.xlabel('Date')
                    plt.ylabel('Transactions')
                    plt.legend()
                    plt.show()
                elif opt == "2":
                    while True:
                        validYears = ["1","2","3","4","5"]
                        print("\n" *100)
                        newuserin = input("Enter the year you'd like to see the graph on\n1. 2013\n2. 2014\n3. 2015\n4. 2016\n5. 2017\n\nYear: ")
                        if newuserin in validYears:
                            if newuserin == "1":
                                newuserin = 2013
                            elif newuserin == "2":
                                newuserin = 2014
                            elif newuserin == "3":
                                newuserin = 2015
                            elif newuserin == "4":
                                newuserin = 2016
                            elif newuserin == "5":
                                newuserin = 2017
                            break

                    if newuserin == 2013:
                        start_date2 = '2013-01-01'
                        end_date2 = '2014-12-31'
                    elif newuserin == 2014:
                        start_date2 = '2014-01-01'
                        end_date2 = '2015-12-31'
                    elif newuserin == 2015:
                        start_date2 = '2015-01-01'
                        end_date2 = '2016-12-31'
                    elif newuserin == 2016:
                        start_date2 = '2016-01-01'
                        end_date2 = '2017-12-31'
                    elif newuserin == 2017:
                        start_date2 = '2017-01-01'
                        end_date2 = '2018-12-31'
                    
                    newFiltered = transactionsDf[(transactionsDf['date'] >= start_date2) & (transactionsDf['date'] <= end_date2)]

                    # Aggregate transactions by date
                    newDaily = newFiltered.groupby('date')['transactions'].sum().reset_index()

                    newDaily['7_day_MA'] = newDaily['transactions'].rolling(window=7).mean()
                    newDaily['14_day_MA'] = newDaily['transactions'].rolling(window=14).mean()
                    newDaily['30_day_MA'] = newDaily['transactions'].rolling(window=30).mean()
                    newDaily['60_day_MA'] = newDaily['transactions'].rolling(window=60).mean()

                    plt.figure(figsize=(12, 6))
                    plt.plot(newDaily['date'], newDaily['transactions'], label='Total Daily Transactions')
                    plt.plot(newDaily['date'], newDaily['7_day_MA'], label='7-Day MA')
                    plt.plot(newDaily['date'], newDaily['14_day_MA'], label='14-Day MA')
                    plt.plot(newDaily['date'], newDaily['30_day_MA'], label='30-Day MA')
                    plt.plot(newDaily['date'], newDaily['60_day_MA'], label='60-Day MA')
                    plt.title('Total Transactions Over Time with Moving Averages')
                    plt.xlabel('Date')
                    plt.ylabel('Transactions')
                    plt.legend()
                    plt.show()
                
                elif opt == "3":
                    start_date2 = '2013-01-01'
                    end_date2 = '2018-12-31'
                    newFiltered = transactionsDf[(transactionsDf['date'] >= start_date2) & (transactionsDf['date'] <= end_date2)]

                    # Aggregate transactions by date
                    newDaily = newFiltered.groupby('date')['transactions'].sum().reset_index()

                    newDaily['7_day_MA'] = newDaily['transactions'].rolling(window=7).mean()
                    newDaily['14_day_MA'] = newDaily['transactions'].rolling(window=14).mean()
                    newDaily['30_day_MA'] = newDaily['transactions'].rolling(window=30).mean()
                    newDaily['60_day_MA'] = newDaily['transactions'].rolling(window=60).mean()

                    plt.figure(figsize=(12, 6))
                    plt.plot(newDaily['date'], newDaily['transactions'], label='Total Daily Transactions')
                    plt.plot(newDaily['date'], newDaily['7_day_MA'], label='7-Day MA')
                    plt.plot(newDaily['date'], newDaily['14_day_MA'], label='14-Day MA')
                    plt.plot(newDaily['date'], newDaily['30_day_MA'], label='30-Day MA')
                    plt.plot(newDaily['date'], newDaily['60_day_MA'], label='60-Day MA')
                    plt.title('Total Transactions Over Time with Moving Averages')
                    plt.xlabel('Date')
                    plt.ylabel('Transactions')
                    plt.legend()
                    plt.show()


            elif final == "4":
                # Predict the data for the testing data.
                x_test = testDf[desiredCategorical + desiredNumerical]
                y_pred = bestModel.predict(x_test)
                testDf['sales'] = y_pred
                output = testDf[['id', 'sales']]
                output.to_csv('predictions.csv', index=False)
            
            elif final == "5":  # Bar Chart
                x = input("Choose the X axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                y = input("Choose the Y axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                xs[x].fillna(0, inplace=True)
                xs[y].fillna(0, inplace=True)
                fig = py.bar(xs, x=x, y=y)
                fig.show()

            elif final == "6":  # Line Chart
                x = input("Choose the X axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                y = input("Choose the Y axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                xs[x].fillna(0, inplace=True)
                xs[y].fillna(0, inplace=True)
                fig = py.line(xs, x=x, y=y)
                fig.show()

            elif final == "7":  # Pie Chart
                column = input("Choose the column for pie chart from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                values = xs[column].value_counts().reset_index()
                values.columns = ['category', 'count']
                fig = py.pie(values, names='category', values='count')
                fig.show()

            elif final == "8":  # Histogram
                column = input("Choose the column for histogram from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                xs[column].fillna(0, inplace=True)
                fig = py.histogram(xs, x=column)
                fig.show()

            elif final == "9":  # Box Plot
                y = input("Choose the Y axis for box plot from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                xs[y].fillna(0, inplace=True)
                fig = py.box(xs, y=y)
                fig.show()
            elif final == "10":
                break


        py.scatter_3d
        exit(0)
    
    elif getModel == 'gradient':
    

        print('Beginning model creating...')
        # Hyper params

        # Initial Parameter Lists
        estParam = [100]
        depthParam = [2]
        featParam = ['sqrt']
        learnParam = [0.1]
        lossParam = ['squared_error']
        critParam = ['friedman_mse']
        purityParam = [0.0]
        splitParam = [2]

        # Function to add or remove items from a list
        def update_list(parameter_list, param_name):
            while True:
                # Display current list
                print(f"\nCurrent {param_name}: {parameter_list}")
                action = input(f"Enter 'add' or 'remove' to modify {param_name} (or 'exit' to exit): ").lower()

                # Exit condition
                if action == 'exit':
                    break

                # Adding a new value
                elif action == 'add':
                    new_value = input(f"Enter a new value to add to {param_name}: ")

                    if new_value == 'None':
                        new_value = None

                    # Check constraints
                    if param_name == 'featParam' and new_value not in ['sqrt', 'log2', 'int', 'float', None]:
                        print("Invalid input. Valid params are 'sqrt', 'log2', 'int', 'float', or 'None'.")
                    elif param_name in ['estParam', 'depthParam'] and int(new_value) < 1:
                        print("Invalid input. Value must be greater than or equal to 1.")
                    elif param_name == 'learnParam' and float(new_value) < 0:
                        print("Invalid input. Value must be greater than or equal to 0.")
                    elif param_name == 'lossParam' and new_value not in ['squared_error', 'absolute_error', 'huber', 'quantile']:
                        print("Invalid input. Valid params are 'squared_error', 'absolute_error', 'huber', 'quantile'.")
                    elif param_name == 'critParam' and new_value not in ['friedman_mse', 'squared_error']:
                        print("Invalid input. Valid params are 'friedman_mse', 'squared_error'.")
                    elif param_name == 'purityParam' and float(new_value) < 0:
                        print("Invalid input. Value must be greater than or equal to 0.")
                    elif param_name == 'splitParam' and int(new_value) < 2:
                        print("Invalid input. Value must be greater than or equal to 2.")
                    else:
                        intLists = ['estParam','depthParam','splitParam']
                        floatLists = ['learnParam','purityParam']
                        
                        if param_name in intLists:
                            new_value = int(new_value)
                        elif param_name in floatLists:
                            new_value = float(new_value)

                        if new_value in parameter_list:
                            print("Value already in list")
                        else:
                            parameter_list.append(new_value)

                # Removing a value
                elif action == 'remove':
                    if len(parameter_list) > 1:
                        value_to_remove = input(f"Enter the value you want to remove from {param_name}: ")
                        if value_to_remove in parameter_list:
                            parameter_list.remove(value_to_remove)
                        else:
                            print("Value not in the list.")
                    else:
                        print("Can't remove value. There must be at least one value in the list.")

                else:
                    print("Invalid input. Please enter 'add', 'remove', or 'exit'.")

        # Main loop for updating parameters
        while True:
            print("\nGradient Boosting Parameters:")
            print(f"1. Estimator Parameters: {estParam}")
            print(f"2. Depth Parameters: {depthParam}")
            print(f"3. Feature Parameters: {featParam}")
            print(f"4. Learning Rate Parameters: {learnParam}")
            print(f"5. Loss Parameters: {lossParam}")
            print(f"6. Criterion Parameters: {critParam}")
            print(f"7. Purity Parameters: {purityParam}")
            print(f"8. Split Parameters: {splitParam}")

            param_choice = input("\nChoose a parameter to update (1-8) or 'run' to run the fit: ")

            if param_choice == 'run':
                break
            elif param_choice == '1':
                update_list(estParam, 'estParam')
            elif param_choice == '2':
                update_list(depthParam, 'depthParam')
            elif param_choice == '3':
                update_list(featParam, 'featParam')
            elif param_choice == '4':
                update_list(learnParam, 'learnParam')
            elif param_choice == '5':
                update_list(lossParam, 'lossParam')
            elif param_choice == '6':
                update_list(critParam, 'critParam')
            elif param_choice == '7':
                update_list(purityParam, 'purityParam')
            elif param_choice == '8':
                update_list(splitParam, 'splitParam')
            else:
                print("Invalid choice. Please enter a number between 1-8 or 'run'.")


        params = {
            'model__n_estimators': estParam, #made this large since data set is in the millions of rows
            'model__max_depth': depthParam,  
            'model__max_features': featParam,  
            'model__learning_rate': learnParam,
            'model__loss': lossParam,
            'model__criterion': critParam,
            'model__min_impurity_decrease': purityParam,
            'model__min_samples_split': splitParam,
        }

        # Creates the xs and ys

        xs = trainDf[desiredCategorical + desiredNumerical]
        ys = np.maximum(trainDf['sales'], 0)
        print(ys)


        # Create the pipeline
        model = GradientBoostingRegressor()
        pipeline = Pipeline([ # Technically only two steps, but the preprocessor contains multiple steps
            ('preprocessor', preprocessor), 
            ('model', model)
        ])

        # Create the grid search
        # Note: Could add user input to choose scoring here, for options that dont crash later
        search = GridSearchCV(pipeline, params, scoring="r2", n_jobs=-1, error_score='raise')

        #fitting
        progress_indicator = ProgressIndicator()
        progress_indicator.start()
        search.fit(xs, ys)
        progress_indicator.stop()
        progress_indicator.join()
        bestModel = search.best_estimator_

        print("Fit Complete")

        while True:
            final = input("Choose Data to Display\n\n1. Scores\n2. 3D Visualization\n3. Transaction Graph\n4. CSV\n5. Bar Chart\n6. Line Chart\n7. Pie Chart\n8. Histogram\n9. Box Plot\n10. Exit Program\n\nInput: ")
            if final == "1":
                print("Best hyperparameters:")
                print(search.best_params_)
                print("Best score")
                print(search.best_score_)
            elif final == "2":

                while True:
                    validList = ['family', 'store_nbr', 'onpromotion', 'is_holiday', 'dcoilwtico']
                    while True:
                        x = input("Choose the X axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                        if x in validList:
                            xs[x].fillna(0,inplace=True)
                            break
                    while True:
                        y = input("Choose the Y axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                        if y in validList:
                            xs[y].fillna(0,inplace=True)
                            break
                    while True:
                        z = input("Choose the Z axis from: 'store_nbr', 'onpromotion', 'is_holiday' ")
                        if z in validList:
                            if z == 'family':
                                pass
                            else:
                                xs[z].fillna(0,inplace=True)
                                break
                    fig = py.scatter_3d(
                    xs,
                    x,
                    y,
                    z,
                    color = z,
                    size_max = 15,
                    size = z
                    )
                    fig.show()
                    
                    new = input("Create new plot or exit?\n\n1.new plot\n2.exit\n\nInput: ")
                    if new == "1":
                        pass
                    elif new == "2":
                        break
                    else:
                        print("Invalid Input. Exiting...")
                        break
                
            elif final == "3":
                opt = input(f"Show graph for {userin} or for new year?\n\n1. {userin}\n2. new year\n3. all years\n4. exit\n\nInput: ")
                if opt == "1":
                    start_date2 = f'{userin}-01-01'
                    temp = int(userin)
                    temp = temp+1
                    temp = str(temp)
                    end_date2 = f'{temp}-12-31'
                    newFiltered = transactionsDf[(transactionsDf['date'] >= start_date2) & (transactionsDf['date'] <= end_date2)]

                    # Aggregate transactions by date
                    newDaily = newFiltered.groupby('date')['transactions'].sum().reset_index()

                    newDaily['7_day_MA'] = newDaily['transactions'].rolling(window=7).mean()
                    newDaily['14_day_MA'] = newDaily['transactions'].rolling(window=14).mean()
                    newDaily['30_day_MA'] = newDaily['transactions'].rolling(window=30).mean()
                    newDaily['60_day_MA'] = newDaily['transactions'].rolling(window=60).mean()

                    plt.figure(figsize=(12, 6))
                    plt.plot(newDaily['date'], newDaily['transactions'], label='Total Daily Transactions')
                    plt.plot(newDaily['date'], newDaily['7_day_MA'], label='7-Day MA')
                    plt.plot(newDaily['date'], newDaily['14_day_MA'], label='14-Day MA')
                    plt.plot(newDaily['date'], newDaily['30_day_MA'], label='30-Day MA')
                    plt.plot(newDaily['date'], newDaily['60_day_MA'], label='60-Day MA')
                    plt.title('Total Transactions Over Time with Moving Averages')
                    plt.xlabel('Date')
                    plt.ylabel('Transactions')
                    plt.legend()
                    plt.show()
                elif opt == "2":
                    while True:
                        validYears = ["1","2","3","4","5"]
                        print("\n" *100)
                        newuserin = input("Enter the year you'd like to see the graph on\n1. 2013\n2. 2014\n3. 2015\n4. 2016\n5. 2017\n\nYear: ")
                        if newuserin in validYears:
                            if newuserin == "1":
                                newuserin = 2013
                            elif newuserin == "2":
                                newuserin = 2014
                            elif newuserin == "3":
                                newuserin = 2015
                            elif newuserin == "4":
                                newuserin = 2016
                            elif newuserin == "5":
                                newuserin = 2017
                            break

                    if newuserin == 2013:
                        start_date2 = '2013-01-01'
                        end_date2 = '2014-12-31'
                    elif newuserin == 2014:
                        start_date2 = '2014-01-01'
                        end_date2 = '2015-12-31'
                    elif newuserin == 2015:
                        start_date2 = '2015-01-01'
                        end_date2 = '2016-12-31'
                    elif newuserin == 2016:
                        start_date2 = '2016-01-01'
                        end_date2 = '2017-12-31'
                    elif newuserin == 2017:
                        start_date2 = '2017-01-01'
                        end_date2 = '2018-12-31'
                    
                    newFiltered = transactionsDf[(transactionsDf['date'] >= start_date2) & (transactionsDf['date'] <= end_date2)]

                    # Aggregate transactions by date
                    newDaily = newFiltered.groupby('date')['transactions'].sum().reset_index()

                    newDaily['7_day_MA'] = newDaily['transactions'].rolling(window=7).mean()
                    newDaily['14_day_MA'] = newDaily['transactions'].rolling(window=14).mean()
                    newDaily['30_day_MA'] = newDaily['transactions'].rolling(window=30).mean()
                    newDaily['60_day_MA'] = newDaily['transactions'].rolling(window=60).mean()

                    plt.figure(figsize=(12, 6))
                    plt.plot(newDaily['date'], newDaily['transactions'], label='Total Daily Transactions')
                    plt.plot(newDaily['date'], newDaily['7_day_MA'], label='7-Day MA')
                    plt.plot(newDaily['date'], newDaily['14_day_MA'], label='14-Day MA')
                    plt.plot(newDaily['date'], newDaily['30_day_MA'], label='30-Day MA')
                    plt.plot(newDaily['date'], newDaily['60_day_MA'], label='60-Day MA')
                    plt.title('Total Transactions Over Time with Moving Averages')
                    plt.xlabel('Date')
                    plt.ylabel('Transactions')
                    plt.legend()
                    plt.show()
                
                elif opt == "3":
                    start_date2 = '2013-01-01'
                    end_date2 = '2018-12-31'
                    newFiltered = transactionsDf[(transactionsDf['date'] >= start_date2) & (transactionsDf['date'] <= end_date2)]

                    # Aggregate transactions by date
                    newDaily = newFiltered.groupby('date')['transactions'].sum().reset_index()

                    newDaily['7_day_MA'] = newDaily['transactions'].rolling(window=7).mean()
                    newDaily['14_day_MA'] = newDaily['transactions'].rolling(window=14).mean()
                    newDaily['30_day_MA'] = newDaily['transactions'].rolling(window=30).mean()
                    newDaily['60_day_MA'] = newDaily['transactions'].rolling(window=60).mean()

                    plt.figure(figsize=(12, 6))
                    plt.plot(newDaily['date'], newDaily['transactions'], label='Total Daily Transactions')
                    plt.plot(newDaily['date'], newDaily['7_day_MA'], label='7-Day MA')
                    plt.plot(newDaily['date'], newDaily['14_day_MA'], label='14-Day MA')
                    plt.plot(newDaily['date'], newDaily['30_day_MA'], label='30-Day MA')
                    plt.plot(newDaily['date'], newDaily['60_day_MA'], label='60-Day MA')
                    plt.title('Total Transactions Over Time with Moving Averages')
                    plt.xlabel('Date')
                    plt.ylabel('Transactions')
                    plt.legend()
                    plt.show()


            elif final == "4":
                # Predict the data for the testing data.
                x_test = testDf[desiredCategorical + desiredNumerical]
                y_pred = bestModel.predict(x_test)
                testDf['sales'] = y_pred
                output = testDf[['id', 'sales']]
                output.to_csv('predictions.csv', index=False)
            
            elif final == "5":  # Bar Chart
                x = input("Choose the X axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                y = input("Choose the Y axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                xs[x].fillna(0, inplace=True)
                xs[y].fillna(0, inplace=True)
                fig = py.bar(xs, x=x, y=y)
                fig.show()

            elif final == "6":  # Line Chart
                x = input("Choose the X axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                y = input("Choose the Y axis from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                xs[x].fillna(0, inplace=True)
                xs[y].fillna(0, inplace=True)
                fig = py.line(xs, x=x, y=y)
                fig.show()

            elif final == "7":  # Pie Chart
                column = input("Choose the column for pie chart from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                values = xs[column].value_counts().reset_index()
                values.columns = ['category', 'count']
                fig = py.pie(values, names='category', values='count')
                fig.show()

            elif final == "8":  # Histogram
                column = input("Choose the column for histogram from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                xs[column].fillna(0, inplace=True)
                fig = py.histogram(xs, x=column)
                fig.show()

            elif final == "9":  # Box Plot
                y = input("Choose the Y axis for box plot from: 'family', 'store_nbr', 'onpromotion', 'is_holiday' ")
                xs[y].fillna(0, inplace=True)
                fig = py.box(xs, y=y)
                fig.show()
            elif final == "10":
                break


        py.scatter_3d
        exit(0)
