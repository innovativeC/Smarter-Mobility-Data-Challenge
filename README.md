# Smarter-Mobility-Data-Challenge
Approach and code files to solve the usecase as a part of the competition organised by Manifeste IA

### <ins> Implementation Summary <br></ins>
#### Data Summary <br>
- A train and test file was given to solve the usecase. The train set data ranged from July 2020 to Feb 2021 and test set ranged from Feb 2021 to March 2021.<br>
- The train dataset consists of 4 target columns which basically demonstrate the charging status of the charging plugs present in the Paris region. They status are as follows -<br>
-- Available <br>
-- Charging<br>
-- Passive<br>
-- Other <br>
- Other columns that are present in the train dataset are Datetime, Station, area, Postcode, Latitude, Longitude, tod(time of day), dow(day of week), trend.<br>
- There were no null values in the plug based columns which depicts that there is a steady capture of real time plug status .

#### Trend analysis <br>
The trend analysis was done on all the 3 levels of data i.e., the station, area and global level. And there were 2 major hypothesis found - <br>
- The availability of plugs decrease during the holiday period i.e., Dec-Jan and gradually rises to its full capacity post the holiday period <br>
- The number of passive plugs also suddenly increase during the holiday period i.e., Dec-Jan and then go back down after that.

#### Data engineering
The following strategies were implemented in the feature engineering space to create a final training set -<br>
- Created a holiday dataset for the timeframe given and joined it with the train set, to test the hypothesis found in the trend analysis.
- Created time groups for 2 levels of data -<br>
-- Time group for station level - Date & Station<br>
-- Time group for area level - Date & Area<br>
- Created 15 lag features based on the time groups for all the plug statuses<br>
- Created an isweekend column to compliment a hypothesis of less available plugs during the weekend<br>
- The test set was joined with the train set before feature engineering so that there is a homogeneous nature across train and test

#### Modeling Building
The following strategies were implemented in the model building stage to build comprehensive models -<br>
- The missing values in the lagged data was filled with 0, so as to 
- Train and validation set selection -<br>
-- At Station level - 
All the data was used until the last 15 minutes for training. The last 15 minutes i.e, 91 data points were used for the validation set.<br>
-- At Area level -
All the data was used until the last 1 hour for training. The last 1 hour/60 minutes i.e, 16 data points were used for the validation set.<br>
-- At Global level -
All the data was used until the last day for training. The last day/24 hours i.e, 96 data points were used for the validation set.<br>
- LightGBM was used as the algorithm to build models for all the plug statuses across all the levels.
- Categorical features in the dataset were also numericised using Labelencoding
- Categorical features were defined at all levels so that categorical feature handling strategies of lightGBM kick in.
- Various iterations of LightGBM were run with different hyperparameter tuning strategies, also to keep the model from overfitting parameters like earlystopping and max_leaves was used.
- The models were then saved to the disk and used for prediction on the transformed test set.

### <ins> Setup Instructions <br></ins>
- The package dependencies are listed in requirements.txt and can be installed via pip3 -r install requirements.txt
- There are 3 code files that are present in this repo -<br>
-- forecasting_evPlugs.ipynb - Consists of all the code and can be used to view the graphs & output in a jupyter notebook<br>
-- forecasting_evPlugs.py - Consists of all the code and can be used as a script to be run without the view of graphs & output, run it on command line via python3 forecasting_evPlugs.py<br>
-- main.py - Used to run the code present in forecasting_evPlugs.py
