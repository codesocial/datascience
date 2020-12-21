#import required python packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_modelselection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

#import dataset
my_df = pd.read_csv("video_game_data.csv")

#split data into input and output objects
X = my_df.drop(["completion_time"], axis = 1)
y = my_df(["completion_time"])

#import required python packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

#import dataset
my_df = pd.read_csv("video_game_data.csv")

#split data into input and output objects
X = my_df.drop(["completion_time"], axis = 1)
y = my_df["completion_time"]

#split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#instantiate the model object
regressor = RandomForestRegressor(random_state = 42)

#train the model
regressor.fit(X_train, y_train)

#asses model accuracy
y_pred = regressor.predict(X_test)

prediction_comparison = pd.DataFrame({"actual" : y_test,"prediction" : y_pred})

r2_score(y_test, y_pred)





