import numpy as np
import pandas as pd

hr_data=pd.read_csv(r"D:\Data Science\MACHINE LEARNING\Assignment\Assignment-3 Linear Models\Employee Exit Prediction\Employee_Exit.csv")
hr_data

	satisfaction_level 	last_evaluation 	number_project 	average_montly_hours 	time_spend_company 	Work_accident 	left 	promotion_last_5years 	sales 	salary
0 	0.38 	0.53 	2 	157 	3 	0 	1 	0 	sales 	low
1 	0.80 	0.86 	5 	262 	6 	0 	1 	0 	sales 	medium
2 	0.11 	0.88 	7 	272 	4 	0 	1 	0 	sales 	medium
3 	0.72 	0.87 	5 	223 	5 	0 	1 	0 	sales 	low
4 	0.37 	0.52 	2 	159 	3 	0 	1 	0 	sales 	low
... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	...
14994 	0.40 	0.57 	2 	151 	3 	0 	1 	0 	support 	low
14995 	0.37 	0.48 	2 	160 	3 	0 	1 	0 	support 	low
14996 	0.37 	0.53 	2 	143 	3 	0 	1 	0 	support 	low
14997 	0.11 	0.96 	6 	280 	4 	0 	1 	0 	support 	low
14998 	0.37 	0.52 	2 	158 	3 	0 	1 	0 	support 	low

14999 rows × 10 columns

hr_data.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14999 entries, 0 to 14998
Data columns (total 10 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   satisfaction_level     14999 non-null  float64
 1   last_evaluation        14999 non-null  float64
 2   number_project         14999 non-null  int64  
 3   average_montly_hours   14999 non-null  int64  
 4   time_spend_company     14999 non-null  int64  
 5   Work_accident          14999 non-null  int64  
 6   left                   14999 non-null  int64  
 7   promotion_last_5years  14999 non-null  int64  
 8   sales                  14999 non-null  object 
 9   salary                 14999 non-null  object 
dtypes: float64(2), int64(6), object(2)
memory usage: 1.1+ MB

hr_data.rename(columns={'sales':'department'}, inplace=True)

hr_data.head()

	satisfaction_level 	last_evaluation 	number_project 	average_montly_hours 	time_spend_company 	Work_accident 	left 	promotion_last_5years 	department 	salary
0 	0.38 	0.53 	2 	157 	3 	0 	1 	0 	sales 	low
1 	0.80 	0.86 	5 	262 	6 	0 	1 	0 	sales 	medium
2 	0.11 	0.88 	7 	272 	4 	0 	1 	0 	sales 	medium
3 	0.72 	0.87 	5 	223 	5 	0 	1 	0 	sales 	low
4 	0.37 	0.52 	2 	159 	3 	0 	1 	0 	sales 	low

# convert string to number

hr_data.department.unique()

array(['sales', 'accounting', 'hr', 'technical', 'support', 'management',
       'IT', 'product_mng', 'marketing', 'RandD'], dtype=object)

hr_data.salary.unique()

array(['low', 'medium', 'high'], dtype=object)

from sklearn.preprocessing import LabelEncoder
for col in ['department', 'salary']:
    le =  LabelEncoder()
    hr_data[col] = le.fit_transform(hr_data[col])

hr_data.head()

	satisfaction_level 	last_evaluation 	number_project 	average_montly_hours 	time_spend_company 	Work_accident 	left 	promotion_last_5years 	department 	salary
0 	0.38 	0.53 	2 	157 	3 	0 	1 	0 	7 	1
1 	0.80 	0.86 	5 	262 	6 	0 	1 	0 	7 	2
2 	0.11 	0.88 	7 	272 	4 	0 	1 	0 	7 	2
3 	0.72 	0.87 	5 	223 	5 	0 	1 	0 	7 	1
4 	0.37 	0.52 	2 	159 	3 	0 	1 	0 	7 	1

hr_data.corr()

	satisfaction_level 	last_evaluation 	number_project 	average_montly_hours 	time_spend_company 	Work_accident 	left 	promotion_last_5years 	department 	salary
satisfaction_level 	1.000000 	0.105021 	-0.142970 	-0.020048 	-0.100866 	0.058697 	-0.388375 	0.025605 	0.003153 	0.011754
last_evaluation 	0.105021 	1.000000 	0.349333 	0.339742 	0.131591 	-0.007104 	0.006567 	-0.008684 	0.007772 	0.013965
number_project 	-0.142970 	0.349333 	1.000000 	0.417211 	0.196786 	-0.004741 	0.023787 	-0.006064 	0.009268 	0.009672
average_montly_hours 	-0.020048 	0.339742 	0.417211 	1.000000 	0.127755 	-0.010143 	0.071287 	-0.003544 	0.003913 	0.007082
time_spend_company 	-0.100866 	0.131591 	0.196786 	0.127755 	1.000000 	0.002120 	0.144822 	0.067433 	-0.018010 	-0.003086
Work_accident 	0.058697 	-0.007104 	-0.004741 	-0.010143 	0.002120 	1.000000 	-0.154622 	0.039245 	0.003425 	-0.002506
left 	-0.388375 	0.006567 	0.023787 	0.071287 	0.144822 	-0.154622 	1.000000 	-0.061788 	0.032105 	-0.001294
promotion_last_5years 	0.025605 	-0.008684 	-0.006064 	-0.003544 	0.067433 	0.039245 	-0.061788 	1.000000 	-0.027336 	-0.001318
department 	0.003153 	0.007772 	0.009268 	0.003913 	-0.018010 	0.003425 	0.032105 	-0.027336 	1.000000 	0.000685
salary 	0.011754 	0.013965 	0.009672 	0.007082 	-0.003086 	-0.002506 	-0.001294 	-0.001318 	0.000685 	1.000000

# co relation between 'left' and other features
fea_corr = hr_data.corr().loc['left']
fea_corr 

satisfaction_level      -0.388375
last_evaluation          0.006567
number_project           0.023787
average_montly_hours     0.071287
time_spend_company       0.144822
Work_accident           -0.154622
left                     1.000000
promotion_last_5years   -0.061788
department               0.032105
salary                  -0.001294
Name: left, dtype: float64

type(hr_data.corr().loc['left'])

pandas.core.series.Series

fea_corr.sort_values()

satisfaction_level      -0.388375
Work_accident           -0.154622
promotion_last_5years   -0.061788
salary                  -0.001294
last_evaluation          0.006567
number_project           0.023787
department               0.032105
average_montly_hours     0.071287
time_spend_company       0.144822
left                     1.000000
Name: left, dtype: float64

np.abs(fea_corr).sort_values

<bound method Series.sort_values of satisfaction_level       0.388375
last_evaluation          0.006567
number_project           0.023787
average_montly_hours     0.071287
time_spend_company       0.144822
Work_accident            0.154622
left                     1.000000
promotion_last_5years    0.061788
department               0.032105
salary                   0.001294
Name: left, dtype: float64>

 

Spliting data into feature & target. Also, train & test

hr_data_x = hr_data.drop(['left'], axis=1)

#target
hr_data_y = hr_data['left']

from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(hr_data_x,hr_data_y)

testY.shape

(3750,)

Apply Machine Learning Classification algorithm

from sklearn.linear_model import LogisticRegression 

from sklearn.naive_bayes import GaussianNB,MultinomialNB

from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier

base_models = [ LogisticRegression(), GaussianNB() , MultinomialNB() ]
base_models_name = ['LogisticRegression','GaussianNB','MultinomialNB']

from sklearn.metrics import accuracy_score

for idx,model in enumerate(base_models):
    model.fit(trainX,trainY)
    y_pred = model.predict(testX)
    print (base_models_name[idx],accuracy_score(testY,y_pred))

LogisticRegression 0.7714666666666666
GaussianNB 0.7928
MultinomialNB 0.7741333333333333

D:\Data Science\edyodha packages\conda.2022\lib\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(

### Feature generation function
def feature_gen_func(data,feature_cols,target_col):
    feature_data = data[feature_cols]
    target_data = data[target_col]
    trainX,testX,trainY,testY = train_test_split(feature_data,target_data)
    return trainX,testX,trainY,testY

trainX,testX,trainY,testY = feature_gen_func(hr_data,['satisfaction_level','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years'], 'left')

trainY.shape

(11249,)

def process_models(hr_data, base_models,base_models_name,feature_cols,target):
    trainX,testX,trainY,testY = feature_gen_func(hr_data,feature_cols,target)
    for idx,model in enumerate(base_models):
        model.fit(trainX,trainY)
        y_pred = model.predict(testX)
        print (base_models_name[idx],accuracy_score(testY,y_pred))

process_models(hr_data, base_models, base_models_name, ['satisfaction_level','number_project','average_montly_hours','time_spend_company','promotion_last_5years'], 'left')

LogisticRegression 0.7669333333333334
GaussianNB 0.8346666666666667
MultinomialNB 0.7626666666666667

D:\Data Science\edyodha packages\conda.2022\lib\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(

base_models = [GradientBoostingClassifier(n_estimators=10),RandomForestClassifier(n_estimators=10)]

base_models_name = ['GradientBoostingClassifier','RandomForestClassifier']

process_models(hr_data,base_models,base_models_name,['satisfaction_level','number_project','average_montly_hours','time_spend_company','promotion_last_5years'], 'left')

GradientBoostingClassifier 0.9666666666666667
RandomForestClassifier 0.9850666666666666



