import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


df = pd.DataFrame(pd.read_csv('diabetes.csv'))

X = df.iloc[:, 0:8]
Y = df.iloc[:, 8:9]

# Normalise X
X_norm = (X-X.std())/(X.mean())

# Get the feature columns into tensorflow
preg = tf.feature_column.numeric_column('Pregnancies')
gluc = tf.feature_column.numeric_column('Glucose')
blpr = tf.feature_column.numeric_column('BloodPressure')
skth = tf.feature_column.numeric_column('SkinThickness')
isln = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
dpdf = tf.feature_column.numeric_column('DiabetesPedigreeFunction')
age = tf.feature_column.numeric_column('Age')

df['Age'].hist(bins=20)

# Make a list of Feature columns
feature_columns = [preg, gluc, blpr, skth, isln, bmi, dpdf, age]
labels = Y


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size=0.3)

# Make the model
input_function = tf.estimator.inputs.pandas_input_fn(x=X_train, y=Y_train, batch_size=10,num_epochs=1000, shuffle=True)

model = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=2)

model.train(input_function,steps=1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=Y_test,batch_size=10, num_epochs=1, shuffle=False)

results = model.evaluate(eval_input_func)

