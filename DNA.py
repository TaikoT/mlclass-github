# Run this cell to import libraries and check that tensorflow is properly installed
import pandas as pd
import numpy as np
import math 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import tensorflow as tf

# TODO Load penguins.csv
data = pd.read_csv("all_classifcation_and_seqs_aln.csv")

#TODO Handle NA Values
data = data.dropna()

# TODO encode string data using LabelEncoder
#make a list of the sequences
#loop through each sequence, encoding each character on its own, and pout into a temporary list
#add that temporary list into another list (list of competely encoded stuff)
le = LabelEncoder()
RawSequence = data["sequence"].tolist()
UpdatedSequence =[]

for rows in RawSequence:
    EncodedSequence =[]     
    for i in range(len(rows)):
        if (rows[i] == "-"):
            EncodedSequence.append(0)

        if (rows[i] == "A"):
            EncodedSequence.append(1)

        if (rows[i] == "T"):
            EncodedSequence.append(2)

        if (rows[i] == "G"):
            EncodedSequence.append(3)

        if (rows[i] == "C"):
            EncodedSequence.append(4)
  
        #86% val acc
    #Random state = 41    
#2143 = 79%
#3412 = 85%
#4321 = 78%
#4213 = 86%
#4123 = 85%

    UpdatedSequence.append(EncodedSequence)
  

     
data["species"] = le.fit_transform(data["species"])   


#TODO Select your features. Select body_mass_g as your "target" (y) and everything else as X
X = np.array(UpdatedSequence)
y=data["species"]


# TODO : Split the data into testing and training data. Use a 20% split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# run this to see if you implemented the above block correctly
assert math.isclose(len(X_train), .8*len(data), rel_tol=1), f"\033[91mExpected {.8*len(data)} but got {len(X_train)}\033[0m"
assert math.isclose(len(X_test), .2*len(data), rel_tol=1), f"\033[91mExpected {.2*len(data)} but got {len(X_test)}\033[0m"

# TODO create a neural network with tensorflow
model = tf.keras.models.Sequential([
   
   
    tf.keras.layers.Dense(1024, input_shape=[27040]),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(200, activation='relu'),

    tf.keras.layers.Softmax()


])


# TODO set your learning rate
lr =0.0001

#  tf.keras.layers.Dense(510, input_shape=[4795]),
#     tf.keras.layers.Dense(260, activation='relu'),
#     tf.keras.layers.Dense(130, activation='relu'),
#     tf.keras.layers.Dense(100, activation='relu'),
#0.000005 -- val acc = .8438
#0.00001 -- val acc = .8333

#TODO Compile your model with a selected optimizer and loss function
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
    metrics =['accuracy']
)

# TODO: fit your model with X_train and Y_train
history = model.fit(X_train, y_train, validation_data =(X_test, y_test), epochs=100)

