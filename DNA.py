# TODO Load DNA.csv
data = pd.read_csv("DNA.csv")

#TODO Handle NA Values
data = data.dropna()

# TODO encode string data using LabelEncoder
#string -> list
data['sequence'] = data['sequence'].apply(list)
#AT, CG... sum=5
base_to_int = {'A':1, 'C':2, 'G':3, 'T':4, '-':0}
data['encoded'] = data['sequence'].apply(lambda s: [base_to_int[b] for b in s])


le = LabelEncoder()
for i in data.select_dtypes(include=['object']).columns:
    data["species"] = le.fit_transform(data["species"])


#TODO Select your features. Select body_mass_g as your "target" (y) and everything else as X
features = ["sequence"]
X = data[features]
y=data["species"]
# TODO : Split the data into testing and training data. Use a 20% split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=50)

# run this to see if you implemented the above block correctly
assert math.isclose(len(X_train), .8*len(data), rel_tol=1), f"\033[91mExpected {.8*len(data)} but got {len(X_train)}\033[0m"
assert math.isclose(len(X_test), .2*len(data), rel_tol=1), f"\033[91mExpected {.2*len(data)} but got {len(X_test)}\033[0m"


# TODO create a neural network with tensorflow
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(6, input_shape=[6]),
    tf.keras.layers.Dense(10, activation='sigmoid'),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(1)
])



# TODO set your learning rate
lr =0.01

#TODO Compile your model with a selected optimizer and loss function
model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
)



# TODO: fit your model with X_train and Y_train
history = model.fit(X_train, y_train, epochs=100)




#Run this cell to graph your loss
df = pd.DataFrame(history.history)['loss']
px.scatter(df).show()



# TODO generate some predictions using Y_test
predictions =predictions = model.predict(X_test)




# Run this cell to calcuate your mean average error based on Y_test
error = Y_test.squeeze() - predictions.ravel()
print("Your average error is: ", error.mean())



if abs(error.mean()) > 100:
    print("\033[91mYour model should be a bit more accurate\033[0m")
else:
    print("\033[92mYour model is accurate enough!\033[0m")