# Import libraries
import Utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Load x, y dataset
x, y = Utils.load_dataset('Data', task='classification')

# Preprocess x input
x = preprocess_input(x)

# Encode y to 0 or 1
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Split x,y dataset
# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=123)
# train-val split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  train_size=0.8,
                                                  random_state=456)

# Build model with ResNet50 
base_model = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(128,128,3))
avg = GlobalAveragePooling2D()(base_model.output)
output = Dense(1, activation='sigmoid')(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)
for layer in base_model.layers:
    layer.trainable = False  # weights frozen for 1st training 
model.summary()

# 1st training
# Compile model
optimizer = tf.keras.optimizers.Adam(lr=0.1, decay=0.01)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_data=(x_val, y_val),
          verbose=2)

# 2nd training
# compile model
for layer in base_model.layers:
    layer.trainable = True  # weights unfrozen for 2nd training 
optimizer = tf.keras.optimizers.Adam(lr=0.01, decay=0.001)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()

# Train model
model.fit(x_train, y_train, epochs=200, batch_size=32,
          validation_data=(x_val, y_val),
          verbose=2)

# Evaluate model
score = model.evaluate(x_test, y_test)
print(score)

# Inference time - use magic command %timeit in IPython or IDE
# %timeit model.predict_on_batch(x_test)