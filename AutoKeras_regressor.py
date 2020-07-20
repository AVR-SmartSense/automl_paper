# Import libraries
import Utils
from sklearn.model_selection import train_test_split
import autokeras as ak

# Load x, y dataset
x, y = Utils.load_dataset('Data', task='regression')

# Split x, y dataset
# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=123)
# train-val split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  train_size=0.8,
                                                  random_state=456)

# ImageRegressor model
model = ak.ImageRegressor(metrics=['mae', 'mape'],
                          max_trials=100,
                          overwrite=True,
                          seed=45)

# Train model
model.fit(x_train, y_train, epochs=200,
          validation_data=(x_val, y_val))

# Evaluate model
score = model.evaluate(x_test, y_test)

# Inference time - use magic command %timeit in IPython or IDE
# %timeit model.predict_on_batch(x_test)