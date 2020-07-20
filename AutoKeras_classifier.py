# Import libraries
import Utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import autokeras as ak

# Load x, y dataset
x, y = Utils.load_dataset('Data', task='classification')

# Encode y to 0,1
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Split datasets
# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=123)
# train-val split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  train_size=0.8,
                                                  random_state=456)

# Build AutoModel classifier
input_node = ak.ImageInput()
output_node = ak.ImageBlock()(input_node)
output_node = ak.ClassificationHead()
model = ak.AutoModel(tuner='bayesian',
                     inputs=input_node,
                     outputs=output_node,
                     max_trials=100,
                     overwrite=True,
                     seed=10)

# Train model
model.fit(x_train, y_train, epochs=200,
          validation_data=(x_val, y_val))    

# Evaluate model
score = model.evaluate(x_test, y_test)
print(score)

# Inference time - use magic command %timeit in IPython or IDE
# %timeit model.predict_on_batch(x_test)