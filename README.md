# Source Codes for AutoML Manuscript
This repository hosts source codes and best models reported in the manuscript 'Automated Machine Learning for High-Throughput 
Image-Based Plant Phenotyping'.

## Getting Started
1. Clone or download the repository. Dataset for images and CSV file available on [Zenodo]().
Please place them into folder named 'Data'.
2. You may execute the python scripts in a terminal or interactively in IPython/IDE. 
3. Best AutoKeras models in the paper are located in the 'Models' folder. Each model is saved in a single folder, 
as per tensorflow SavedModel format. To load models:

```
# Load model
model = tf.keras.models.load_model('Models/AutoKeras_classifier')

# Evaluate model on test data
score = model.evaluate(x_test, y_test)
print(score)
```

## Prerequisites
Source codes tested on:
* Python 3.7
* AutoKeras 1.0.1
* Tensorflow-GPU 2.1.0
* Pandas 1.0.3
* Numpy 1.16
* Scikit-learn 0.22.2
* Pillow 7.1.1

## Reference
Joshua C.O. Koh, German Spangenberg, Surya Kant. 'Automated Machine Learning for High-Throughput 
Image-Based Plant Phenotyping'.  
