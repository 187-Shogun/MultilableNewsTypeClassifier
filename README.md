### News Categories Classification Model

This is a quick demo to showcase how to build a multi-label
classification system that can predict the type of news based on the headline and a short description text. 

The data comes from Kaggle. 
You can get it from this [link](https://storage.googleapis.com/open-ml-datasets/news-categories-dataset/News_Category_Dataset_v2.zip).

### Instructions:

Simply run the main.py file and wait till it returns a zero code. The script downloads a dataset, preprocess it and then use it to train a neural network model. Once that is ready, it saves the model in H5 format and the training results on a logs folder. The logs folder can be accessed using the tensorboard UI.

### Env Requirements:

Python 3.9\
Use pip install -r ./requirements.txt