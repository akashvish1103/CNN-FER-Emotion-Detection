# CNN-FER-Emotion-Detection
![Alt Text](https://projectmaker.in/admin/product/emotion-Recognition-system-using-raspberry-pi.webp)

Goal : Emotion Detection ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

Dataset used : FER 2013 dataset from kaggle 
Dataset link : https://www.kaggle.com/datasets/msambare/fer2013

I trained a CNN and Extracted Feature from CNN, and then feed it into a SVM for classification of emotion label.
After that I saved both models, and then used it to predict emotions from the live webcam feed.

Accuracy : 75 %

Trained Model atteched in the repo :<br>
CNN saved model --> cnn_svm_FER_model.h5<br>
SVM saved model --> svm_model.pkl<br>



