Human facial recognition is a classical task of computer vision. It has important applications in access control, attendance counting, facial security checks, etc. With the outbreak of the Covid-19, wearing masks has become mandatory for most public areas, which pose challenges for conventional facial recognition solutions. To solve this issue, our team plans to use a CNN-based deep learning model to achieve the masked face identity recognition task.



Our current progress is two-folds:

- We employed [MTCNN](https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb) -- a facial detection framework, to crop the target facial area. Facial detection in video streams are supported, which enables testing in real-time.
- We use Alexnet as a pretrained network followed by a simple CNN to perform the (unmasked) human facial recognition task. The dataset we use for now is [the ORL dataset](https://www.kaggle.com/tavarez/the-orl-database-for-training-and-testing).



Our plan for the next stage:

- In the next stage, we plan to crop the eye and forehead area with MTCNN and use this to train our neural network to do the masked facial recognition task. The most important idea is to exclude mask in the input feature so as not to let the neural network to learn the shape of the mask instead of the user identity.
- We also notice there are some traditional computer vision algorithms that could preprocess (e.g. Autolevel) the image or perform facial recognition tasks (e.g. local binary pattern). Therefore, we are also interested in whether incorporating these methods into our DNN could improve the model performance.