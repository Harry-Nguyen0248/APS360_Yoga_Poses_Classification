# Yoga Pose Classification Using ResNet-Based CNN

This project implements deep learning to classify yoga poses from images using a ResNet-inspired Convolutional Neural Network (CNN). The model aims to help yoga enthusiasts improve their form and track their progress by identifying poses accurately. 

# Description
The goal of this project is to use deep learning to classify yoga poses from images,
providing a tool that could help users refine their practice. The model utilizes a CNN architecture inspired by ResNet, which allows for effective feature extraction while avoiding the vashing gradient problem. The project is built on a dataset of 2923 images covering 55 yoga poses, with the data being processed through OpenPose for skeletonization. 

# Installation
1. Clone the Repository
```git clone https://github.com/your-username/yoga-pose-classification.git cd yoga-pose-classification```
3. Install Dependencies
```pip install -r requirements.txt ```
4. Download Dataset
- Download the yoga pose dataset from Kaggle [here](https://www.kaggle.com/datasets/nhgh08/skeletonized-final)

#Model Architecture
- Residual Blocks: To maintain gradient flow and improve training efficiency.
- Two Residual Classes: Increase depth while managing complexity.
- Skeletonization: Images are processed through OpenPose to focus on pose structure rather than appearance.

  #Results
  - Baseline Model: Achieved 41% accuracy using a Random Forest classifier.
  - Primary Model: Achieved 69.25% accuracy using the ResNet-inspired CNN.
  - Challenges: The model experienced some overfitting and difficulty with certain poses  due to their similarities and unclarity.
 

 #License 
 This project is licensed under the MIT License

 #Acknowledgments
 Special thanks to the University of Toronto APS360 Team:
 - Marcus Hong
 - Rosalie Pampolina
 - George Wang
and to the creators of the Kaggle original dataset and OpenPose for their valuable resources.
