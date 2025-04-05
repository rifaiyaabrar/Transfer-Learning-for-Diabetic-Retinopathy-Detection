# Transfer Learning for Diabetic Retinopathy Detection

This repository contains our implementation of transfer learning and deep neural networks for automated diabetic retinopathy (DR) detection. Diabetic retinopathy is a leading cause of blindness worldwide, and early detection is crucial for effective treatment.

## Overview

Our project focuses on applying transfer learning techniques to improve the performance of convolutional neural networks for classifying diabetic retinopathy severity. We implemented several approaches:

1. **Single Fine-Tuning**: Testing how general-purpose models perform on a specific medical imaging task
2. **Two-Stage Training**: Pre-training on additional DR-specific datasets before fine-tuning on DeepDRiD to improve sensitivity and accuracy

## Model Architecture

We experimented with multiple backbone architectures:
- ResNet18
- ResNet34
- EfficientNet

Our methodology integrated:
- Two-stage transfer learning
- Combined channel and spatial attention mechanisms
- Ensemble learning techniques
- Clinically-inspired preprocessing pipeline

## Results

Our best model achieved a **Cohen Kappa score of 0.8563** with ResNet18 using the two-stage training approach. Here's a summary of our results:

| Model | Cohen-Kappa | Accuracy | Precision | Recall |
|-------|------------|----------|-----------|--------|
| ResNet18 | 0.8022 | 64.25% | 62.09% | 64.25% |
| ResNet18 (Task-b) | 0.8563 | 69.00% | 72.18% | 69.00% |
| ResNet34 | 0.8434 | 67.25% | 64.51% | 67.25% |
| EfficientNet | 0.8453 | 71.75% | 65.16% | 71.75% |

After ensemble techniques (Task-d):
- Weighted average kappa: 0.8638
- Max voting kappa: 0.8467
- Stacking kappa: 0.8638
- Bagging kappa: 0.8638

## Dataset and Preprocessing

We utilized:
- **DeepDRiD Dataset**: Contains 512×512 retinal images with 5 severity levels
- **Supplementary Datasets**: Kaggle DR Resized and APTOS Blindness Detection

### Preprocessing Pipeline:
- Image resizing to 512×512 pixels
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Normalization using ImageNet mean and standard deviation
- Data augmentation techniques:
  - Random cropping
  - Horizontal and vertical flipping
  - Brightness, contrast, saturation, and color jitter adjustments
  - Random rotation (up to 30 degrees)
  - Random padding to maintain spatial integrity

## Training Details

- **Batch size**: 16-24
- **Learning rate**: 0.0001-0.001
- **Epochs**: 20-25
- **Optimizer**: Adam with step-based learning rate scheduler
- **Loss function**: Weighted cross-entropy (to handle class imbalance)

## Challenges

1. Imbalanced data distribution across severity classes
2. High computational costs for training and hyperparameter optimization
3. Difficulty in detection of severe DR cases due to class imbalance

## Future Work

- Explore more advanced architectures (DenseNet, VGG16)
- Combine multiple datasets to enhance model generalization
- Implement advanced techniques for addressing class imbalance (oversampling, synthetic data generation)
- Further improve detection of severe DR cases

## Contributors

- S M Rifaiya Abrar - University of Vaasa
- Minhaz Uddin - University of Vaasa
- Sushil Bhusal - University of Vaasa

## References

1. [DeepDRiD dataset](https://www.kaggle.com/t/41e0944a6839469fadd529fabab45e06)
2. [APTOS 2019 Blindness Detection Dataset](https://www.kaggle.com/datasets/mariaherrerot/aptos2019)
3. [Diabetic Retinopathy Resized Dataset](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized)
4. [Cohen's Kappa Explanation](https://datatab.net/tutorial/cohens-kappa)
5. [What is the Grad-CAM method?](https://datascientest.com/en/what-is-the-grad-cam-method)
6. [What is fine-tuning?](https://www.techtarget.com/searchenterpriseai/definition/fine-tuning)
7. [Image augmentation for creating datasets using PyTorch](https://anushsom.medium.com/image-augmentation-for-creating-datasets-using-pytorch-for-dummies-by-a-dummy-a7c2b08c5bcb)
8. [A comprehensive guide to ensemble learning](https://www.geeksforgeeks.org/a-comprehensive-guide-to-ensemble-learning/)
9. [CLAHE (contrast limited adaptive histogram equalization) in OpenCV](https://www.geeksforgeeks.org/clahe-histogram-eqalization-opencv/)

## Models

- [Models on Google Drive](https://drive.google.com/drive/folders/your_folder_link_here)
- [Test Predictions](https://drive.google.com/drive/folders/your_folder_link_here)

## License

This project is available under the [MIT License](LICENSE).
