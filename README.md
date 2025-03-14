This study utilizes a dataset from the Kaggle platform ("rsna-pneumonia-detection-challenge"), comprising 26,684 radiological lung images. 
The objective is to classify each image as either showing pneumonia or being normal, based on the presence of the condition.

Before training the model, the image resolutions were adjusted using various interpolation methodes to assess the performance of the Convolutional Neural Network (CNN) across different resolution-method combinations.

The resolutions tested included 32x32, 64x64, 128x128, and 256x256, while the following four interpolation methods were applied:
Nearest Neighbor Interpolation
Bilinear Interpolation
Bicubic Interpolation
Lanczos Interpolation

The aim was to identify which combination of resolution and interpolation method delivered the optimal results for pneumonia detection.
