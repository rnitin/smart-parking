# smart-parking
A deep learning system to identify vacant and occupied lots in outdoor car parkings.

<h2>1. Background</h2>

This program implements a Convolutional Neural Network through transfer learning on the Resnet50 model and additional fully-connected layers. The model is trained using the PyTorch library on the CNRPark dataset. The CNRPark dataset was downloaded and sorted into train, test and validation sets irrespective of the weather conditions and camera angles of the overall image.
OpenCV is used to read lot video and capture the individual frames. At a regular defined interval, the captured frames are evaluated based on defined lot coordinate values to determine vacant and occupied parking lots. 
The status is reflected by drawing red and green grids on the video.
The evaluated status of the lots can be written to a Firebase Realtime database, and this status can be used to update the Android app at github.com/rnitin/smart-parking-app

<h2>2. Dependencies</h2>
The programs were evaluated on Python 3.8 with the following versions of the dependencies:

| Package | Version |
| --- | --- |
|torch|1.5.0|
|torchvision|0.6.0|
|opencv_python|4.2.0|
|numpy|1.18.1|
|matplotlib|2.2.0|
|pandas|1.0.3|
|Pillow|7.1.2|  
  
  
|Optional Package|Version |
|---|---|
|torchsummary|1.5.1  |
|Pyrebase |3.0.27|

<h2>4. References and Acknowledgements</h2>
To be updated
