
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

<h2>3. Executing the Programs</h2>

<h4>Preparing the programs and the dataset</h4>

1. Clone this repository  
`git clone https://github.com/rnitin/smart-parking.git`
2. Prepare the dataset
	Download the CNRPark+EXT dataset and separate it into training, validation and test sets in:  
	*./train/dataset/carpark/training_set/*, *./train/dataset/carpark/valid_set/*, and *./train/dataset/carpark/test_set/*
	
	
<h4>Training the CNN model</h4> 

1. Change the value of model_no each time to determine the result file names.
2. Execute the python script *smartpark-train.py*  
`python smartpark-train.py`

<h4>Using the vacancy detection system</h4>  

1. Execute the python script *smartpark-test.py*  
`python smartpark-test.py`
2. To evalute a different parking lot video:
	2.1 Replace *./test-data/lot.mp4* with the new video
	2.2 Identify the top-left and bottom-right coordinates of the individual parking lots present in the video and update the *./test-data/lot-coords.csv* file.
3. (Optional) To upload the evaluated status to Firebase:
	3.1 Create a Firebase Realtime database
	3.2 Update the values of `apiKey`, `authDomain`, `databaseURL` and `storageBucket` in the `config` dictionary of the script.
	3.3 Uncomment relevant lines from the `update_output` function.

<h2>4. References and Acknowledgements</h2>

[CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/)  
[CNRPark+EXT Database](http://cnrpark.it/)  
[PyTorch](https://pytorch.org/)  
[LearnOpenCV](https://github.com/spmallick/learnopencv)
