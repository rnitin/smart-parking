# Program to evaluate input video and determine vacant and occupied lots

# Update to Firebase has been disabled in this code. Enable by uncommenting.

# ***** Import libraries *****
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import time
from timeit import default_timer as timer
from threading import Thread
import pyrebase
import json


# ***** Function definitions *****

def predict(image, model):
    '''
    Function to determine image class
    Parameter: image (Torch tensor), model (Torch model - trained)
    Returns: probs (float), classes (int)   
    '''
    output = model.forward(image) # pass the image through model
    output = torch.exp(output) # reverse the log function in output
    probs, classes = output.topk(1, dim=1) # Get the top predicted class its probability
    return probs.item(), classes.item()

def process_image(cv2_img):
    '''
    Function to process image to normalised tensor
    Parameter: cv2_img (cv2 image array)
    Returns: image (Torch tensor)  
    '''
    img = Image.fromarray(cv2_img)
    width, height = img.size # read image width and height
        
    # Set the coordinates to do a center crop of 224 x 224
    left = int((width - 224)/2)
    top = int((height - 224)/2)
    right = int((width + 224)/2)
    bottom = int((height + 224)/2)
    img = img.crop((left, top, right, bottom)) # crop image to 224 x 224
    
    img = np.array(img) # turn image into numpy array  
    img = img.transpose((2, 0, 1)) # make the color channel dimension first instead of last
    img = img/255 # make all values between 0 and 1

    # Normalize based on the ImageNet mean and standard deviation
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    img = img[np.newaxis,:] # add a fourth dimension to the beginning to indicate batch size
    image = torch.from_numpy(img) # turn into a torch tensor
    image = image.float()
    return image

def draw_lots(disp): # input parameter : cv2 image
    '''
    Function to add status grids to the displayed image
    Parameter: disp (cv2 image array)
    Returns: none
    '''
    global n_lots, lot_status # sets n_lots and lot_status as global variables across threads
    lot = 0
    while(lot in range(n_lots)): # iterate through each lot
        if lot_status[lot][0] == -1: # unchecked lot
            disp = cv2.rectangle(disp, loca[lot], locb[lot], (255,0,0), 1) # draw blue rectangle
        elif lot_status[lot][0] == 0: # occupied lot
            disp = cv2.rectangle(disp, loca[lot], locb[lot], (0,0,255), 1) # draw red rectangle
        elif lot_status[lot][0] == 1: # vacant lot
            disp = cv2.rectangle(disp, loca[lot], locb[lot], ( 0,255,0), 1) # draw green rectangle
        msg = ("ID:" + str(lot) + "P:" + str(round(lot_status[lot][1]*100))) # message to be displayed above the lot
        disp = cv2.putText(disp, msg, loca[lot], font, fontScale, fontColor, lineType) # add message above lot
        lot += 1  
        
def eval_lots(img):
    '''
    Function to evaluate lot status
    Parameter: img (cv2 image array)
    Returns: status (list)  
    '''
    status = list((-1, 0.0) for arr in range(n_lots)) # -1 if unchecked, 0 if occupied, 1 if free; second row: probability  
    lot = 0
    while(lot in range(n_lots)): # iterate through each lot

        # Read individual lot image
        lot_img = img[loca[lot][1]:locb[lot][1], loca[lot][0]:locb[lot][0]] # read lot image from video frame
        lot_img = cv2.resize(lot_img, dim, interpolation = cv2.INTER_CUBIC) # resize lot
        image = process_image(lot_img) # process input image
        
        # Predict image class
        top_prob, top_class = predict(image, model) # determine lot class and probability
        status[lot] = (top_class, top_prob) # add class and probability to status list
        lot += 1
    return status

def recheck(img, lot_status, new_status):
    '''
    Function to recheck lot status
    Parameter: img (cv2 image array), lot_status (list), new_status (list)
    Returns: status (list)  
    '''
    status = list((-1, 0.0) for arr in range(n_lots)) # -1 if unchecked, 0 if occupied, 1 if free; second row: probability
    lot = 0
    while(lot in range(n_lots)): # iterate through each lot

        if (new_status[lot][0] != lot_status[0]): # if lot status has changed, recheck
            lot_img = img[loca[lot][1]:locb[lot][1], loca[lot][0]:locb[lot][0]] # read lot image from video frame
            lot_img = cv2.resize(lot_img, dim, interpolation = cv2.INTER_CUBIC) # resize lot
            image = process_image(lot_img) # process input image

            top_prob, top_class = predict(image, model) # predict image class
            if (top_class == new_status[lot][0]): # if double check successful
                status[lot] = (top_class, top_prob)
            else: # if double check failed
                status[lot] = lot_status[lot] # continue with previous double check status

        else:
            status[lot] = new_status[lot]
        lot += 1
    return status

def read_video(): 
    '''
    Function to capture input video stream
    Parameter: none
    Returns: none
    '''
    global disp, img # sets disp and img cv2 arrays as global variables across threads
    global video_flag, input_flag # sets video_flag and input_flag as global variables across threads
    cap = cv2.VideoCapture(video_path) # open video stream (15 fps)
    while(cap.isOpened()): # when video stream is opened

        if input_flag == 0: # if first frame is being read
            ret, disp = cap.read()
            ret, img = cap.read()
            input_flag = 1 # input_flag set to 1 when video begins

        draw_lots(disp) # add status lots to frame
        cv2.imshow('Video', disp) # display the frame to user
        if cv2.waitKey(132) == 27: # frame delay set to (2/15)s, not (1/15)s, as 2 frames are read per iteration
            break # break when 'Esc' key is pressed

        ret, disp = cap.read() # read frame for display
        ret, img = cap.read() # read frame for evaluation
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT): # break when frame is last video frame
            break
      
    video_flag = 0 # video_flag set to 0 when video ends
    print('end of video') 
    cap.release() # release video capture object

def update_output(lot_status):
    '''
    Function to update lot status to firebase
    Parameter: lot_status (list)
    Returns: none
    '''
    update_status = list ((-1) for arr in range(n_lots)) # list to store each lot'ss class
    lot = 0
    while(lot in range(n_lots)): # iterate through each lot
        update_status[lot] = lot_status[lot][0] # store class alone in update_status
        lot += 1
    output["status"] = str(update_status).replace(' ', '') # remove space in the list
    # result = db.child("TestData").push(output) # push data to firebase
    # print(result)


# ***** Declarations ******

# Initialise pyrebase wrapper for accessing Firebase REST API
config = {
    "apiKey": " ",
    "authDomain": " ",
    "databaseURL": " ",
    "storageBucket": " "
}
firebase = pyrebase.initialize_app(config)
db = firebase.database() # create pyrebase database object

# Declare a dictionary to store output
output = {
    "lotID": 20,
    "length": 0,
    "status": "",
    "timestamp": {".sv": "timestamp"},
}

# Set path of trained model, test video, lot coordinates
model_path = './test-data/model.pt'
video_path = './test-data/lot.mp4'
lot_path = './test-data/lot_coords.csv'

# Formatting for status display
font        = cv2.FONT_HERSHEY_SIMPLEX
fontScale   = 0.3
fontColor   = (255,255,255)
lineType    = 1

# Variables and values
dim = (250, 250)        # dimension to resize individual lot sinput
t_start, t_check = 0, 0 # initialise start time and check time variables as 0
flag = 0                # flag for double check
video_flag = 1          # flag to check video stream presence
input_flag = 0          # flag to check video frame availability
disp, img = 0, 0        # global declaration for disp and img cv2 frames


# ***** Program routine *****

# Load model and initialise parameters
model = torch.load(model_path, map_location=torch.device('cpu')) # load model (tained in GPU) and map to CPU
for param in model.parameters(): # turn off training for model parameters
    param.requires_grad = False
device = torch.device('cpu') # set computing device as CPU
model.to(device) # move model to device
criterion = nn.NLLLoss() # set the model error function
model.eval() # set model to evaluate mode

# Read and assign lot coordinates
lot_list = pd.read_csv(lot_path).values # read lot coordinates from csv file
n_lots = len(lot_list) # total no. of lots
loca = [(0,0) for arr in range(n_lots)] # loca (left-top coordinate of each lot) set as 0
locb = [(0,0) for arr in range(n_lots)] # locb (right-bottom coordinate of each lot) set as 0
lot = 0
while lot < n_lots: # iterate through each lot
    loca[lot] = (lot_list[lot][1], lot_list[lot][2]) # assign loca from coordinate list
    locb[lot] = (lot_list[lot][3], lot_list[lot][4]) # assign locb from coordinate list
    lot += 1
output["length"] = n_lots # update lot length in output dictionary
lot_status = list((-1, 0.0) for arr in range(n_lots)) # first member: -1 if unchecked, 0 if occupied, 1 if free; second member: probability
new_status = list((-1, 0.0) for arr in range(n_lots)) # first member: -1 if unchecked, 0 if occupied, 1 if free; second member: probability

# Begin a new thread for video capture
vid_thread = Thread(target = read_video) # create thread for video capture
vid_thread.start() # start video stream capture
print('start')

# Read input video stream and evaluate status
while(video_flag): # continue till video runs
   
    t_start = timer() # set loop timer start
    
    if input_flag == 0: # no frame read
        time.sleep(0.05)
        print('wait')

    else:

        if t_start - t_check >= 10: # check every 10 seconds
            t_check = timer() # set check timer
            print('check')
            new_status = eval_lots(img)
            flag = 1 # to recheck
        
        if ((flag == 1) and (t_start - t_check >= 1)): # recheck after second
            flag = 0 # recheck done
            if (new_status != lot_status):
                print('recheck')
                lot_status = recheck(img, lot_status, new_status)
                update_output(lot_status) # update to firebase

cv2.destroyAllWindows()