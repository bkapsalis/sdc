#Libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import glob
from skimage.feature import hog
from skimage import color, exposure
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split











#Functions

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a function to compute color histogram features  
#def color_hist(img, nbins=32, bins_range=(0, 256)):
#    # Compute the histogram of the RGB channels separately
#    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
#    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
#    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
#    # Generating bin centers
#    bin_edges = rhist[1]
#    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
#    # Concatenate the histograms into a single feature vector
#    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
#    # Return the individual histograms, bin_centers and feature vector
#    return rhist, ghist, bhist, bin_centers, hist_features

# Define a function to compute color histogram features   Now produces a float
def color_hist(img,  nbins=32, bins_range=(0, 256), channel_1 = True, channel_2 = True, channel_3 = True):
         
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = channel1_hist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2   
    
    
    #modify to combine any 1, 2 or 3 channels.
    
    if(channel_1 == True & channel_2 == True & channel_3 == True):
    # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    elif((channel_1 == True) & (channel_2 == True) & (channel_3 == False)):
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0]))
    elif((channel_1 == True) & (channel_2 == False) & (channel_3 == True)):
        hist_features = np.concatenate((channel1_hist[0], channel3_hist[0]))
    elif((channel_1 ==False) & (channel_2 == True) & (channel_3 == True)):
        hist_features = np.concatenate(( channel2_hist[0], channel3_hist[0]))
    elif((channel_1 == True) & (channel_2 == False) & (channel_3 == False)):    
        #hist_features = np.concatenate(( channel1_hist[0]))
        hist_features = channel1_hist[0]    
    elif((channel_1 == False) & (channel_2 == True) & (channel_3 == False)):    
        #hist_features = np.concatenate(( channel2_hist[0]))
        hist_features = channel2_hist[0]    
    elif((channel_1 == False) & (channel_2 == False) & (channel_3 == True)):  
        #hist_features = np.concatenate(( channel3_hist[0]))
        hist_features = channel3_hist[0]


    # Return the individual histograms, bin_centers and feature vector
    return channel1_hist, channel2_hist, channel3_hist, bin_centers, hist_features


def plot_three_bar_charts(rh, gh, bh, bincen, color_space = 'RGB'):
                     
    cs_list = list(color_space)
    if rh is not None:
        fig = plt.figure(figsize=(4,3))
        plt.subplot(131)
        plt.bar(bincen, rh[0])
        plt.xlim(0, 256)
        plt.title(cs_list[0] + ' Histogram')
        plt.subplot(132)
        plt.bar(bincen, gh[0])
        plt.xlim(0, 256)
        plt.title(cs_list[1] + ' Histogram')
        plt.subplot(133)
        plt.bar(bincen, bh[0])
        plt.xlim(0, 256)
        plt.title(cs_list[2] + ' Histogram')        
        fig.tight_layout()
    else:
        print('Your function is returning None for at least one variable...')
        
        
def plot_three_bar_charts_car_notcar(car_rh, car_gh, car_bh, notcar_rh, notcar_gh, notcar_bh, bincen, color_space = 'RGB'):
                     
    cs_list = list(color_space)
    if car_rh is not None:
        fig = plt.figure(figsize=(18,3))
        plt.subplot(161)
        plt.bar(bincen, car_rh[0])
        plt.xlim(0, 256)
        plt.title('Car ' + cs_list[0] + ' Histogram')
        plt.subplot(162)
        plt.bar(bincen, car_gh[0])
        plt.xlim(0, 256)
        plt.title('Car ' + cs_list[1] + ' Histogram')
        plt.subplot(163)
        plt.bar(bincen, car_bh[0])
        plt.xlim(0, 256)
        plt.title('Car ' + cs_list[2] + ' Histogram') 
        
        plt.subplot(164)
        plt.bar(bincen, notcar_rh[0])
        plt.xlim(0, 256)
        plt.title('NotCar ' + cs_list[0] + ' Histogram')
        plt.subplot(165)
        plt.bar(bincen, notcar_gh[0])
        plt.xlim(0, 256)
        plt.title('NotCar ' + cs_list[1] + ' Histogram')
        plt.subplot(166)
        plt.bar(bincen, notcar_bh[0])
        plt.xlim(0, 256)
        plt.title('NotCar ' + cs_list[2] + ' Histogram')          
        
        fig.tight_layout()
    else:
        print('Your function is returning None for at least one variable...')        
        
        
def plot3d(pixels, colors_rgb, axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation        
        
# Define a function to compute color histogram features  
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
#32 * 32 * 3 =3072
def bin_spatial(img, size=(32, 32), channel_1 = True, channel_2 = True, channel_3 = True):
    #print(img.shape, 'resize pre image size')
    # Use cv2.resize().ravel() to create the feature vector

    #modify to combine any 1, 2 or 3 channels.
    
    if(channel_1 == True & channel_2 == True & channel_3 == True):
    # Concatenate the histograms into a single feature vector
        features = cv2.resize(img, size).ravel()  
    elif((channel_1 == True) & (channel_2 == True) & (channel_3 == False)):
        features = cv2.resize(img[:,:, 0:2], size).ravel()  
    elif((channel_1 == True) & (channel_2 == False) & (channel_3 == True)):
        features = cv2.resize(img[:,:, (0,2)], size).ravel()  
    elif((channel_1 ==False) & (channel_2 == True) & (channel_3 == True)):
        features = cv2.resize(img[:,:,1:3], size).ravel()  
    elif((channel_1 == True) & (channel_2 == False) & (channel_3 == False)):    
        features = cv2.resize(img[:,:,0], size).ravel()    
    elif((channel_1 == False) & (channel_2 == True) & (channel_3 == False)):    
        features = cv2.resize(img[:,:,1], size).ravel()     
    elif((channel_1 == False) & (channel_2 == False) & (channel_3 == True)):  
        features = cv2.resize(img[:,:,2], size).ravel()    
    #features = img.ravel() # Remove this line!
    # Return the feature vector
    return features        
 
    
    
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    img = cv2.resize(img, (64, 64))                                                            #!!!!!Added!!!!!
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features    
    
    
# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict



# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0, gray_flag = False, 
                        chan_1 = True, chan_2 = True, chan_3 = True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'Yrb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size,
                                       channel_1 = chan_1, channel_2 = chan_2, channel_3 = chan_3)

        # Apply color_hist() also with a color space option now
        #hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        _,_,_,_,hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range,
                                           channel_1 = chan_1, channel_2 = chan_2, channel_3 = chan_3)
        # Call get_hog_features() with vis=False, feature_vec=True
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if gray_flag:
            hog_features = get_hog_features(gray, orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)        
        
        # Append the new feature vector to the features list

        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors
    return features




    # Define a function to extract features from an image
# Have this function call bin_spatial() and color_hist()
def extract_features_from_image(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0, gray_flag = False,
                        chan_1 = True, chan_2 = True, chan_3 = True):
    # Create a list to append feature vectors to

    # Iterate through the list of images

    # Read in each one by one
    image = imgs
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'Yrb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)            
    else: feature_image = np.copy(image)  
    
    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image,  size=spatial_size,
                                   channel_1 = chan_1, channel_2 = chan_2, channel_3 = chan_3)
    ###print(len(spatial_features), 'spatial_features')
    # Apply color_hist() also with a color space option now
    #hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
    _,_,_,_,hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range,
                                       channel_1 = chan_1, channel_2 = chan_2, channel_3 = chan_3)
    ###print(len(hist_features), 'hist_features')
    # Call get_hog_features() with vis=False, feature_vec=True
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if gray_flag:
        hog_features = get_hog_features(gray, orient, 
                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)    
    
    
    
    #print(len(hog_features), 'hog_features')
    # Append the new feature vector to the features list
    features = np.concatenate((spatial_features, hist_features, hog_features))
    #print(len(features), 'features')
    
    # Return list of feature vectors
    return features


# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
            
 
            
            
            
            
    # Return the list of windows
    return window_list

