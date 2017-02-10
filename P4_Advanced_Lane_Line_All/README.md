## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:  

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply the distortion correction to the raw image.  
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view"). 
* Detect lane pixels and fit to find lane boundary.
* Determine curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---




Examples are give throughout.

Functions were written to get the high, med and low thresholds of RBG, HLS , Magnitude and Direction Gradients and create a binary images. An example of the visualization demonstrates that the high threshold for the S channel of HLS was one of the best for visualizing the lane lines in shadowy areas. 
Many combinations were tried to best visualize the lane lines. A final combination included S channel with a high threshold, gradient in the x direction with a low threshold, magnitude and directional gradients. Masking was used.

It will be easier to work with the binary lane line images if we had a bird eye view. The cv2.getPerspectiveTransform() was use to write a birds_eye() function to transform the lane line image.

A Line class and a sliding window were used to find the lane line pixels. A histogram of the sum of the number of the pixels, in the lower half of the image, in each x positions was used to find the highest number of pixels and therefor the probable location of the starting point of the lane lines. Two histograms were used, one for the right half of the image and one for the left half. To save processing time two smaller windows were placed over the determined start points of each lane line. x and y postilions were recoded for each pixels in each row of the small windows. The average of the x postilions were used the calculate the next starting point of the next window. This resulted in the recording of x and y positions in a left or right Line class object.  


I used a few methods to find the lane line boundaries. First if  the current value of A, B or C were very different form the average of the past n values then the new image values must be wrong and the current values were not used. For this I created a ratio of current values divided by the average of the previous n images.  Second if the left and right A, B or C values were very different the line was probably not a lane line and not used. Lastly I used distance the lane line is from the center of the lane and if the current difference were very different form the previous n values the current values were not used.

The ranges of each factor were determined by collecting all results and getting the max, min and average. This gave me an idea were to set the cutoffs to remove data. With trial and error about 10 percent of values were excluded. Only results with noticeable positive changes were used. 

In the event a current image values were not used the average of the previous n values were used.

This was an awesome topic. This was my first real implementation of a class that was useful. The notebook is designed to be a grand pipeline. So I can go back when time permits and improve and visualize the improvements as I go. Later I plan to try the challenges. I want to print out all RGB, and directional and magnitude gradients the way I did here with S channel. This will help me pick a better combination to create a binary image of the lanes. Also, I was thinking the vertices for the masking now is 4 points. If I could make the points in the vertices similar to the point in the lane lines of the previous image but slightly wider apart this would create a dynamic masking that closely mimics the previous images lane lines but father apart. 


