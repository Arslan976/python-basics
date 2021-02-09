# author Arslan Faisal CSU ID : 2809340

# Simple Instruction for running this program
# When you run the program it will ask you for any input out of 1 2 3
# These number are according to the task
# You can run one task at a time, for the other task you would have to run program again
# If you want to see output for another image uncomment the lines of function call 
# In return comment any other function call
# And run the program 


import cv2 as cv
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Task 1
def get_histogram_grey_scale(src):
    
    imageone = Image.open(src)
    imageone.show()
    img = ImageOps.grayscale(imageone) 
    img1 = np.asarray(img)
    
    flat = img1.flatten()
    plt.hist(flat, bins=250)
    def get_histogram(image, bins):
        histogram = np.zeros(bins)  
        for pixel in image:
            histogram[pixel] += 1
            
        return histogram
    
    hist = get_histogram(flat, 256)
    
    # create a cumulative sum function
    def comulative_sum(val):
        val = iter(val)
        b = [next(val)]
        for i in val:
            b.append(b[-1] + i)
        return np.array(b)

    # function exceution
    cs = comulative_sum(hist)
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()

    # re-normalize the comulative sum
    cs = nj / N

    # cast it back to uint8 since we can't use floating point values in images
    cs = cs.astype('uint8')
    plt.plot(cs)
   
# Task 2
def get_histogram_color(src):
    def separation_of_rgb_contrast(rgb_img):
        # show the image
        colorimg = Image.open(src)
        colorimg.show()
        # segregate color streams
        blue,green,red = cv.split(rgb_img)
        h_blue, bin_blue = np.histogram(blue.flatten(), 256, [0, 256])
        h_green, bin_green = np.histogram(green.flatten(), 256, [0, 256])
        h_red, bin_red = np.histogram(red.flatten(), 256, [0, 256])
        # CDF Calculation   
        cdf_blue = np.cumsum(h_blue)  
        cdf_green = np.cumsum(h_green)
        cdf_red = np.cumsum(h_red)
        
        # Mask all images of zero value and replace it with mean of the pixel values 
        cdf_m_b = np.ma.masked_equal(cdf_blue,0)
        cdf_m_b = (cdf_m_b - cdf_m_b.min())*255/(cdf_m_b.max()-cdf_m_b.min())
        cdf_final_b = np.ma.filled(cdf_m_b,0).astype('uint8')
        
        cdf_m_g = np.ma.masked_equal(cdf_green,0)
        cdf_m_g = (cdf_m_g - cdf_m_g.min())*255/(cdf_m_g.max()-cdf_m_g.min())
        cdf_final_g = np.ma.filled(cdf_m_g,0).astype('uint8')
        cdf_m_r = np.ma.masked_equal(cdf_red,0)
        cdf_m_r = (cdf_m_r - cdf_m_r.min())*255/(cdf_m_r.max()-cdf_m_r.min())
        cdf_final_r = np.ma.filled(cdf_m_r,0).astype('uint8')
        # Merging of images into three channels
        img_b = cdf_final_b[blue]
        img_g = cdf_final_g[green]
        img_r = cdf_final_r[red]
        
        img_out = cv.merge((img_b, img_g, img_r))
        # validation
        equ_b = cv.equalizeHist(blue)
        equ_g = cv.equalizeHist(green)
        equ_r = cv.equalizeHist(red)
        cv.merge((equ_b, equ_g, equ_r))
        hist,bins = np.histogram(img_out.flatten(),256,[0,256])
        plt.hist(img_out.flatten(),256,[0,256], color = 'r')
        plt.xlim([0,256])
        plt.show()

    colored_img_in = cv.imread(src, 0)
    converted_rgb_img = cv.cvtColor(colored_img_in, cv.COLOR_BGR2RGB)
    separation_of_rgb_contrast(converted_rgb_img)
    
def get_gradient_threshold():
    
     img = cv.imread('colored.jpg')
     arrayinput = asarray(img)
     print(np.shape(arrayinput))
     input_data = np.float32(arrayinput) / 255.0
    
     # Calculate Gradient of Gx and Gy
     gx = cv.Sobel(input_data, cv.CV_32F, 1, 0, ksize=1)
     gy = cv.Sobel(input_data, cv.CV_32F, 0, 1, ksize=1)
    
     print('Gradient of X:')
     print(gx)
     print('-------------------------------------------')
     print('Gradient of Y:')
     print(gy)
     print('-------------------------------------------')
    
     # Calculating Magnitude and Angle
     gradient_magnitude,angle = cv.cartToPolar(gx, gy, angleInDegrees=True)
      
     ret, thresh1 = cv.threshold(gradient_magnitude, 120, 255, cv.THRESH_BINARY) 
     print('Final Answer')
     cv.imshow('Binary Threshold', thresh1) 
     cv.waitKey(0)
     cv.destroyAllWindows() 
    
print("Hi this is the project 1. Please Enter Task Number You Want To See:")
print ("Task 1: Print Histogram for grey scale image")
print ("Task 2: Print Histogram for colored image")
print ("Task 3: Compute gradient x , y and its magnitude")
val = input("Enter your value: ")
if val == '1':
   get_histogram_grey_scale('firstgreyimage.jpg')
   # Extreme light
   #get_histogram_grey_scale('secondgreyimage.jpg')
   # Dark
   #get_histogram_grey_scale('thridgreyimage.jpg')
   # Extreme Dark
   #get_histogram_grey_scale('forthgreyimage.jpg')
elif val == '2':
    get_histogram_color('colored.jpg')
    # Extreme Light
    #get_histogram_color('coloredlight.jpg')
    # Dark 
    #get_histogram_color('coloreddark.jpg')
    # Extreme Dark
    #get_histogram_color('coloredextremedark.jpg')
elif val == '3':
    get_gradient_threshold()
else:
    print('You Enter a wrong value!')
    print ("Task 1: Print Histogram for grey scale image")
    print ("Task 2: Print Histogram for colored image")
    print ("Task 3: Compute gradient x , y and its magnitude")
    decision = input("Do you want to run another task? (y/n): ")
    if decision == 'n' or decision == 'N':
        i = False
    elif decision == 'y' or decision == 'Y':
        i = True
    else:
        print('You need to run the program again wrong input!')