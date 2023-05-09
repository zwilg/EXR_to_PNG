import cv2
from os import listdir
import os.path
import shutil
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
    return src ** invGamma

def exr2png(filename:str, path_exr:Path, path_png:Path):
    # Function provided by Mauhing Yip, NTNU
    # 65536 = 2*16 - 1
    exr_file_path = path_exr+"/"+filename+".exr"
    png_file_path = path_png+"/"+filename+".png"
    #exr_file_path = (path_exr/filename).with_suffix(".exr")
    #png_file_path = (path_png/filename).with_suffix(".png")
    im=cv2.imread(str(exr_file_path),-1)
    im=im*65536
    im[im>65535]=65535
    im=np.uint16(im)
    print("Writing to: ", str(png_file_path))
    cv2.imwrite(str(png_file_path),im)

def image_histogram_scaling(image, number_bins=256, scaling_upper_percentile=0.975):
    """
    image:                      2D array of iamge luminance values
    number_bins:                Number of bins used for intensity sorting. Higher count --> Higher accuracy
    scaling_upper_percentile:   Percentage of pixels which is not clipped. I.e. a value of 0.975 
                                will clip the brightest 0.025*100%=2.5% of values to pure white. 
    returns:
        scaling_factor:         Scaling factor to multiply the image RGB values such that the luminance level 
                                corresponds to the desired scaling specified in scaling_upper_percentile.

    This function sorts the luminance-values of the image into a histogram with the specified number of bins. 
    It then calculates how many bins to clip to the brightest value (1, on a scale of [0,1]) through counting from
    the number of pixels in each bin and summarizing these, starting from bin 0 (darkest values). 
    When 97.5% (default value) of pixels are counted, the current bin is stored.
    The total number of bins is then divided with the current bin number to return the scaling factor. 
    """

    # Create a histogram of image intensities
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=False) # density=False such that we can access the number of samples in each bin.
    
    # Find total number of pixels and number of pixels in the upper percentile
    number_of_pixels = np.prod(np.size(image))
    number_of_pixels_upper_percentile = int(number_of_pixels*scaling_upper_percentile)
    
    # For each iteration, add the number of pixels in each bin together.
    # Once the pixel count reaches the upper percentile of pixels, return the number of the bin where this percentile was found.
    pixel_count = 0
    scaling_factor_bin_number = 0
    for i in range(number_bins-1):
        pixel_count = pixel_count + image_histogram[i]
        if pixel_count >= number_of_pixels_upper_percentile:
            scaling_factor_bin_number = i
            break

    # Find the scaling factor by dividing the total number of bins by the bin number where the upper percentile threshold was reached.
    scaling_factor = number_bins/scaling_factor_bin_number
    print(scaling_factor)
    return scaling_factor

def find_scaling_factor(image_luminance, percentile):
    """ NOT IMPLEMENTED!!!
        DO NOT USE!!!!!!!! 
        ################## """
    # Define upper and lower percentiles
    percentile_lower = percentile
    percentile_upper = 1-percentile

    # Reshape to a 1D vector
    pixel_values = np.reshape(image_luminance, -1)

    # Number of pixels in the lower and upper percentile
    num_pixels = pixel_values.size
    num_pixels_percentile = int(num_pixels*percentile)

    # Sort such that lowest pixel intensities are first in the list, largest last
    pixel_values = np.sort(pixel_values, axis=None)

    # Find lowest and highest values
    pixel_value_low = pixel_values[num_pixels_percentile-1]
    pixel_value_high = pixel_values[num_pixels-num_pixels_percentile-1]

    # Scaling-factors
    scaling_factor_low = 1 # TODO - White level scaling
    scaling_factor_high = 1 # TODO - Black level scaling

    return scaling_factor_low*scaling_factor_high

def exr2png_equalized(filename:str, path_exr:Path, path_png:Path, gamma, scaling_factor_old, scaling_upper_percentile, scaling_factor_delta_limit):
    """
    filename:                   Filename of image
    path_exr:                   Path to EXR image folder
    path_png:                   Path to PNG image folder
    gamma:                      Gamma-value, standard value for srgb color-space is 2.2
    scaling_factor_old:         Scaling factor used for the previous auto-exposure iteration. Used to limit the exposure 
                                    change between each iteration.
    scaling_upper_percentile:   Upper percentile of values which is clipped during the auto-exposure process. 
    scaling_factor_delta_limit: Maximum change of the scaling factor between two iterations. A low value will result in 
                                    more smooth auto-exposure between images (less abrubt changes in exposure between 
                                    frames), however the risk of over/underexposure increases. 
    """
    # Function provided by Mauhing Yip, NTNU - Modified by Peder Zwilgmeyer

    exr_file_path = path_exr+"/"+filename
    png_file_path = path_png+"/"+filename[:-4]+"_equalized.png"
    im=cv2.imread(str(exr_file_path),-1)

    # Create grayscale image for luminance check on image w/ linear values
    _height = len(im[:,0,0]) # y, height
    _width= len(im[0,:,0]) # x, width
    im_luminance = np.zeros((_height, _width))
    im_luminance = np.dot(im[:,:,:3],rgb_luminance_weights)

    # Autoexposure using a luminance histogram to determine the scaling_factor
    _scaling_factor = image_histogram_scaling(im_luminance, number_bins=2**16, scaling_upper_percentile=scaling_upper_percentile)
    # _scaling_factor = find_scaling_factor(im_luminance, scaling_upper_percentile)
    if scaling_factor_old == 0:
        scaling_factor = _scaling_factor
    else:
        # Ensure that scaling_factor is max XX% different from the last image to prevent abrubt exposure changes.
        if _scaling_factor > scaling_factor_old*(1+scaling_factor_delta_limit):
            scaling_factor = scaling_factor_old*(1+scaling_factor_delta_limit)
        elif _scaling_factor < scaling_factor_old*(1-scaling_factor_delta_limit):
            scaling_factor = scaling_factor_old*(1-scaling_factor_delta_limit)
        else:
            scaling_factor = _scaling_factor
    
    # Scale image
    im = im*scaling_factor

    # Apply gamma correction to .exr image with 32bit float precision
    im_gamma = gammaCorrection(im, gamma)

    # Convert image to 16bit integer values
    im_gamma = im_gamma*65536
    im_gamma[im_gamma>65535]=65535
    im_gamma=np.uint16(im_gamma)

    # Save image
    print("Saving image to: ", str(png_file_path))
    cv2.imwrite(str(png_file_path),im_gamma)
    return scaling_factor


if __name__ == "__main__":
    # Setup absolute filepaths, using this file's path as a reference. 
    dir_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    path_exr = dir_path + "/images_exr/"
    path_png = dir_path + "/images_png_converted-OpenCV/"

    # Create a list of the filenames for all files in the path_exr folder
    filenames = os.listdir(path_exr)

    gamma = 2.2 # Gamma-value used for gamma correction. Higher value = brighter image. 
                # See the following link for more info: https://learnopengl.com/Advanced-Lighting/Gamma-Correction
                # gamma = 2.2 is the most common default value. 

    # Luminance weights to find the grayscale luminance of a given pixel value. 
    rgb_luminance_weights = np.array([0.2126, 0.7152, 0.0722])  # Y = 0.2126 R + 0.7152 G + 0.0722 B 

    # Create a list of scaling_factors, this is essentially the amount of exposure correction for every image.
    # This is used for the functionality further described by scaling_factor_delta_limit.
    scaling_factors = np.zeros(len(filenames)+1)

    # Using scaling_upper_percentile=0.995 is equivalent to clipping the top 0.005*100% = 0.5% most bright pixels. 
    # This means that the top 0.5% of the pixels in the final image will be saturated white (max pixel intensity)
    scaling_upper_percentile = 0.995

    # The largest acceptable difference in the scaling factor from one image to another. A larger value will allow
    # for more significant differences in the "exposure correction" between two successive frames.
    # I.e. with a low framerate, the "auto exposure" responds slowly. With a high value, it will respond very quick
    # which can result in "flashing" effects when the images are played back as a video.
    scaling_factor_delta_limit=0.10

    # Iterator
    scaling_factor_it = 0

    for filename in filenames:
        #exr2png(filename, path_exr, path_png)
        print("filename: ", filename)
        scaling_factors[scaling_factor_it+1] = exr2png_equalized(filename, path_exr, path_png, gamma, scaling_factors[scaling_factor_it], scaling_upper_percentile, scaling_factor_delta_limit)
        scaling_factor_it = scaling_factor_it + 1
