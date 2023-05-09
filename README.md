This code reads a directory of OpenEXR images, applies white level scaling and gamma correction, before exporting them as 16bit PNG files. 

The scipt consists of the following:

First the parameters are defined. 
These include:
- gamma: Gamma-value used for gamma correction, where the default is gamma=2.2.
- rgb_luminance_weights: Weights which is used to sum the RGB intensities of a single pixel to find the luminance (grayscale intensity) value. This is used as the basis for white level scaling.
- scaling_factors: Create a list of scaling_factors, this is essentially the amount of exposure correction for every image. This is used for the functionality further described by scaling_factor_delta_limit.
- scaling_upper_percentile: Using scaling_upper_percentile=0.995 is equivalent to clipping the top 0.005*100% = 0.5% most bright pixels. This means that the top 0.5% of the pixels in the final image will be saturated white (max pixel intensity)
- The largest acceptable difference in the scaling factor from one image to another. A larger value will allow for more significant differences in the "exposure correction" (white level scaling) between two successive frames. I.e. with a low framerate, the "auto exposure" responds slowly. With a high value, it will respond very quick which can result in "flashing" effects when the images are played back as a video.

During the main loop, the script iterates through all OpenEXR files in the "images_exr" folder, doing the following:

1. The image is loaded using OpenCV, and the luminance is calculated. 
2. Gamma-correction is applied
3. The scaling factor for white level scaling is determined. This is done by sorting the luminance values into a histogram. This function sorts the luminance-values of the image into a histogram with the specified number of bins. 
It then calculates how many bins to clip to the brightest value (1, on a scale of [0,1]) through counting from the number of pixels in each bin and summarizing these, starting from bin 0 (darkest values). When 97.5% (default value) of pixels are counted, the current bin is stored. The total number of bins is then divided with the current bin number to return the scaling factor.  
4. White level scaling factor is checked to be equal to or below a maximum allowed delta, or "change", from the previous scaling factor value. If the value exceeds this maximum allowed delta, the scaling factor is truncated to be equal to the maximum allowed delta. This is no to ensure that the perceived exposure-changes are not unrealistically abrubt.
5. White level scaling factor is applied to all RGB values of the original OpenEXR image.
6. Scaled image is cast from 32bit float values in [0,1] to integer values in [0,65535] and saved as 16bit PNG files. 

