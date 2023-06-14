[//]: # (TODO: add some pictures to readme)
# Interactive CDI
An interactive applet that demonstrates principles of coherent diffraction imaging (CDI), specifically phase retrieval, in a hands-on environment. 

## Why is this useful?
Most coherent image reconstruction software works something like this:

1. User gives the program a "recipe" which dictates every step of the reconstruction.
2. Software executes the reconstruction as rapidly as possible.
3. User sees the end result, often by loading a results file in another application.

This process is designed to maximize efficiency for both the end-user and the computer, and, in the hands of an expert, it does just that. However, in the hands of a newcomer, it is both overwhelming and largely opaque. This project uses a very different approach, which trades efficiency for transparency:

1. User gives the program a command (e.g. apply a single operator, iterate a single constraint, or iterate a set of constraints until told to stop, etc.)
2. Software executes the command, displaying the results in real-time.
3. User may repeat *ad desiderium cordis*.

While not designed for high performance, this approach has great educational value in at least the following contexts:
* A high-school physics teacher could demonstrate the advanced applications of diffraction.
* A physics lab professor could use it for a coherent diffraction imaging experiment.
* A new member of a CDI research group could use it to build understanding and intuition applicable to more advanced software.
* An experienced researcher might still gain new insights by watching phase retrieval occur in realtime. (I know I have...)

## Installation
If you are running on Windows, the simplest way to run Interactive CDI is to download and extract the official release, and run the `cdi_live.exe` executable within. As of right now, this software is only compiled for Windows.

If you're not on Windows, or if you just want to run Interactive CDI from its Python source code, download the appropriate files from GitHub. Make sure that you have installed Python v3.9 or later. Then run the following (in a virtual environment, if desired):

    pip install numpy scipy scikit-image matplotlib Pillow

Once that's done, you should be able to run the live app:

    python src/cdi_live.py

## Using the applet
### Understanding phase retrieval
Coherent diffraction imaging (CDI) is an indirect imaging method that works by back-propagating the light field in a diffraction pattern to its source. If you know the amplitude and phase of the light at every point in the diffraction plane, you can calculate the amplitude and phase at any other plane. While this is usually a pretty painful calculation, there are some special cases in which it can be greatly simplified. One of these is propagating from a coherently illuminated aperture/object to a far-field diffraction pattern and vice versa, which amounts to a slightly modified Fourier transform.[^1]

Now, if we could measure both the amplitude and phase of the light at each point, reconstructing the object would be easy---just take the Fourier transform of the complex diffraction pattern. However, since we only measure the intensity (amplitude squared) we have to use various phase retrieval algorithms to reconstruct the object. A full description of these algorithms is beyond the scope of a README.

In summary, iterative phase retrieval consists of alternately projecting the same data between direct and reciprocal space, applying constraints with each projection. In reciprocal space, the amplitude of each pixel is known by measurement, while the phase is unknown. In direct space, we assume that the object or aperture exists in a compact region of space (known as the "support region"), and pixels outside that region must have zero intensity. The projection between the two spaces maps amplitude and phase information from every pixel in one space to every pixel in the other. Thus, enforcing a ground truth on part of the direct-space image pushes the entire reciprocal-space image in the right direction, and vice versa. 

### The images
When you first open the app, it will load one of the simulated example files. The amplitude image shows the measured diffraction amplitude in grayscale, and the phase image shows a random initial phase in color. Because phase is a cyclic quantity ($e^{i(\phi+2\pi)}=e^{i\phi}$), it is often represented on a color wheel (as opposed to a linear color scale), which goes from red all the way to violet, before starting over at red. In this way, phase wraps do not appear as discontinuous jumps, but as a smooth transition back around the cycle.

Depending on the state of the reconstruction and which tab is active, the app will show either a direct-space (physical) or a reciprocal-space (diffraction) representation of the illuminated object or aperture. The "Data" tab will always show the reciprocal-space representation, the "Auto" tab will always show the direct-space representation, and the "Manual" tab may show either.

### The "Data" tab
This is where you load and pre-process your diffraction pattern. It has the following features:

**Load data**: Opens a dialog to select one or more diffraction image files. If multiple images are selected, they will be summed pixel-wise. Some basic pre-processing is also applied at this stage: (1) color images are converted to grayscale, (2) the brightest point in the image is shifted to the center, (3) if the image is not a square, the long dimension is cropped to match the short dimension, (4) images larger than 1024x1024 pixels are downsampled to that size (to keep computational times reasonable), and (5) the square root of the image is taken, which converts intensity to amplitude.

**Load background**: Opens a dialog to select one or more background image files. These are (ideally) images measured under the exact same conditions as the diffraction data, but without the coherent light source. Such a measurement allows you to characterize the noise in your experiment, e.g. electrical noise in the camera or stray light in the room. The same pre-processing is applied to these images as to the diffraction images, with two exceptions. First, the shifting step re-uses the shift from the diffraction images (that is, it doesn't check for the brightest noise). Second, the background intensity is scaled to match the exposure of the diffraction; for example, if there are 10 summed diffraction images and only 5 background images, the background intensity will be scaled by a factor of 2. _Note that this requires the actual exposure levels to remain constant throughout the experiment._

**Subtract background**: Toggles whether the background is subtracted from the diffraction data. If no background is loaded, this does nothing. Background subtraction is probably the most important pre-processing feature in CDI, and should always be used when possible.

**Bin pixels**: Applies pixel binning (downsampling by summing adjacent pixels) to the image. For sufficiently oversampled data (many pixels per smallest diffraction fringe) this can drastically reduce the computational time of each iteration. However, if the fringes lose fidelity, the reconstruction will fail.

**Gaussian blur**: Applies gaussian blurring to the image. The slider below it adjusts the amount of blurring (gaussian sigma in pixels). This can be useful for smoothing out grainy data, though it is usually better to simply sum over more images.

**Threshold**: Applies thresholding to the image. The slider adjusts the threshold value; for a value of 0.25, the dimmest 25% of all pixels will be set to zero. This can sometimes be an effective way of removing noise in a dataset when long exposures and/or background subtraction are not options.

**Vignette**: Applies vignetting to the image. The slider adjusts the level of vignetting. If the diffraction pattern has high-intensity fringes extending well beyond the edges of the image, it may cause artifacts to appear in the direct-space reconstruction. Vignetting may make these artifacts smaller at the cost of reduced resolution.

It's worth noting that, while these and other pre-processing steps (which may or may not be implemented in the future) can help make bad data better, they are almost never an adequate substitute for simply getting better data.

### The "Manual" tab
The "Manual" tab lets you control each constraint and projection during the phase retrieval process.

**Shrinkwrap**: Applies shrinkwrap[^2] to the support region. This sets the support region to include only those pixels above a certain fractional threshold in a gaussian-filtered version of the direct-space amplitude. In effect, this shrinks the support region to fit tightly around the object (hence the name). _Applying shrinkwrap switches the amplitude image to show the updated support region._ So don't panic when it changes. 

**Hybrid input-output**: Applies the hybrid input-output (HIO) constraint[^3] to the direct-space amplitude and phase. The HIO constraint sets everything outside the support region to be whatever it was on the previous iteration, minus some fraction (beta) of its value on the current iteration. This provides a feedback which prevents stagnation and can allow the support region to grow if needed. However, by putting energy outside the support region, it prevents itself from strongly converging to any one solution.

**Error reduction**: Applies the error reduction (ER) constraint[^3] to the direct-space amplitude and phase. The ER constraint sets everything outside the support region to zero. This is a highly convergent method, essentially never taking a step that will increase the error of the reconstruction. However, as is often the case with such optimization techniques, it is also very prone to stagnation.

**Forward propagate (FFT)**: Projects the current image from direct space to reciprocal space using a fast Fourier transform, or FFT. Pressing this button will also disable all direct-space buttons in this tab, including itself.

**Back propagate (IFFT)**: Projects the current image from reciprocal space to direct space using an inverse fast Fourier transform, or IFFT. Pressing this button will also disable all reciprocal-space buttons in this tab, including itself.

**Replace modulus**: Replaces the amplitude (known in the biz as the "modulus") of the current image with the measured amplitude.

**Parameters**: These three sliders control the gaussian sigma and relative threshold for shrinkwrap as well as the beta coefficient for HIO. They are synced with those on the auto tab.

### The "Auto" tab
Iterative phase retrieval is, well, iterative. As such, it can get tedious to do every step by hand. This tab iterates automatically so that you can watch the reconstruction take place on a much faster timescale. Each iteration is equivalent to the following sequence on the manual tab:
1. Forward propagate (FFT)
2. Replace modulus
3. Back propagate (IFFT)
4. Hybrid input-output
5. Shrinkwrap

**Start**: Begins iterative phase retrieval.

**Stop**: Completes the current iteration, then stops iterating.

**Stop w/ ER**: Completes the current iteration, performs a single iteration with the ER constraint replacing HIO, then stops iterating. If pressed when the reconstruction is not iterating, 

**Re-center**: Shifts the direct-space object so that the center-of-mass of the support region is centered on the image. If currently iterating, this will happen between iterations.

**Remove twin**: Re-centers the direct-space image (see previous), then sets everything outside the top-left quadrant to zero. This abrupt removal of symmetry can help break out of stagnation. If currently iterating, this will happen between iterations.

**Gaussian blur**: Passes the amplitude and phase through a gaussian filter (sigma = 1 pixel). Again, this abrupt change in the "texture" of the object can help break out of stagnation. If currently iterating, this will happen between iterations.

**Reset**: Resets the reconstruction to its initial state, with random phases in reciprocal space and a centered square support region half the size of the image. Also resets the SW and HIO parameters. If currently iterating, this will happen between iterations.

**Save results**: Opens a dialog to select a directory in which to save the current state of the reconstruction. You are strongly encouraged to create a new folder, as it will overwrite existing output files. Once a directory is selected, it will save seven files: an amplitude, phase, and composite image for direct space, the same for reciprocal space, and a raw numpy array file containing the actual complex values in direct space.

## How to contribute
For bug tracking and feature requests, just use the issues tab on GitHub. If you want to contribute more directly, feel free to fork the repository and submit a pull request. Be aware that the release version is compiled using PyInstaller, which [only supports certain packages](https://github.com/pyinstaller/pyinstaller/wiki/Supported-Packages). In particular, `scikit-image` is not supported, so image processing must be done using something else. Most (if not all) of `scikit-image`'s functionality can be replicated with `scipy.ndimage`.

[^1]: Ware, M. and Peatross, J. (2015) ‘Fraunhofer Approximation’, sec 10.4 in _Physics of Light and Optics_, pp. 264–265. Available at: https://optics.byu.edu/textbook.

[^2]: Marchesini, S. et al. (2003) ‘X-ray image reconstruction from a diffraction pattern alone’, _Physical Review B_, 68(14), p. 140101. Available at: https://doi.org/10.1103/PhysRevB.68.140101.

[^3]: Fienup, J.R. (1982) ‘Phase retrieval algorithms: a comparison’, _Applied Optics_, 21(15), pp. 2758–2769. Available at: https://doi.org/10.1364/AO.21.002758.


