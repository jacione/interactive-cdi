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
3. User may repeat *ad satisfactio*.

While not designed for high performance, this approach has great educational value in at least the following contexts:
* A high-school physics teacher could demonstrate the advanced applications of diffraction.
* A physics lab professor could use it for a coherent diffraction imaging experiment.
* A new member of a CDI research group could use it to build understanding and intuition applicable to more advanced software.
* An experienced researcher might still gain new insights by watching phase retrieval occur in realtime.

## Installation & Use
First, make sure that you have installed Python v3.9 or later. Then run the following (in a virtual environment, if desired):

    pip install numpy scipy scikit-image matplotlib

Once that's done, you should be able to run either the stepper app or the live app:

    python src/cdi_step.py
    # OR 
    python src/cdi_live.py


