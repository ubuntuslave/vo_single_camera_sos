# Visual Odometry with a Single-Camera Stereo Omnidirectional System

Proof of concept of the visual odometry (VO) application via a single-camera catadioptric *stereo omnidirectional system* (SOS) and performance comparison against a popular RGB-D camera.

- **Author**: Carlos Jaramillo
- **Contact**: <omnistereo@gmail.com>

*Copyright (C)* 2018 under the *Gnu Public License version 3 (GPL3)*

**WARNING**: This code is at an experimental stage, and has only been tested under **Python 3.6**  on   **Mac OS X 10.12.6 (Sierra)**

## Setup

### Python Modules:

#### Required modules:

- Scipy
- Numpy
- Vispy
  - Tkinter
  - PyQt5
  
And these packages with their respective Python bindings (instructions below):

- OpenCV
- OpenGV

For instance Using Python 3.6, with `pip3`, install the following modules:

    $ pip3 install numpy
    $ pip3 install scipy
    $ pip3 install sympy

For visualization:

    $ pip3 install matplotlib
    $ pip3 install mpldatacursor

#### (Optional) For 3D Visualization:

    $ pip3 install vispy

In Linux (Ubuntu), you may need `vispy` with *Tkinter* and *PyQt*):
 
    $ sudo apt-get install python3-tk python3-pyqt5
    $ sudo -H pip3 install vispy
    
For some reason, I had to install *PyQt5* from pip:

    $ sudo -H pip3 install pyqt5

***NOTE***: In Linux, you may get *Permission Denied* errors that can be solved by:

    $ sudo rm -rf ~/.cache/pip/


#### OpenCV 3

The following guide uses `Homebrew `for Mac OS X:

##### Requirements:

Make sure you have installed the XCode CLI tools are all installed:

    $ ls -lah /usr/include/sys/types.h 

If not, try:

    $ xcode-select --install

The last release of OpenCV's [Deep Neural Network module](http://docs.opencv.org/master/d6/d0f/group__dnn.html) requires Google's [Protocol Buffers](https://developers.google.com/protocol-buffers/)    

    $ brew install protobuf    
    
However, it *may fail* to find a "suitable threading library available." If so, disable DNN within the CMake configuration (in the next steps)

##### Install OpenCV 3

###### The Easy Way Via Homebrew (Mac OS X only):
    
    $ brew install opencv3
    
###### The Hard (more powerful) Way from Source Code:
    
Clone *OpenCV 3* from the repo

    $ cd ~/src
    $ git clone git@github.com:Itseez/opencv.git

And also the ***contributed*** modules:

    $ git clone git@github.com:Itseez/opencv_contrib.git    
    
Configure the installation vis CMake (or Curses CMake as I'd like to do): 

    $ cd opencv/..
    $ mkdir opencv-build
    $ cd opencv-build
    $ ccmake ../opencv -G"Eclipse CDT4 - Unix Makefiles"

The extra modules path is set like :

```
OPENCV_EXTRA_MODULES_PATH        /PATH_TO_/src/opencv_contrib/modules
```

For example, Python paths should be set to:

```
 PYTHON2_EXECUTABLE               /usr/local/bin/python2.7
 PYTHON2_INCLUDE_DIR              /usr/local/Cellar/python/2.7.14/Frameworks/Python.framework/Versions/Current/include/python2.7
 PYTHON2_INCLUDE_DIR2
 PYTHON2_LIBRARY                  /usr/local/Cellar/python/2.7.14/Frameworks/Python.framework/Versions/Current/lib/libpython2.7.dylib
 PYTHON2_LIBRARY_DEBUG
 PYTHON2_NUMPY_INCLUDE_DIRS       /usr/local/lib/python2.7/site-packages/numpy/core/include
 PYTHON2_PACKAGES_PATH            lib/python2.7/site-packages
 PYTHON3_EXECUTABLE               /usr/local/bin/python3
 PYTHON3_INCLUDE_DIR              /usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/include/python3.6m
 PYTHON3_INCLUDE_DIR2
 PYTHON3_LIBRARY                  /usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/lib/libpython3.6m.dylib
 PYTHON3_LIBRARY_DEBUG
 PYTHON3_NUMPY_INCLUDE_DIRS       /usr/local/lib/python3.6/site-packages/numpy/core/include
 PYTHON3_PACKAGES_PATH            /Users/carlos/src/opencv-build/lib/python3.6/site-packages
```

Then, compile OpenCV 3, and install as usual:

    $ make
    $ make install


###### In Ubuntu Linux

In Ubuntu 16.04, using the `ros-kinetic-opencv3` version and workaround to use it with `python3`:

1) Install OpenCV3 for ROS:

    $ sudo apt-get install ros-kinetic-opencv3

2) Install python3 versions of rospkg and catkin 
    
    $ sudo -H pip3 install rospkg catkin_pkg


**NOTE for Python3**: Once the usual `ros-kinetic-opencv3` package is installed and working for python2.7, you can get it working in python3 by *removing the catkin* sourced in the `.bashrc` file. Otherwise, just adjust your IDE not to have the ROS python paths for the `$PYTHONPATH`.

In other words:

  - Problems are caused by ROS adding /opt/ros/kinetic/lib/python2.7/dist-packages to the `$PYTHONPATH`. This actually happens when you activate ROS with the command `source /opt/ros/kinetic/setup.bash`. This line is often added at the end of your `.bashrc` file, in `/home/username/.bashrc`.

  - A **workaround** is to remove this line from the bashrc file. This way the `python3` `opencv` packages will be correctly used, and you can still run `source /opt/ros/kinetic/setup.bash` to use ROS. However, this does mean **you cannot use ROS and python3 from the same environment**.


3) You need to install the following module via `pip3`:

    $ sudo -H pip3 install opencv-python

3) Test OpenCV under python3:

    $ python3 
    >>> import cv2
    
    
#### OpenGV (for Visual Odometry):


`pyopengv` depends on **Boost** >= 1.66 and **Boost.python.numpy**, so make sure it gets installed before proceeding. 

In **Mac OS X**, with Homebrew:

    $ brew install boost
    $ brew install boost-python3
    
In **Ubuntu**, you may need to compile boost 1.66+ by downloading its source as indicated in this [tutorial](https://www.boost.org/doc/libs/1_67_0/more/getting_started/unix-variants.html#get-boost). I did:

    $ cd path/to/boost_1_67_0
    $ ./bootstrap.sh --help

Select your configuration options and invoke ./bootstrap.sh again, for example:

    $ ./bootstrap.sh --with-python=python3

Finally,

    $ sudo ./b2 install

##### Compile and install OpenGV

Assuming, **Boost**  >= 1.66 and **Boost.python.numpy* were already successfully installed.

Clone the fork from [opengv](https://github.com/ubuntuslave/opengv.git) because this has Python bindings for the *non-central camera* case, which we need for tracking the omnistereo system pose:

    $ cd ~/src
    $ git clone https://github.com/ubuntuslave/opengv.git
    $ cd opengv
    $ git checkout non_central-python
    $ cd .. && mkdir opengv-build && cd opengv-build
    
Now build the project, making sure 

- The correct paths for *Python* get selected, and
- The `boost.python.numeric` path is set correctly.

    $ ccmake ../opengv

If compiling with `python3` support, you must *toggle* the advanced configuration and set the appropriate *Python* paths. 

For example, in **Mac OS X**, I set:

    Boost_NUMPY3_LIBRARY_DEBUG       /usr/local/lib/libboost_numpy36-mt.dylib                        
    Boost_NUMPY3_LIBRARY_RELEASE     /usr/local/lib/libboost_numpy36-mt.dylib                        
    Boost_NUMPY_LIBRARY_DEBUG        /usr/local/lib/libboost_numpy-mt.dylib                          
    Boost_NUMPY_LIBRARY_RELEASE      /usr/local/lib/libboost_numpy-mt.dylib
 
    PYTHON_EXECUTABLE             /usr/local/bin/python3
    PYTHON_INCLUDE_DIR            /usr/local/Cellar/python3/3.6.5/Frameworks/Python.framework/Versions/3.6/include/python3.6m
    PYTHON_INSTALL_DIR            /usr/local/lib/python3.6/site-packages/
    PYTHON_LIBRARY                /usr/local/Cellar/python3/3.6.5/Frameworks/Python.framework/Versions/3.6/lib/libpython3.6m.dylib
    
For example, in **Ubuntu 16.04**, I had:

    PYTHON_EXECUTABLE             /usr/bin/python3
    PYTHON_INCLUDE_DIR            /usr/include/python3.5
    PYTHON_INSTALL_DIR            /usr/local/lib/python3.5/dist-packages
    PYTHON_LIBRARY                /usr/lib/x86_64-linux-gnu/libpython3.5m.so

In Ubuntu, I had to also configure the proper location of the *installed-from-source* Boost 1.66:

    Boost_DIR                        /home/carlos/src/boost_1_66_0                                                  
    Boost_INCLUDE_DIR                /usr/local/include                                                             
    Boost_LIBRARY_DIR_DEBUG          /usr/local/lib                                                                 
    Boost_LIBRARY_DIR_RELEASE        /usr/local/lib                                                                 
    Boost_PYTHON3_LIBRARY_DEBUG      /usr/local/lib/libboost_python3.so                                             
    Boost_PYTHON3_LIBRARY_RELEASE    /usr/local/lib/libboost_python3.so                                             
    Boost_PYTHON_LIBRARY_DEBUG       /usr/lib/x86_64-linux-gnu/libboost_python.so                                   
    Boost_PYTHON_LIBRARY_RELEASE     /usr/lib/x86_64-linux-gnu/libboost_python.so 
        
Once the Cmakefiles are *generated*, compile and install as usual:

    $ make
    $ sudo make install
    $ sudo ldconfig
    
*NOTE*: I had to run `$ sudo ldconfig` for the `$ python3 -c "import pyopengv"` to work.

    
### Omnistereo package

Install the omnistereo package by running:

    $ cd path_to_the_cloned_repo
    $ sudo -H pip install -e .

### Running the Demos

#### Add `omnistereo` to your $PYTHONPATH:

To your `.bashrc` file, you may add:

    # Omnistereo project stuff for Python
    OMNISTEREO=~/src/vo_single_camera_sos
    if [ -z $PYTHONPATH ]
    then
        export PYTHONPATH="$OMNISTEREO"
    else
        export PYTHONPATH="$PYTHONPATH:$OMNISTEREO"
    fi

Save, close, and reopen your Terminal

#### To run the VO demo with the RGB-D camera:

```
usage: demo_vo_rgbd.py [-h] [--is_synthetic IS_SYNTHETIC]
                       [--hand_eye_transformation HAND_EYE_TRANSFORMATION]
                       sequence_path

Demo of frame-to-frame visual odometry for the RGB-D images collected for the
vo_single_camera_sos project.

positional arguments:
  sequence_path         The path to the sequence where the rgbd folders is
                        located.

required arguments:
  --is_synthetic IS_SYNTHETIC
                        Determines whether the data is real or synthetic. This
                        is necessary to use the appropriate camera intrinsic
                        parameters.

optional arguments:
  --hand_eye_transformation HAND_EYE_TRANSFORMATION
                        (Optional) If real-life data, this indicates the
                        complete path and name to hand-eye transformation file
```

For example, using the `free_style` sequence,

    $ python3 demo_vo_rgbd.py "PATH_TO_MY_SEQUENCE/free_style" --is_synthetic=false --hand_eye_transformation="PATH_TO_APPROXIMATED/rgbd_hand_eye_transformation.txt"


#### To run the VO demo with the single-camera SOS:

```
usage: demo_vo_sos.py [-h] [--calibrated_gums_file CALIBRATED_GUMS_FILE]
                      sequence_path

Demo of frame-to-frame visual odometry for the Single-camera SOS images
collected for the vo_single_camera_sos project.

positional arguments:
  sequence_path         The path to the sequence where the omni folder is
                        located.

required arguments:
  --calibrated_gums_file CALIBRATED_GUMS_FILE
                        Indicates the complete path and name of the calibrated
                        GUMS pickle file
```

For example, using the `free_style` sequence,

    $ python3 demo_vo_sos.py "PATH_TO_MY_SEQUENCE/free_style" --calibrated_gums_file="PATH_TO_CALIBRATED_MODEL/gums-calibrated.pkl"

