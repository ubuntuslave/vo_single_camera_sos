# -*- coding: utf-8 -*-
# common_tools.py

# Copyright (c) 2012-2018, Carlos Jaramillo
# Produced at the Laboratory for Robotics and Intelligent Systems of the City College of New York
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import division
from __future__ import print_function

import warnings
import numpy as np
# import cPickle as pickle

import os
import errno

def get_current_os_name():
    from sys import platform as _platform
    if _platform == "linux" or _platform == "linux2":
        return "linux"  # linux
    elif _platform == "darwin":
        return "mac"  # OS X
    elif _platform == "win32" or _platform == "cygwin":
        return "windows"  # Windows...

def str2bool(v):
    # Used for the argsparser to simulate a boolean type
    return v.lower() in ("yes", "true", "t", "on", "1")

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def get_namestr(obj, namespace):
    '''
    Finds and returns the variable name for the object instance in question in the desired namespace.
    For example

        >>> get_namestr(my_wow, globals())
        Out: 'my_wow'
    '''
    if namespace:
        results = [name for name in namespace if namespace[name] is obj]
        if len(results) == 1:
            return results[0]
        else:
            return results
    else:
        return ""

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def get_theoretical_params_from_file(filename, file_units = "cm", model_units = "m"):
    # NOTE: This only works assuming the model in this project's implementation uses milimeters as the units
    scale = get_length_units_conversion_factor(input_units = file_units, output_units = model_units)
    scales = [scale, scale, 1., 1., scale, scale, scale, scale]  # Recall, k1 and k2 are dimensionless
    params_file = open(filename, 'r')
    params_line = params_file.readline().rstrip('\n')  # Read first and only line of this params file
    params_file.close()
    c1, c2, k1, k2, d, r_sys, r_reflex, r_cam = scales * np.array(params_line.split(","), dtype = np.float64)  # The rows, cols, width_row, width_col, margin are saved on the first line
    return c1, c2, k1, k2, d, r_sys, r_reflex, r_cam

def save_obj_in_shelve(obj_instance, filename, namespace = None):
    import shelve

    print("Saving %s instance to shelve file %s " % (get_namestr(obj_instance, namespace), filename))

    db = shelve.open('filename')
    db["calibrator"] = obj_instance
    db.close()

    print("done!")

def save_obj_in_pickle(obj_instance, filename, namespace = None):
    '''
    Save object to pickle for Python3. Recall, that the default pickling protocol used by default in Python 3 is incompatible with the protocol used by Python 2
    '''
    import pickle
    print("Saving %s instance to pickle (for Python 3) file %s " % (get_namestr(obj_instance, namespace), filename))

    f = open(filename, 'wb')  # Create external f
    pickle.dump(obj_instance, f, protocol = pickle.DEFAULT_PROTOCOL)
    f.close()

    print("done!")

def save_obj_in_pickle2(obj_instance, filename, namespace = None):
    '''
    Save object to pickle for Python2. Recall, that the default pickling protocol used by default in Python 3 is incompatible with the protocol used by Python 2
    '''
    import pickle
    print("Saving %s instance to pickle (for Python 2) file %s " % (get_namestr(obj_instance, namespace), filename))

    f = open(filename, 'wb')  # Create external f
    pickle.dump(obj_instance, f, protocol = 2)
    f.close()

    print("done!")

def load_obj_from_pickle(filename, namespace = None):
    import pickle
    f = open(filename, 'rb')  # Create external f
    obj_instance = pickle.load(f,)
    print("Loading %s instance from pickle file %s ... " % (get_namestr(obj_instance, locals()), filename), end = "")
    f.close()
    print("done!")
    return obj_instance

def load_obj_from_shelve(filename, namespace = None):
    import shelve
    db = shelve.open(filename)
    obj_instance = db['calibrator']
    print("Loading %s instance from pickle file %s ... " % (get_namestr(obj_instance, locals()), filename), end = "")
    db.close()
    print("done!")
    return obj_instance

def blind_copy(objfrom, objto):
    from inspect import getmembers
    for n, v in getmembers(objfrom):
        setattr(objto, n, v)

def copy_only_attributes(objfrom, objto, exclude_names = []):
    from inspect import getmembers
    for n, v in getmembers(objfrom):
        if "__" in n or n in exclude_names:
            continue
        setattr(objto, n, v)

def copy_some(objfrom, objto, names):
    for n in names:
        if hasattr(objfrom, n):
            v = getattr(objfrom, n)
            setattr(objto, n, v)

def solve_quadratic_functions_system_exact(func1_coeffs, func2_coeffs, verbose = False):
    '''
    Solves the system of two quadratic functions (exact). Thus, each quadratic function is in the form of f(x) = a*x^2 + b*x + c

    @param func1_coeffs: the list of [a, b, c] coefficients for the quadratic function 1
    @param func2_coeffs: the list of [a, b, c] coefficients for the quadratic function 2

    @return: The list of evaluated solutions (if any) arranged as tuples, either [], [(x1,f(x1))], or [(x1,f(x1)),(x2,f(x2))]
    '''
    a1, b1, c1 = func1_coeffs
    a2, b2, c2 = func2_coeffs

    a = a1 - a2
    b = b1 - b2
    c = c1 - c2

    x_solution = solve_quadratic_equation(a, b, c, verbose = verbose)
    num_solns = len(x_solution)

    if num_solns == 0:
        if verbose:
            print("No solution exists!")
        return []
    elif num_solns == 1:
        x = x_solution[0]
        y = a1 * x ** 2 + b1 * x + c1
        if verbose:
            print("Single solution (x, f(x)) = (%f, %f)" % (x, y))
        return [(x, y)]
    elif num_solns == 2:
        x1 = x_solution[0]
        y1 = a1 * x1 ** 2 + b1 * x1 + c1
        if verbose:
            print("Solution 1: [x, f(x)] = [%f, %f]" % (x1, y1))
        x2 = x_solution[1]
        y2 = a1 * x2 ** 2 + b1 * x2 + c1
        if verbose:
            print("Solution 2: [x, f(x)] = [%f, %f]" % (x2, y2))
        return [(x1, y1), (x2, y2)]

def solve_quadratic_equation(a, b, c, verbose = False):
    '''
    Solves the quadratic equation in the form of a*x^2 + b*x + c = 0

    @param a: second order coefficient
    @param b: first order coefficient
    @param c: scalar coefficient

    @return: a tuple of possible solutions, such as none, 1, or 2 answers
    '''
    import math

    d = b ** 2 - 4 * a * c  # discriminant

    if d < 0:
        if verbose:
            print("This equation has no real solution")
        return tuple()
    elif d == 0:
        x = (-b + math.sqrt(d)) / (2 * a)
        if verbose:
            print("This equation has one solutions: ", x)
        return (x)
    else:
        x1 = (-b + math.sqrt(d)) / (2 * a)
        x2 = (-b - math.sqrt(d)) / (2 * a)
        if verbose:
            print("This equation has two solutions: ", x1, " and", x2)
        return (x1, x2)

def pdf(point, cons, mean, det_sigma):
    if isinstance(mean, np.ndarray):
        return cons * np.exp(-(np.dot((point - mean), det_sigma) * (point - mean)) / 2.)
    else:
        return cons * np.exp(-((point - mean) / det_sigma) ** 2 / 2.)

def reverse_axis_elems(arr, k = 0):
    '''
    @param arr: the numpy ndarray to be reversed
    @param k: The axis to be reversed
        Reverse the order of rows: set axis k=0
        Reverse the order of columns: set axis k=1

    @return: the reversed numpy array
    '''
    reversed_arr = np.swapaxes(np.swapaxes(arr, 0, k)[::-1], 0, k)
    return reversed_arr

def flatten(x):
    '''
    Iteratively flattens the elements of a list of lists (or any other iterable such as tuples)
    '''
    import collections
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def rms(x, axis = None):
    return np.sqrt(np.mean(np.square(x), axis = axis))

def nanrms(x, axis = None):
    '''
    If you have nans in your data, you can do
    '''
    return np.sqrt(np.nanmean(np.square(x), axis = axis))

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights = weights)
    variance = np.average((values - average) ** 2, weights = weights)  # Fast and numerically precise
    return average, np.sqrt(variance)

def mean_and_std(values):
    """
    Return the arithmetic mean or unweighted average and the standard deviation.

    @param values: Numpy ndarrays of values
    """
#     average = np.mean(values)
    average = np.mean(values)
    std_dev = np.std(values)

    return average, std_dev

def huber_loss_pseudo(error, delta):
    """
    The Pseudo-Huber loss function is a smooth approximation of the Huber loss function so that derivatives are continuous for all degrees. Defined as follows:
    \f[
        L_{\delta }(a)=\delta ^{2}({\sqrt  {1+(a/\delta )^{2}}}-1).
    \f]

    @param error: The residual or numpy array of residuals to compute the losses for.
    @param delta: A threshold for the \p error so that this function approximates to a quadratic loss of \f$ (error^2)/2 \f$ for small values of \p error, and approximates a straight line with slope \p \delta for large values of the error.
    """
    return (delta ** 2) * (np.sqrt(1. + (error / delta) ** 2) - 1.)

def error_analysis_simple(errors, units):
    mean, std_dev = mean_and_std(np.array(errors))
    max_error = errors[np.argmax(errors) % 3]
    print("ERROR ANALYSIS: mean = {mean:.4f} [{units}], std. dev = {std:.4f} [{units}], MAX = {max:.4f} [{units}]".format(mean = mean, std = std_dev, max = max_error, units = units))

def get_transformation_matrix(pose_vector):
    '''
    Concatenates a transformation matrix using the rotation (quaternion) and translation components of the pose_vector

    @param pose_vector: A list (7-vector) of the rotation quaternion components [w, x, y, z] that imply w+ix+jy+kz and  translation components [tx, ty, tz].

    @attention:  the pose vector is NOT in TUM format!

    @return: the homogeneous transformation matrix encoding the rotation followed by a translation given as the pose_vector
    '''
    import omnistereo.transformations as tr
    # Normalize to unit quaternion:
    q = pose_vector[0:4]
    q_normalized = q / np.linalg.norm(q)
    rot_matrix = tr.quaternion_matrix(q_normalized)
    trans_matrix = tr.translation_matrix(pose_vector[4:])
    T = tr.concatenate_matrices(trans_matrix, rot_matrix)
    return T

def get_normalized_pose_vector(pose_vectors_list):
    '''
    Normalizes the rotation quaternion portion of each 7-vector pose

    @param pose_vector: A list of (7-vector) poses encoding a rotation quaternion components [w, x, y, z] that imply w+ix+jy+kz and the translation components [tx, ty, tz].

    @return: the pose list with the normalized quaternion for the rotation
    '''

    n_components_in_pose = 7  # Because we are assuming each pose is encoded as [q0, q1, q2, q3, tx, ty, tz]

    if len(pose_vectors_list) % n_components_in_pose != 0:
        raise ValueError("Length of `pose_vectors_list` is not exact.")

    n_poses = int(len(pose_vectors_list) / n_components_in_pose)

    poses_normalized = []

    for pose_idx in range(n_poses):
        current_pose_offset = pose_idx * n_components_in_pose
        current_pose = pose_vectors_list[current_pose_offset:current_pose_offset + n_components_in_pose]
        t = current_pose[4:]  # Translation vector
        # Normalize to unit quaternion:
        q = current_pose[:4]
        q_normalized = q / np.linalg.norm(q)

        new_pose = list(q_normalized) + list(t)
        poses_normalized += new_pose

    return poses_normalized

def get_inversed_transformation(T):
    '''
    Let's call T = [R|t] transformation matrix, applies the rule T_inv = [R^transpose | -R^transpose * t]
    @param T: 4x4 matrix [R|t] as the mixture of 3x3 rotation matrix R and translation 3D vector t.

    @note: This gives the same results as the inverse_matrix from the transformations.py module or simply numpy.linalg.inv(matrix)
    '''
    dims = len(T)
    T_inv = np.identity(dims)
    R_transposed = T[:dims - 1, :dims - 1].T
    translation = T[:dims - 1, dims - 1]
    T_inv[:dims - 1, :dims - 1] = R_transposed
    T_inv[:dims - 1, dims - 1] = -np.dot(R_transposed, translation)

    return T_inv

def get_2D_points_normalized(points_3D):
    '''
    @param points_3D: Numpy array of points 3D coordinates
    @return: new_pts_2D -  The array of transformed 2D homogeneous coordinates where the scaling parameter is normalised to 1.
             T      -  The 3x3 transformation matrix, such that new_pts_2D <-- T * points_normalized
    '''
    # Ensure homogeneous coords have scale of 1 because of normalization type INF
    # Similar to calling the OpenCV normalize function, such as
    # >>>> dst = normalize(points_3D[...,:3], norm_type=cv2.NORM_INF)
    pts_normalized = np.ones_like(points_3D[..., :3])
    pts_normalized[..., :2] = points_3D[..., :2] / np.abs(points_3D[..., 2, np.newaxis])

    centroid = np.mean(pts_normalized[..., :2].reshape(-1, 2), axis = (0))
    # Shift origin to centroid.
    pts_offset = pts_normalized[..., :2] - centroid
    meandist = np.mean(np.linalg.norm(pts_offset))
    scale = np.sqrt(2) / meandist  # because sqrt(1^2 + 1^2)

    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])

    new_pts_2D = np.einsum("ij, mnj->mni", T, pts_normalized)
    # TEST:
    # np.all([np.allclose(np.dot(T, pts_normalized[i,j]),new_pts_2D[i,j]) for (i,j), x in np.ndenumerate(new_pts_2D[...,0])])

    return new_pts_2D, T

import sys

PYTHON_V3 = sys.version_info >= (3, 0, 0) and sys.version_info < (4, 0, 0)

def get_screen_resolution(measurement = "px"):
    """
    Tries to detect the screen resolution from the system.
    @attention: # OpenCV HighGUI windows should be closed before invoking this function due to some QT threading issues!
    @param measurement: The measurement to describe the screen resolution in. Can be either 'px', 'inch' or 'mm'.
    @return: (screen_width,screen_height) where screen_width and screen_height are int types according to measurement.
    """
    mm_per_inch = 25.4
    px_per_inch = 72.0  # most common
    try:  # Platforms supported by GTK3, Fx Linux/BSD
        from gi.repository import Gdk
        screen = Gdk.Screen.get_default()
        if measurement == "px":
            width = screen.get_width()
            height = screen.get_height()
        elif measurement == "inch":
            width = screen.get_width_mm() / mm_per_inch
            height = screen.get_height_mm() / mm_per_inch
        elif measurement == "mm":
            width = screen.get_width_mm()
            height = screen.get_height_mm()
        else:
            raise NotImplementedError("Handling %s is not implemented." % measurement)
        return (width, height)
    except:
        try:
            from PyQt4 import QtGui
            import sys
            app = QtGui.QApplication(sys.argv)
            screen_rect = app.desktop().screenGeometry()
            width, height = screen_rect.width(), screen_rect.height()
            return (width, height)
        except:
            try:
                from PyQt5.QtWidgets import QApplication, QDesktopWidget
                import sys
                app = QApplication(sys.argv)
                screen_rect = QDesktopWidget().availableGeometry()
                width, height = screen_rect.width(), screen_rect.height()
                return (width, height)
            except:
                try:  # Probably the most OS independent way
                    import tkinter
                    root = tkinter.Tk()
                    if measurement == "px":
                        width = root.winfo_screenwidth()
                        height = root.winfo_screenheight()
                    elif measurement == "inch":
                        width = root.winfo_screenmmwidth() / mm_per_inch
                        height = root.winfo_screenmmheight() / mm_per_inch
                    elif measurement == "mm":
                        width = root.winfo_screenmmwidth()
                        height = root.winfo_screenmmheight()
                    else:
                        raise NotImplementedError("Handling %s is not implemented." % measurement)
                    return (width, height)
                except:
                    try:  # Windows only
                        from win32api import GetSystemMetrics
                        width_px = GetSystemMetrics (0)
                        height_px = GetSystemMetrics (1)
                        if measurement == "px":
                            return (width_px, height_px)
                        elif measurement == "inch":
                            return (width_px / px_per_inch, height_px / px_per_inch)
                        elif measurement == "mm":
                            return (width_px / mm_per_inch, height_px / mm_per_inch)
                        else:
                            raise NotImplementedError("Handling %s is not implemented." % measurement)
                    except:
                        try:  # Windows only
                            import ctypes
                            user32 = ctypes.windll.user32
                            width_px = user32.GetSystemMetrics(0)
                            height_px = user32.GetSystemMetrics(1)
                            if measurement == "px":
                                return (width_px, height_px)
                            elif measurement == "inch":
                                return (width_px / px_per_inch, height_px / px_per_inch)
                            elif measurement == "mm":
                                return (width_px / mm_per_inch, height_px / mm_per_inch)
                            else:
                                raise NotImplementedError("Handling %s is not implemented." % measurement)
                        except:
                            try:  # Mac OS X only
                                import AppKit
                                for screen in AppKit.NSScreen.screens():
                                    width_px = screen.frame().size.width
                                    height_px = screen.frame().size.height
                                    if measurement == "px":
                                        return (width_px, height_px)
                                    elif measurement == "inch":
                                        return (width_px / px_per_inch, height_px / px_per_inch)
                                    elif measurement == "mm":
                                        return (width_px / mm_per_inch, height_px / mm_per_inch)
                                    else:
                                        raise NotImplementedError("Handling %s is not implemented." % measurement)
                            except:
                                try:  # Linux/Unix
                                    import Xlib.display
                                    resolution = Xlib.display.Display().screen().root.get_geometry()
                                    width_px = resolution.width
                                    height_px = resolution.height
                                    if measurement == "px":
                                        return (width_px, height_px)
                                    elif measurement == "inch":
                                        return (width_px / px_per_inch, height_px / px_per_inch)
                                    elif measurement == "mm":
                                        return (width_px / mm_per_inch, height_px / mm_per_inch)
                                    else:
                                        raise NotImplementedError("Handling %s is not implemented." % measurement)
                                except:
                                    try:  # Linux/Unix
                                        if not self.is_in_path("xrandr"):  # FIXME: implement is_in_path function
                                            raise ImportError("Cannot read the output of xrandr, if any.")
                                        else:
                                            args = ["xrandr", "-q", "-d", ":0"]
                                            proc = subprocess.Popen(args, stdout = subprocess.PIPE)
                                            for line in iter(proc.stdout.readline, ''):
                                                if isinstance(line, bytes):
                                                    line = line.decode("utf-8")
                                                if "Screen" in line:
                                                    width_px = int(line.split()[7])
                                                    height_px = int(line.split()[9][:-1])
                                                    if measurement == "px":
                                                        return (width_px, height_px)
                                                    elif measurement == "inch":
                                                        return (width_px / px_per_inch, height_px / px_per_inch)
                                                    elif measurement == "mm":
                                                        return (width_px / mm_per_inch, height_px / mm_per_inch)
                                                    else:
                                                        raise NotImplementedError("Handling %s is not implemented." % measurement)
                                    except:
                                        # Failover
                                        screensize = 1366, 768
                                        sys.stderr.write("WARNING: Failed to detect screen size. Falling back to %sx%s" % screensize)
                                        if measurement == "px":
                                            return screensize
                                        elif measurement == "inch":
                                            return (screensize[0] / px_per_inch, screensize[1] / px_per_inch)
                                        elif measurement == "mm":
                                            return (screensize[0] / mm_per_inch, screensize[1] / mm_per_inch)
                                        else:
                                            raise NotImplementedError("Handling %s is not implemented." % measurement)

def convert_to_tuple(iterable_vals):
    '''
    Convert a 1D numpy array to be expressed as a tuple, even for one-dimensional shapes.

    @param iterable_vals: an int, or iterable of ints

    @return: a tuple converted from the iterable_vals
    '''
    try:
        i = int(iterable_vals)
        return (i,)
    except TypeError:
        # iterable_vals was not a number
        pass

    try:
        t = tuple(iterable_vals)
        return t
    except TypeError:
        # iterable_vals was not iterable
        pass

    raise TypeError('iterable_vals must be an int, or a tuple of ints')

def get_length_units_conversion_factor(input_units, output_units):
    units_scale_factor = 1.0
    if input_units == "cm":
        if output_units == "mm":
            units_scale_factor = 10.0
        elif output_units == "m":
            units_scale_factor = 0.01
    elif input_units == "mm":
        if output_units == "cm":
            units_scale_factor = 0.1
        elif output_units == "m":
            units_scale_factor = 0.001
    elif input_units == "m":
        if output_units == "mm":
            units_scale_factor = 1000.0
        elif output_units == "cm":
            units_scale_factor = 100.0
    return units_scale_factor

def save_as_tum_poses_to_file(output_tum_filename, poses_7_list, input_units, input_format, output_units = "m"):
    '''
    @param poses_7_list:  A list of pose lists (7-vector) indicating. Based on the input_format parameter, the order of components will change
    @param input_format: When "tr" the order of components is assumed to be the rotation quaternion components [q_w, q_i, q_j, q_k] that imply q_w+i*q_i+j*q_j+k*q_k and translation components [tx, ty, tz].
                         When "tum" the order of components is assumed to be the translation + rotation quaternion, such as: [tx, ty, tz, q_i, q_j, q_k, q_w]

    '''
    unit_conversion_factor = get_length_units_conversion_factor(input_units = input_units, output_units = output_units)

    f_tum_out = open(output_tum_filename, 'w')

    # Add a fake time stamp
    for time_stamp, pose_7 in enumerate(poses_7_list):
        if "tr" in input_format.lower():
            qw, qi, qj, qk = pose_7[:4]
            tx, ty, tz = pose_7[4:]
        elif "tum" in input_format.lower():
            tx, ty, tz = pose_7[:3]
            qi, qj, qk, qw = pose_7[3:]

        print("%d %.9f %.9f %.9f %.9f %.9f %.9f %.9f" % (time_stamp, unit_conversion_factor * tx, unit_conversion_factor * ty, unit_conversion_factor * tz, qi, qj, qk, qw), file = f_tum_out)  # FIXME: quaternion is wrong!

    f_tum_out.close()

def get_poses_from_file(poses_filename, input_units = "m", output_working_units = "m", indices = None, pose_format = "tum", zero_up_wrt_origin = False, initial_T = None, delimiter = None):
    '''
    @param poses_filename: the comma separated file for the pose information at each instance
    @param indices: the indices for the working images
    @param pose_format: the format in which the poses are given. It could be "povray" for the POV-Ray way encoding translation and euler angles in degrees. Or "tum" for the more standard encoding.
    @param zero_up_wrt_origin: We remove any initial ofsets, so that the first pose is given as identity
    @param initial_T: Any extra transform to be applied to every pose before anything else

    @return A list of pose lists (7-vector in TUM format ordering) indicating the translation components [tx, ty, tz] and the rotation quaternion components [q_i, q_j, q_k, w] that implies w+i*q_i+j*q_j+k*q_k.
    @return A list of transform matrices (homogeneous matrix) encoding the rotation (first) followed by a translation for the given frames wrt to the scene (reference frame)
    '''
    import omnistereo.transformations as tr

    if pose_format.lower() == "povray":
        if delimiter is None:
            delimiter = ","
        cols_tuple = (0, 1, 2, 3, 4, 5)
    elif pose_format.lower() == "tum":
        if delimiter is None:
            delimiter = " "
        cols_tuple = (0, 1, 2, 3, 4, 5, 6, 7)

    if isinstance(poses_filename, str):  # It's a filename
        grid_poses_from_file = np.loadtxt(poses_filename, delimiter = delimiter, usecols = cols_tuple, comments = "#", unpack = False)
        if len(grid_poses_from_file) == 0:
            print("Couldn't find poses in file", poses_filename)
            print("Exiting from", __name__)
            exit(1)
        if np.ndim(grid_poses_from_file) == 1:  # Means that only a single row entry existed
            grid_poses_from_file = grid_poses_from_file[np.newaxis, ...]

    unit_conversion_factor = get_length_units_conversion_factor(input_units = input_units, output_units = output_working_units)

    if indices is None or len(indices) == 0:
        list_len = len(grid_poses_from_file)
        indices = range(list_len)

    list_len = len(indices)
    poses_7_tum_format_list = list_len * [None]
    transform_matrices_list = list_len * [None]
    T_init_offset_wrt_S = tr.identity_matrix()
    if zero_up_wrt_origin:
        has_T_initial = False

    apply_init_T = True
    if initial_T is None:
        initial_T = tr.identity_matrix()
        apply_init_T = False

    for pose_number in indices:
        try:
            pose_info_list = grid_poses_from_file[pose_number]
            # pose info will be given as a list (7-vector) of the rotation quaternion components [w, x, y, z] followed by translation components [tx, ty, tz].

            if np.any(np.isnan(pose_info_list)):
                # Just create an invalid entry full of nans
                pose_info_tum = 7 * [np.nan]
            else:
                pose_info_tum = 7 * [0.0]
                if pose_format.lower() == "povray":
                    # We grab the translation values (given as RHS)
                    pose_info_tum[0] = unit_conversion_factor * float(pose_info_list[0])
                    pose_info_tum[1] = unit_conversion_factor * float(pose_info_list[1])
                    pose_info_tum[2] = unit_conversion_factor * float(pose_info_list[2])
                    # Camera frame [C] pose wrt to Scene frame [S]
                    CwrtS_angle_rot_x = np.deg2rad(float(pose_info_list[3]))  # because rotations in POV-Ray are given in degrees
                    CwrtS_angle_rot_y = np.deg2rad(float(pose_info_list[4]))  # because rotations in POV-Ray are given in degrees
                    CwrtS_angle_rot_z = np.deg2rad(float(pose_info_list[5]))  # because rotations in POV-Ray are given in degrees
                    #  In our RHS, the order of rotations are rotX --> rotY --> rotZ

                    CwrtS_rot_q = tr.quaternion_from_euler(CwrtS_angle_rot_x, CwrtS_angle_rot_y, CwrtS_angle_rot_z, 'sxyz')
                    [pose_info_tum[6], pose_info_tum[3], pose_info_tum[4], pose_info_tum[5]] = CwrtS_rot_q
                elif pose_format.lower() == "tum":
                    # We grab the translation values in TUM format: timestamp, tx, ty, tz, qi, qj, qk, qw
                    # Translation
                    pose_info_tum[0] = unit_conversion_factor * float(pose_info_list[1])
                    pose_info_tum[1] = unit_conversion_factor * float(pose_info_list[2])
                    pose_info_tum[2] = unit_conversion_factor * float(pose_info_list[3])
                    # Quaternion as expected by the "transformation" module:
                    pose_info_tum[3] = float(pose_info_list[4])  # qi
                    pose_info_tum[4] = float(pose_info_list[5])  # qj
                    pose_info_tum[5] = float(pose_info_list[6])  # qk
                    pose_info_tum[6] = float(pose_info_list[7])  # qw

            # Fill up the result lists:
            poses_7_tum_format_list[pose_number] = pose_info_tum  # A list (7-vector) of the TUM format
            transform_matrices_list[pose_number] = tr.transform44_from_TUM_entry(pose_info_tum, scale_translation = 1.0, has_timestamp = False)
            # Just to test:
            # >>> pose_from_tum_tr_approach = tr.transform44_from_TUM_entry_tr_approach(pose_info_list, scale_translation=unit_conversion_factor, has_timestamp=True)
            # >>> np.allclose(pose_from_tum_tr_approach, transform_matrices_list[pose_number])
            # >>> pose_from_tum = tr.transform44_from_TUM_entry(pose_info_list, scale_translation=unit_conversion_factor, has_timestamp=True)
            # >>> np.allclose(pose_from_tum, transform_matrices_list[pose_number])

            if zero_up_wrt_origin and not np.any(np.isnan(transform_matrices_list[pose_number])) or apply_init_T:
                if not has_T_initial:
                    T_init_offset_wrt_S = tr.inverse_matrix(transform_matrices_list[pose_number])
                    has_T_initial = True
                if has_T_initial or apply_init_T:
                    # Remove initial offset
                    transform_matrices_list[pose_number] = tr.concatenate_matrices(T_init_offset_wrt_S, transform_matrices_list[pose_number], initial_T)
                    pose_7_tum_without_offset = 7 * [0.0]
                    if np.any(np.isnan(transform_matrices_list[pose_number])):
                        pose_7_tum_without_offset[6], pose_7_tum_without_offset[3], pose_7_tum_without_offset[4], pose_7_tum_without_offset[5] = np.nan, np.nan, np.nan, np.nan
                    else:
                        pose_7_tum_without_offset[6], pose_7_tum_without_offset[3], pose_7_tum_without_offset[4], pose_7_tum_without_offset[5] = tr.quaternion_from_matrix(matrix = transform_matrices_list[pose_number], isprecise = False)  # It's safer to set "isprecise<--False" here
                    pose_7_tum_without_offset[:3] = tr.translation_from_matrix(matrix = transform_matrices_list[pose_number])
                    poses_7_tum_format_list[pose_number] = pose_7_tum_without_offset
        except:  # catch *all* exceptions
            err_msg = sys.exc_info()[1]
            warnings.warn("Warning...%s" % (err_msg))
            print("Exiting from", __name__)
            sys.exit(1)

    return poses_7_tum_format_list, transform_matrices_list

def get_time_stamps_from_tum_file(poses_filename, indices = None):
    '''
    @param poses_filename: the comma separated file for the pose information at each instance
    @param indices: the indices for the working images

    @return The list of time stamps
    '''
    import numpy as np

    delimiter = " "
    cols_tuple = (0,)

    if isinstance(poses_filename, str):  # It's a filename
        all_times_tamps_from_file = np.loadtxt(poses_filename, delimiter = delimiter, usecols = cols_tuple, comments = "#", unpack = False)
        if len(all_times_tamps_from_file) == 0:
            print("Couldn't find poses in file", poses_filename)
            print("Exiting from", __name__)
            exit(1)

    if indices is not None or len(indices) > 0:
        list_len = len(all_times_tamps_from_file)
        timestamps_list = list_len * [None]
        for count_idx, desired_idx in enumerate(indices):
            timestamps_list[count_idx] = all_times_tamps_from_file[desired_idx]
        return timestamps_list
    else:
        return all_times_tamps_from_file

def test_corner_detection_individually(model, calibrator, images_path_as_template, corner_extraction_args, img_indices):
    from omnistereo.calibration import draw_detected_points_manually
    calibrator.calibrate(model, images_path_as_template, img_indices, corner_extraction_args)
    draw_detected_points_manually(model.calibrator.omni_monos[0].omni_image, model.calibrator.omni_monos[0].image_points, -3, show = True)

#     print("DEBUG: %s image_points" % (model.mirror_name), model.calibrator.omni_monos[0].image_points)
#     print("DEBUG: %s Top obj_points" % (model.mirror_name), model.calibrator.omni_monos[0].obj_points)

def test_corner_detection_stereo(omnistereo_model, calibrator, images_path_as_template, corner_extraction_args_top, corner_extraction_args_bottom, img_indices):
    # Corner selection test:
    calibrator.calibrate(omnistereo_model, images_path_as_template, img_indices, corner_extraction_args_top, corner_extraction_args_bottom)

#     print("DEBUG: image_points", calibrator.calib_top.calibration_pairs[0].calib_img_top.image_points)
#     print("DEBUG: obj_points", calibrator.calib_bottom.calibration_pairs[0].calib_img_top.obj_points)

    if len(omnistereo_model.top_model.calibrator.omni_monos) > 0 and len(omnistereo_model.bot_model.calibrator.omni_monos) > 0:
        from omnistereo.calibration import draw_detected_points_manually
        calibrator.calib_top.omni_monos[0].visualize_points(window_name = omnistereo_model.panorama_top.name + " corners " + str(0))
        calibrator.calib_bottom.omni_monos[0].visualize_points(window_name = omnistereo_model.panorama_bot.name + " corners " + str(0))
        draw_detected_points_manually(calibrator.calib_top.omni_monos[0].omni_image, calibrator.calib_top.omni_monos[0].image_points, 5, show = True)
        draw_detected_points_manually(calibrator.calib_bottom.omni_monos[0].omni_image, calibrator.calib_bottom.omni_monos[0].image_points, 5, show = True)

def test_space2plane_mono(omni_model, point_3D_wrt_C, visualize = False, draw_fiducial_rings = False, z_offsets = [0]):
    # Arbitrary point in space

    offset_points = np.repeat(point_3D_wrt_C, len(z_offsets), axis = 0)
    offset_points[..., 2] = (offset_points[..., 2].T + z_offsets).T
    print("Projecting:",)
    u, v, uv_coords = omni_model.get_pixel_from_3D_point_wrt_C(offset_points)
    print("Space to Plane test: to (u,v)", uv_coords)

    max_num_of_rings_top = 4
    if visualize:
        import cv2
        from omnistereo.common_cv import draw_points
        # Make copy of omni_image
        img = omni_model.current_omni_img.copy()
        ring_count = 0
        fiducial_rings_radii = []
        for m in uv_coords:
            if ring_count < max_num_of_rings_top:
                draw_points(img, points_uv_coords = m[..., :2], color = (0, 0, 255), thickness = 10)

            if draw_fiducial_rings:
                # circle centers
                img_center_point = omni_model.precalib_params.center_point
                center_pixel = (int(img_center_point[0]), int(img_center_point[1]))
                cv2.circle(img, center_pixel, 3, (255, 0, 0), -1, 8, 0)

                if ring_count < max_num_of_rings_top:
                    # Draw top ring:
                    r = np.linalg.norm(m[..., :2] - img_center_point, axis = -1)
                    cv2.circle(img, center_pixel, int(r), (255, 0, 0), 3, 8, 0)
                    fiducial_rings_radii.append(r)
                ring_count += 1

            win_name = "TEST: Projection from 3D Space onto omnidirectional image"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, img)
            cv2.waitKey(1)

        return fiducial_rings_radii

def test_space2plane(omnistereo_model, point_3D_wrt_C, visualize = False, draw_fiducial_rings = False, z_offsets = [0]):
    # Arbitrary point in space

    offset_points = np.repeat(point_3D_wrt_C, len(z_offsets), axis = 0)
    offset_points[..., 2] = (offset_points[..., 2].T + z_offsets).T
    print("Projecting:",)
    u_top, v_top, m_top = omnistereo_model.top_model.get_pixel_from_3D_point_wrt_C(offset_points)
    print("TOP: Space to Plane test: to (u,v)", m_top)

    u_bottom, v_bottom, m_bottom = omnistereo_model.bot_model.get_pixel_from_3D_point_wrt_C(offset_points)
    print("BOTTOM: Space to Plane test: to (u,v)", m_bottom)

    max_num_of_rings_top = 4
    if visualize:
        import cv2
        from omnistereo.common_cv import draw_points
        # Make copy of omni_image
        img = omnistereo_model.current_omni_img.copy()
        ring_count = 0
        fiducial_rings_radii_top = []
        fiducial_rings_radii_bottom = []
        for m_t, m_b in zip(m_top, m_bottom):
            draw_points(img, points_uv_coords = m_b[..., :2], color = (255, 0, 0), thickness = 10)
            if ring_count < max_num_of_rings_top:
                draw_points(img, points_uv_coords = m_t[..., :2], color = (0, 0, 255), thickness = 10)

            if draw_fiducial_rings:
                # circle centers
                img_center_point_top = omnistereo_model.top_model.precalib_params.center_point
                center_top = (int(img_center_point_top[0]), int(img_center_point_top[1]))
                cv2.circle(img, center_top, 3, (255, 0, 0), -1, 8, 0)
                img_center_point_bottom = omnistereo_model.bot_model.precalib_params.center_point
                center_bottom = (int(img_center_point_bottom[0]), int(img_center_point_bottom[1]))
                cv2.circle(img, center_bottom, 3, (0, 0, 255), -1, 8, 0)

                # Draw bottom ring:
                r_bottom = np.linalg.norm(m_b[..., :2] - img_center_point_bottom, axis = -1)
                cv2.circle(img, center_bottom, int(r_bottom), (0, 0, 255), 3, 8, 0)
                fiducial_rings_radii_bottom.append(r_bottom)

            if ring_count < max_num_of_rings_top:
                # Draw top ring:
                r_top = np.linalg.norm(m_t[..., :2] - img_center_point_top, axis = -1)
                cv2.circle(img, center_top, int(r_top), (255, 0, 0), 3, 8, 0)
                fiducial_rings_radii_top.append(r_top)
            ring_count += 1

            win_name = "TEST: 3D Space to omni_image plane (Projection)"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, img)
            cv2.waitKey(1)

        return fiducial_rings_radii_top, fiducial_rings_radii_bottom

def test_pixel_lifting(gums):
    import omnistereo.euclid as euclid
    u_offset = -300
    v_offset = -300

    u0_top, v0_top = gums.top_model.get_center()
    (u, v) = (1200, 600)
#     u = u0_top + u_offset
#     v = v0_top + v_offset
    print("\nTOP Lifting test with pixel(%f,%f):" % (u, v))
    Ps_top = gums.top_model.lift_pixel_to_unit_sphere_wrt_focus(u, v, visualize = True, debug = True)
    X = Ps_top[..., 0]
    Y = Ps_top[..., 1]
    Z = Ps_top[..., 2]
    lifted_point_on_sphere_top = euclid.Point3(X, Y, Z)
    print("as point in sphere = %s with elevation of %f degrees" % (lifted_point_on_sphere_top, np.rad2deg(np.arcsin(lifted_point_on_sphere_top.z))))

    u0_bottom, v0_bottom = gums.bot_model.get_center()
    (u, v) = (1000, 600)
#     u = u0_bottom + u_offset
#     v = v0_bottom + v_offset
    print("\nBOTTOM Lifting test with pixel(%f,%f):" % (u, v))
    Ps_bottom = gums.bot_model.lift_pixel_to_unit_sphere_wrt_focus(u, v, visualize = True, debug = True)
    X = Ps_bottom[..., 0]
    Y = Ps_bottom[..., 1]
    Z = Ps_bottom[..., 2]
    lifted_point_on_sphere_bottom = euclid.Point3(X, Y, Z)
    print("as point in sphere = %s with elevation of %f degrees" % (lifted_point_on_sphere_bottom, np.rad2deg(np.arcsin(lifted_point_on_sphere_bottom.z))))

def convert_steradian_to_radian(steredian):
    # FIXME: choose the correct conversion
    radian = 2 * np.arccos(1 - (steredian / (2 * np.pi)))  # According to Wikipedia, which seems more correct!
#     radian = 2 * np.arcsin(np.sqrt(steredian / np.pi))  # According to the Self Study Manual on Optial Radiation Measurements
    return radian

def convert_resolution_units(pixel_length, in_pixels, in_radians, use_spatial_resolution, eta, in_2D = False):
    if in_2D:
        pow_factor = 1.0
    else:
        pow_factor = 2.0

    if use_spatial_resolution:
        if in_pixels:
            eta = eta / (pixel_length ** pow_factor)  # [mm/rad]/[mm/px] ==> [px/rad]
        if in_radians == False:
            eta = eta / ((180 / np.pi) ** pow_factor)  # [len / rad] / [deg/rad]  ==> [len / deg]
    else:
        if in_pixels:
            eta = eta * (pixel_length ** pow_factor)  # [rad/mm]*[mm/px] ==> [rad/px]
        if in_radians == False:
            eta = eta * ((180 / np.pi) ** pow_factor)  # [deg/rad] * [rad/len] ==> [deg/len]
    return eta

def unit_test(val1, val2, decimals):
    val1_approx = np.zeros_like(val1)
    val2_approx = np.zeros_like(val2)
    np.round(val1, decimals, val1_approx)
    np.round(val2, decimals, val2_approx)
    print("UNIT TEST:")
    print("val1:", val1_approx)
    print("val2:", val2_approx)
    print("val1 != val2? -->", "Pass!" if np.count_nonzero(val1_approx != val2_approx) == 0 else "Fail")

def get_list_of_T_Ggt_wrt_Cest_from_Rvicon_transforms(T_G_wrt_C_est_list, T_Ggt_wrt_Rvicon_as_ground_truth_list):
    # Compute all the transforms of the model's common frame [C] with respect to the Vicon's Rig frame [Rvicon] based on the estimated [G_g] poses wrt [C]
    from omnistereo import transformations as tr
    T_Cest_wrt_Rvicon_list = len(T_G_wrt_C_est_list) * [np.identity(4) * np.nan]
    idx = 0
    for T_Ggt_wrt_Rvicon, T_G_wrt_C_est in zip(T_Ggt_wrt_Rvicon_as_ground_truth_list, T_G_wrt_C_est_list):
        if not np.all(np.isnan(T_G_wrt_C_est)):
            T_Cest_wrt_G = tr.inverse_matrix(T_G_wrt_C_est)
            T_Cest_wrt_Rvicon = tr.concatenate_matrices(T_Ggt_wrt_Rvicon, T_Cest_wrt_G)
            T_Cest_wrt_Rvicon_list[idx] = T_Cest_wrt_Rvicon
        idx = idx + 1
    # Plot original frame situations:
    # draw_frame_poses(T_Ggt_wrt_Rvicon_as_ground_truth_list + T_G_wrt_C_est_list, show_grid=False, pause_for_each_frame=False)
    # Find the average Transform of omnistereo model common frame [C] wrt Vicon's Rig frame [Rvicon]
    #=======================================================================
    # T_Cest_wrt_Rvicon_avg = tr.pose_average(poses_list=T_Cest_wrt_Rvicon_list, weights=weights, use_birdal_method=True)
    # T_Ggt_wrt_Rvicon_from_Cest_transformed_list = [tr.concatenate_matrices(T_Cest_wrt_Rvicon_avg, T_G_wrt_Cest) for T_G_wrt_Cest in T_G_wrt_C_est_list ]
    # # and plot results for alignment comparison
    # draw_frame_poses(T_Ggt_wrt_Rvicon_as_ground_truth_list + T_Ggt_wrt_Rvicon_from_Cest_transformed_list, show_grid=False, pause_for_each_frame=False)
    #=======================================================================

    T_Cest_wrt_Rvicon_superimposed = np.identity(4) * np.nan  # Initialize solution as a matrix NaNs

    T_G_wrt_Cest_valid_list_of_tuples = [(idx, T_G_wrt_Cest) for idx, T_G_wrt_Cest in enumerate(T_G_wrt_C_est_list) if not np.all(np.isnan(T_G_wrt_Cest))]
    valid_indices_list = [idx for idx, T_G_wrt_Cest in T_G_wrt_Cest_valid_list_of_tuples]
    T_G_wrt_Cest_valid_list = [T_G_wrt_Cest for idx, T_G_wrt_Cest in T_G_wrt_Cest_valid_list_of_tuples]
    T_G_wrt_Rvicon_valid_list = [T_Ggt_wrt_Rvicon_as_ground_truth_list[idx] for idx in valid_indices_list]

    #=======================================================================
    # Alternatively (SEEMS better!), getting the T_Cest_wrt_Rvicon_avg via ICP
    # Let's use just the origins of the grid frames to see what we get.
    T_G_wrt_Cest_translations_as_list = [T_G_wrt_Cest[:3, 3] for T_G_wrt_Cest in T_G_wrt_Cest_valid_list]
    T_G_wrt_Cest_translations = np.array(T_G_wrt_Cest_translations_as_list).T
    T_G_wrt_Rvicon_translations_as_list = [T_Ggt_wrt_Rvicon_valid[:3, 3] for T_Ggt_wrt_Rvicon_valid in T_G_wrt_Rvicon_valid_list]
    T_G_wrt_Rvicon_translations = np.array(T_G_wrt_Rvicon_translations_as_list).T
    if len(valid_indices_list) >= 3:
        T_Cest_wrt_Rvicon_superimposed = tr.superimposition_matrix(v0 = T_G_wrt_Cest_translations, v1 = T_G_wrt_Rvicon_translations, scale = False, usesvd = True)
        # TEST: >>> scale, shear, angles, trans, persp = tr.decompose_matrix(T_Cest_wrt_Rvicon_superimposed)

    # Using origin and axes points (versors), because superposition can work with a single grid
    #=======================================================================
    axis_end_pts_homo = np.array([[1 , 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])  # Column vectors
    T_G_wrt_Cest_versor_points_separated = np.dot(T_G_wrt_Cest_valid_list, axis_end_pts_homo)
    # T_G_wrt_Cest_versor_points = np.hstack([T_G_wrt_C[:3] for T_G_wrt_C in T_G_wrt_Cest_versor_points_separated if not np.all(np.isnan(T_G_wrt_C))])
    T_G_wrt_Cest_versor_points = np.hstack([T_G_wrt_C[:3] for T_G_wrt_C in T_G_wrt_Cest_versor_points_separated])
    T_G_wrt_Rvicon_versor_points_separated = np.dot(T_G_wrt_Rvicon_valid_list, axis_end_pts_homo)
    # T_G_wrt_Rvicon_versor_points = np.hstack([T_G_wrt_R[:3] for T_G_wrt_R in T_G_wrt_Rvicon_versor_points_separated if not np.all(np.isnan(T_G_wrt_R))])
    T_G_wrt_Rvicon_versor_points = np.hstack([T_G_wrt_R[:3] for T_G_wrt_R in T_G_wrt_Rvicon_versor_points_separated])
    if len(T_G_wrt_Cest_versor_points) >= 3:
        T_Cest_wrt_Rvicon_superimposed = tr.superimposition_matrix(v0 = T_G_wrt_Cest_versor_points, v1 = T_G_wrt_Rvicon_versor_points, scale = False, usesvd = True)
    #=======================================================================
    #=======================================================================
    # Testing as wrt to [Rvicon]
    # T_Ggt_wrt_Rvicon_from_Cest_transformed_via_superimposition_list = [tr.concatenate_matrices(T_Cest_wrt_Rvicon_superimposed, T_G_wrt_Cest) for T_G_wrt_Cest in T_G_wrt_Cest_valid_list ]
    # draw_frame_poses(T_G_wrt_Rvicon_valid_list + T_Ggt_wrt_Rvicon_from_Cest_transformed_via_superimposition_list, show_grid=False, pause_for_each_frame=False)
    #=======================================================================
    #=======================================================================
    # Testing as wrt to [C] because [~C] is the reference frame used for CVPR data
    T_Rvicon_wrt_Cest_superimposed = tr.inverse_matrix(T_Cest_wrt_Rvicon_superimposed)
    # T_Ggt_wrt_Cest_from_Rvicon_transformed_via_superimposition_list = [tr.concatenate_matrices(T_Rvicon_wrt_Cest_superimposed, T_Ggt_wrt_Rvicon) for T_Ggt_wrt_Rvicon in T_G_wrt_Rvicon_valid_list ]
    T_Ggt_wrt_Cest_from_Rvicon_transformed_via_superimposition_list = len(T_G_wrt_C_est_list) * [None]
    for idx, T_Ggt_wrt_Rvicon in zip(valid_indices_list, T_G_wrt_Rvicon_valid_list):
        T_Ggt_wrt_Cest_from_Rvicon_transformed_via_superimposition_list[idx] = tr.concatenate_matrices(T_Rvicon_wrt_Cest_superimposed, T_Ggt_wrt_Rvicon)

    return T_Ggt_wrt_Cest_from_Rvicon_transformed_via_superimposition_list, T_Cest_wrt_Rvicon_superimposed
