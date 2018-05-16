# -*- coding: utf-8 -*-
# common_cv.py

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

'''
This module contains some common routines based on OpenCV
'''
from __future__ import print_function
from __future__ import division
import warnings
import numpy as np
import cv2
import os
from contextlib import contextmanager
import itertools as it
from omnistereo.common_tools import get_length_units_conversion_factor

image_extensions = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pbm', '.pgm', '.ppm']

def has_opencv():
    try:
        import cv2 as lib
        return True
    except ImportError:
        return False

def is_cv2():
    # if we are using OpenCV 2, then our cv2.__version__ will start
    # with '2.'
    return check_opencv_version("2.")

def is_cv3():
    # if we are using OpenCV 3.X, then our cv2.__version__ will start
    # with '3.'
    return check_opencv_version("3.")

def check_opencv_version(major, lib = None):
    # if the supplied library is None, import OpenCV
    if lib is None:
        try:
            import cv2 as lib
        except ImportError:
            return False
    # return whether or not the current OpenCV version matches the
    # major version number
    return lib.__version__.startswith(major)

def get_cv_img_as_RGB(image):
    cv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv_rgb

def clean_up(wait_key_time = 0):
    cv2.waitKey(wait_key_time)
    cv2.destroyAllWindows()

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def draw_str(img_dst, coords, s, font_size = 1.0, color_RGB = (255, 255, 255)):
    xc, yc = coords
    cv2.putText(img_dst, s, (xc + 1, yc + 1), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 0), thickness = 2, lineType = cv2.LINE_AA)
    cv2.putText(img_dst, s, (xc, yc), cv2.FONT_HERSHEY_PLAIN, font_size, rgb2bgr_color(color_RGB), lineType = cv2.LINE_AA)

def getsize(img):
    h, w = img.shape[:2]
    return w, h

def mdot(*args):
    from functools import reduce
    return reduce(np.dot, args)

def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
            xc, yc = kp.pt
            cv2.circle(vis, (int(xc), int(yc)), 2, color)

def filter_correspondences_manually(train_img, query_img, train_kpts, query_kpts, colors_RGB, first_row_to_crop_bottom = 0, do_filtering = False, win_name = 'Current Matches'):
    '''
    @param do_filtering: if True, the manual filtering is excuted, other wise only the matches are drawn and shown.
    '''
    valid_indices = np.ones(shape = (len(query_kpts)), dtype = "bool")  # All are True initially
    # Visualize inlier matches
    if train_img is not None:
        if train_img.ndim == 3:
            top_pano_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
        else:
            top_pano_gray = train_img.copy()
    top_pano_gray_vis = cv2.cvtColor(top_pano_gray, cv2.COLOR_GRAY2BGR)

    if query_img is not None:
        if query_img.ndim == 3:
            bot_pano_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        else:
            bot_pano_gray = query_img.copy()
    bot_pano_gray_vis = cv2.cvtColor(bot_pano_gray, cv2.COLOR_GRAY2BGR)

    row_offset = top_pano_gray.shape[0]  # Rows

    for top_kpt, bot_kpt, random_RGB_color in zip(train_kpts, query_kpts, colors_RGB):
        top_pano_gray_vis = cv2.drawKeypoints(top_pano_gray_vis, [top_kpt], outImage = top_pano_gray_vis, color = rgb2bgr_color(random_RGB_color))
        bot_pano_gray_vis = cv2.drawKeypoints(bot_pano_gray_vis, [bot_kpt], outImage = bot_pano_gray_vis, color = rgb2bgr_color(random_RGB_color))
    matches_img = np.vstack((top_pano_gray_vis, bot_pano_gray_vis[first_row_to_crop_bottom:, ...]))  # ATTENTION: Bottom panorama may be cropped.
    num_of_points = len(query_kpts)
    index_counter = 0
    while index_counter < num_of_points:
        top_kpt = train_kpts[index_counter]
        bot_kpt = query_kpts[index_counter]
        random_RGB_color = colors_RGB[index_counter]
        top_pt = (int(top_kpt.pt[0]), int(top_kpt.pt[1]))  # Recall, pt is given as (u,v)
        bot_pt = (int(bot_kpt.pt[0]), int(bot_kpt.pt[1] + row_offset - first_row_to_crop_bottom))
        if do_filtering:
            current_query_kpts = query_kpts[:index_counter + 1]
            current_train_kpts = train_kpts[:index_counter + 1]
            current_colors_RGB = colors_RGB[:index_counter + 1]
            _, matches_img = filter_correspondences_manually(train_img = train_img, query_img = query_img, query_kpts = current_query_kpts[valid_indices[:index_counter + 1]], train_kpts = current_train_kpts[valid_indices[:index_counter + 1]], colors_RGB = current_colors_RGB[valid_indices[:index_counter + 1]], first_row_to_crop_bottom = first_row_to_crop_bottom, do_filtering = False)
            ch_pressed_waitkey = cv2.waitKey(0)
            if (ch_pressed_waitkey & 255) == ord('v'):  # To save as VALID mactch
                valid_indices[index_counter] = True
                print("Added index %d to the valid list" % index_counter)
            elif (ch_pressed_waitkey & 255) == ord('r'):  # Rewind
                if index_counter > 0:
                    index_counter = index_counter - 1
                    valid_indices[index_counter] = True  # because it needs to be drawn again
                index_counter = index_counter - 1  # To ask again
            elif ch_pressed_waitkey == 27:  # Stop filtering at this point
                break
            else:
                valid_indices[index_counter] = False
        else:
            matches_img = cv2.line(matches_img, top_pt, bot_pt, color = rgb2bgr_color(random_RGB_color), thickness = 1, lineType = cv2.LINE_8)
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, matches_img)
            cv2.waitKey(1)
        index_counter += 1

    return valid_indices, matches_img

def filter_pixel_correspondences(matched_points_top, matched_points_bot, min_rectified_disparity, max_horizontal_diff):
    '''
    @param matched_points_top: A numpy array of points Not keypoints! Use "cv2.KeyPoint_convert(some_keypoints_list)" to get the keypoints as regular point coordinates
    @param matched_points_bot: A numpy array of points Not keypoints! Use "cv2.KeyPoint_convert(some_keypoints_list)" to get the keypoints as regular point coordinates
    @param min_rectified_disparity: Inclussive limit. This disparity helps to check for point correspondences with positive disparity (which should be the case for rectified stereo), such that (top_kpt.v_coord - bot_kpt.v_coord) >= min_rectified_disparity for the vertical stereo case
    @param max_horizontal_diff: Inclussive limit. This pixel distance on the u-axis. Ideally, this shouldn't be an issue for rectified panoramas, but it's useful to set when panoramas are not perfectly aligned.

    @return: a list of validity based on the checks performed

    '''
    if max_horizontal_diff > 0:
        validation_hor_diff = np.abs(matched_points_top[..., 0] - matched_points_bot[..., 0]) <= max_horizontal_diff
    else:  # no check, assume all true
        validation_hor_diff = np.ones(shape = (matched_points_top.shape[:-1]), dtype = "bool")

    if min_rectified_disparity >= 0:  # Not sure if this check should be conditional
        validation_min_disparity = matched_points_top[..., 1] - matched_points_bot[..., 1] >= min_rectified_disparity
        validation_list = np.logical_and(validation_hor_diff, validation_min_disparity)
    else:
        validation_list = validation_hor_diff

    return validation_list

class ArucoDetectionParamsTuner(object):

    def __init__(self, img, win_name = "Aruco Detection"):
        # NOTE: we can't pickle any of the Aruco objects, so we create class variables instead:
        # DEFAULT parameter values:
        #===================================================================
        self.adaptiveThreshConstant = 7.0
        self.adaptiveThreshWinSizeMax = 23
        self.adaptiveThreshWinSizeMin = 3
        self.adaptiveThreshWinSizeStep = 10
        self.cornerRefinementMaxIterations = 30
        self.cornerRefinementMinAccuracy = 0.1
        self.cornerRefinementWinSize = 5
        self.doCornerRefinement = False
        self.errorCorrectionRate = 0.6
        self.markerBorderBits = 1
        self.maxErroneousBitsInBorderRate = 0.35
        self.minMarkerPerimeterRate = 0.03
        self.maxMarkerPerimeterRate = 4.0
        self.minCornerDistanceRate = 0.05
        self.minDistanceToBorder = 3
        self.minMarkerDistanceRate = 0.05
        self.minOtsuStdDev = 5.0
        self.perspectiveRemoveIgnoredMarginPerCell = 0.13
        self.perspectiveRemovePixelPerCell = 4
        self.polygonalApproxAccuracyRate = 0.03
        #===================================================================

        self.window_name = win_name
        self.reset_images(img)

    def reset_images(self, img):
        self.img_raw = img
        self._setup_gui()
        self.needs_update = True

    def start_tuning(self, aruco_dictionary, win_name = "", save_file_name = "data/ArucoDetectionParamsTuner.pkl", tune_live = False):
        self.save_file_name = save_file_name
        if win_name:
            self.window_name = win_name
            self._setup_gui()

        ch_pressed_waitkey = cv2.waitKey(10)
        while not(ch_pressed_waitkey == 27):  # Pressing the Escape key breaks the loop
            ch_pressed_waitkey = cv2.waitKey(10)
            if (ch_pressed_waitkey & 255) == ord('s'):  # Save Tuner to pickle
                from omnistereo.common_tools import save_obj_in_pickle
                save_obj_in_pickle(self, self.save_file_name, locals())

            if self.needs_update:
                # We need to set the aruco_DetectorParameters object everytime, because we cannot picke it
                aruco_detection_params = self.get_current_aruco_detection_params()

                # Detect ChArUco corners:
                corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(self.img_raw, dictionary = aruco_dictionary, parameters = aruco_detection_params)

                vis_img_detectected_corners = self.img_raw.copy()
                vis_img_rejected_corners = self.img_raw.copy()
                if ids is not None and len(ids) > 0:
                    for i, c in enumerate(corners):
                        vis_img_detectected_corners = cv2.aruco.drawDetectedCornersCharuco(vis_img_detectected_corners, charucoCorners = c, cornerColor = (0, 255, 0))
                        #=======================================================
                        # print("Aruco id = %d" % (ids[i]))
                        # cv2.imshow(self.window_name, vis_img_detectected_corners)
                        # cv2.waitKey(0)
                        #=======================================================

                    vis_img_rejected_corners = vis_img_detectected_corners.copy()
                    for c in rejectedImgPoints:
                        vis_img_rejected_corners = cv2.aruco.drawDetectedCornersCharuco(vis_img_rejected_corners, charucoCorners = c, cornerColor = (0, 0, 255))

                cv2.imshow(self.window_name, vis_img_detectected_corners)
                cv2.imshow(self.window_name_rejected, vis_img_rejected_corners)

                self.needs_update = False

                if tune_live:
                    break

        return corners, ids, rejectedImgPoints

    def get_current_aruco_detection_params(self):
        aruco_detection_params = cv2.aruco.DetectorParameters_create()

        aruco_detection_params.adaptiveThreshConstant = self.adaptiveThreshConstant
        aruco_detection_params.adaptiveThreshWinSizeMax = self.adaptiveThreshWinSizeMax
        aruco_detection_params.adaptiveThreshWinSizeMin = self.adaptiveThreshWinSizeMin
        aruco_detection_params.adaptiveThreshWinSizeStep = self.adaptiveThreshWinSizeStep
        aruco_detection_params.cornerRefinementMaxIterations = self.cornerRefinementMaxIterations
        aruco_detection_params.cornerRefinementMinAccuracy = self.cornerRefinementMinAccuracy
        aruco_detection_params.cornerRefinementWinSize = self.cornerRefinementWinSize
        aruco_detection_params.doCornerRefinement = self.doCornerRefinement
        aruco_detection_params.errorCorrectionRate = self.errorCorrectionRate
        aruco_detection_params.markerBorderBits = self.markerBorderBits
        aruco_detection_params.maxErroneousBitsInBorderRate = self.maxErroneousBitsInBorderRate
        aruco_detection_params.minMarkerPerimeterRate = self.minMarkerPerimeterRate
        aruco_detection_params.maxMarkerPerimeterRate = self.maxMarkerPerimeterRate
        aruco_detection_params.minCornerDistanceRate = self.minCornerDistanceRate
        aruco_detection_params.minDistanceToBorder = self.minDistanceToBorder
        aruco_detection_params.minMarkerDistanceRate = self.minMarkerDistanceRate
        aruco_detection_params.minOtsuStdDev = self.minOtsuStdDev
        aruco_detection_params.perspectiveRemoveIgnoredMarginPerCell = self.perspectiveRemoveIgnoredMarginPerCell
        aruco_detection_params.perspectiveRemovePixelPerCell = self.perspectiveRemovePixelPerCell
        aruco_detection_params.polygonalApproxAccuracyRate = self.polygonalApproxAccuracyRate

        return aruco_detection_params

    def _setup_gui(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.window_name_rejected = self.window_name + " - Rejected"
        cv2.namedWindow(self.window_name_rejected, cv2.WINDOW_NORMAL)

        # General Trackbars (sliders)
        self.button_name_doCornerRefinement = "Do corner refinement"
        cv2.createTrackbar(self.button_name_doCornerRefinement, self.window_name, self.doCornerRefinement, 1, self.doCornerRefinement_callback)

        self.tb_name_adaptiveThreshConstant = "Adaptive Threshold Constant"
        cv2.createTrackbar(self.tb_name_adaptiveThreshConstant, self.window_name, int(self.adaptiveThreshConstant), 100, self.adaptiveThreshConstant_callback)

        self.tb_name_adaptiveThreshWinSizeMax = "Adaptive Threshold WinSizeMax"
        cv2.createTrackbar(self.tb_name_adaptiveThreshWinSizeMax, self.window_name, self.adaptiveThreshWinSizeMax, 100, self.adaptiveThreshWinSizeMax_callback)

        self.tb_name_adaptiveThreshWinSizeMin = "Adaptive Thresh WinSizeMin"
        cv2.createTrackbar(self.tb_name_adaptiveThreshWinSizeMin, self.window_name, self.adaptiveThreshWinSizeMin, 100, self.adaptiveThreshWinSizeMin_callback)

        self.tb_name_adaptiveThreshWinSizeStep = "Adaptive Thresh WinSizeStep"
        cv2.createTrackbar(self.tb_name_adaptiveThreshWinSizeStep, self.window_name, self.adaptiveThreshWinSizeStep, 100, self.adaptiveThreshWinSizeStep_callback)

        self.tb_name_cornerRefinementMaxIterations = "Corner Refinement Max Iterations"
        cv2.createTrackbar(self.tb_name_cornerRefinementMaxIterations, self.window_name, self.cornerRefinementMaxIterations, 100, self.cornerRefinementMaxIterations_callback)

        self.tb_name_cornerRefinementMinAccuracy = "Corner Refinement Min Accuracy (x100)"
        cv2.createTrackbar(self.tb_name_cornerRefinementMinAccuracy, self.window_name, int(100 * self.cornerRefinementMinAccuracy), 100, self.cornerRefinementMinAccuracy_callback)

        self.tb_name_cornerRefinementWinSize = "Corner Refinement Win Size"
        cv2.createTrackbar(self.tb_name_cornerRefinementWinSize, self.window_name, self.cornerRefinementWinSize, 100, self.cornerRefinementWinSize_callback)

        self.tb_name_errorCorrectionRate = "Error Correction Rate (x10)"
        cv2.createTrackbar(self.tb_name_errorCorrectionRate, self.window_name, int(10 * self.errorCorrectionRate), 100, self.errorCorrectionRate_callback)

        self.tb_name_markerBorderBits = "Marker Border Bits"
        cv2.createTrackbar(self.tb_name_markerBorderBits, self.window_name, self.markerBorderBits, 10, self.markerBorderBits_callback)

        self.tb_name_maxErroneousBitsInBorderRate = "Max Erroneous Bits In Border Rate (x100)"
        cv2.createTrackbar(self.tb_name_maxErroneousBitsInBorderRate, self.window_name, int(100 * self.maxErroneousBitsInBorderRate), 100, self.maxErroneousBitsInBorderRate_callback)

        self.tb_name_minMarkerPerimeterRate = "Min Marker Perimeter Rate (x100)"
        cv2.createTrackbar(self.tb_name_minMarkerPerimeterRate, self.window_name, int(100 * self.minMarkerPerimeterRate), 100, self.minMarkerPerimeterRate_callback)

        self.tb_name_maxMarkerPerimeterRate = "Max Marker Perimeter Rate (x100)"
        cv2.createTrackbar(self.tb_name_maxMarkerPerimeterRate, self.window_name, int(100 * self.maxMarkerPerimeterRate), 1000, self.maxMarkerPerimeterRate_callback)

        self.tb_name_minCornerDistanceRate = "Min Corner Distance Rate (x100)"
        cv2.createTrackbar(self.tb_name_minCornerDistanceRate, self.window_name, int(100 * self.minCornerDistanceRate), 100, self.minCornerDistanceRate_callback)

        self.tb_name_minDistanceToBorder = "Min Distance To Border"
        cv2.createTrackbar(self.tb_name_minDistanceToBorder, self.window_name, self.minDistanceToBorder, 100, self.minDistanceToBorder_callback)

        self.tb_name_minMarkerDistanceRate = "Min Marker Distance Rate (x100)"
        cv2.createTrackbar(self.tb_name_minMarkerDistanceRate, self.window_name, int(100 * self.minMarkerDistanceRate), 100, self.minMarkerDistanceRate_callback)

        self.tb_name_minOtsuStdDev = "Min Otsu Std. Dev."
        cv2.createTrackbar(self.tb_name_minOtsuStdDev, self.window_name, int(self.minOtsuStdDev), 100, self.minOtsuStdDev_callback)

        self.tb_name_perspectiveRemoveIgnoredMarginPerCell = "Perspective Remove Ignored Margin Per Cell (x100)"
        cv2.createTrackbar(self.tb_name_perspectiveRemoveIgnoredMarginPerCell, self.window_name, int(100 * self.perspectiveRemoveIgnoredMarginPerCell), 50, self.perspectiveRemoveIgnoredMarginPerCell_callback)

        self.tb_name_perspectiveRemovePixelPerCell = "Perspective Remove Pixel Per Cell"
        cv2.createTrackbar(self.tb_name_perspectiveRemovePixelPerCell, self.window_name, self.perspectiveRemovePixelPerCell, 100, self.perspectiveRemovePixelPerCell_callback)

        self.tb_name_polygonalApproxAccuracyRate = "Polygonal Approx. Accuracy Rate (x100)"
        cv2.createTrackbar(self.tb_name_polygonalApproxAccuracyRate, self.window_name, int(100 * self.polygonalApproxAccuracyRate), 100, self.polygonalApproxAccuracyRate_callback)

    def doCornerRefinement_callback(self, pos):
        if pos == 0:
            self.doCornerRefinement = False
        else:
            self.doCornerRefinement = True

        self.needs_update = True

    def adaptiveThreshConstant_callback(self, pos):
        self.adaptiveThreshConstant = pos
        self.needs_update = True

    def adaptiveThreshWinSizeMax_callback(self, pos):
        if pos > self.adaptiveThreshWinSizeMin:
            self.adaptiveThreshWinSizeMax = pos
            self.needs_update = True

    def adaptiveThreshWinSizeMin_callback(self, pos):
        if 2 < pos < self.adaptiveThreshWinSizeMax:
            self.adaptiveThreshWinSizeMin = pos
            self.needs_update = True

    def adaptiveThreshWinSizeStep_callback(self, pos):
        if 0 < pos:
            self.adaptiveThreshWinSizeStep = pos
            self.needs_update = True

    def cornerRefinementMaxIterations_callback(self, pos):
        if pos > 0:
            self.adaptiveThreshWinSizeStep = pos
            self.needs_update = True

    def cornerRefinementMinAccuracy_callback(self, pos):
        self.cornerRefinementMinAccuracy = float(pos / 100.)
        self.needs_update = True

    def cornerRefinementWinSize_callback(self, pos):
        self.cornerRefinementWinSize = pos
        self.needs_update = True

    def errorCorrectionRate_callback(self, pos):
        self.errorCorrectionRate = float(pos / 10.)
        self.needs_update = True

    def markerBorderBits_callback(self, pos):
        if pos > 0:
            self.markerBorderBits = pos
            self.needs_update = True

    def maxErroneousBitsInBorderRate_callback(self, pos):
        self.maxErroneousBitsInBorderRate = float(pos / 100.)
        self.needs_update = True

    def minMarkerPerimeterRate_callback(self, pos):
        if pos > 0:
            self.minMarkerPerimeterRate = float(pos / 100.)
            self.needs_update = True

    def maxMarkerPerimeterRate_callback(self, pos):
        if pos > 0:
            self.maxMarkerPerimeterRate = float(pos / 100.)
            self.needs_update = True

    def minCornerDistanceRate_callback(self, pos):
        self.minCornerDistanceRate = float(pos / 100.)
        self.needs_update = True

    def minDistanceToBorder_callback(self, pos):
        self.minDistanceToBorder = pos
        self.needs_update = True

    def minMarkerDistanceRate_callback(self, pos):
        self.minMarkerDistanceRate = float(pos / 100.)
        self.needs_update = True

    def minOtsuStdDev_callback(self, pos):
        self.minOtsuStdDev = float(pos)
        self.needs_update = True

    def perspectiveRemoveIgnoredMarginPerCell_callback(self, pos):
        self.perspectiveRemoveIgnoredMarginPerCell = float(pos / 100.)
        self.needs_update = True

    def perspectiveRemovePixelPerCell_callback(self, pos):
        if pos > 0:
            self.perspectiveRemovePixelPerCell = pos
            self.needs_update = True

    def polygonalApproxAccuracyRate_callback(self, pos):
        if pos > 0:
            self.polygonalApproxAccuracyRate = float(pos / 100.)
            self.needs_update = True

class StereoMatchTuner(object):

    def __init__(self, left_img, right_img, rotate_images = False, method = "sgbm", win_name = "Disparity Map", disp_first_valid_row = 0, disp_last_valid_row = -1):
        self.reset_images(left_img, right_img, rotate_images)
        self.window_name = win_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.disp_first_valid_row = disp_first_valid_row
        self.disp_last_valid_row = disp_last_valid_row
        self.matching_method = method
        self.disparity_map = None
        self.disparity_img = None
        self.needs_update = True
        self.pano_mask = None
        # TODO: try several presets available:
        # //   CV_STEREO_BM_NORMALIZED_RESPONSE,
        # //   CV_STEREO_BM_BASIC,
        # //   CV_STEREO_BM_FISH_EYE,
        # //   CV_STEREO_BM_NARROW
        # // the preset is one of ..._PRESET above.
        # // the disparity search range. For each pixel algorithm will find the best disparity from 0 (default minimum disparity) to n disparities. The search range can then be shifted by changing the minimum disparity.

        if self.matching_method == "bm":
            self.texture_threshold = 10
            self.pre_filter_size = 9

        if self.matching_method == "sgbm":
            # Parameters for SYNTHETIC experiments:
            self.smooth_P1 = 50  # The first parameter controlling the disparity smoothness", P1, 0, 4096
            self.smooth_P2 = 1000  # The second parameter controlling the disparity smoothness", P2, 0, 32768
            self.full_dyn_prog = False  # not used by VAR stereo) Use Dynamic Programming? (DP uses more memory)", True)

        if self.matching_method == "bm" or self.matching_method == "sgbm":
            self.SAD_window_size = 7  # Matched block size. It must be an odd number >=1", SAD_window_size_value, 1, 128)
            self.median_kernel_size = 0
            self.num_of_disparities = 64  # The size of the disparity search window. Together with min_disparity, this defines the horopter (the 3D volume that is visible to the stereo algorithm). This parameter must be divisible by 16", Range: 16, 256) # MAX disp shouldn't exceed the image's height
            self.min_disparity = 1  # Minimum disparity (controls the offset from the x-position of the left pixel at which to begin searching)" (Range: -128, 128)
            self.disp_12_max_diff = -1  # left.shape[1] // 2  # TODO: compute with the nearest point: Maximum allowed difference (in integer pixel units) in the left-right disparity check (How many pixels to slide the window over). The larger it is, the larger the range of visible depths, but more computation is required. Set it to a non-positive value to disable the check.", Range: (-400, 400)
            self.pre_filter_cap = 31  # Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval.", (1, 256)
            self.uniqueness_ratio = 15  # Margin in percentage by which the best (minimum) computed cost function value should win the second best value to consider the found match correct", (1, 100). Normally, a value within the 5-15 range is good enough
            self.speckle_window_size = 0  # Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.", (0, 600)
            self.speckle_range = 0  # Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
            self.use_WLS_filter = False  # Using the WLS filter from the contributed ximgproc module
            self.wls_lambda = 8000
            self.wls_sigma = 100  # This value is divided by 100. when set

            #===================================================================
            # # Parameters For REAL panoramas
            # self.median_kernel_size = 5
            # self.SAD_window_size = 10  # Matched block size. It must be an odd number >=1", SAD_window_size_value, 1, 128)
            # self.min_disparity = 16 - 129  # Minimum disparity (controls the offset from the x-position of the left pixel at which to begin searching)" (Range: -128, 128)
            # # self.num_of_disparities = ((self.rows/8) + 15) & -16
            # self.num_of_disparities = 240  # The size of the disparity search window. Together with min_disparity, this defines the horopter (the 3D volume that is visible to the stereo algorithm). This parameter must be divisible by 16", Range: 16, 256) # MAX disp shouldn't exceed the image's height
            # self.disp_12_max_diff = 0  # self.num_of_disparities - self.min_disparity  # left.shape[1] // 2  # TODO: compute with the nearest point: Maximum allowed difference (in integer pixel units) in the left-right disparity check (How many pixels to slide the window over). The larger it is, the larger the range of visible depths, but more computation is required. Set it to a non-positive value to disable the check.", Range: (-400, 400)
            # self.smooth_P1 = 4096  # 50 The first parameter controlling the disparity smoothness", P1, 0, 4096
            # self.smooth_P2 = 32768  # 100 The second parameter controlling the disparity smoothness", P2, 0, 32768
            # self.pre_filter_cap = 0  # Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval.", (1, 256)
            # self.uniqueness_ratio = 8  # Margin in percentage by which the best (minimum) computed cost function value should win the second best value to consider the found match correct", (1, 100)
            # self.speckle_window_size = 425  # Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.", (0, 600)
            # self.speckle_range = 16  # Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value.", (0, 32)
            # self.full_dyn_prog = True  # not used by VAR stereo) Use Dynamic Programming? (DP uses more memory)", True)
            #===================================================================
        if self.matching_method == "var":
            self.median_kernel_size = 5
            self.min_disparity = 128 - 128
            self.max_disparity = 192  # Default is 16
            self.var_cycles_dict = {"CYCLE_0":0, "CYCLE_V":1}
            self.var_cycle = "CYCLE_V"  #  "CYCLE_O", O-cycle or null-cycle: performs significantly more iterations on the coarse grids (faster convergence)
                                # "CYCLE_V" : The V-cycles makes one recursive call of a two-grid cycle per level.")
            self.var_penalizations_dict = {"P_TICHONOV":0, "P_CHARBONNIER":1, "P_PERONA_MALIK":2}
            self.var_penalization = "P_CHARBONNIER"  # Penalization regulizer method: "P_TICHONOV",  "P_CHARBONNIER", "P_PERONA_MALIK"
            self.var_levels = 8  # number of multigrid levels" (1, 10)
            self.var_pyrScale = 0.80  # VAR stereo: pyramid scale", (0.4, 1.0) Default: 0.5
            self.var_poly_n = 5  # degree of polynomial (see paper)", (1, 7) Default: 3
            self.var_poly_sigma = 0.64  # sigma value in polynomial (see paper) (0, 1.0)  # TODO: Find proper bounds
            self.var_fi = 80  # fi value (see paper)", (1.0, 100.0) Default: 25.
            self.var_lambda = 1.1  # lambda value (see paper): (0, 2.0) Default: 0.03 # TODO: Find proper bounds
            self.var_nIt = 15  # The number of iterations the algorithm does at each pyramid level.  (If the flag USE_SMART_ID is set, the number of iterations will be redistributed in such a way, that more iterations will be done on more coarser levels.)

            # TODO:
#             VAR_flag_auto_params = True
#             VAR_flag_init_disp = False  # USE_INITIAL_DISPARITY?", False
#             VAR_flag_eq_hist = False  # USE_EQUALIZE_HIST?", False
#             VAR_flag_smart_id = False  # USE_SMART_ID?", True
#             VAR_flag_median_filter = False  # USE_MEDIAN_FILTERING?", True
#             var_auto = 0;
#             var_init_params = 0;
#             var_eq_hist = 0;
#             var_smart_id = 0;
#             var_median_filter = 0;
#             if VAR_flag_auto_params:
#                 var_auto = cv2.StereoVar.USE_AUTO_PARAMS
#             if VAR_flag_init_disp:
#                 var_init_params = cv2.StereoVar.USE_INITIAL_DISPARITY
#             if VAR_flag_eq_hist:
#                 var_eq_hist = cv2.StereoVar.USE_EQUALIZE_HIST
#             if VAR_flag_smart_id:
#                 var_smart_id = cv2.StereoVar.USE_SMART_ID
#             if VAR_flag_median_filter:
#                 var_median_filter = cv2.StereoVar.USE_MEDIAN_FILTERING

#             var_flags = var_auto | var_init_params | var_eq_hist | var_smart_id | var_median_filter;
#             panoramic_stereo.flags = var_flags

        self._setup_gui()

    def reset_images(self, left_img, right_img, rotate_images = True, disp_first_valid_row = 0, disp_last_valid_row = -1):
        self.left_raw = left_img
        self.right_raw = right_img
        self.rotate_images = rotate_images
        self.disp_first_valid_row = disp_first_valid_row
        self.disp_last_valid_row = disp_last_valid_row
        self.rows, self.cols = left_img.shape[0:2]
        self.needs_update = True

    def start_tuning(self, win_name = "", save_file_name = "data/StereoMatchTuner.pkl", tune_live = False, pano_mask = None, use_heat_map = True):
        self.pano_mask = pano_mask
        self.save_file_name = save_file_name
        if win_name:
            self.window_name = win_name
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._setup_gui()

#         if self.matching_method == "sgbm":
        self.disparity_map = np.zeros((self.rows, self.cols), dtype = "float64")
#         else:
#             self.disparity_map = np.zeros((self.rows, self.cols), dtype="float32")

        self.disparity_img = np.zeros((self.rows, self.cols), dtype = "uint8")

#===============================================================================
#         if self.matching_method == "bm":
# #             panoramic_stereo = cv2.StereoBM()
#             ch_pressed_waitkey = cv2.waitKey(10)
#
#             while not(ch_pressed_waitkey == 27):  # Pressing the Escape key breaks the loop
#                 ch_pressed_waitkey = cv2.waitKey(10)
#                 if (ch_pressed_waitkey & 255) == ord('s'):  # Save Tuner to pickle
#                     from omnistereo.common_tools import save_obj_in_pickle
#                     save_obj_in_pickle(self, self.save_file_name, locals())
#
#                 if self.needs_update:
#                     panoramic_stereo = cv2.StereoBM_create(numDisparities=self.num_of_disparities, blockSize=self.SAD_window_size)
#                     disparity = panoramic_stereo.compute(self.left, self.right)
#                     disparity_rotated = cv2.flip(cv2.transpose(disparity), flipCode=1)
#                     # It contains disparity values scaled by 16. So, to get the floating-point disparity map, you need to divide each disp element by 16.
# #                     self.disparity_map = disparity_rotated / 16. + 1
# #                     self.disparity_img = np.uint8((self.disparity_map - self.disparity_map.min()) * 255. / (self.disparity_map.max() - self.disparity_map.min()))
# #                     self.disparity_map[self.disp_first_valid_row:] = (disparity_rotated / 16.)[self.disp_first_valid_row:]
# #                     self.disparity_img[self.disp_first_valid_row:] = np.uint8((self.disparity_map[self.disp_first_valid_row:] - self.disparity_map.min()) * 255. / (self.disparity_map.max() - self.disparity_map.min()))
#                     self.disparity_map = (disparity_rotated / 16.)
#                     self.disparity_img = np.uint8((self.disparity_map - self.disparity_map.min()) * 255. / (self.disparity_map.max() - self.disparity_map.min()))
#
#                     disp_img_color = cv2.cvtColor(self.disparity_img, cv2.COLOR_GRAY2BGR)
#                     # Draw min/max bounds (valid region of disparities)
#                     line_thickness = 2
#                     line_color = (0, 255, 0)  # Green in BGR
# #                     cv2.line(img=disp_img_color, pt1=(0, self.disp_first_valid_row), pt2=(self.cols - 1, self.disp_first_valid_row), color=line_color, thickness=line_thickness , lineType=cv2.LINE_AA)
# #                     stereo_match_view = np.vstack((disp_img_color[self.disp_first_valid_row:self.disp_last_valid_row + 1], self.divider_line_img, self.stereo_view_img))
#                     stereo_match_view = np.vstack((disp_img_color, self.divider_line_img, self.stereo_view_img))
#                     cv2.imshow(self.window_name, stereo_match_view)
#                     self.needs_update = False
#
#                     if tune_live:
#                         break
#===============================================================================

        if self.matching_method == "bm" or self.matching_method == "sgbm":
            number_of_image_channels = self.left.ndim
            ch_pressed_waitkey = cv2.waitKey(10)
            while not(ch_pressed_waitkey == 27):  # Pressing the Escape key breaks the loop
                ch_pressed_waitkey = cv2.waitKey(10)
                if (ch_pressed_waitkey & 255) == ord('s'):  # Save Tuner to pickle
                    from omnistereo.common_tools import save_obj_in_pickle
                    save_obj_in_pickle(self, self.save_file_name, locals())

                if self.needs_update:
                    if self.matching_method == "sgbm":
                        if self.full_dyn_prog:
                            mode = cv2.STEREO_SGBM_MODE_HH  # to run the full-scale two-pass dynamic programming algorithm
                        else:
                            mode = cv2.STEREO_SGBM_MODE_SGBM
    #                         mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

                        # self.smooth_P1 = 2 * number_of_image_channels * self.SAD_window_size * self.SAD_window_size
                        # self.smooth_P2 = 8 * number_of_image_channels * self.SAD_window_size * self.SAD_window_size
                        # panoramic_stereo.P1 = self.smooth_P1
                        # panoramic_stereo.P2 = self.smooth_P2
                        panoramic_stereo = cv2.StereoSGBM_create(minDisparity = self.min_disparity, numDisparities = self.num_of_disparities, blockSize = self.SAD_window_size, \
                                                                 P1 = 0, P2 = 0, disp12MaxDiff = self.disp_12_max_diff, preFilterCap = self.pre_filter_cap, uniquenessRatio = self.uniqueness_ratio, \
                                                                 speckleWindowSize = self.speckle_window_size, speckleRange = self.speckle_range, mode = mode\
                                                                 )

    #                     panoramic_stereo.minDisparity = self.min_disparity
    #                     panoramic_stereo.numDisparities = self.num_of_disparities
    #                     panoramic_stereo.SADWindowSize = self.SAD_window_size
                        #===========================================================
                        # panoramic_stereo.disp12MaxDiff = self.disp_12_max_diff  # TODO: value for nearest triangulated point
                        # panoramic_stereo.preFilterCap = self.pre_filter_cap
                        # panoramic_stereo.speckleRange = self.speckle_range
                        # panoramic_stereo.speckleWindowSize = self.speckle_window_size
                        # panoramic_stereo.uniquenessRatio = self.uniqueness_ratio
                        # panoramic_stereo.fullDP = self.full_dyn_prog
                        #===========================================================

                    elif self.matching_method == "bm":
                        panoramic_stereo = cv2.StereoBM_create(numDisparities = self.num_of_disparities, blockSize = self.SAD_window_size)
                        panoramic_stereo.setMinDisparity(self.min_disparity)
                        panoramic_stereo.setDisp12MaxDiff(self.disp_12_max_diff)
                        if 0 < self.pre_filter_cap < self.num_of_disparities:
                            panoramic_stereo.setPreFilterCap(self.pre_filter_cap)
                        panoramic_stereo.setPreFilterSize(self.pre_filter_size)
                        panoramic_stereo.setUniquenessRatio(self.uniqueness_ratio)
                        panoramic_stereo.setSpeckleWindowSize(self.speckle_window_size)
                        panoramic_stereo.setSpeckleRange(self.speckle_range)
                        panoramic_stereo.setTextureThreshold(self.texture_threshold)

                    disparity = panoramic_stereo.compute(self.left, self.right)

                    if self.use_WLS_filter:
                        from cv2.ximgproc import createDisparityWLSFilter, createRightMatcher
                        panoramic_stereo_WLS = createDisparityWLSFilter(panoramic_stereo)
                        panoramic_stereo_WLS.setLambda(self.wls_lambda)
                        panoramic_stereo_WLS.setSigmaColor(self.wls_sigma / 100.)

                        right_matcher = createRightMatcher(panoramic_stereo)
                        right_disparity = right_matcher.compute(self.right, self.left)

                        filtered_disparity_map = panoramic_stereo_WLS.filter(disparity_map_left = disparity, left_view = self.left, disparity_map_right = right_disparity, right_view = self.right)  # TODO: set optional ROI param?
                        disparity_rotated = cv2.flip(cv2.transpose(filtered_disparity_map), flipCode = 1)
                    else:
                        disparity_rotated = cv2.flip(cv2.transpose(disparity), flipCode = 1)
                    # It contains disparity values scaled by 16. So, to get the floating-point disparity map, you need to divide each disp element by 16.
#                     self.disparity_map[self.disp_first_valid_row:self.disp_last_valid_row + 1] = (disparity_rotated / 16.)[self.disp_first_valid_row:self.disp_last_valid_row + 1]
#                     self.disparity_img[self.disp_first_valid_row:self.disp_last_valid_row + 1] = np.uint8((self.disparity_map[self.disp_first_valid_row:self.disp_last_valid_row + 1] - self.disparity_map.min()) * 255. / (self.disparity_map.max() - self.disparity_map.min()))
#                     self.disparity_map[self.disp_first_valid_row:] = (disparity_rotated / 16.)[self.disp_first_valid_row:]
#                     self.disparity_img[self.disp_first_valid_row:] = np.uint8((self.disparity_map[self.disp_first_valid_row:] - self.disparity_map.min()) * 255. / (self.disparity_map.max() - self.disparity_map.min()))
                    disparity_map_normalized = (disparity_rotated / 16.)
                    # It's not good to do the normalization as follows because it becomes inconsistent:
                    # disparity_img_normalized = np.uint8((disparity_map_normalized - disparity_map_normalized.min()) * 255. / (disparity_map_normalized.max() - disparity_map_normalized.min()))
                    if self.use_WLS_filter:
                        min_disp_for_vis = -1  # disparity_map_normalized.min()  # because it could be -1
                    else:
                        min_disp_for_vis = self.min_disparity
                    disparity_img_normalized = np.uint8((disparity_map_normalized - min_disp_for_vis) * 255. / (self.num_of_disparities - min_disp_for_vis))

                    # (my TRICK) Filter out of bound values by applying panoramic mask to the disparity image and depth map using radial bounds:
                    self.disparity_map = np.zeros_like(disparity_map_normalized)
                    self.disparity_map = cv2.bitwise_and(src1 = disparity_map_normalized, src2 = disparity_map_normalized, dst = self.disparity_map, mask = self.pano_mask)
                    self.disparity_img = np.zeros_like(disparity_img_normalized)
                    self.disparity_img = cv2.bitwise_and(src1 = disparity_img_normalized, src2 = disparity_img_normalized, dst = self.disparity_img, mask = self.pano_mask)

                    if use_heat_map:
                        disp_img_color = cv2.applyColorMap(self.disparity_img, cv2.COLORMAP_JET)
                        disp_img_color[self.disparity_map < self.min_disparity] = 0  # Keep invalid disparity as black
                    else:
                        disp_img_color = cv2.cvtColor(self.disparity_img, cv2.COLOR_GRAY2BGR)

                    # Draw min/max bounds (valid region of disparities)
                    line_thickness = 2
                    line_color = (0, 255, 0)  # Green in BGR
#                     cv2.line(img=disp_img_color, pt1=(0, self.disp_first_valid_row), pt2=(self.cols - 1, self.disp_first_valid_row), color=line_color, thickness=line_thickness , lineType=cv2.LINE_AA)

#                     stereo_match_view = np.vstack((disp_img_color[self.disp_first_valid_row:self.disp_last_valid_row + 1], self.divider_line_img, self.stereo_view_img))
#                     stereo_match_view = np.vstack((disp_img_color[self.disp_first_valid_row:], self.divider_line_img, self.stereo_view_img))
                    stereo_match_view = np.vstack((disp_img_color, self.divider_line_img, self.stereo_view_img))

                    cv2.imshow(self.window_name, stereo_match_view)
                    self.needs_update = False

                    if tune_live:
                        break

        # NOTE: not longer supported by OpenCV 3
        # WARNING: it seems that one needs to run it twice (at first) for the algorithm to work better (as expected)
#===============================================================================
#         if self.matching_method == "var":
#             panoramic_stereo = cv2.StereoVar()
# #             while count < 2:
#             while not(cv2.waitKey(10) == 27):
#                 if self.needs_update:
#                     panoramic_stereo.levels = self.var_levels
#                     panoramic_stereo.pyrScale = self.var_pyrScale
#                     panoramic_stereo.nIt = self.var_nIt
#                     panoramic_stereo.minDisp = self.min_disparity
#                     panoramic_stereo.maxDisp = self.max_disparity
#                     panoramic_stereo.poly_n = self.var_poly_n
#                     panoramic_stereo.poly_sigma = self.var_poly_sigma
#                     panoramic_stereo.fi = self.var_fi
#                     # panoramic_stereo.lambda  = self.var_lambda # TODO: not allowed to call this in Python, but for some reason the class has it
#                     panoramic_stereo.penalization = self.var_penalizations_dict[self.var_penalization]
#                     panoramic_stereo.cycle = self.var_cycles_dict[self.var_cycle]
#
#                     # panoramic_stereo.flags = var_flags # TODO: How?
#
#                     disparity_rotated = panoramic_stereo.compute(self.left, self.right)
#                     num_of_disparities = float(panoramic_stereo.maxDisp - panoramic_stereo.minDisp)
#     #                 self.disparity_img = np.zeros_like(self.disparity_map, dtype=np.uint8)
#     #                 self.disparity_img = np.uint8((self.disparity_map + 1) * 255. / num_of_disparities)
#                     self.disparity_map[self.disp_first_valid_row:] = cv2.flip(cv2.transpose(disparity_rotated), flipCode=1)[self.disp_first_valid_row:]
#                     # Recall that ranges don't include the last index (so we want to include it with +1)
#                     # FIXME: it seems that the some rows are exceeding the rows of the image while resolving the match coordinates on the other panorama
#                     self.disparity_img[self.disp_first_valid_row:] = np.uint8((self.disparity_map[self.disp_first_valid_row:] - self.disparity_map.min()) * 255. / (self.disparity_map.max() - self.disparity_map.min()))
#
#                     disp_img_color = cv2.cvtColor(self.disparity_img, cv2.COLOR_GRAY2BGR)
#                     stereo_match_view = np.vstack((disp_img_color[self.disp_first_valid_row:], self.divider_line_img, self.stereo_view_img))
#                     cv2.imshow(self.window_name, stereo_match_view)
#                     self.needs_update = False
#                     if tune_live:
#                         break
#                     # FIXME: it seems that the image buffer (or the np.array is being reused while displaying)
#                     # Also, related to above problem forcing to run it twice (at first)
#                     # TODO: crop the noise bottom results from the depth map (since anything beyond the black stripe is incorrect)
#===============================================================================

        return self.disparity_map, self.disparity_img

    def _setup_gui(self):
        # General Trackbars (sliders)
        self.tb_name_median_filter = "Median Filter"
        cv2.createTrackbar(self.tb_name_median_filter, self.window_name, self.median_kernel_size, 32, self.on_median_filter_callback)

        if self.matching_method == "bm" or self.matching_method == "sgbm":
            self.tb_name_SAD_window_size = "SAD window size"
            cv2.createTrackbar(self.tb_name_SAD_window_size, self.window_name, self.SAD_window_size, int(self.rows / 2), self.on_SAD_win_size_callback)
            self.tb_name_num_of_disparities = "N Disparities"
            cv2.createTrackbar(self.tb_name_num_of_disparities, self.window_name, self.num_of_disparities, (self.rows // 16) * 16, self.on_num_of_disps_callback)
            self.tb_name_min_disp = "Min Disp [-128,128]"
            cv2.createTrackbar(self.tb_name_min_disp, self.window_name, 128 + self.min_disparity, 256, self.on_min_disp_callback)
            self.tb_name_disp12MaxDiff = "disp12MaxDiff"
            if self.disp_12_max_diff < 0:
                disp_12_max_diff_pos = 0
            else:
                disp_12_max_diff_pos = self.disp_12_max_diff
            cv2.createTrackbar(self.tb_name_disp12MaxDiff, self.window_name, disp_12_max_diff_pos, self.rows, self.on_disp_12_max_diff_callback)
            self.tb_name_pre_filter_cap = "PreFilter Cap"
            cv2.createTrackbar(self.tb_name_pre_filter_cap, self.window_name, self.pre_filter_cap, 256, self.on_pre_filter_cap_callback)
            self.tb_name_uniqueness_ratio = "Uniqueness Ratio"
            cv2.createTrackbar(self.tb_name_uniqueness_ratio, self.window_name, self.uniqueness_ratio, 100, self.on_uniqueness_ratio_callback)
            self.tb_name_speckle_range = "Speckle Range"
            cv2.createTrackbar(self.tb_name_speckle_range, self.window_name, self.speckle_range, 32, self.on_speckle_range_callback)
            self.tb_name_speckle_win_size = "Speckle Window Size"
            cv2.createTrackbar(self.tb_name_speckle_win_size, self.window_name, self.speckle_window_size, 1000, self.on_speckle_win_size_callback)
            self.button_name_WLS_filter = "WLS Filter"
            cv2.createTrackbar(self.button_name_WLS_filter, self.window_name, self.use_WLS_filter, 1, self.WLS_filter_callback)
            self.tb_name_WLS_lambda = "WLS Lambda"
            cv2.createTrackbar(self.tb_name_WLS_lambda, self.window_name, self.wls_lambda, 20000, self.WLS_lambda_callback)
            self.tb_name_WLS_sigma = "WLS Sigma (x100)"
            cv2.createTrackbar(self.tb_name_WLS_sigma, self.window_name, self.wls_sigma, 1000, self.WLS_sigma_callback)
            # I don't like these Buttons because they get hidden in the property menu and they are meant to be used globally!
            # cv2.createButton(self.button_name_WLS_filter, self.WLS_filter_callback, self.use_WLS_filter, cv2.QT_CHECKBOX, False)

        if self.matching_method == "bm":
            self.tb_name_prefilter_size = "PreFilter Size"
            cv2.createTrackbar(self.tb_name_prefilter_size, self.window_name, self.pre_filter_size, 255, self.on_prefilter_size_callback)
            self.tb_name_texture_threshold = "Texture Threshold"
            cv2.createTrackbar(self.tb_name_texture_threshold, self.window_name, self.texture_threshold, 100, self.on_texture_threshold_callback)

        if self.matching_method == "sgbm":
            self.button_name_full_dyn_prog = "Full DP"
            cv2.createTrackbar(self.button_name_full_dyn_prog, self.window_name, self.full_dyn_prog, 1, self.full_DP_callback)

        if self.matching_method == "var":
            self.tb_name_min_disp = "Min Disp [-128,128]"
            cv2.createTrackbar(self.tb_name_min_disp, self.window_name, 128 + self.min_disparity, 256, self.on_min_disp_callback)
            self.tb_name_max_disp = "Max Disp"
            cv2.createTrackbar(self.tb_name_max_disp, self.window_name, self.max_disparity, 256, self.on_max_disp_callback)
            self.tb_name_cycle = "CYCLE"
            cv2.createTrackbar(self.tb_name_cycle, self.window_name, self.var_cycles_dict[self.var_cycle], 1, self.on_cycle_callback)
            self.tb_name_penalization = "PENALIZATION"
            cv2.createTrackbar(self.tb_name_penalization, self.window_name, self.var_penalizations_dict[self.var_penalization], 2, self.on_penalization_callback)
            self.tb_name_levels = "Levels"
            cv2.createTrackbar(self.tb_name_levels, self.window_name, self.var_levels, 10, self.on_levels_callback)
            self.tb_name_pyr_scale = "Pyr.Sc x100"
            cv2.createTrackbar(self.tb_name_pyr_scale, self.window_name, int(self.var_pyrScale * 100), 100, self.on_pyr_scale_callback)
            self.tb_name_poly_n = "Poly Num"
            cv2.createTrackbar(self.tb_name_poly_n, self.window_name, self.var_poly_n, 7, self.on_poly_n_callback)
            self.tb_name_poly_sigma = "Poly Sigma"
            cv2.createTrackbar(self.tb_name_poly_sigma, self.window_name, int(self.var_poly_sigma * 100), 100, self.on_poly_sigma_callback)
            self.tb_name_fi = "Fi"
            cv2.createTrackbar(self.tb_name_fi, self.window_name, self.var_fi, 100, self.on_fi_callback)
#             self.tb_name_lambda = "Lambda"
#             cv2.createTrackbar(self.tb_name_lambda, self.window_name, int(self.var_lambda * 100), 200, self.on_lambda_callback)
            self.tb_name_num_iters = "N Iters"
            cv2.createTrackbar(self.tb_name_num_iters, self.window_name, self.var_nIt, 50, self.on_num_iters_callback)

        self.preprocess_images()  # Applies median filter and rotation if any

    def preprocess_images(self):
        if self.median_kernel_size > 0:
            left_img = cv2.medianBlur(self.left_raw, self.median_kernel_size)
            right_img = cv2.medianBlur(self.right_raw, self.median_kernel_size)
        else:
            left_img = self.left_raw
            right_img = self.right_raw

        if self.rotate_images:
            # Produce vertical panoramas
            self.left = cv2.flip(cv2.transpose(left_img), flipCode = 0)
            self.right = cv2.flip(cv2.transpose(right_img), flipCode = 0)

            # Composite horizontal visualization top above bottom
            if left_img.ndim < 3:
                left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
            if right_img.ndim < 3:
                right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)

            line_thickness = 4
            line_color = (0, 255, 255)  # yellow in BGR
            self.divider_line_img = np.zeros_like(left_img)[:line_thickness]
            cv2.line(img = self.divider_line_img, pt1 = (0, 0), pt2 = (self.cols - 1, 0), color = line_color, thickness = line_thickness , lineType = cv2.LINE_AA)

#             self.stereo_view_img = np.vstack((left_img[self.disp_first_valid_row:], self.divider_line_img, right_img))
            self.stereo_view_img = np.vstack((left_img, self.divider_line_img, right_img))
        else:
            # WISH: side to side stereo view
            self.left = left_img
            self.right = right_img

    def on_median_filter_callback(self, pos):
        if pos > 0 and pos % 2 == 0:  # Even values are not allowed
            self.median_kernel_size = pos + 1  # Jump ahead to an odd value
            cv2.setTrackbarPos(self.tb_name_median_filter, self.window_name, self.median_kernel_size)
        else:
            self.median_kernel_size = pos

        self.preprocess_images()
        self.needs_update = True

    def on_SAD_win_size_callback(self, pos):
        if pos > 0:
            if self.matching_method == "bm":
                if pos < 5:
                    pos = 5
                if pos % 2 == 0:  # ADWindowSize must be odd
                    pos = pos + 1  # Jump ahead to an odd value
                cv2.setTrackbarPos(self.tb_name_SAD_window_size, self.window_name, pos)

            self.SAD_window_size = pos
            self.needs_update = True

    def on_min_disp_callback(self, pos):
        self.min_disparity = pos - 128  # because value range goes from -128 to 128
        self.needs_update = True

    def on_max_disp_callback(self, pos):
        self.max_disparity = pos
        self.needs_update = True

    def on_num_of_disps_callback(self, pos):
        if pos == 0:
            self.num_of_disparities = 16
        else:
            self.num_of_disparities = (pos // 16) * 16
        cv2.setTrackbarPos(self.tb_name_num_of_disparities, self.window_name, self.num_of_disparities)
        self.needs_update = True

    def on_disp_12_max_diff_callback(self, pos):
        if pos == 0:
            self.disp_12_max_diff = -1
        else:
            self.disp_12_max_diff = pos

        self.needs_update = True

    def on_pre_filter_cap_callback(self, pos):
        self.pre_filter_cap = pos
        self.needs_update = True

    def on_prefilter_size_callback(self, pos):
        if pos > 0:
            if self.matching_method == "bm":
                if pos < 5:
                    pos = 5
                if pos % 2 == 0:  # Prefilter Size must be odd
                    pos = pos + 1  # Jump ahead to an odd value
                cv2.setTrackbarPos(self.tb_name_prefilter_size, self.window_name, pos)
                self.pre_filter_size = pos
                self.needs_update = True

    def on_speckle_range_callback(self, pos):
        if pos > 0:
            self.speckle_range = pos
            self.needs_update = True

    def on_speckle_win_size_callback(self, pos):
        self.speckle_window_size = pos
        self.needs_update = True

    def on_uniqueness_ratio_callback(self, pos):
        self.uniqueness_ratio = pos
        self.needs_update = True

    def on_texture_threshold_callback(self, pos):
        if pos > 0:
            self.texture_threshold = pos
            self.needs_update = True

    def full_DP_callback(self, pos):
        if pos == 0:
            self.full_dyn_prog = False
        else:
            self.full_dyn_prog = True
        self.needs_update = True

    def WLS_filter_callback(self, pos):
        if pos == 0:
            self.use_WLS_filter = False
        else:
            self.use_WLS_filter = True
        self.needs_update = True

    def WLS_lambda_callback(self, pos):
        self.wls_lambda = pos
        self.needs_update = True

    def WLS_sigma_callback(self, pos):
        if pos > 0:
            self.wls_sigma = pos
            self.needs_update = True

    def on_cycle_callback(self, pos):
        self.var_cycle = self.var_cycles_dict.keys()[self.var_cycles_dict.values().index(pos)]
        self.needs_update = True

    def on_penalization_callback(self, pos):
        self.var_penalization = self.var_penalizations_dict.keys()[self.var_penalizations_dict.values().index(pos)]
        self.needs_update = True

    def on_levels_callback(self, pos):
        if pos == 0:
            self.var_levels = 1
            cv2.setTrackbarPos(self.tb_name_levels, self.window_name, self.var_levels)
        else:
            self.var_levels = pos
        self.needs_update = True

    def on_pyr_scale_callback(self, pos):
        min_perc_scale = 10
        if pos < min_perc_scale:
            self.var_pyrScale = min_perc_scale / 100.
            cv2.setTrackbarPos(self.tb_name_pyr_scale, self.window_name, int(self.var_pyrScale * 100))
        else:
            self.var_pyrScale = pos / 100.
        self.needs_update = True

    def on_poly_n_callback(self, pos):
        if pos % 2 == 0:  # Even number of polynomial terms are not allowed
            self.var_poly_n = pos + 1  # Make it odd
            cv2.setTrackbarPos(self.tb_name_poly_n, self.window_name, self.var_poly_n)
        else:
            self.var_poly_n = pos
        self.needs_update = True

    def on_poly_sigma_callback(self, pos):
        self.var_poly_sigma = pos / 100.
        self.needs_update = True

    def on_fi_callback(self, pos):
        if pos == 0:
            self.var_fi = 1
            cv2.setTrackbarPos(self.tb_name_fi, self.window_name, self.var_fi)
        else:
            self.var_fi = pos
        self.needs_update = True

    def on_lambda_callback(self, pos):
        self.var_lambda = pos / 100.
        self.needs_update = True

    def on_num_iters_callback(self, pos):
        self.var_nIt = pos
        self.needs_update = True

class PointClicker(object):

    def __init__(self, win_name, max_clicks = 1, save_path = "", draw_polygon_clicks = False):
        self.window_name = win_name
        self.save_path = save_path
        self.click_counter = 0
        self.img_save_number = 0
        self.is_new_mouse_click = False
        self.max_number_of_clicks = max_clicks
        self.clicked_points = np.ndarray((self.max_number_of_clicks, 2), dtype = int)
        self.shift_mouse_pos = None
        self.verbose = True
        self.draw_lines = draw_polygon_clicks
        self.lines = self.max_number_of_clicks * [None]  # To Contain list of line pairs for example: [[(x0,y0),(x1,y1)], [(x1,y1),(x2,y2)],[(x2,y2),(x_curr,y_curr)]]
        cv2.setMouseCallback(self.window_name, self.on_mouse_callback)

    def on_mouse_callback(self, event, xc, yc, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            if flags == cv2.EVENT_FLAG_SHIFTKEY:
#         if flags == (cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY):
                self.shift_mouse_pos = (xc, yc)
            if self.draw_lines:
                if self.click_counter > 0 and self.click_counter != self.max_number_of_clicks:
                    self.lines[self.click_counter - 1] = [tuple(self.clicked_points[self.click_counter - 1]), (xc, yc)]
                    self.is_new_mouse_click = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.click_counter += 1
            if self.click_counter > self.max_number_of_clicks:
                self.click_counter = 1  # Reset counter
                self.lines = self.max_number_of_clicks * [None]  # Reset all lines
            self.clicked_points[self.click_counter - 1] = (xc, yc)
            if self.draw_lines:
                if self.click_counter > 1:
                    self.lines[self.click_counter - 2] = [tuple(self.clicked_points[self.click_counter - 2]), tuple(self.clicked_points[self.click_counter - 1])]
                    if self.click_counter == self.max_number_of_clicks:  # Close the loop
                        self.lines[self.click_counter - 1] = [tuple(self.clicked_points[self.click_counter - 1]), tuple(self.clicked_points[0])]
            if self.verbose:
                print("Clicked on (u,v) = ", self.clicked_points[self.click_counter - 1])
            self.is_new_mouse_click = True

    def get_clicks_uv_coords(self, img, verbose = True):
        '''
        @return: the np array of valid points clicked. NOTE: the arrangement is in the (u,v) coordinates
        '''
        self.verbose = verbose
        cv2.imshow(self.window_name, img)

        # while cv2.waitKey(1) == -1:  # While not any key has been pressed
        ch_pressed_waitkey = cv2.waitKey(1)
        while not(ch_pressed_waitkey == 27):  # Pressing the Escape key breaks the loop
            if (ch_pressed_waitkey & 255) == ord('r'):
                self.click_counter = 0  # reset count
                self.is_new_mouse_click = True
                self.lines = self.max_number_of_clicks * [None]

            # Grab a point
            if self.is_new_mouse_click:
                channels = img.ndim
                    # img_copy = img.copy()  # Keep the original image
                if channels < 3:
                    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    vis = img.copy()

                draw_points(vis, self.clicked_points, num_valid_points = self.click_counter)
                draw_lines(vis, self.lines, thickness = 5)
                draw_str(vis, (10, 20), "Keyboard Commands:", color_RGB = (255, 0, 0))
                draw_str(vis, (20, 40), "'R': restart", color_RGB = (255, 0, 0))
                draw_str(vis, (20, 60), "'(1-9)': rewind by #", color_RGB = (255, 0, 0))
                draw_str(vis, (20, 80), "'Esc': finish", color_RGB = (255, 0, 0))
                cv2.imshow(self.window_name, vis)

            # To indicate to go back to the previous case (useful when detecting corner detection errors)
            if 48 < ch_pressed_waitkey < 58:  # Indicate to go back to the previous case (useful when detecting corner detection errors)
                return ch_pressed_waitkey - 48  # Because 48 is mapped to key 0

            self.is_new_mouse_click = False  # Reset indicator
            ch_pressed_waitkey = cv2.waitKey(10)

#         cv2.destroyWindow(self.window_name)

        return self.clicked_points[:self.click_counter]

    def get_clicks_uv_coords_for_stereo(self, stereo_model, show_correspondence_on_circular_img = False, min_disparity = 1, max_disparity = 0, verbose = False):
        '''
        @return: the two np arrays of valid points clicked and its correspondences. NOTE: the arrangement is in the (u,v) coordinates
        '''
        self.verbose = verbose
        target_window_name = 'Target Point Correspondences'
        cv2.namedWindow(target_window_name, cv2.WINDOW_NORMAL)

        target_coords = None
        reference_coords = None
        img_reference = stereo_model.top_model.panorama.panoramic_img  # Acting as the right image
        img_target = stereo_model.bot_model.panorama.panoramic_img
        if show_correspondence_on_circular_img:
            omni_top_coords = None
            omni_bot_coords = None
            img_omni = stereo_model.current_omni_img
            omni_window_name = 'Correspondences on Omni Image'
            cv2.namedWindow(omni_window_name, cv2.WINDOW_NORMAL)

        cv2.imshow(self.window_name, img_reference)
        # while cv2.waitKey(1) == -1:  # While not any key has been pressed
        ch_pressed_waitkey = cv2.waitKey(1)
        while not(ch_pressed_waitkey == 27):  # Pressing the Escape key breaks the loop
            if (ch_pressed_waitkey & 255) == ord('r'):
                self.click_counter = 0  # reset count
                self.is_new_mouse_click = False
                cv2.imshow(self.window_name, img_reference)
                cv2.imshow(target_window_name, img_target)
                if show_correspondence_on_circular_img:
                    cv2.imshow(omni_window_name, img_omni)

            # Grab a point
            if self.is_new_mouse_click:
                channels = img_reference.ndim
                    # img_copy = img_reference.copy()  # Keep the original image
                if channels < 3:
                    vis_ref = cv2.cvtColor(img_reference, cv2.COLOR_GRAY2BGR)
                    vis_target = cv2.cvtColor(img_target, cv2.COLOR_GRAY2BGR)
                    if show_correspondence_on_circular_img:
                        vis_omni = cv2.cvtColor(img_omni, cv2.COLOR_GRAY2BGR)
                else:
                    vis_ref = img_reference.copy()
                    vis_target = img_target.copy()
                    if show_correspondence_on_circular_img:
                        vis_omni = img_omni.copy()

                # Find correspondence
                reference_coords, target_coords, disparities = stereo_model.resolve_pano_correspondences_from_disparity_map(self.clicked_points[:self.click_counter], min_disparity = min_disparity, max_disparity = max_disparity, verbose = verbose)
                # Update clicks
                self.click_counter = int(np.count_nonzero(reference_coords) / 2)
                self.clicked_points[:self.click_counter] = reference_coords

                # Write instructions on image
                draw_str(vis_ref, (10, 20), "Keyboard Commands:")
                draw_str(vis_ref, (20, 40), "'R': to restart")
                draw_str(vis_ref, (20, 60), "'Esc': to finish")
                # Draw points on panoramas
                ref_pts_color = (255, 0, 0)  # RGB = red
                tgt_pts_color = (0, 0, 255)  # RGB = blue
                pt_thickness = 5
                draw_points(vis_ref, reference_coords.reshape(-1, 2), color = ref_pts_color, thickness = pt_thickness)
                cv2.imshow(self.window_name, vis_ref)
                draw_points(vis_target, reference_coords.reshape(-1, 2), color = ref_pts_color, thickness = pt_thickness)
                # Coloring a blue dot at the proper target location:
                draw_points(vis_target, target_coords.reshape(-1, 2), color = tgt_pts_color, thickness = pt_thickness)
                cv2.imshow(target_window_name, vis_target)
                if show_correspondence_on_circular_img and self.click_counter > 0 and self.verbose:
                    _, _, omni_top_coords = stereo_model.top_model.panorama.get_omni_pixel_coords_from_panoramic_pixel(reference_coords)
                    _, _, omni_bot_coords = stereo_model.bot_model.panorama.get_omni_pixel_coords_from_panoramic_pixel(target_coords)
                    print("Omni pixel coords: TOP %s, BOT %s" % (omni_top_coords[0, self.click_counter - 1], omni_bot_coords[0, self.click_counter - 1]))
                    draw_points(vis_omni, omni_top_coords[..., :2].reshape(-1, 2), color = ref_pts_color, thickness = pt_thickness)
                    draw_points(vis_omni, omni_bot_coords[..., :2].reshape(-1, 2), color = tgt_pts_color, thickness = pt_thickness)
                    cv2.imshow(omni_window_name, vis_omni)

            self.is_new_mouse_click = False  # Reset indicator
            ch_pressed_waitkey = cv2.waitKey(1)

        return reference_coords, target_coords, disparities

    def save_image(self, img, img_name = None, num_of_zero_padding = 6):
        if img_name:
            name_prefix = img_name
        else:
            name_prefix = "img"

        n = str(self.img_save_number)
        img_name = '%s-%s.png' % (name_prefix, n.zfill(num_of_zero_padding))

        if self.save_path:
            complete_save_name = self.save_path + img_name
        else:
            complete_save_name = img_name

        print('Saving', complete_save_name)
        cv2.imwrite(complete_save_name, img)

        self.img_save_number += 1  # Increment save counter

def rgb2bgr_color(rgb_color):
    return (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))

def draw_points(img_input, points_uv_coords, num_valid_points = None, color = None, thickness = 1):
    '''
    @param img_input: The image on which points will be drawn to (NOTE: it doesn't preserve the image)
    @param points_uv_coords: FIXME: the uv coordinates list or ndarray must be of shape (n, 2) for n points.
    Note that the coordinates will be expressed as integers while visualizing
    @param color: a 3-tuple of the RGB color for these points
    '''
    if color == None:
        color = (0, 0, 255)  # Red because BGR(B,G,R)
    else:  # Swap the passed color from RGB into BGR
        color = rgb2bgr_color(color)

    if num_valid_points == None:
        num_valid_points = len(points_uv_coords)

    for i in range(num_valid_points):
        pt = points_uv_coords[i]
        if np.isnan(pt[0]) or np.isnan(pt[1]):
            print("nan cannot be drawn!")
        else:
            try:  # TODO: also, out of border points cannot be drawn!
                pt_as_tuple = (int(pt[0]), int(pt[1]))  # Recall: (pt[0],pt[1]) # (x, u or col and y, v or row)
                cv2.circle(img_input, pt_as_tuple, 2, color, thickness, 8, 0)
            except:
                print("Point", pt_as_tuple, "cannot be drawn!")

def draw_lines(img_input, lines_list, color = None, thickness = 2):
    '''
    @param img_input: The image on which points will be drawn to (NOTE: it doesn't preserve the image)
    @param lines_list: A list of point pairs such as [[(x0,y0),(x1,y1)], [(x1,y1),(x2,y2)],[(x2,y2),(x_last,y_Last)], None, None]
    @param color: a 3-tuple of the RGB color for these points
    '''
    if color == None:
        color = (0, 0, 255)  # Red because BGR(B,G,R)
    else:  # Swap the passed color from RGB into BGR
        color = rgb2bgr_color(color)

    for pts in lines_list:
        if pts is not None:
            [pt_beg, pt_end] = pts
            cv2.line(img_input, pt_beg, pt_end, color, thickness = thickness, lineType = cv2.LINE_AA)

def get_masked_omni_image_single_center(img_input, center_point, outer_radius, inner_radius = 0.0, color_RGB = None):
    '''
    @param color_RGB: A tuple specifying the desired background as (Red,Green,Blue). If None, the background is black
    '''
    mask = np.zeros(img_input.shape[0:2], dtype = np.uint8)  # Black, single channel mask

    # Paint outer perimeter:
    cv2.circle(mask, center_point, int(outer_radius), (255, 255, 255), -1, 8, 0)

    # Paint inner perimeter:
    if inner_radius > 0:
        cv2.circle(mask, center_point, int(inner_radius), (0, 0, 0), -1, 8, 0)

    # Apply mask
    masked_img = np.zeros(img_input.shape)
    masked_img = cv2.bitwise_and(src1 = img_input, src2 = img_input, dst = masked_img, mask = mask)

    if color_RGB is not None:  # Paint the masked area other than black
        background_img = np.zeros_like(masked_img)
        color_BGR = np.array([color_RGB[2], color_RGB[1], color_RGB[0]], dtype = "uint8")
        background_img[..., :] += color_BGR  # Paint the B-G-R channels for OpenCV
        mask_inv = cv2.bitwise_not(src = mask)
        # Apply the background using the inverted mask
        masked_img = cv2.bitwise_and(src1 = background_img, src2 = background_img, dst = masked_img, mask = mask_inv)
        #=======================================================================
        # mask2 = np.zeros(img_input.shape[0:2], dtype=np.uint8) + 255  # Now, we start on a white canvas
        # # Paint outer perimeter (a black filled circle):
        # cv2.circle(mask2, center_point, int(outer_radius), (0, 0, 0), -1, 8, 0)
        # # Paint a white inner perimeter:
        # if inner_radius > 0:
        #     cv2.circle(mask2, center_point, int(inner_radius), (255, 255, 255), -1, 8, 0)
        # # Apply mask
        # masked_img = cv2.bitwise_and(src1=background_img, src2=background_img, dst=masked_img, mask=mask2)
        #=======================================================================

    return masked_img

def get_images(filename_template, indices_list = [], show_images = False, return_names_only = False):
    '''
    @param indices_list: Returns only those images indices from the entire list. If this list is empty (default), all images read are returned
    @note: all images files acquired by glob will be read and shown (however), but only those indexed in the list (if any) will be returned
    @return: A list of the retrieved images (based on an index list, if any) from the filenames template. If the return_names_only is set to True, only the names of the images will be retrieved
    '''
    #===========================================================================
    # from glob import glob
    # img_names = glob(filename_template)
    #===========================================================================
    # It's faster to retrieve files from a directory with "fnmatch":
    import fnmatch
    from os import listdir
    from os.path import split, join
    path_to_files, pattern_filename = split(filename_template)
    img_names = fnmatch.filter(listdir(path_to_files), pattern_filename)

    if indices_list is None or len(indices_list) == 0:
        l = len(img_names)
        indices_list = range(l)
    else:
        l = len(indices_list)

    img_names_list_all = [join(path_to_files, img_name) for img_name in img_names]
    img_names_list = l * [None]
    for i, img_index in enumerate(indices_list):
        img_names_list[i] = img_names_list_all[img_index]

    if return_names_only:
        return img_names_list

    images = l * [None]
    for i, fn in enumerate(img_names_list):
        try:
            # fn = img_names[i] # when using glob
            # fn = join(path_to_files, img_names[i])  # When using fnmatch
            print('Reading %s...' % fn, end = "")
            img = cv2.imread(fn)
            if img is not None:
                print("success")
                images[i] = img
                if show_images:
                    path, name, ext = splitfn(fn)
                    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                    cv2.imshow(name, img)
            else:
                print("failed!")
        except:
            warnings.warn("Warning...image index %d not found at %s" % (i, __name__))

    if show_images:
        cv2.waitKey(1)

    # We want the whole list!, so this is not good any more
#     if len(indices_list) == 0:
#         return images  # all
#     else:
#         return list(np.take(images, indices_list, 0))

    return images  # all even if None

def get_feature_matches_data_from_files(filename_template, indices_list = []):
    '''
    @param indices_list: Returns only those frame indices from the entire list. If this list is empty (default), all existing pickles are read are returned
    @note: all feature_data files acquired by glob will be read and shown (however), but only those indexed in the list (if any) will be returned
    @return: A list of the retrieved feature_data (based on an index list, if any) from the filenames template. NOTE that cv2.KeyPoints have been serialized as tuples.
    Each data entry in the returned list is organized as: (matched_m_top, matched_kpts_top_serial, matched_desc_top), (matched_m_bot, matched_kpts_bot_serial, matched_desc_bot), random_colors_RGB
    '''
    from omnistereo.common_tools import load_obj_from_pickle

    if indices_list == None or len(indices_list) == 0:
        from glob import glob
        pickle_names = glob(filename_template)
        l = len(pickle_names)
        indices_list = range(l)
    else:
        l = indices_list[-1] + 1  # Assuming indices are ordered increasingly

    feature_data = l * [None]

    for i in indices_list:
        features_data_filename = filename_template.replace("*", str(i), 1)
        try:
            print('Reading %s...' % features_data_filename, end = "")
            data = load_obj_from_pickle(filename = features_data_filename)
            feature_data[i] = data
        except:
            warnings.warn("Warning...file index %d not found at %s" % (i, __name__))

    return feature_data  # all even if None

def get_masked_images_mono(unmasked_images, camera_model, img_indices = [], show_images = False, color_RGB = None):
    '''
    @param color_RGB: A tuple specifying the desired background as (Red,Green,Blue). If None, the background is black
    '''

    l = len(unmasked_images)
    masked_images = l * [None]
    if img_indices is None or len(img_indices) == 0:
        img_indices = range(l)  # Use all images

    if hasattr(camera_model, "outer_img_radius"):
        use_circular_mask = True
    else:
        use_circular_mask = False

    for i in img_indices:
        try:
            img = unmasked_images[i]
            if use_circular_mask:
                masked_img = camera_model.get_masked_image(omni_img = img, view = False, color_RGB = color_RGB)
                # OLDER method was incomplete:
                # masked_img = get_masked_omni_image_single_center(img, (int(u0), int(v0)), camera_model.outer_img_radius, camera_model.inner_img_radius, color_RGB=color_RGB)
            else:
                masked_img = img.copy()

            if masked_img is not None:
                masked_images[i] = masked_img

                if show_images:
                    win_name = "%s (masked) - [%d]" % (camera_model.mirror_name , i)
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(win_name, masked_img)
        except:
            print("Problems with masking image_%d" % (i))

    if show_images:
        cv2.waitKey(1)

    return masked_images

def get_masked_images_as_pairs(unmasked_images, omnistereo_model, img_indices = [], show_images = False, color_RGB = None):
    '''
    @param color_RGB: A tuple specifying the desired background as (Red,Green,Blue). If None, the background is black
    '''

    l = len(unmasked_images)
    masked_images = l * [None]
    if img_indices is None or len(img_indices) == 0:
        img_indices = range(l)  # Use all images

    for i in img_indices:
        try:
            img = unmasked_images[i]
            masked_img_top, masked_img_bottom = omnistereo_model.get_fully_masked_images(omni_img = img, view = False, color_RGB = color_RGB)
            masked_images[i] = (masked_img_top, masked_img_bottom)

            if show_images:
                win_name_top = "Top masked - [%d]" % (i)
                win_name_bot = "Bottom masked - [%d]" % (i)
                cv2.namedWindow(win_name_top, cv2.WINDOW_NORMAL)
                cv2.namedWindow(win_name_bot, cv2.WINDOW_NORMAL)
                cv2.imshow(win_name_top, masked_img_top)
                cv2.imshow(win_name_bot, masked_img_bottom)
        except:
            print("Problems with masking image [%d]" % (i))

    #===========================================================================
    # masked_images_top = get_masked_images_mono(unmasked_images, omnistereo_model.top_model, img_indices=[], show_images=show_images, color_RGB=color_RGB)
    # masked_images_bottom = get_masked_images_mono(unmasked_images, omnistereo_model.bot_model, img_indices=[], show_images=show_images, color_RGB=color_RGB)
    #===========================================================================

    if show_images:
        cv2.waitKey(1)

    return masked_images  # zip(masked_images_top, masked_images_bottom)

def create_arbitrary_mask(img_input, points, preview = False):
    mask = np.zeros(img_input.shape[0:2], dtype = np.uint8)  # Black, single channel mask
    masked_img = np.zeros(img_input.shape)

    cv2.fillConvexPoly(mask, points, color = (255, 255, 255), lineType = 8, shift = 0)
    masked_img = cv2.bitwise_and(img_input, img_input, masked_img, mask = mask)

    if preview:
        resulting_mask_window_name = "Resulting Mask"
        cv2.namedWindow(resulting_mask_window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(resulting_mask_window_name, masked_img)
        cv2.waitKey(0)
        cv2.destroyWindow(resulting_mask_window_name)

    return masked_img

def refine_radial_bounds_mono(omni_img, initial_values = []):
    '''
    @param initial_values: A list of initial values: [center_pixel, outer_radius, inner_radius]
    '''
    if len(initial_values) == 3:
        center_pixel, outer_radius, inner_radius = initial_values
    else:
        center_pixel, outer_radius, inner_radius = None, None, None

    # Find circular boundaries
    win_name_outter = "Outter boundary"
    center_pixel, outer_radius = extract_circular_bound(omni_img, win_name_outter, center_coords = center_pixel, radius = outer_radius)
    win_name_inner = "Inner boundary"
    center_pixel, inner_radius = extract_circular_bound(omni_img, win_name_inner, center_coords = center_pixel, radius = inner_radius)

    from cv2 import destroyWindow
    destroyWindow(win_name_outter)
    destroyWindow(win_name_inner)

    return (center_pixel, outer_radius, inner_radius)

def refine_radial_bounds(omni_img, top_values, bottom_values):
    [[center_pixel_top, outer_radius_top, inner_radius_top], [center_pixel_bottom, outer_radius_bottom, inner_radius_bottom]] = find_center_and_radial_bounds(omni_img, initial_values = [top_values, bottom_values], save_to_file = False)
    return (outer_radius_top, inner_radius_top), (outer_radius_bottom, inner_radius_bottom)

def find_center_and_radial_bounds(omni_img, initial_values = [], radial_bounds_filename = "", save_to_file = True, fiducial_rings_radii_top = [], fiducial_rings_radii_bottom = [], is_stereo = True, use_auto_circle_finder = True, use_existing_radial_bounds = False):
    from cv2 import destroyWindow

    # TODO: Load existing filename if any:
    # if exists, use it to initialize the data
    import os.path
    file_exists = os.path.isfile(radial_bounds_filename)
    if file_exists:
        from omnistereo.common_tools import load_obj_from_pickle
    if save_to_file:
        from omnistereo.common_tools import save_obj_in_pickle

    if is_stereo:
        if file_exists:
            [[center_pixel_top_outer, center_pixel_top_inner, outer_radius_top, inner_radius_top], [center_pixel_bottom_outer, center_pixel_bottom_inner, outer_radius_bottom, inner_radius_bottom]] = load_obj_from_pickle(radial_bounds_filename)
        else:
            if len(initial_values) > 0:
                # use initial values and do testing
                [[center_pixel_top_outer, center_pixel_top_inner, outer_radius_top, inner_radius_top], [center_pixel_bottom_outer, center_pixel_bottom_inner, outer_radius_bottom, inner_radius_bottom]] = initial_values
            else:
                [[center_pixel_top_outer, center_pixel_top_inner, outer_radius_top, inner_radius_top], [center_pixel_bottom_outer, center_pixel_bottom_inner, outer_radius_bottom, inner_radius_bottom]] = [[None, None, None, None], [None, None, None, None]]

        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Find center and radial boundaries
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #=======================================================================
        # Find circular boundaries
        win_name_top_outter = "Top mirror - outter boundary"
        win_name_top_inner = "Top mirror - inner boundary"
        # NOTE: trusting only the centers extracted from the outer radius
        win_name_bottom_outter = "Bottom mirror - outter boundary"
        win_name_bottom_inner = "Bottom mirror - inner boundary"

        if not use_existing_radial_bounds:
            if use_auto_circle_finder:
                center_pixel_top_outer, outer_radius_top = extract_circular_bound_auto(omni_img, win_name_top_outter, center_coords = center_pixel_top_outer, radius = outer_radius_top)

                if not file_exists:  # Init values based on previous results
                    center_pixel_top_inner = center_pixel_top_outer
                    inner_radius_top = 0.75 * outer_radius_top
                center_pixel_top_inner, inner_radius_top = extract_circular_bound_auto(omni_img, win_name_top_inner, center_coords = center_pixel_top_inner, radius = inner_radius_top)

                if not file_exists:  # Init values based on previous results
                    center_pixel_bottom_outer = center_pixel_top_inner
                    outer_radius_bottom = inner_radius_top
                center_pixel_bottom_outer, outer_radius_bottom = extract_circular_bound_auto(omni_img, win_name_bottom_outter, center_coords = center_pixel_bottom_outer, radius = outer_radius_bottom)

                if not file_exists:  # Init values based on previous results
                    center_pixel_bottom_inner = center_pixel_bottom_outer
                    inner_radius_bottom = 0.75 * outer_radius_bottom
                center_pixel_bottom_inner, inner_radius_bottom = extract_circular_bound_auto(omni_img, win_name_bottom_inner, center_coords = center_pixel_bottom_inner, radius = inner_radius_bottom)

            else:
                center_pixel_top_outer, outer_radius_top = extract_circular_bound(omni_img, win_name_top_outter, center_coords = center_pixel_top_outer, radius = outer_radius_top, ring_fiducials_radii = fiducial_rings_radii_top)
                center_pixel_top_inner, inner_radius_top = extract_circular_bound(omni_img, win_name_top_inner, center_coords = center_pixel_top_outer, radius = inner_radius_top, ring_fiducials_radii = fiducial_rings_radii_top)
                center_pixel_bottom_outer, outer_radius_bottom = extract_circular_bound(omni_img, win_name_bottom_outter, center_coords = center_pixel_bottom_outer, radius = outer_radius_bottom, ring_fiducials_radii = fiducial_rings_radii_bottom)
                center_pixel_bottom_inner, inner_radius_bottom = extract_circular_bound(omni_img, win_name_bottom_inner, center_coords = center_pixel_bottom_outer, radius = inner_radius_bottom, ring_fiducials_radii = fiducial_rings_radii_bottom)

            # NOTE: trusting only the center extracted from the outer radius
            if save_to_file:
                save_obj_in_pickle([[center_pixel_top_outer, center_pixel_top_inner, outer_radius_top, inner_radius_top], [center_pixel_bottom_outer, center_pixel_bottom_inner, outer_radius_bottom, inner_radius_bottom]], radial_bounds_filename, locals())

        return [[np.array(center_pixel_top_outer), np.array(center_pixel_top_inner), outer_radius_top, inner_radius_top], [np.array(center_pixel_bottom_outer), np.array(center_pixel_bottom_inner), outer_radius_bottom, inner_radius_bottom]]
    else:
        if file_exists:
            [center_pixel_outer, center_pixel_inner, outer_radius, inner_radius] = load_obj_from_pickle(radial_bounds_filename)
        else:
            if len(initial_values) > 0:
                # use initial values and do testing
                [center_pixel_outer, center_pixel_inner, outer_radius, inner_radius] = initial_values
            else:
                [center_pixel_outer, center_pixel_inner, outer_radius, inner_radius] = [None, None, None, None]

        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Find center and radial boundaries
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #=======================================================================
        # Find circular boundaries
        win_name_outter = "Outter boundary"
        win_name_inner = "Inner boundary"
        if use_auto_circle_finder:
            center_pixel_outer, outer_radius = extract_circular_bound_auto(omni_img, win_name_outter, center_coords = center_pixel_outer, radius = outer_radius)

            if not file_exists:  # Init values based on previous results
                center_pixel_inner = center_pixel_outer
                inner_radius = 0.75 * outer_radius
            center_pixel_inner, inner_radius = extract_circular_bound_auto(omni_img, win_name_inner, center_coords = center_pixel_inner, radius = inner_radius)
        else:
            center_pixel_outer, outer_radius = extract_circular_bound(omni_img, win_name_outter, center_coords = center_pixel_outer, radius = outer_radius)
            center_pixel_inner, inner_radius = extract_circular_bound(omni_img, win_name_inner, center_coords = center_pixel_inner, radius = inner_radius)

        if save_to_file:
            save_obj_in_pickle([center_pixel_outer, center_pixel_inner, outer_radius, inner_radius], radial_bounds_filename, locals())

        return [np.array(center_pixel_outer), np.array(center_pixel_inner), outer_radius, inner_radius]

def create_rectangular_mask(img_input, points, preview = False):
    mask = np.zeros(img_input.shape[0:2], dtype = np.uint8)  # Black, single channel mask
    masked_img = np.zeros(img_input.shape)

    cv2.rectangle(img = mask, pt1 = points[0], pt2 = points[1], color = (255, 255, 255), thickness = cv2.FILLED)
    masked_img = cv2.bitwise_and(img_input, img_input, masked_img, mask = mask)

    if preview:
        resulting_mask_window_name = "Resulting Rectangular Mask"
        cv2.namedWindow(resulting_mask_window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(resulting_mask_window_name, masked_img)
        cv2.waitKey(1)

    return masked_img

def extract_circular_bound_auto(omni_img, win_name = "Circle Extraction", center_coords = None, radius = None):
    from omnistereo.mirror_boundary import ContourExtractorApp
    print("USER: attempting boundary extraction on " + win_name + "...")
    current_center, current_circle_radius = ContourExtractorApp(omni_img, window_name = win_name, center_point_init = center_coords, best_radius_init = radius).run()
    return current_center, current_circle_radius

def extract_circular_bound(omni_img, win_name = "Circle Extraction", center_coords = None, radius = None, ring_fiducials_radii = []):
    from omnistereo.common_tools import get_screen_resolution
    window_width, window_height = get_screen_resolution(measurement = "px")  # PROBLEM: other OpenCV windows should be closed before invoking this function due to some QT threading issues!

    print("USER: attempting " + win_name + "...")
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # We want to resize the window to occupy the entire screen
    cv2.resizeWindow(win_name, window_width, window_height)
    pos_x = 0
    pos_y = 0
    cv2.moveWindow(win_name, pos_x, pos_y)

    win_handler = PointClicker(win_name)
    if omni_img.ndim == 3:
        omni_img_gray = cv2.cvtColor(omni_img, cv2.COLOR_BGR2GRAY)
    else:
        omni_img_gray = omni_img.copy()

    omnin_vis = cv2.cvtColor(omni_img_gray, cv2.COLOR_GRAY2BGR)

    try_again = True
    if center_coords is not None:
        current_center = np.array(center_coords)
    else:
        current_center = np.array([(omnin_vis.shape[1] / 2) - 1, (omnin_vis.shape[0] / 2) - 1])

    if radius is None:
        current_circle_radius = 500  # @ISH: this default value is arbitrary
    else:
        current_circle_radius = radius

    while try_again:
        omni_img_drawn = omnin_vis.copy()

        if win_handler.click_counter > 0:
            current_center = win_handler.clicked_points[0]
        if win_handler.shift_mouse_pos:  # Basically a SHIFT and move the mouse
            rp = win_handler.shift_mouse_pos
            current_circle_radius = int(np.sqrt((current_center[0] - rp[0]) ** 2 + (current_center[1] - rp[1]) ** 2))
            win_handler.shift_mouse_pos = None  # Clear

        # draw the circle center
        draw_points(omni_img_drawn, [current_center], color = (255, 0, 0), thickness = 3)

        # draw the circle outline
        circle_outline_thickness = 2
        current_center_as_int = (int(current_center[0]), int(current_center[1]))
        cv2.circle(omni_img_drawn, current_center_as_int, current_circle_radius, (0, 0, 255), circle_outline_thickness, 8, 0)

        # Draw ring fiducials:
        fiducials_line_color = (0, 255, 255)  # yellow in BGR
        for fid_radius in ring_fiducials_radii:
            cv2.circle(omni_img_drawn, current_center_as_int, fid_radius, fiducials_line_color, circle_outline_thickness, 8, 0)

        cv2.imshow(win_name, omni_img_drawn)

        ch_pressed_waitkey = cv2.waitKey(10)
        # Vim style motion commands for center adjustment
        if (ch_pressed_waitkey & 255) == ord('i'):  # move up
            current_center = (current_center[0], current_center[1] - 1)
        if (ch_pressed_waitkey & 255) == ord('k'):  # move down
            current_center = (current_center[0], current_center[1] + 1)
        if (ch_pressed_waitkey & 255) == ord('j'):  # move left
            current_center = (current_center[0] - 1, current_center[1])
        if (ch_pressed_waitkey & 255) == ord('l'):  # move right
            current_center = (current_center[0] + 1, current_center[1])
        # Update manual adjustement of center
        win_handler.clicked_points[0] = current_center

        # Resize circle radius
        if (ch_pressed_waitkey & 255) == ord('+') or (ch_pressed_waitkey & 255) == ord('='):
            current_circle_radius += 1
        if (ch_pressed_waitkey & 255) == ord('-'):
            current_circle_radius -= 1

        # Save image
        if (ch_pressed_waitkey & 255) == ord('s'):
            win_handler.save_image(omni_img_drawn, "test_center")

        # Quit
        if (ch_pressed_waitkey == 27) or (ch_pressed_waitkey & 255) == ord('q'):  # Pressing the Escape key breaks the loop
            break

    cv2.destroyWindow(win_name)
    return current_center, current_circle_radius

def mask_rect_min_area(base_img, input_image, points, use_all_points = True):
    padding = 2 * np.linalg.norm(points[0, 0] - points[0, 1])  # Use the double of distance between 2 points
    if use_all_points:
        min_area_rect = cv2.minAreaRect(points[..., :2].reshape(-1, 2).astype("int32"))
    else:  # Only use the four courners
        num_rows, num_cols = points.shape[:2]
        corners_4_pattern_indices = np.array([[0, 0], [num_rows - 1, 0], [num_rows - 1, num_cols - 1], [0, num_cols - 1]])
        corners_4_top = points[corners_4_pattern_indices[:, 0], corners_4_pattern_indices[:, 1]].astype("int32")
        min_area_rect = cv2.minAreaRect(corners_4_top)

    # Add padding to box size
    min_area_rect = (min_area_rect[0], (min_area_rect[1][0] + padding, min_area_rect[1][1] + padding), min_area_rect[2])
    box = cv2.boxPoints(min_area_rect).astype("int32")
    # Masking the on the top view:
    mask = np.zeros(input_image.shape[0:2], dtype = np.uint8)  # Black, single channel mask
    # draw rotated rectangle (as filled contours)
    cv2.drawContours(mask, [box], 0, (255, 255, 255), lineType = 8, thickness = -1)
    mask_inverted = np.zeros(input_image.shape[0:2], dtype = np.uint8) + 255  # White, single channel mask
    cv2.drawContours(mask_inverted, [box], 0, (0, 0, 0), lineType = 8, thickness = -1)
    masked_img = np.zeros_like(input_image)
    masked_img = cv2.bitwise_and(input_image, input_image, dst = masked_img, mask = mask)
    masked_img_base = np.zeros_like(input_image)
    masked_img_base = cv2.bitwise_and(base_img, base_img, masked_img_base, mask = mask_inverted)
    base_img = cv2.bitwise_or(masked_img_base, masked_img)  # Update resulting image as the new base_img

    return base_img

def overlay_all_chessboards(omni_model, calibrator, indices = [], draw_detection = False, visualize = False):

    from omnistereo.camera_models import OmniStereoModel
    if visualize:
        win_name = "Overlayed Chessboard Images"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 800, 600)

    base_img = None
    base_img_idx = None
    if isinstance(omni_model, OmniStereoModel):
        omni_calib_list = calibrator.calibration_pairs
        is_stereo = True
    else:
        omni_calib_list = calibrator.omni_monos
        is_stereo = False

    if indices is None or len(indices) == 0:
        for idx, oc in enumerate(omni_calib_list):
            if oc is not None:
                if oc.found_points:
                    base_img = oc.omni_image.copy()
                    base_img_idx = idx
                    break
        indices = list(range(len(omni_calib_list)))
    else:
        base_img = omni_calib_list[indices[0]].omni_image.copy()
        base_img_idx = indices[0]

    if base_img_idx is not None:
        for idx in indices[base_img_idx:]:  # Start from the next one on
            oc = omni_calib_list[idx]
            if hasattr(oc, "found_points") and oc.found_points:
                if is_stereo:
                    pts_top = oc.mono_top.image_points  # Must be a float32 for OpenCV to work!
                    base_img = mask_rect_min_area(base_img, oc.omni_image, pts_top)
                    pts_bottom = oc.mono_bottom.image_points  # Must be a float32 for OpenCV to work!
                    base_img = mask_rect_min_area(base_img, oc.omni_image, pts_bottom)
                    if draw_detection:
                        # SIMPLER: draw_points(base_img, pts_top[..., :2].reshape(-1, 2), color=(255, 0, 0), thickness=2)
                        cv2.drawChessboardCorners(base_img, oc.mono_top.pattern_size_applied, pts_top.reshape(-1, 2), oc.found_points)
                        # SIMPLER: draw_points(base_img, pts_bottom[..., :2].reshape(-1, 2), color=(0, 0, 255), thickness=2)
                        cv2.drawChessboardCorners(base_img, oc.mono_bottom.pattern_size_applied, pts_bottom.reshape(-1, 2), oc.found_points)
                else:  # mono
                    pts = oc.image_points  # Must be a float32 for OpenCV to work!
                    base_img = mask_rect_min_area(base_img, oc.omni_image, pts)
                    if draw_detection:
                        # SIMPLER: draw_points(base_img, pts[..., :2].reshape(-1, 2), color=(255, 0, 0), thickness=2)
                        cv2.drawChessboardCorners(base_img, oc.pattern_size_applied, pts.reshape(-1, 2), oc.found_points)

    if visualize:
        cv2.imshow(win_name, base_img)
        cv2.waitKey(1)

    return base_img

# Older way of overlaying by projection from 3D:
#===============================================================================
# def overlay_all_chessboards(omni_model, calibrator, indices=[], draw_detection=False, visualize=False):
#
#     from camera_models import OmniStereoModel
#     if visualize:
#         win_name = "Overlayed Chessboard Images"
#         cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
#
#     base_img = None
#     base_img_idx = None
#     if isinstance(omni_model, OmniStereoModel):
#         omni_calib_list = calibrator.calibration_pairs
#         is_stereo = True
#     else:
#         omni_calib_list = calibrator.omni_monos
#         is_stereo = False
#
#     if len(indices) == 0:
#         for idx, oc in enumerate(omni_calib_list):
#             if oc is not None:
#                 if oc.found_points:
#                     base_img = oc.omni_image.copy()
#                     base_img_idx = idx
#                     break
#         indices = list(range(len(omni_calib_list)))
#     else:
#         base_img = omni_calib_list[indices[0]].omni_image.copy()
#         base_img_idx = indices[0]
#
#     if base_img_idx is not None and calibrator.has_chesboard_pose_info:
#         for idx in indices[base_img_idx:]:  # Start from the next one on
#             oc = omni_calib_list[idx]
#             if hasattr(oc, "found_points") and oc.found_points:
#                 if is_stereo:
#                     T_G_wrt_F_top = calibrator.calib_top.T_G_wrt_F_list[idx]
#                     T_G_wrt_F_bottom = calibrator.calib_bottom.T_G_wrt_F_list[idx]
#                     all_points_wrt_G = oc.mono_top.obj_points_homo
#                     # Recall on [G] points are on the XZ plane (Find margin from the difference of coordinates between consecutive points)
#                     margin_x = 3 * (all_points_wrt_G[0, 1, 0] - all_points_wrt_G[0, 0, 0])
#                     margin_z = 3 * (all_points_wrt_G[1, 0, 2] - all_points_wrt_G[0, 0, 2])
#                     # Create a mask_top offset from the outer corners of the grid
#                     points_wrt_G = np.ndarray((1, 4, 4))
#                     points_wrt_G[0, 0] = all_points_wrt_G[0, 0] + np.array([-margin_x, 0, -margin_z, 0])  # ORIGIN: Lower left corner
#                     points_wrt_G[0, 1] = all_points_wrt_G[-1, 0] + np.array([-margin_x, 0, +margin_z, 0])  # Upper left corner
#                     points_wrt_G[0, 2] = all_points_wrt_G[-1, -1] + np.array([+margin_x, 0, +margin_z, 0])  # Upper right corner
#                     points_wrt_G[0, 3] = all_points_wrt_G[0, -1] + np.array([+margin_x, 0, -margin_z, 0])  # lower right corner
#                     obj_pts_wrt_M_top = np.einsum("ij, mnj->mni", T_G_wrt_F_top, points_wrt_G)
#                     obj_pts_wrt_M_bottom = np.einsum("ij, mnj->mni", T_G_wrt_F_bottom, points_wrt_G)
#                     # Project the 4 margin chessboard corners as mask_top
#                     _, _, projected_corners_top = omni_model.top_model.get_pixel_from_3D_point_homo(obj_pts_wrt_M_top)
#                     _, _, projected_corners_bottom = omni_model.bot_model.get_pixel_from_3D_point_homo(obj_pts_wrt_M_bottom)
#
#                     # Masking the on the top view:
#                     mask_top = np.zeros(oc.omni_image.shape[0:2], dtype=np.uint8)  # Black, single channel mask
#                     cv2.fillConvexPoly(mask_top, projected_corners_top[..., :2].reshape(-1, 2).astype("int32"), color=(255, 255, 255), lineType=8, shift=0)
#                     mask_inverted_top = np.zeros(oc.omni_image.shape[0:2], dtype=np.uint8) + 255  # White, single channel mask
#                     cv2.fillConvexPoly(mask_inverted_top, projected_corners_top[..., :2].reshape(-1, 2).astype("int32"), color=(0, 0, 0), lineType=8, shift=0)
#                     masked_img_top = np.zeros_like(oc.omni_image)
#                     masked_img_top = cv2.bitwise_and(oc.omni_image, oc.omni_image, dst=masked_img_top, mask=mask_top)
#                     masked_img_base_top = np.zeros_like(oc.omni_image)
#                     masked_img_base_top = cv2.bitwise_and(base_img, base_img, masked_img_base_top, mask=mask_inverted_top)
#                     base_img = cv2.bitwise_or(masked_img_base_top, masked_img_top)  # Update resulting image as the new base_img
#
#                     # Masking the on the bottom view:
#                     mask_bottom = np.zeros(oc.omni_image.shape[0:2], dtype=np.uint8)  # Black, single channel mask
#                     cv2.fillConvexPoly(mask_bottom, projected_corners_bottom[..., :2].reshape(-1, 2).astype("int32"), color=(255, 255, 255), lineType=8, shift=0)
#                     mask_inverted_bottom = np.zeros(oc.omni_image.shape[0:2], dtype=np.uint8) + 255  # White, single channel mask
#                     cv2.fillConvexPoly(mask_inverted_bottom, projected_corners_bottom[..., :2].reshape(-1, 2).astype("int32"), color=(0, 0, 0), lineType=8, shift=0)
#                     masked_img_bottom = np.zeros_like(oc.omni_image)
#                     masked_img_bottom = cv2.bitwise_and(oc.omni_image, oc.omni_image, dst=masked_img_bottom, mask=mask_bottom)
#                     masked_img_base_bottom = np.zeros_like(oc.omni_image)
#                     masked_img_base_bottom = cv2.bitwise_and(base_img, base_img, masked_img_base_bottom, mask=mask_inverted_bottom)
#                     base_img = cv2.bitwise_or(masked_img_base_bottom, masked_img_bottom)  # Update resulting image as the new base_img
#                     if draw_detection:
#                         det_corner_pixels_top = oc.mono_top.image_points  # Must be a float32 for OpenCV to work!
#                         # SIMPLE: draw_points(base_img, det_corner_pixels_top[..., :2].reshape(-1, 2), color=(255, 0, 0), thickness=2)
#                         cv2.drawChessboardCorners(base_img, oc.mono_top.pattern_size_applied, det_corner_pixels_top.reshape(-1, 2), oc.found_points)
#
#                         det_corner_pixels_bottom = oc.mono_bottom.image_points  # Must be a float32 for OpenCV to work!
#                         # SIMPLE: draw_points(base_img, det_corner_pixels_bottom[..., :2].reshape(-1, 2), color=(0, 0, 255), thickness=2)
#                         cv2.drawChessboardCorners(base_img, oc.mono_bottom.pattern_size_applied, det_corner_pixels_bottom.reshape(-1, 2), oc.found_points)
#                 else:
#                     T_G_wrt_F = calibrator.T_G_wrt_F_list[idx]
#                     all_points_wrt_G = oc.obj_points_homo
#                     # Recall on [G] points are on the XZ plane (Find margin from the difference of coordinates between consecutive points)
#                     margin_x = 3 * (all_points_wrt_G[0, 1, 0] - all_points_wrt_G[0, 0, 0])
#                     margin_z = 3 * (all_points_wrt_G[1, 0, 2] - all_points_wrt_G[0, 0, 2])
#                     # Create a mask offset from the outer corners of the grid
#                     points_wrt_G = np.ndarray((1, 4, 4))
#                     points_wrt_G[0, 0] = all_points_wrt_G[0, 0] + np.array([-margin_x, 0, -margin_z, 0])  # ORIGIN: Lower left corner
#                     points_wrt_G[0, 1] = all_points_wrt_G[-1, 0] + np.array([-margin_x, 0, +margin_z, 0])  # Upper left corner
#                     points_wrt_G[0, 2] = all_points_wrt_G[-1, -1] + np.array([+margin_x, 0, +margin_z, 0])  # Upper right corner
#                     points_wrt_G[0, 3] = all_points_wrt_G[0, -1] + np.array([+margin_x, 0, -margin_z, 0])  # lower right corner
#                     obj_pts_wrt_M = np.einsum("ij, mnj->mni", T_G_wrt_F, points_wrt_G)
#                     # Project the 4 margin chessboard corners as mask
#                     _, _, projected_corners = omni_model.get_pixel_from_3D_point_homo(obj_pts_wrt_M)
#
#                     # Masking the single view:
#                     mask = np.zeros(oc.omni_image.shape[0:2], dtype=np.uint8)  # Black, single channel mask
#                     cv2.fillConvexPoly(mask, projected_corners[..., :2].reshape(-1, 2).astype("int32"), color=(255, 255, 255), lineType=8, shift=0)
#                     mask_inverted = np.zeros(oc.omni_image.shape[0:2], dtype=np.uint8) + 255  # White, single channel mask
#                     cv2.fillConvexPoly(mask_inverted, projected_corners[..., :2].reshape(-1, 2).astype("int32"), color=(0, 0, 0), lineType=8, shift=0)
#                     masked_img = np.zeros_like(oc.omni_image)
#                     masked_img = cv2.bitwise_and(oc.omni_image, oc.omni_image, dst=masked_img, mask=mask)
#                     masked_img_base = np.zeros_like(oc.omni_image)
#                     masked_img_base = cv2.bitwise_and(base_img, base_img, masked_img_base, mask=mask_inverted)
#                     base_img = cv2.bitwise_or(masked_img_base, masked_img)  # Update resulting image as the new base_img
#                     if draw_detection:
#                         det_corner_pixels = oc.image_points  # Must be a float32 for OpenCV to work!
#                         # SIMPLE: draw_points(base_img, det_corner_pixels[..., :2].reshape(-1, 2), color=(255, 0, 0), thickness=2)
#                         cv2.drawChessboardCorners(base_img, oc.pattern_size_applied, det_corner_pixels.reshape(-1, 2), oc.found_points)
#
#     if visualize:
#         cv2.imshow(win_name, base_img)
#         cv2.waitKey(1)
#
#     return base_img
#===============================================================================

def detect_ChArUco_board(gum_model, chessboard_params_filename, input_units = "cm", aruco_params_filename = "", load_aruco_params_from_pickle = True):
    '''
    Generates and saves the image of a ChArUco board pattern used for camera calibration
    @param image: The image from where to detected the ChArUco board
    @param chessboard_params_filename:  the comma separated file for chessboard sizing (first row) and pose information for each pattern (used by POV-Ray, not needed here)
    @param show: Indicates whether to show the image of the generated board in a window.
    '''
    vis_img_omni = gum_model.current_omni_img.copy()
    pano_img = gum_model.panorama.get_panoramic_image(vis_img_omni, set_own = False, border_RGB_color = (0, 255, 0))

    if pano_img is not None:
        print("Image seems valid.")

        # Note we did not need more than 50 markers, and we used the 4x4_50 dictionary so that it provides the highest inter-marker distance to increase the detection robustness
#         aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

        aruco_detector_params_tuner = None

        if aruco_detector_params_tuner is None:
            import os.path as osp
            if load_aruco_params_from_pickle:
                file_exists = osp.isfile(aruco_params_filename)
                if file_exists:
                    from omnistereo.common_tools import load_obj_from_pickle
                    aruco_detector_params_tuner = load_obj_from_pickle(aruco_params_filename)
                    aruco_detector_params_tuner.reset_images(img = pano_img)
                else:
                    aruco_detector_params_tuner = ArucoDetectionParamsTuner(img = pano_img, win_name = "Aruco Detection - Params tuner")
            else:
                aruco_detector_params_tuner = ArucoDetectionParamsTuner(img = pano_img, win_name = "Aruco Detection - Params tuner")
        else:
            aruco_detector_params_tuner.reset_images(img = pano_img)

        # Detect ChArUco corners:
        from omnistereo.calibration import CornerDetector
        corner_detector_masker = CornerDetector()
        tune_dynamically = True  # Allows for generation of masks before setting parameters in the tuner
        valid_manual_mask = True
        charuco_found = False
        while (not charuco_found) and valid_manual_mask:  # Pressing the Escape key breaks the loop
            # if img_number is not None: print('Re-processing omni_image %s using manual mask...' % (img_number))
            manually_masked_img, valid_manual_mask, roi_position = corner_detector_masker.get_masked_omni_image_manual(pano_img, crop_result = True)
            if valid_manual_mask:
                aruco_detector_params_tuner.reset_images(img = manually_masked_img)
                corners, ids, rejectedImgPoints = aruco_detector_params_tuner.start_tuning(aruco_dictionary = aruco_dict, save_file_name = aruco_params_filename)
                charuco_found = len(corners) > 4
                if charuco_found:  # And because we are cropping
                    roi_position = roi_position.astype(corners[0].dtype)  # Make sure data types are compatible
                    corners = corners + roi_position

        if ids is not None and len(ids) > 0:
            # Subpixel refinement in the ChArUco corners to find the traditional chessboard corners:
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            info_file = open(chessboard_params_filename, 'r')
            info_content_lines = info_file.readlines()  # Read contents is just a long string
            info_file.close()
            chessboard_size_info_list = info_content_lines[0].split(",")  # The rows, cols, width_row, width_col, margin are saved on the first line
            rows = int(chessboard_size_info_list[0])
            cols = int(chessboard_size_info_list[1])
            width_row = float(chessboard_size_info_list[2])  # The square length on the y-direction
            width_col = float(chessboard_size_info_list[3])  # The square length on the x-direction
            square_length_in = width_col
            # Convert to meters (units expected by ChaRuCo library)
            unit_conversion_factor = get_length_units_conversion_factor(input_units = input_units, output_units = "m")

            square_length = square_length_in * unit_conversion_factor  # in meters
            aruco_marker_length = square_length * .75

            charuco_board = cv2.aruco.CharucoBoard_create(squaresX = cols, squaresY = rows, squareLength = square_length, markerLength = aruco_marker_length, dictionary = aruco_dict)
            charuco_min_adjacent_markers = 2  # number of adjacent markers that must be detected to return a charuco corner
            # ids = np.arange(len(ids))
            # Filter out wrong ids (NOT helping):
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            #===================================================================
            # valid_aruco_corners_list = []
            # valid_aruco_ids_list = []
            # for index, id in enumerate(ids):
            #     if ids[index] < 22:  # Arbitrary for now
            #         valid_aruco_corners_list.append(corners[index])
            #         valid_aruco_ids_list.append(ids[index, 0])
            # retval, charucoCornersInterpolated, charucoIdsInterpolated = cv2.aruco.interpolateCornersCharuco(markerCorners=valid_aruco_corners_list, markerIds=np.array(valid_aruco_ids_list), image=pano_img, board=charuco_board, minMarkers=charuco_min_adjacent_markers)
            #===================================================================

            retval, charucoCornersInterpolated, charucoIdsInterpolated = cv2.aruco.interpolateCornersCharuco(markerCorners = corners, markerIds = ids, image = pano_img, board = charuco_board, minMarkers = charuco_min_adjacent_markers)

            vis_img_pano = cv2.aruco.drawDetectedCornersCharuco(pano_img, charucoCorners = charucoCornersInterpolated, charucoIds = charucoIdsInterpolated)
            win_name_pano = "ChArUco Resulting Corner Detection (on Panoramic Image)"
            cv2.namedWindow(win_name_pano, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name_pano, vis_img_pano)

            u_omni_coord, v_omni_coord, points_on_omni_from_pano = gum_model.panorama.get_omni_pixel_coords_from_panoramic_pixel(m_pano = charucoCornersInterpolated, use_LUTs = False)
            charucoCornersInterpolated_on_Omni = points_on_omni_from_pano[..., :2]
            # For some reason the above is not working!
            vis_img_omni = cv2.aruco.drawDetectedCornersCharuco(vis_img_omni, charucoCorners = charucoCornersInterpolated_on_Omni, charucoIds = charucoIdsInterpolated, cornerColor = (127, 0, 0))
            draw_points(vis_img_omni, charucoCornersInterpolated_on_Omni[..., :2].reshape(-1, 2), color = (255, 0, 0), thickness = 9)

            win_name_omni = "ChArUco Resulting Corner Detection (on Omnidirectional Image)"
            cv2.namedWindow(win_name_omni, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name_omni, vis_img_omni)
            cv2.waitKey(0)

def generate_ChArUco_board(board_filename, chessboard_params_filename, input_units = "cm", show = False):
    '''
    Generates and saves the image of a ChArUco board pattern used for camera calibration
    @param board_filename_template: The file name this generated ChArUco board will be saved as
    @param chessboard_params_filename:  the comma separated file for chessboard sizing (first row) and pose information for each pattern (used by POV-Ray, not needed here)
    @param show: Indicates whether to show the image of the generated board in a window.
    '''

    info_file = open(chessboard_params_filename, 'r')
    info_content_lines = info_file.readlines()  # Read contents is just a long string
    info_file.close()

    chessboard_size_info_list = info_content_lines[0].split(",")  # The rows, cols, width_row, width_col, margin are saved on the first line
    rows = int(chessboard_size_info_list[0])
    cols = int(chessboard_size_info_list[1])
    width_row = float(chessboard_size_info_list[2])  # The square length on the y-direction
    width_col = float(chessboard_size_info_list[3])  # The square length on the x-direction
    square_length_in = width_col
    margin = float(chessboard_size_info_list[4])
    # IMPORTANT: the number of corners with intersection is always 1 less than the number of row and cols of the pattern!
    pattern_size_str = '(' + str(rows - 1) + ', ' + str(cols - 1) + ')'
    # Convert to meters (units expected by ChaRuCo library)
    if input_units == "cm":
            unit_conversion_factor = 1. / 100.0
    elif input_units == "mm":
            unit_conversion_factor = 1. / 1000.0
    else:
        unit_conversion_factor = 1.

    square_length = square_length_in * unit_conversion_factor  # in meters
    aruco_marker_length = square_length * .75

    # Note we don't need more than 50 markers, so this 4x4_50 dictionary provides the highest inter-marker distance to increase the detection robustness
#     aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DI CT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    # Using the newer ChArUco pattern
    board = cv2.aruco.CharucoBoard_create(squaresX = cols, squaresY = rows, squareLength = square_length, markerLength = aruco_marker_length, dictionary = aruco_dict)
    # Testing the older Aruco board pattern
    #===========================================================================
    # marker_separation_length = square_length - aruco_marker_length
    # board = cv2.aruco.GridBoard_create(markersX=cols, markersY=rows, markerLength=aruco_marker_length, markerSeparation=marker_separation_length, dictionary=aruco_dict)
    #===========================================================================

    inch_to_meter = 0.0254  # 1 inch =  0.0254 meters
    # meter_to_inch = 39.37  # 1 meter = 39.37 inches
    ppi_standard = 72  # Standard PPI = 72
    one_meter_in_pixels = ppi_standard / inch_to_meter
    pixels_per_square = int(one_meter_in_pixels * square_length)
    board_size_in_pixels = (pixels_per_square * cols, pixels_per_square * rows)
    board_img = board.draw(board_size_in_pixels)

    if show:
        board_window_name = 'ChArUco Board'
        cv2.namedWindow(board_window_name, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(board_window_name, board_img)
        cv2.waitKey(0)

    # Dump the calibration board to a file
    cv2.imwrite(board_filename, board_img)

def get_depthmap_img_for_visualization(numpy_depthmap_float, depth_units_conversion_factor_to_m = 1.0, max_range_in_m = None, exponential_range_decay = 1.0, cam_model = None):
    from numpy import where, power, uint8, nanmax

    depthmap_inversed = numpy_depthmap_float * depth_units_conversion_factor_to_m
    if cam_model is not None:
        # Convert Euclidean depth values to Z-values
        depthmap_inversed = cam_model.get_depth_Z(depth = depthmap_inversed)

    if max_range_in_m is None:
        max_depth = nanmax(depthmap_inversed)
    else:
        # max_depth = min([max_range_in_m, max_depth])
        max_depth = max_range_in_m
    depthmap_inversed = where(depthmap_inversed <= max_depth, depthmap_inversed, max_depth)

    max_depth = power(max_depth, exponential_range_decay)
    depthmap_inversed = power(depthmap_inversed, exponential_range_decay)

    # Because the convention is to assign black (0) to the max depth
    depthmap = 255.0 * (max_depth - depthmap_inversed) / max_depth
    depthmap_uchar = depthmap.astype(uint8)
    return depthmap_uchar

def get_depthmap_img_for_visualization_from_png(depth_img_filename, depth_units_conversion_factor_to_m, max_range_in_m = None, exponential_range_decay = 1.0, cam_model = None):
    from numpy import where, power, uint8, nanmax
    from cv2 import imread, IMREAD_UNCHANGED

    depthmap_inversed = imread(depth_img_filename, IMREAD_UNCHANGED) * depth_units_conversion_factor_to_m

    if cam_model is not None:
        # Convert Euclidean depth values to Z-values
        depthmap_inversed = cam_model.get_depth_Z(depth = depthmap_inversed)

    if max_range_in_m is None:
        max_depth = nanmax(depthmap_inversed)
    else:
        # max_depth = min([max_range_in_m, max_depth])
        max_depth = max_range_in_m
    depthmap_inversed = where(depthmap_inversed <= max_depth, depthmap_inversed, max_depth)

    max_depth = power(max_depth, exponential_range_decay)

    depthmap_inversed = power(depthmap_inversed, exponential_range_decay)
    # Because the convention is to assign black (0) to the max depth
    depthmap = 255.0 * (max_depth - depthmap_inversed) / max_depth
    depthmap_uchar = depthmap.astype(uint8)
    return depthmap_uchar

def get_depthmap_img_for_visualization_from_txt(depth_filename, depth_units_conversion_factor_to_m, width, height, max_range_in_m = None, exponential_range_decay = 1.0, cam_model = None):
    from numpy import loadtxt, where, power, uint8, nanmax
    depthmap_inversed = loadtxt(depth_filename).reshape(height, width) * depth_units_conversion_factor_to_m

    if cam_model is not None:
        # Convert Euclidean depth values to Z-values
        depthmap_inversed = cam_model.get_depth_Z(depth = depthmap_inversed)

    if max_range_in_m is None:
        max_depth = nanmax(depthmap_inversed)
    else:
        # max_depth = min([max_range_in_m, max_depth])
        max_depth = max_range_in_m

    depthmap_inversed = where(depthmap_inversed <= max_depth, depthmap_inversed, max_depth)

    max_depth = power(max_depth, exponential_range_decay)
    depthmap_inversed = power(depthmap_inversed, exponential_range_decay)

    # Because the convention is to assign black (0) to the max depth
    depthmap = 255.0 * (max_depth - depthmap_inversed) / max_depth
    depthmap_uchar = depthmap.astype(uint8)
    return depthmap_uchar

def get_depthmap_img_from_txt(depth_filename, width, height, conversion_factor = 1.0):
    '''
    @param conversion_factor: The dimension conversion is related to the units in which the depth file was interpreted as
    '''
    from numpy import loadtxt, uint16
    depthmap_in = loadtxt(depth_filename).reshape(height, width)
    depthmap_uint16 = uint16(conversion_factor * depthmap_in)
    return depthmap_uint16

def get_depthmap_float32_from_png(depth_img_filename, conversion_factor = 1.0):
    '''
    @param conversion_factor: The dimension conversion is related to the units in which the depth file was interpreted as
    '''
    from numpy import float32
    from cv2 import imread, IMREAD_UNCHANGED
    depthmap_in = imread(depth_img_filename, IMREAD_UNCHANGED)
    depthmap_float32 = float32(conversion_factor * depthmap_in)
    return depthmap_float32
