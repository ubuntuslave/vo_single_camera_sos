# -*- coding: utf-8 -*-
# demo_vo_rgbd.py

# Copyright (c) 2018, Carlos Jaramillo
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
Demo of frame-to-frame visual odometry for the RGB-D images collected for the vo_single_camera_sos project

@author: Carlos Jaramillo
@contact: omnistereo@gmail.com
'''

from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import os.path as osp
from os import listdir
import fnmatch
from omnistereo.common_tools import make_sure_path_exists, str2bool
from omnistereo.camera_models import RGBDCamModel

from argparse import ArgumentParser
parser = ArgumentParser(description = 'Demo of frame-to-frame visual odometry for the RGB-D images collected for the vo_single_camera_sos project.')
parser.register('type', 'bool', str2bool)  # add type 'bool' keyword to the registry because it doesn't exist by default!

parser.add_argument('sequence_path', nargs = 1, help = 'The path to the sequence where the rgbd folders is located.')

parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--is_synthetic', help = 'Determines whether the data is real or synthetic. This is necessary to use the appropriate camera intrinsic parameters.', type = 'bool')
optional.add_argument('--hand_eye_transformation', help = '(Optional) If real-life data, this indicates the complete path and name to hand-eye transformation file', default = 'rgbd_hand_eye_transformation.txt', type = str)
optional.add_argument('--visualize_VO', help = '(Optional) Indicates whether to visualize the estimated 3D trajectory (and ground-truth if available).', default = True, type = 'bool')

parser.print_help()
args = parser.parse_args()

def main_rgbd_vo():
    scene_path = osp.realpath(osp.expanduser(args.sequence_path[0]))
    visualize_VO = args.visualize_VO

    hand_eye_T = None
    if args.is_synthetic:
        focal_length_m = 1. / 1000.0  # in [m]
        depth_is_Z = False  # The synthetic dataset encodes the radial (Euclidean) distance as depth
        # POV-Ray synthetic
        fx = 554.256258
        fy = 554.256258
        center_x = 319.5
        center_y = 239.5
        scaling_factor = 1. / 1000.0  # [mm] to [m]
        do_undistortion = False
        distortion_coeffs = np.array([0., 0., 0., 0., 0.])
        #=======================================================================
        # Align the optical axis of a perspective camera is the +Z axis
        xaxis, yaxis, zaxis = [1, 0, 0], [0, 1, 0], [0, 0, 1]
        # Recall, for the RGB-D sensor we need to rotate -90 degrees around the scene's [S] x-axis
        from omnistereo.transformations import rotation_matrix
        hand_eye_T = rotation_matrix(-np.pi / 2.0, xaxis)  # Not longer in place if the hand-eye transformation is given
        #=======================================================================
    else:
        focal_length_m = 1. / 1000.0  # in [m] DUMMY!
        depth_is_Z = True
        fx = fy = 525.0
        center_x = 319.5
        center_y = 239.5
        scaling_factor = 1. / 1000.0  # [mm] to [m]
        do_undistortion = False
        distortion_coeffs = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])

    # HARD-CODED flags:
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    only_visualize_frames_from_existing_VO_poses_file = False  # <<< SETME: Will not run VO if True, but just reload the file from an existing experiment
    # NOTE: For DEBUG, set use_multithreads_for_VO <-- False
    use_multithreads_for_VO = True  # <<< 3D Visualization interactivity (is more responsive) when running the VO as a threads
    step_for_scene_images = 1
    first_image_index = 0
    last_image_index = -1  # -1 for up to the last one
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    if not args.is_synthetic:
        hand_eye_transformation_filename = osp.realpath(osp.expanduser(args.hand_eye_transformation))
        hand_eye_transformation_file_exists = osp.isfile(hand_eye_transformation_filename)
        if hand_eye_transformation_file_exists:
            from omnistereo.common_tools import get_poses_from_file
            T_Cest_wrt_Rgt_poses_7_tum_format_list, T_Cest_wrt_Rgt_transform_matrices_list = get_poses_from_file(poses_filename = hand_eye_transformation_filename, input_units = "m", output_working_units = "m", indices = None, pose_format = "tum", zero_up_wrt_origin = False, initial_T = None)
            hand_eye_T = T_Cest_wrt_Rgt_transform_matrices_list[0]  # There should be a single transformation in this file

    scene_prefix_filename = "*.png"
    scene_path_rgbd = osp.join(scene_path, "rgbd")
    scene_rgb_img_filename_template = osp.join(scene_path_rgbd, "rgb", scene_prefix_filename)
    num_scene_images = len(fnmatch.filter(listdir(osp.join(scene_path_rgbd, "rgb")), scene_prefix_filename))
    scene_depth_filename_template = osp.join(scene_path_rgbd, "depth", scene_prefix_filename)
    scene_path_vo_results = osp.join(scene_path, "results-rgbd")  # VO Results path
    make_sure_path_exists(scene_path_vo_results)
    path_to_scene_name, scene_name = osp.split(scene_path)

    rgbd_cam_model = RGBDCamModel(fx = fx, fy = fy, center_x = center_x, center_y = center_y, scaling_factor = scaling_factor, do_undistortion = do_undistortion, depth_is_Z = depth_is_Z, focal_length_m = focal_length_m)
    rgbd_cam_model.T_Cest_wrt_Rgt = hand_eye_T  # Apply hand-eye transformation for the camera (as found from some preceding procedure)
    # Use the thread_name parameter to name the VO Visualization window:
    vis_name = "%s-%s" % (scene_name, "RGB-D")
    if only_visualize_frames_from_existing_VO_poses_file:
        from omnistereo.common_plot import replay_VO_visualization
        replay_VO_visualization(scene_path_vo_results = scene_path_vo_results, first_image_index = first_image_index, last_image_index = last_image_index, step_for_poses = step_for_scene_images, vis_name = vis_name)
    else:
        from omnistereo.pose_est_tools import driver_VO
        driver_VO(camera_model = rgbd_cam_model, scene_path = scene_path_rgbd, scene_path_vo_results = scene_path_vo_results, scene_img_filename_template = scene_rgb_img_filename_template, depth_filename_template = scene_depth_filename_template, num_scene_images = num_scene_images, visualize_VO = visualize_VO, use_multithreads_for_VO = use_multithreads_for_VO, step_for_scene_images = step_for_scene_images, first_image_index = first_image_index, last_image_index = last_image_index, thread_name = vis_name)

    from omnistereo.common_cv import clean_up
    clean_up(wait_key_time = 1)
    print("GOODBYE!")

if __name__ == '__main__':
    main_rgbd_vo()
