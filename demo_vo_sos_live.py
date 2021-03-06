# -*- coding: utf-8 -*-
# demo_vo_sos_live.py

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
Live demo of frame-to-frame visual odometry for the single-camera omnistereo images collected for the vo_single_camera_sos project

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
from omnistereo.common_tools import make_sure_path_exists, str2bool, load_obj_from_pickle

from argparse import ArgumentParser
parser = ArgumentParser(description = 'Demo of frame-to-frame visual odometry for the Single-camera SOS images collected for the vo_single_camera_sos project.')
parser.register('type', 'bool', str2bool)  # add type 'bool' keyword to the registry because it doesn't exist by default!

parser.add_argument('results_path', nargs = 1, help = 'The path where the results will be save into.')

parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--calibrated_gums_file', help = 'Indicates the complete path and name of the calibrated GUMS pickle file', default = 'gums-calibrated.pkl', type = str)
optional.add_argument('--visualize_VO', help = '(Optional) Indicates whether to visualize the estimated 3D trajectory (and ground-truth if available).', default = True, type = 'bool')

parser.print_help()
args = parser.parse_args()

def main_sos_vo():
    scene_path = osp.realpath(osp.expanduser(args.results_path[0]))
    gums_calibrated_filename = osp.realpath(osp.expanduser(args.calibrated_gums_file))
    visualize_VO = args.visualize_VO

    # HARD-CODED flags:
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    use_perfect_model_VO = False  # WARNING: It may be used only when using SYNTHETIC images
    only_visualize_frames_from_existing_VO_poses_file = False  # <<< SETME: Will not run VO if True, but just reload the file from an existing experiment
    # NOTE: For DEBUG, set use_multithreads_for_VO <-- False
    use_multithreads_for_VO = True  # <<< 3D Visualization interactivity (is more responsive) when running the VO as a threads
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    scene_path_vo_results = osp.join(scene_path, "results-omni")  # VO Results path
    make_sure_path_exists(scene_path_vo_results)
    vis_name = "LIVE-SOS"
    if only_visualize_frames_from_existing_VO_poses_file:
        from omnistereo.common_plot import replay_VO_visualization
        replay_VO_visualization(scene_path_vo_results = scene_path_vo_results, first_image_index = 0, last_image_index = -1, step_for_poses = 1, vis_name = vis_name)
    else:
        from omnistereo.pose_est_tools import driver_VO_live
        from omnistereo.webcam_live import WebcamLive, CamAsWorkingThread
        cam = WebcamLive(cam_index = 0, mirror_image = False, file_name = "", cam_model = "BLACKFLY", show_img = True)
        success, omni_frame = cam.get_single_frame()
        # Use camera as a working thread in order to obtain the most recent image from the buffer:
        cam_working_thread = CamAsWorkingThread(cam = cam)

        # Use the thread_name parameter to name the VO Visualization window:
        if use_perfect_model_VO:
            print("Using 'theoretical' model for", vis_name)
            from omnistereo.cata_hyper_model import  get_theoretical_OmniStereo
            from omnistereo.common_cv import get_images
            mirror_images = get_images(calib_img_filename_template, indices_list = [0], show_images = False, return_names_only = False)
            use_existing_radial_bounds = True
            hyperbolic_model_theoretical = get_theoretical_OmniStereo(omni_img = mirror_images, radial_bounds_filename = radial_bounds_filename, theoretical_params_filename = theoretical_params_filename, model_version = model_version, is_synthetic = is_synthetic, use_existing_radial_bounds = use_existing_radial_bounds)
            driver_VO_live(camera_model = hyperbolic_model_theoretical, scene_path_vo_results = scene_path_vo_results, cam_working_thread = cam_working_thread, visualize_VO = visualize_VO, use_multithreads_for_VO = use_multithreads_for_VO, thread_name = vis_name)
        else:
            # Attempting to just load the calibrated GUMS model
            gums_calibrated = load_obj_from_pickle(gums_calibrated_filename)
            driver_VO_live(camera_model = gums_calibrated, scene_path_vo_results = scene_path_vo_results, cam_working_thread = cam_working_thread, visualize_VO = visualize_VO, use_multithreads_for_VO = use_multithreads_for_VO, thread_name = vis_name)

    from omnistereo.common_cv import clean_up
    clean_up(wait_key_time = 1)
    print("GOODBYE!")

if __name__ == '__main__':
    main_sos_vo()
