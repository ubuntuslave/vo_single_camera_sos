# -*- coding: utf-8 -*-
# pose_est_tools.py

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
@package omnistereo_sensor_design
Tools for pose estimation from multiple views

@author: Carlos Jaramillo
'''

from __future__ import division
from __future__ import print_function

import cv2
import pyopengv
import numpy as np
from math import log10, sqrt
from time import process_time
from matplotlib.pyplot import get_cmap
from warnings import warn
from sys import exc_info
from omnistereo.common_tools import make_sure_path_exists, get_length_units_conversion_factor
from os.path import realpath, expanduser, join

def normalized(x):
    return x / np.linalg.norm(x)

def pose_relative_ransac_2D_to_2D(bearing_vectors1, bearing_vectors2, model_error_threshold = 0.001, rel_pose_est_algorithm = "STEWENIUS", outlier_fraction_known = 0.50):
    '''
    @param model_error_threshold: Defines the model's error threshold for the pose fitting model, which in this context, it's about twice the angle projection error because we reproject the triangulated 3D points to each view. In other words, the relative RANSAC pose estimation model computes the angular distance between the bearing (back projection) and forward projection angles. We do this for each view, so we they get added up as total measure of fitness quality.
    '''
    threshold = model_error_threshold
    n_points_for_model = -1
    w = 1.0 - outlier_fraction_known  # w = number of inliers in data / number of points in data
    # The probability that the RANSAC algorithm in some iteration selects only inliers from the input data set
    # when it chooses the n points from which the model parameters are estimated:
    # The number of iterations, N, is chosen high enough to ensure that the probability
    # (usually set to 0.99) that at least one of the sets of random samples does not include an outlier.
    desired_prob_only_inlier_selection = 0.99
    if rel_pose_est_algorithm == "NISTER" or rel_pose_est_algorithm == "STEWENIUS":
        n_points_for_model = 5
    elif rel_pose_est_algorithm == "SEVENPT":
        n_points_for_model = 7
    elif rel_pose_est_algorithm == "EIGHTPT":
        n_points_for_model = 8
    num_of_iters = log10(1.0 - desired_prob_only_inlier_selection) / log10(1.0 - w ** n_points_for_model)
    std_of_k = sqrt(1.0 - w ** n_points_for_model) / (w ** n_points_for_model)
    # Add the std. dev in order to gain additional confidence
#     max_iterations = 1000
    max_iterations = int(num_of_iters + 3 * std_of_k)

    ransac_transformation, indices_inliers = pyopengv.relative_pose_ransac(bearing_vectors1[..., :3], bearing_vectors2[..., :3], rel_pose_est_algorithm, threshold, max_iterations)
    # The following is not longer necessary because I modified "pyopengv" to also return the inlier indices
    #===========================================================================
    # ransac_transformation = pyopengv.relative_pose_ransac(bearing_vectors1[..., :3], bearing_vectors2[..., :3], rel_pose_est_algorithm, threshold, max_iterations)
    # indices_all = np.arange(len(bearing_vectors1))
    # # A final selection of inlier correspondences would be:
    # indices_inliers, indices_outliers = select_inliers_within_distance(model_coefficients=ransac_transformation, indices_all=indices_all, threshold=threshold, bearing_vectors1=bearing_vectors1[..., :3], bearing_vectors2=bearing_vectors2[..., :3], is_relative_2D_to_2D_case=True)
    #===========================================================================

    ransac_transformation_homo = np.identity(4)
    ransac_transformation_homo[:3] = ransac_transformation

    return ransac_transformation_homo, indices_inliers

def pose_absolute_ransac_3D_to_2D(bearing_vectors, points3D, model_error_threshold = 0.001, pose_est_algorithm = "EPNP", outlier_fraction_known = 0.50, max_iterations = -1):
    '''
    @param model_error_threshold: Defines the model's error threshold for the pose fitting model, which in this context, it's about twice the angle projection error because we reproject the triangulated 3D points to each view. In other words, the relative RANSAC pose estimation model computes the angular distance between the bearing (back projection) and forward projection angles. We do this for each view, so we they get added up as total measure of fitness quality.
    @param pose_est_algorithm: Absolute pose estimation implemented algorithms are: "TWOPT", "KNEIP", "GAO", "EPNP" and "GP3P"
    @param max_iterations: When the number of iteration is -1, this number will be computed online
    '''
    threshold = model_error_threshold
    if max_iterations < 0:
        n_points_for_model = -1
        w = 1.0 - outlier_fraction_known  # w = number of inliers in data / number of points in data
        # The probability that the RANSAC algorithm in some iteration selects only inliers from the input data set
        # when it chooses the n points from which the model parameters are estimated:
        # The number of iterations, N, is chosen high enough to ensure that the probability
        # (usually set to 0.99) that at least one of the sets of random samples does not include an outlier.
        desired_prob_only_inlier_selection = 0.99
        if pose_est_algorithm == "TWOPT":
            n_points_for_model = 2
        else:
            n_points_for_model = 3
        num_of_iters = log10(1.0 - desired_prob_only_inlier_selection) / log10(1.0 - w ** n_points_for_model)
        std_of_k = sqrt(1.0 - w ** n_points_for_model) / (w ** n_points_for_model)
        # Add the std. dev in order to gain additional confidence
        max_iterations = int(num_of_iters + 3 * std_of_k)

    ransac_transformation, indices_inliers = absolute_pose_ransac(bearing_vectors[..., :3], points3D[..., :3], pose_est_algorithm, threshold, max_iterations)
    # The following is not longer necessary because I modified "pyopengv" to also return the inlier indices
    #===========================================================================
    # ransac_transformation = absolute_pose_ransac(bearing_vectors[..., :3], points3D[..., :3], pose_est_algorithm, threshold, max_iterations)
    # indices_all = np.arange(len(bearing_vectors))
    # # A final selection of inlier correspondences would be:
    # indices_inliers_test, indices_outliers = select_inliers_within_distance(model_coefficients=ransac_transformation, indices_all=indices_all, threshold=threshold, bearing_vectors1=points3D[..., :3], bearing_vectors2=bearing_vectors[..., :3], is_relative_2D_to_2D_case=False)
    #===========================================================================

    ransac_transformation_homo = np.identity(4)
    ransac_transformation_homo[:3] = ransac_transformation

    return ransac_transformation_homo, indices_inliers

# Get the set of inliers that correspond to the best model found so far
def select_inliers_within_distance(model_coefficients, indices_all, threshold, bearing_vectors1, bearing_vectors2, is_relative_2D_to_2D_case):
    '''
    @param model_error_threshold: Defines the model's error threshold for the pose fitting model, which in this context, it's about twice the angle projection error because we reproject the triangulated 3D points to each view. In other words, the relative RANSAC pose estimation model computes the angular distance between the bearing (back projection) and forward projection angles. We do this for each view, so we they get added up as total measure of fitness quality.
    '''
    dist_scores_list = get_selected_distances_to_model(model_coefficients, indices_all, bearing_vectors1, bearing_vectors2, is_relative_2D_to_2D_case)
    dist_scores_list_as_nparray = np.array(dist_scores_list)
    inlier_test_indices = dist_scores_list_as_nparray < threshold
    inliers_indices = indices_all[inlier_test_indices]
    indices_outliers = indices_all[np.invert(inlier_test_indices)]
    # Loop-based way (non-efficiente)
    #===========================================================================
    # inliers_indices = []
    # for i in indices_all:
    #     if dist_scores_list[i] < threshold:
    #         inliers_indices.append(i)
    #===========================================================================

    return inliers_indices, indices_outliers

def get_selected_distances_to_model(model, indices, bearing_vectors1, bearing_vectors2, is_relative_2D_to_2D_case, verbose_debug = False):
    '''
    @param is_relative_2D_to_2D_case: Triangulation is necessary when the bearing_vectors1 are not 3D points.
    '''
    scores = []  # Will be the scores distances to return
    translation = model[:3, 3]
    rotation = model[:3, :3]

    inverse_solution_matrix = np.identity(4)
    inverse_solution_matrix[:3, :3] = rotation.T
    inverse_solution_matrix[:3, 3] = -inverse_solution_matrix[:3, :3].dot(translation[:3])

    if is_relative_2D_to_2D_case:
        p_all = pyopengv.triangulation_triangulate2(bearing_vectors1[..., :3], bearing_vectors2[..., :3], translation, rotation)
    else:
        p_all = bearing_vectors1.copy()

    for i in indices:  # TODO: vectorize this loop!
        p_homo = np.zeros(4)
        p_homo[:3] = p_all[i]
        p_homo[3] = 1.0  # TODO: doing this in the loop is inefficient (specially when we allready have 3D points in homogenous coordinates)

        if is_relative_2D_to_2D_case:
            reprojection1 = p_homo[:3]
            reprojection1 = normalized(reprojection1)
            bearing_f1 = bearing_vectors1[i, :3]

        reprojection2 = inverse_solution_matrix.dot(p_homo)[:3]
        reprojection2 = normalized(reprojection2)
        bearing_f2 = bearing_vectors2[i, :3]

        # bodyReprojection = inverseSolution * p_hom;
        # CHECKME: I think this extra conversion is for the non-central case only
        # reprojection2 =_adapter.getCamRotation(indices[i]).transpose() * (bodyReprojection - _adapter.getCamOffset(indices[i]));
        # where getCamRotation is the rotation from the corresponding camera back to the viewpoint origin.
        # reprojection2 = reprojection2 / reprojection2.norm();

        # bearing-vector based outlier criterium (select threshold accordingly):
        # RECALL: alpha is the angle between 2 unit vectors.
        # 1-(f1'*f2) = 1-cos(alpha)
        if is_relative_2D_to_2D_case:  # Relative pose case # Absolute pose case (as implemented in OpenGV's "CentralRelativePoseSacProblem.cpp"
            reproj_error1 = 1.0 - bearing_f1.T.dot(reprojection1)
            reproj_error2 = 1.0 - bearing_f2.T.dot(reprojection2)
            if verbose_debug:
                print("Reprojection Errors [%d]: 1:%.8f, 2:%.8f" % (i, reproj_error1, reproj_error2))
            # This reprojection error can be in the range [0:4]
            reproj_error_score = reproj_error1 + reproj_error2  # Sum of both reprojection errors
        else:  # Absolute pose case (as implemented in OpenGV's "AbsolutePoseSacProblem.cpp"
            # This reprojection error can be in the range [0:2]
            reproj_error_score = 1.0 - bearing_f2.T.dot(reprojection2)

        scores.append(reproj_error_score)

    return scores

def proportional_check(x, y, tol = 1e-2):
    xn = normalized(x)
    yn = normalized(y)
    return (np.allclose(xn, yn, rtol = 1e20, atol = tol) or
            np.allclose(xn, -yn, rtol = 1e20, atol = tol))

def match_features_frame_to_frame(cam_model, train_kpts, train_desc, query_kpts, query_desc, random_colors_RGB, max_horizontal_diff = -1, max_descriptor_distance_radius = -1, keypts_as_points_train = None, keypts_as_points_query = None, pano_img_train = None, pano_img_query = None, show_matches = False, win_name = 'Matches (Frame-to-Frame)'):
    '''
    @param max_descriptor_distance_radius: This distance means here is descriptor metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured in Pixels)!
    '''
    matches = cam_model.feature_matcher_for_motion.match(query_descriptors = query_desc, train_descriptors = train_desc, max_descriptor_distance_radius = max_descriptor_distance_radius)

    matched_train_indices = []
    matched_query_indices = []
    matched_kpts_train = []
    matched_kpts_query = []
    matched_desc_train = []
    matched_desc_query = []
    random_colors = []

    num_of_good_matches = int(cam_model.feature_matcher_for_motion.percentage_good_matches * len(matches))

    if num_of_good_matches > 0:
        matched_train_indices_all = [m.trainIdx for m in matches[:num_of_good_matches]]
        matched_query_indices_all = [m.queryIdx for m in matches[:num_of_good_matches]]
        matched_kpts_train_all = train_kpts[matched_train_indices_all]
        matched_kpts_query_all = query_kpts[matched_query_indices_all]
        # Filter matches according to keypoint distance constraints
        # TODO: only doing the constraint on the horizontal distances (u-coords)
        # CHECKME: Do we weed to take care of the margins (left and right) that wrap around. Write another function that fulfills this requirement.
        # TODO: pass the m coordinates already computed in the PanoramicCorrespondences class!
        if keypts_as_points_train is None:
            keypts_as_points_train = cv2.KeyPoint_convert(matched_kpts_train_all).astype(np.float)
        else:
            keypts_as_points_train = keypts_as_points_train[matched_train_indices_all]
        if keypts_as_points_query is None:
            keypts_as_points_query = cv2.KeyPoint_convert(matched_kpts_query_all).astype(np.float)
        else:
            keypts_as_points_query = keypts_as_points_query[matched_query_indices_all]

        if max_horizontal_diff >= 0:
            from omnistereo.common_cv import filter_pixel_correspondences
            validation_list = filter_pixel_correspondences(matched_points_top = keypts_as_points_train, matched_points_bot = keypts_as_points_query, min_rectified_disparity = -1, max_horizontal_diff = max_horizontal_diff)
            # Slice results
            matched_train_indices = np.array(matched_train_indices_all)[validation_list]
            matched_query_indices = np.array(matched_query_indices_all)[validation_list]
            matched_kpts_train = matched_kpts_train_all[validation_list]
            matched_kpts_query = matched_kpts_query_all[validation_list]
        else:
            matched_train_indices = np.array(matched_train_indices_all)
            matched_query_indices = np.array(matched_query_indices_all)
            matched_kpts_train = matched_kpts_train_all
            matched_kpts_query = matched_kpts_query_all

        matched_desc_train = train_desc[matched_train_indices]
        matched_desc_query = query_desc[matched_query_indices]
        try:
            random_colors = random_colors_RGB[matched_train_indices]  # because we are passing the colors from the reference frame (a.k.a. train)
        except:
            print("Problem, it only has", len(random_colors_RGB))
    if show_matches:
        from omnistereo.common_plot import draw_matches_between_frames
        draw_matches_between_frames(pano_img_train = pano_img_train, pano_img_query = pano_img_query, matched_kpts_train = matched_kpts_train, matched_kpts_query = matched_kpts_query, random_colors = random_colors, win_name = win_name)

    return (matched_train_indices, matched_kpts_train, matched_desc_train), (matched_query_indices, matched_kpts_query, matched_desc_query), random_colors

class StereoPanoramicFrame(object):

    def __init__(self, stereo_camera_model, frame_id, **kwargs):
        '''
        @param stereo_camera_model: The GUMS with an omnidirectional image already set and the corresponding "feature_matcher_for_static_stereo"
        '''
        self.frame_id = frame_id
        self.parent_id = kwargs.get("parent_id", -1)  # If no parent exists, assign -1
        self.T_frame_wrt_tracking_ref_frame = np.identity(4)  # Relative pose of the frame wrt keyframe (a.k.a. tracking reference frame). Identity by default
        self.panoramic_image_top = stereo_camera_model.top_model.panorama.panoramic_img.copy()
        self.panoramic_image_bottom = stereo_camera_model.bot_model.panorama.panoramic_img.copy()
        # Only applicable to the top-2-bottom omnistereo calibration
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        self.use_midpoint_triangulation = True  # More robust!
        self.use_opengv_triangulation = False
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        self.conversion_factor_length_to_m = get_length_units_conversion_factor(input_units = stereo_camera_model.units, output_units = "m")

        self.show_pano_keypoints = False
        self.show_stereo_pano_outlier_matches = False
        self.show_stereo_pano_correspondences_initial = False
        self.show_stereo_pano_correspondences_final = False  # <<< SET: True in order to visualize the top-2-bottom correspondences
        self.first_row_to_crop_bottom = 0  # <<<< SETME (TEMP) Used 20 for Journal Paper (Sensors)

        self.total_time = 0.  # To be used with time statistics

        self.median_win_size = 1  # Image processing applied before matching
        self.min_disp = 1  # for v-axis
        # for u-axis in order to filter vertically aligned features.
        # BECAREFUL, if the triangulation is not the midpoint method, then this should only be using exactly aligned matches on the same column
        if self.use_midpoint_triangulation:  # This is a constraint only for the top-2-bottom omnistereo
            self.max_u_dist = 1.5
        else:
            self.max_u_dist = 0.5
        # For filtering triangulated points according to range
        conversion_factor_dummy = get_length_units_conversion_factor(input_units = "m", output_units = stereo_camera_model.units)
        self.min_range = 0.5 * conversion_factor_dummy  # written in [m] but converted to whatever the units expected for the model are
        self.max_range = 20.0 * conversion_factor_dummy  # written in [m] but converted to whatever the units expected for the model are

        # Eventually, we compute this:
        self.pano_correspondences = None
        self.num_valid_keypoints = 0  # Number of static stereo correspondences
        self.establish_stereo_correspondences(omnistereo_model = stereo_camera_model)

    def _set_frame(self, frame):  # Instantiate blind copy from existing frame
        if type(frame) is StereoPanoramicFrame:
            from omnistereo.common_tools import copy_only_attributes
            copy_only_attributes(objfrom = frame, objto = self)

    def establish_stereo_correspondences(self, omnistereo_model, collect_time_statistics = False):
        if collect_time_statistics:
            start_total_time_frame = process_time()

        if collect_time_statistics:
            start_feature_detection_time = process_time()
        keypts_list_top, descriptors_list_top = omnistereo_model.top_model.detect_sparse_features_on_panorama(feature_detection_method = omnistereo_model.feature_matcher_for_static_stereo.feature_detection_method, num_of_features = omnistereo_model.feature_matcher_for_static_stereo.num_of_features, median_win_size = self.median_win_size, show = self.show_pano_keypoints)
        keypts_list_bot, descriptors_list_bot = omnistereo_model.bot_model.detect_sparse_features_on_panorama(feature_detection_method = omnistereo_model.feature_matcher_for_static_stereo.feature_detection_method, num_of_features = omnistereo_model.feature_matcher_for_static_stereo.num_of_features, median_win_size = self.median_win_size, show = self.show_pano_keypoints)
        if collect_time_statistics:
            end_feature_detection_time = process_time()
            feature_detection_time = end_feature_detection_time - start_feature_detection_time

        if collect_time_statistics:
            start_correspondences_setup_time = process_time()
        pano_feats_original = PanoramicCorrespondences(kpts_top_list = keypts_list_top, desc_top_list = descriptors_list_top, kpts_bot_list = keypts_list_bot, desc_bot_list = descriptors_list_bot, m_top_array = None, m_bot_array = None, random_colors_RGB_list = None, do_flattening = True)
        if collect_time_statistics:
            end_correspondences_setup_time = process_time()
            correspondences_setup_time = end_correspondences_setup_time - start_correspondences_setup_time

        (matched_m_top_initial, matched_kpts_top_initial, matched_desc_top_initial), (matched_m_bot_initial, matched_kpts_bot_initial, matched_desc_bot_initial), random_colors_RGB_initial = omnistereo_model.match_features_panoramic_top_bottom(keypts_list_top = keypts_list_top, desc_list_top = descriptors_list_top, keypts_list_bot = keypts_list_bot, desc_list_bot = descriptors_list_bot, min_rectified_disparity = self.min_disp, max_horizontal_diff = self.max_u_dist, show_matches = self.show_stereo_pano_correspondences_initial, win_name = 'Original Top-to-Bottom Matches t=%d' % (self.frame_id))

        # Get xyz and rgb sparse points
        # WISH: Get the floating point coordinates instead of int, so we can use precision elevation without LUTs
        # NOTE: at the moment, angles are being resolved discretely, which can add error to the triangulation
        az1, el1 = omnistereo_model.top_model.panorama.get_direction_angles_from_pixel_pano(matched_m_top_initial, use_LUTs = False)
        az2, el2 = omnistereo_model.bot_model.panorama.get_direction_angles_from_pixel_pano(matched_m_bot_initial, use_LUTs = False)

        # Get bearing vectors (normalized to sphere viewpoints):
        bearing_vectors_top = omnistereo_model.top_model.get_3D_point_from_angles_wrt_focus(azimuth = az1, elevation = el1)
        bearing_vectors_bottom = omnistereo_model.bot_model.get_3D_point_from_angles_wrt_focus(azimuth = az2, elevation = el2)
        # Reduce the extra dimension on the head
        bearing_vectors_1 = bearing_vectors_top[0, ..., :3]
        bearing_vectors_2 = bearing_vectors_bottom[0, ..., :3]
        bearing_vectors_inliers_top = bearing_vectors_1
        bearing_vectors_inliers_bottom = bearing_vectors_2

        if self.use_opengv_triangulation:
            T_bot_wrt_top_translation = omnistereo_model.T_bot_wrt_top[:3, 3]
            T_bot_wrt_top_rotation_matrix = omnistereo_model.T_bot_wrt_top[:3, :3]
            xyz_points_initial_wrt_top = triangulation_triangulate2(bearing_vectors_inliers_top, bearing_vectors_inliers_bottom, T_bot_wrt_top_translation, T_bot_wrt_top_rotation_matrix)
            # CHECKME: a bit redundant as this gets only needed for the Keyframe and for now I'm recalculating it for the VO as 3D-to-2D fashion
            ones_matrix = np.ones(xyz_points_initial_wrt_top.shape[:-1])[..., np.newaxis]
            xyz_points_initial_wrt_top_homo = np.concatenate((xyz_points_initial_wrt_top, ones_matrix), axis = -1)
            xyz_points_initial_wrt_C = np.einsum("ij, nj->ni", omnistereo_model.top_model.T_model_wrt_C, xyz_points_initial_wrt_top_homo)
        else:
            xyz_points_initial_wrt_C = omnistereo_model.get_triangulated_point_from_direction_angles(dir_angs_top = (az1, el1), dir_angs_bot = (az2, el2), use_midpoint_triangulation = self.use_midpoint_triangulation)
            xyz_points_initial_wrt_C = xyz_points_initial_wrt_C[0]  # Reduce the extra dimension on the head
            # CHECKME: a bit redundant as this gets only needed for the Keyframe and for now I'm recalculating it for the VO as 3D-to-2D fashion
            # xyz_points_initial_wrt_top_homo = np.einsum("ij, nj->ni", omnistereo_model.T_omnistereo_C_wrt_top, xyz_points_initial_wrt_C)

        # Filter outlier feature correspondences by projecting 3D points and measuring pixel norm to matched_m_top and matched_m_bot, so only pixels under a certain distance threshold remain.
        # good_points_indices = omnistereo_model.filter_panoramic_points_due_to_reprojection_error(matched_m_top, matched_m_bot, xyz_points_initial_wrt_C, pixel_error_threshold=pixel_error_threshold)
        good_points_indices = omnistereo_model.filter_panoramic_points_due_to_range(xyz_points_initial_wrt_C, min_3D_range = self.min_range, max_3D_range = self.max_range)
        num_of_inliers = np.count_nonzero(good_points_indices)
        # print(num_of_inliers, "inliers out of", num_point_correspondences_pano, "intial correspondences from panoramas.")
        self.num_valid_keypoints = num_of_inliers
        xyz_points_wrt_C = xyz_points_initial_wrt_C[good_points_indices]
        # xyz_points_wrt_top = xyz_points_initial_wrt_top_homo[good_points_indices]
        self.bearing_vectors_top_stereo_triangulated = bearing_vectors_inliers_top[good_points_indices]
        self.bearing_vectors_bottom_stereo_triangulated = bearing_vectors_inliers_bottom[good_points_indices]
        matched_m_top = matched_m_top_initial[good_points_indices]
        matched_m_bot = matched_m_bot_initial[good_points_indices]
        random_colors_RGB = random_colors_RGB_initial[good_points_indices]
        matched_kpts_top = matched_kpts_top_initial[good_points_indices]
        matched_desc_top = matched_desc_top_initial[good_points_indices]
        matched_kpts_bot = matched_kpts_bot_initial[good_points_indices]
        matched_desc_bot = matched_desc_bot_initial[good_points_indices]

        # Just show the resulting matches:
        # TODO: Provide real colors (not just the randomly provided for matching visualization!)
        if self.show_stereo_pano_correspondences_final:
            _, _ = filter_correspondences_manually(train_img = omnistereo_model.top_model.panorama.panoramic_img, query_img = omnistereo_model.bot_model.panorama.panoramic_img, train_kpts = matched_kpts_top, query_kpts = matched_kpts_bot, colors_RGB = random_colors_RGB, first_row_to_crop_bottom = self.first_row_to_crop_bottom, do_filtering = False, win_name = 'Final Matches t=%d' % (self.frame_id))

        # Only needed if saving point cloud as PCD
        # (points_3D, rgb_points), (xyz_PCL_cloud, rgb_PCL_cloud) = omnistereo_model.generate_point_clouds(xyz_points_wrt_C, matched_m_top, rgb_colors=random_colors_RGB, use_PCL=True, export_to_pcd=save_pcl, cloud_path=points_3D_path, cloud_index=self.frame_id)

        # Put correspondence data in list of PanoramicCorrespondences
        self.pano_correspondences = PanoramicCorrespondences(kpts_top_list = matched_kpts_top, desc_top_list = matched_desc_top, kpts_bot_list = matched_kpts_bot, desc_bot_list = matched_desc_bot, points_3D = xyz_points_wrt_C, m_top_array = matched_m_top, m_bot_array = matched_m_bot, random_colors_RGB_list = random_colors_RGB, do_flattening = False)

        if collect_time_statistics:
            end_total_time_frame = process_time()
            self.total_time = end_total_time_frame - start_total_time_frame
            print("Stablishing stereo correspondences took: {time:.8f} [s]".format(time = self.total_time))

class RGBDFrame(object):

    def __init__(self, rgbd_camera_model, frame_id, **kwargs):
        '''
        @param stereo_camera_model: The GUMS with an omnidirectional image already set and the corresponding "feature_matcher_for_static_stereo"
        '''
        self.rgbd_camera_model = rgbd_camera_model
        self.frame_id = frame_id
        self.parent_id = kwargs.get("parent_id", -1)  # If no parent exists, assign -1

        # Usually the optical axis of a perspective camera is the +Z axis
#         xaxis, yaxis, zaxis = [1, 0, 0], [0, 1, 0], [0, 0, 1]
        # Recall, for the RGB-D sensor we need to rotate -90 degrees around the scene's [S] x-axis
#         T_C_wrt_S_init_default = tr.rotation_matrix(-np.pi / 2.0, xaxis)  # Not longer in place if the hand-eye transformation is given
        T_C_wrt_S_init_default = np.identity(4)
        self.T_frame_wrt_tracking_ref_frame = kwargs.get("T_wrt_ref", T_C_wrt_S_init_default)  # Relative pose of the frame wrt keyframe (a.k.a. tracking reference frame). Identity by default
        self.conversion_factor_length_to_m = get_length_units_conversion_factor(input_units = rgbd_camera_model.units, output_units = "m")

        self.total_time = 0.  # To be used with time statistics

        self.show_keypoints = False
        self.show_depth_img = False

        self.median_win_size = 0  # Image processing applied before matching
        # RECALL: Manufacturer states the range (distance) is between 0.8m and 3.5m for the ASUS Xtion Pro
        self.min_range = 0.8  # in [m]
        self.max_range = 7.  # in [m]: Officially is 3.5 m

        self.mask = kwargs.get("mask", None)
        self.rgb_img = kwargs.get("rgb_img", None)
        self.depth_map = kwargs.get("depth_map", None)

        # Eventually, we compute these:
        self.num_valid_keypoints = 0
        self.keypoints_and_descriptors = None
        self.bearing_vectors = None
        self.keypoints_3D_points = None
        self.current_depth = None
        if (self.rgb_img is not None) and (self.depth_map is not None):
            self.establish_keypoints(rgb = self.rgb_img, depth = self.depth_map)

    def _set_frame(self, frame):  # Instantiate blind copy from existing frame
        if type(frame) is RGBDFrame:
            from omnistereo.common_tools import copy_only_attributes
            copy_only_attributes(objfrom = frame, objto = self)

    def detect_sparse_features(self, rgb_img, feature_detection_method = "ORB", num_of_features = 50, mask = None, median_win_size = 0, show = True):
        '''
        @param median_win_size: Window size for median blur filter. If value is 0, there is no filtering.
        @return: List of list of kpts_top and ndarray of descriptors. The reason we use a list of each result is for separating mask-wise results for matchin.
        '''
        # Possible Feature Detectors are:
        # SIFT, SURF, DAISY
        # ORB:  oriented BRIEF
        # BRISK
        # AKAZE
        # KAZE
        # AGAST: Uses non-maximal suppression (NMS) by default
        # FAST: Uses non-maximal suppression (NMS) by default
        # GFT: Good Features to Track
        # MSER: http://docs.opencv.org/master/d3/d28/classcv_1_1MSER.html#gsc.tab=0

        # Possible Feature Descriptors are:
        # SIFT, SURF: work to describe FAST and AGAST kpts_top
        # ORB:  oriented BRIEF
        # BRISK
        # AKAZE
        # KAZE
        # DAISY: works to describe FAST and AGAST kpts_top
        # FREAK: is essentially a descriptor.
        # LATCH: is essentially a descriptor.
        # LUCID: The CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching

        if feature_detection_method.upper() == "ORB":
            detector = cv2.ORB_create(nfeatures = num_of_features)
            descriptor = detector
#             descriptor = cv2.xfeatures2d.LATCH_create(rotationInvariance=True)  # because ORB kpts_top provide the patch orientation
#             descriptor = cv2.xfeatures2d.FREAK_create()
            # detector.setPatchSize(10)  # TODO: set according to mask's width
        if feature_detection_method.upper() == "LUCID":
            detector = cv2.xfeatures2d.LUCID_create()
            descriptor = detector
        if feature_detection_method.upper() == "AKAZE":
            detector = cv2.AKAZE_create()
            descriptor = detector
        if feature_detection_method.upper() == "KAZE":
            detector = cv2.KAZE_create()
            descriptor = detector
        if feature_detection_method.upper() == "BRISK":
            detector = cv2.BRISK_create()
            descriptor = detector
        if feature_detection_method.upper() == "SIFT":
            detector = cv2.xfeatures2d.SIFT_create()
            descriptor = detector
        if feature_detection_method.upper() == "SURF":
            detector = cv2.xfeatures2d.SURF_create()
            descriptor = detector
        if feature_detection_method.upper() == "FAST":
            detector = cv2.FastFeatureDetector_create()  # Uses non-maximal suppression (NMS) by default
            detector.setNonmaxSuppression(True)
            descriptor = cv2.ORB_create(nfeatures = num_of_features)
#             descriptor = cv2.xfeatures2d.DAISY_create()
#             descriptor = cv2.xfeatures2d.FREAK_create()
#             descriptor = cv2.xfeatures2d.SIFT_create() # FIXME: ORB and BRISK don't seem to work
        if feature_detection_method.upper() == "AGAST":
            detector = cv2.AgastFeatureDetector_create()  # Uses non-maximal suppression (NMS) by default
            detector.setNonmaxSuppression(True)
            descriptor = cv2.ORB_create(nfeatures = num_of_features)
#             descriptor = cv2.BRISK_create()
#             descriptor = cv2.xfeatures2d.DAISY_create()
#             descriptor = cv2.xfeatures2d.FREAK_create()
#             descriptor = cv2.xfeatures2d.SIFT_create()
        if feature_detection_method.upper() == "GFT":
            useHarrisDetector = False  # I prefer not to use Harris Corners because not many matches occur on them
#             descriptor = cv2.xfeatures2d.DAISY_create()
#             descriptor = cv2.xfeatures2d.LATCH_create(rotationInvariance = False)  # we are setting this rotationInvariance parameter to False because GFT doesn't provide the patch orientation
            descriptor = cv2.ORB_create(nfeatures = num_of_features)
#             descriptor = cv2.BRISK_create()
#             descriptor = cv2.xfeatures2d.FREAK_create()
#             descriptor = cv2.xfeatures2d.LUCID_create()

        img = rgb_img.copy()
        if img is not None:
            if median_win_size > 0:
                img = cv2.medianBlur(img, median_win_size)
            if feature_detection_method.upper() == "GFT":
                if img.ndim == 3:
                    img_detect_with_GFT = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # GFT needs a grayscale image
                else:
                    img_detect_with_GFT = img.copy()

        keypts_detected_list = []
        descriptors_list = []
        if mask is None:
            mask = self.mask

        #=======================================================================
        # start_detection_time = process_time()
        #=======================================================================
        if feature_detection_method.upper() == "GFT":
            pts_GFT = cv2.goodFeaturesToTrack(image = img_detect_with_GFT, maxCorners = num_of_features, qualityLevel = 0.01, minDistance = 5, mask = mask, useHarrisDetector = useHarrisDetector)
            keypts_detected_on_mask = cv2.KeyPoint_convert(pts_GFT)
        else:
            keypts_detected_on_mask = detector.detect(image = img, mask = mask)
        #=======================================================================
        # end_detection_time = process_time()
        # detection_time = end_detection_time - start_detection_time
        #=======================================================================

        keypts_detected, descriptors = descriptor.compute(image = img, keypoints = keypts_detected_on_mask)

        keypts_detected_list.append(keypts_detected)
        descriptors_list.append(descriptors)
        # kpts_top are put in a list of length n
        # descriptor are ndarrays of n x desc_size

        if show:
            img_with_keypts = img.copy()
            img_with_keypts = cv2.drawKeypoints(img, keypts_detected, outImage = img_with_keypts)
            detected_kpts_win_name = "Detected Keypoints"
            cv2.namedWindow(detected_kpts_win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(detected_kpts_win_name, img_with_keypts)
            cv2.waitKey(1)

        return keypts_detected_list, descriptors_list

    def filter_3D_points_due_to_range(self, xyz_points_wrt_C, min_3D_range = 0, max_3D_range = 0.):
        '''
        Filter outlier feature correspondences by projecting 3D points under a certain range threshold remain.

        @param xyz_points_wrt_C: The coordinates of the estimated points wrt to the common frame [C]
        @param min_3D_range: The minimum euclidean norm to be considered a valid point. 0 by default
        @param max_3D_range: The maximum euclidean norm to be considered a valid point. If 0 (defaul), this filtering is bypassed.

        @return: a Boolean list related to the validity of the indices of good points from set
        '''
        valid_indices = np.ones(shape = (xyz_points_wrt_C.shape), dtype = "bool")
        if min_3D_range > 0 or max_3D_range > 0:
            norm_of_3D_points = np.linalg.norm(xyz_points_wrt_C, axis = 0, keepdims = True)
            norm_of_3D_points_clean_nans = np.nan_to_num(norm_of_3D_points, copy = True)
            if min_3D_range > 0:
                valid_min_ranges = np.where(norm_of_3D_points_clean_nans >= min_3D_range, True, False)
                valid_indices = np.logical_and(valid_indices, valid_min_ranges)

            if max_3D_range > 0:
                valid_max_ranges = np.where(norm_of_3D_points_clean_nans <= max_3D_range, True, False)
                valid_indices = np.logical_and(valid_indices, valid_max_ranges)

        return valid_indices

    def get_depth_img_visualization(self):
        from omnistereo.common_cv import get_depthmap_img_for_visualization
        depth_img = get_depthmap_img_for_visualization(numpy_depthmap_float = self.current_depth, max_range_in_m = self.max_range, cam_model = self.rgbd_camera_model)
        depth_img_vis = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
        return depth_img_vis

    def establish_keypoints(self, rgb, depth):
        from omnistereo.camera_models import KeyPointAndDescriptor, get_normalized_points
        self.current_depth = depth
        if self.show_depth_img:
            depth_img_vis = self.get_depth_img_visualization()
            cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Depth Image", depth_img_vis)
            cv2.waitKey(1)

        keypts_list, descriptors_list = self.detect_sparse_features(rgb_img = rgb, feature_detection_method = self.rgbd_camera_model.feature_matcher_for_motion.feature_detection_method, num_of_features = self.rgbd_camera_model.feature_matcher_for_motion.num_of_features, median_win_size = self.median_win_size, show = self.show_keypoints)
        all_keypoints_and_descriptors = KeyPointAndDescriptor(kpts_list = keypts_list[0], desc_list = descriptors_list[0], coords_array = None, random_colors_RGB_list = None, do_flattening = False)
        # For all kpts_top's coordinates in the image, compute their XYZ values
        all_keypoints_3D_points = self.rgbd_camera_model.get_XYZ(depth = self.current_depth, u_coords = all_keypoints_and_descriptors.pixel_coords[..., 0].astype(np.uint), v_coords = all_keypoints_and_descriptors.pixel_coords[..., 1].astype(np.uint))
        # Find those indices on not NANs:
        valid_keypoints_indices = np.logical_not(np.isnan(all_keypoints_3D_points[..., 2]))
        # Filter due to range:
        points_within_range_indices = self.filter_3D_points_due_to_range(all_keypoints_3D_points[..., 2], min_3D_range = self.min_range, max_3D_range = self.max_range)
        num_points_within_range = np.count_nonzero(points_within_range_indices)
        # print(num_points_within_range, "points within range")
        valid_indices = np.logical_and(valid_keypoints_indices, points_within_range_indices)
        self.keypoints_3D_points = all_keypoints_3D_points[valid_indices]  # Remove those keypoints, descriptor without depth (with NANs)
        self.bearing_vectors = get_normalized_points(self.keypoints_3D_points)
        self.keypoints_and_descriptors = KeyPointAndDescriptor(kpts_list = np.array(all_keypoints_and_descriptors.keypoints)[valid_indices[0]], desc_list = all_keypoints_and_descriptors.descriptors[valid_indices[0]], coords_array = all_keypoints_and_descriptors.pixel_coords[valid_indices], random_colors_RGB_list = all_keypoints_and_descriptors.random_colors_RGB[valid_indices[0]], do_flattening = False)
        self.num_valid_keypoints = len(self.keypoints_3D_points)

class StereoPanoramicKeyFrame(StereoPanoramicFrame):

    def __init__(self, frame, **kwargs):
        '''
        @param frame: The StereoPanoramicFrame for this keyframe inheriting all properties from the child frame
        '''
        self._set_frame(frame)
        self.children_ids = []

class RGBDKeyFrame(RGBDFrame):

    def __init__(self, frame, **kwargs):
        '''
        @param frame: The StereoPanoramicFrame for this keyframe inheriting all properties from the child frame
        '''
        self._set_frame(frame)
        self.children_ids = []

from omnistereo.common_plot import draw_matches_between_frames
from  omnistereo.common_cv import filter_correspondences_manually
from omnistereo.camera_models import FeatureMatcher, PanoramicCorrespondences
import omnistereo.transformations as tr
from pyopengv import triangulation_triangulate, triangulation_triangulate2
from pyopengv import absolute_pose_noncentral_ransac, absolute_pose_noncentral_optimize_nonlinear
from pyopengv import absolute_pose_ransac, absolute_pose_optimize_nonlinear

class TrackerSE3(object):

    def __init__(self, camera_model, show_3D_points = False, **kwargs):
        self.camera_model = camera_model
        self.show_3D_points = show_3D_points
        self.save_correspondence_images = kwargs.get("save_correspondence_images", False)
        self.results_path = kwargs.get("results_path", "~/temp")
        self.correspondence_images_path = ""
        if self.save_correspondence_images:
            self.correspondence_images_path = join(realpath(expanduser(self.results_path)), "final_correspondences_images")
            make_sure_path_exists(self.correspondence_images_path)
        self.T_C_wrt_S_init = tr.identity_matrix()
        self.T_C_curr_frame_wrt_S_est = tr.identity_matrix()
        self.xyz_homo_points_wrt_C_inliers = []  # XYZ homogeneous coordinates of inliers for the estimated pose
        self.rgb_points_inliers = []  # Colors for inlier points
        self.num_tracked_correspondences = 0  # RANSAC inliers of current tracking
        self.inlier_tracked_correspondences_ratio = 0.  # RANSAC inliers of current tracking
        self.number_of_cams = 1  # Just a default number that should be changed if needed
        self.T_Ckey_wrt_S_est_list = []  # This should only contain a list of keyframe poses
        self.set_global_parameters_for_tracking()  # This is done here because we are attempting to unify the trackers inheriting from this class

    def set_global_parameters_for_tracking(self):
        self.show_f2f_correspondences_initial = False
        self.show_f2f_correspondences_final = False
        self.backprojection_score_threshold_3D_to_2D_in_degrees = 0.5  # used for filtering some points due to triangulation
        self.backprojection_score_threshold_3D_to_2D = 1.0 - np.cos(np.deg2rad(self.backprojection_score_threshold_3D_to_2D_in_degrees))
        #=======================================================================
        # self.use_midpoint_triangulation = True
        # self.use_opengv_triangulation = True
        # self.backprojection_score_threshold_2D_to_2D_in_degrees = 0.5  # Used for filtering some points due to triangulation
        # self.backprojection_score_threshold_2D_to_2D = 1.0 - np.cos(np.deg2rad(self.backprojection_score_threshold_2D_to_2D_in_degrees))
        #=======================================================================

        self.detection_method = "GFT"
        self.matching_type = "BF"  # "BF" for brute force matching, or "FLANN" for matching vie the Fast Library for Approximate Nearest Neighbors
        self.k_best_matches = 1  # To set the number of k nearest matches in the matcher
        self.percentage_good_matches = 1.0
        # For motion-stage matching:
        # NOTE: because we are matching already detected/described features from the panorama, the only settings applicable here are the k_best and the percentage
        self.use_descriptor_radius_match_for_motion = False  # This name is deceiving because the radius distance threshold is actually a descriptor metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured in Pixels)!
        self.num_features_detection_for_motion = 1000  # It doesn't apply to the StereoPanoramicFrame if it has bucketting masks
        self.max_horizontal_search_ratio = 0.50  # A factor that multiplies by the width of the image to be a search constraint on the initial matches
                                                # NOTE: this factor is divided in 2 for the panoramic images
        # RANSAC related:
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        self.pose_est_algorithm = "EPNP"  # NOTE: only used for the "central" absolute_pose_ransac. The non-central case will ALWAYS use "GP3P"
        self.n_points_for_RANSAC_model = 3  # Usually for the non-central absolute pose model (P3P)
        # TODO: we need to compute this statistically to determine the max_ransac_iterations
        # IDEA: adjust the number of iterations based on the statistics of outliers from history.
        #       Then, have a threshold of change for when to adjust this value
        self.correspondences_outliers_fraction = 0.90  # Conservative value based on statistics from real data (the synthetic case could be just 0.60)
        self.max_ransac_iterations_3D_to_2D = -1  # With -1, it will be computed once initially
        # TODO: adjust the number of iterations based on the statistics of outliers from history.
        # IDEA: have a threshold of change to adjust this value
        if self.max_ransac_iterations_3D_to_2D < 0:
            self.max_ransac_iterations_3D_to_2D = self.compute_num_of_iterations_RANSAC(n_points_for_model = self.n_points_for_RANSAC_model, correspondences_outliers_fraction = self.correspondences_outliers_fraction)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def compute_num_of_iterations_RANSAC(self, n_points_for_model, correspondences_outliers_fraction):
        w = 1.0 - correspondences_outliers_fraction  # w = number of inliers in data / number of points in data
        # The probability that the RANSAC algorithm in some iteration selects only inliers from the input data set
        # when it chooses the n points from which the model parameters are estimated:
        # The number of iterations, N, is chosen high enough to ensure that the probability
        # (usually set to 0.99) that at least one of the sets of random samples does not include an outlier.
        desired_prob_only_inlier_selection = 0.998
        num_of_iters = log10(1.0 - desired_prob_only_inlier_selection) / log10(1.0 - w ** n_points_for_model)
        std_of_k = sqrt(1.0 - w ** n_points_for_model) / (w ** n_points_for_model)
        # Add the std. dev in order to gain additional confidence
        max_ransac_iterations = int(num_of_iters + 3 * std_of_k)
        return max_ransac_iterations

class TrackerStereoSE3(TrackerSE3):

    def __init__(self, camera_model, show_3D_points = False, **kwargs):
        TrackerSE3.__init__(self, camera_model, show_3D_points, **kwargs)
        self.omnistereo_model = self.camera_model  # Renamed as to remind of its usage
        self.number_of_cams = 2  # Omnistereo camera model has 2 views: top and bottom
        self.bootstrap_tracker()

        if self.save_correspondence_images:
            self.correspondence_images_path_top = join(self.correspondence_images_path, "top")
            make_sure_path_exists(self.correspondence_images_path_top)
            self.correspondence_images_path_bottom = join(self.correspondence_images_path, "bottom")
            make_sure_path_exists(self.correspondence_images_path_bottom)

    def track_frame(self, reference_frame, current_frame):
        self.num_tracked_correspondences = 0
        self.inlier_tracked_correspondences_ratio = 0.
        # 1) Find matches between views at frames at time k-1 and k
        # Find matches from top between frames at time k-1 and k
        (matched_train_indices_pano_top, matched_kpts_train_pano_top, matched_desc_train_pano_top), (matched_query_indices_pano_top, matched_kpts_query_pano_top, matched_desc_query_pano_top), random_colors_pano_top = match_features_frame_to_frame(cam_model = self.omnistereo_model, train_kpts = reference_frame.pano_correspondences.kpts_top, train_desc = reference_frame.pano_correspondences.desc_top, query_kpts = current_frame.pano_correspondences.kpts_top, query_desc = current_frame.pano_correspondences.desc_top, random_colors_RGB = reference_frame.pano_correspondences.random_colors_RGB,
                                                                                                                                                                                                                                                                 keypts_as_points_train = reference_frame.pano_correspondences.m_top, keypts_as_points_query = current_frame.pano_correspondences.m_top,
                                                                                                                                                                                                                                                                 max_horizontal_diff = self.max_horizontal_diff_f2f_matches, max_descriptor_distance_radius = -1, pano_img_train = reference_frame.panoramic_image_top,
                                                                                                                                                                                                                                                                 pano_img_query = current_frame.panoramic_image_top, show_matches = self.show_f2f_correspondences_initial, win_name = 'All Triangulated Matches (Top-to-Top) t=%d' % (current_frame.frame_id))
        # Find matches from bottom between frames at time k-1 and k
        (matched_train_indices_pano_bottom, matched_kpts_train_pano_bottom, matched_desc_train_pano_bottom), (matched_query_indices_pano_bottom, matched_kpts_query_pano_bottom, matched_desc_query_pano_bottom), random_colors_pano_bottom = match_features_frame_to_frame(cam_model = self.omnistereo_model, train_kpts = reference_frame.pano_correspondences.kpts_bot, train_desc = reference_frame.pano_correspondences.desc_bot, query_kpts = current_frame.pano_correspondences.kpts_bot, query_desc = current_frame.pano_correspondences.desc_bot, random_colors_RGB = reference_frame.pano_correspondences.random_colors_RGB,
                                                                                                                                                                                                                                                                 keypts_as_points_train = reference_frame.pano_correspondences.m_bot, keypts_as_points_query = current_frame.pano_correspondences.m_bot,
                                                                                                                                                                                                                                                                 max_horizontal_diff = self.max_horizontal_diff_f2f_matches, max_descriptor_distance_radius = -1, pano_img_train = reference_frame.panoramic_image_bottom,
                                                                                                                                                                                                                                                                 pano_img_query = current_frame.panoramic_image_bottom, show_matches = self.show_f2f_correspondences_initial, win_name = 'All Triangulated Matches (Bottom-to-Bottom) t=%d' % (current_frame.frame_id))

        # 2) estimate a (rigid) transformation between camera poses (motion estimate) and minimize error metric
        bearing_vectors_of_tracked_frame_top = current_frame.bearing_vectors_top_stereo_triangulated[matched_query_indices_pano_top]
        ref_3D_points_wrt_C_top = reference_frame.pano_correspondences.points_3D_coords_homo[matched_train_indices_pano_top]

        bearing_vectors_of_tracked_frame_bottom = current_frame.bearing_vectors_bottom_stereo_triangulated[matched_query_indices_pano_bottom]
        ref_3D_points_wrt_C_bottom = reference_frame.pano_correspondences.points_3D_coords_homo[matched_train_indices_pano_bottom]

        # Making temporary lists just for convenience
        bearing_vectors_of_tracked_frame_list = [bearing_vectors_of_tracked_frame_top, bearing_vectors_of_tracked_frame_bottom]
        ref_3D_points_wrt_C_list = [ref_3D_points_wrt_C_top[..., :3], ref_3D_points_wrt_C_bottom[..., :3]]
        # We need to create the array camera correspondences indices:
        cam_indices_array_all = np.empty((0, 1))
        bearing_vectors_array_all = np.empty((0, 3))
        points_array_all = np.empty((0, 3))
        indices_ranges_tuples = self.number_of_cams * [None]
        last_cutting_index = 0
        num_initial_matches = 0
        for cam_idx in range(self.number_of_cams):
            num_of_pt_correspondences = len(ref_3D_points_wrt_C_list[cam_idx])
            indices_ranges_tuples[cam_idx] = (last_cutting_index, last_cutting_index + num_of_pt_correspondences - 1)  # Inclusive!
            last_cutting_index = last_cutting_index + num_of_pt_correspondences
            cam_indices_array = np.zeros((num_of_pt_correspondences, 1)) + float(cam_idx)
            cam_indices_array_all = np.vstack((cam_indices_array_all, cam_indices_array))
            # Just stack the bearing vectors
            bearing_vectors_array_all = np.vstack((bearing_vectors_array_all, bearing_vectors_of_tracked_frame_list[cam_idx]))
            # Just stack the points
            points_array_all = np.vstack((points_array_all, ref_3D_points_wrt_C_list[cam_idx]))
            num_initial_matches = num_initial_matches + num_of_pt_correspondences

        if num_initial_matches < 2 * self.n_points_for_RANSAC_model * (0.33 * self.number_of_cams):  # We penalize this threhold by the 1/3 number of cameras (since points may be redundant)
            # For example, we need least 6 unique keypoint features used for both the generation and verification of the SE3 pose hypothesis via RANSAC and the P3P algorithm.
            return False, "Cannot track on only %d point correspondences" % (num_initial_matches)

        # 3) Estimate some initial absolute pose using RANSAC
        ransac_transformation_wrt_C, indices_inliers_combined = absolute_pose_noncentral_ransac(bearing_vectors_array_all, cam_indices_array_all, points_array_all, self.cam_offsets, self.cam_rotations, self.backprojection_score_threshold_3D_to_2D, self.max_ransac_iterations_3D_to_2D)
        # scale_decomposed_wrt_C, shear_decomposed_wrt_C, angles_decomposed_wrt_C, translation_decomposed_wrt_C, persp_decomposed_wrt_C = tr.decompose_matrix(ransac_transformation_wrt_C)
        # Assuming indices are sorted:
        index_begin_top, index_end_top = indices_ranges_tuples[0]
        index_inlier_slice_begin_top = np.argmax(index_begin_top <= indices_inliers_combined)
        if np.any(indices_inliers_combined > index_end_top):
            index_inlier_slice_end_top = np.argmin(indices_inliers_combined <= index_end_top)
        else:
            index_inlier_slice_end_top = len(indices_inliers_combined)
        indices_inliers_top_raw = indices_inliers_combined[index_inlier_slice_begin_top:index_inlier_slice_end_top]
        indices_inliers_top = indices_inliers_top_raw - index_begin_top
        assert np.all(indices_inliers_top >= 0)
        matched_kpts_train_top = np.array(matched_kpts_train_pano_top)[indices_inliers_top]
        matched_kpts_query_top = np.array(matched_kpts_query_pano_top)[indices_inliers_top]
        random_colors_final_pano_top = np.array(random_colors_pano_top)[indices_inliers_top]

        index_begin_bottom, index_end_bottom = indices_ranges_tuples[1]
        if np.any(indices_inliers_combined >= index_begin_bottom):
            index_inlier_slice_begin_bottom = np.argmax(index_begin_bottom <= indices_inliers_combined)
            index_inlier_slice_end_bottom = index_end_bottom + 1
            indices_inliers_bottom_raw = indices_inliers_combined[index_inlier_slice_begin_bottom:index_inlier_slice_end_bottom]
            indices_inliers_bottom = indices_inliers_bottom_raw - index_begin_bottom
        else:
            indices_inliers_bottom = np.array([], dtype = "int")
        assert np.all(indices_inliers_bottom >= 0)
        matched_kpts_train_bottom = np.array(matched_kpts_train_pano_bottom)[indices_inliers_bottom]
        matched_kpts_query_bottom = np.array(matched_kpts_query_pano_bottom)[indices_inliers_bottom]
        random_colors_final_pano_bottom = np.array(random_colors_pano_bottom)[indices_inliers_bottom]
        if self.show_f2f_correspondences_final or self.save_correspondence_images:
            matchings_img_top = draw_matches_between_frames(pano_img_train = reference_frame.panoramic_image_top, pano_img_query = current_frame.panoramic_image_top, matched_kpts_train = matched_kpts_train_top, matched_kpts_query = matched_kpts_query_top, random_colors = random_colors_final_pano_top, win_name = 'Matches after RANSAC (Top-to-Top) t=%d' % (current_frame.frame_id), show_window = self.show_f2f_correspondences_final)
            matchings_img_bottom = draw_matches_between_frames(pano_img_train = reference_frame.panoramic_image_bottom, pano_img_query = current_frame.panoramic_image_bottom, matched_kpts_train = matched_kpts_train_bottom, matched_kpts_query = matched_kpts_query_bottom, random_colors = random_colors_final_pano_bottom, win_name = 'Matches after RANSAC (Bottom-to-Bottom) t=%d' % (current_frame.frame_id), show_window = self.show_f2f_correspondences_final)
            if self.save_correspondence_images:
                num_of_zero_padding = 6
                current_frame_id_as_str = str(current_frame.frame_id).zfill(num_of_zero_padding)
                cv2.imwrite("%s/%s.png" % (self.correspondence_images_path_top, current_frame_id_as_str), matchings_img_top)
                cv2.imwrite("%s/%s.png" % (self.correspondence_images_path_bottom, current_frame_id_as_str), matchings_img_bottom)

        self.num_tracked_correspondences = len(indices_inliers_combined)
        self.inlier_tracked_correspondences_ratio = float(self.num_tracked_correspondences) / float(num_initial_matches)
        # 4) Nonlinear optimization (BA pose refinement) using all the RANSAC inliers
        cam_indices_array_inliers = cam_indices_array_all[indices_inliers_combined]
        bearing_vectors_of_tracked_frame_inliers = bearing_vectors_array_all[indices_inliers_combined]
        ref_3D_points_wrt_C_inliers = points_array_all[indices_inliers_combined]  # Here, points are still in whatever units the camera model is using!
        t_ransac = ransac_transformation_wrt_C[:3, 3]
        R_ransac = ransac_transformation_wrt_C[:3, :3]
        nonlinear_transformation = absolute_pose_noncentral_optimize_nonlinear(bearing_vectors_of_tracked_frame_inliers, cam_indices_array_inliers, ref_3D_points_wrt_C_inliers, self.cam_offsets, self.cam_rotations, t_ransac, R_ransac)
        nonlinear_transformation_homo = np.identity(4)
        nonlinear_transformation_homo[:3] = nonlinear_transformation
        nonlinear_transformation_homo[:3, 3] = nonlinear_transformation[:3, 3] * current_frame.conversion_factor_length_to_m  # Converting translation to [m]
        current_frame.T_frame_wrt_tracking_ref_frame = nonlinear_transformation_homo
        # 5) use the rigid transformation to rotate/translate the source onto the target
        # The pose of current tracked frame wrt to the fixed [S]cene frame
        self.T_C_curr_frame_wrt_S_est = tr.concatenate_matrices(self.T_Ckey_wrt_S_est_list[-1], current_frame.T_frame_wrt_tracking_ref_frame)

        if self.show_3D_points:
            # Get the following from the "query" which is the current frame being tracked
            # Only using the inlier points
            self.xyz_homo_points_wrt_C_inliers = np.hstack((ref_3D_points_wrt_C_inliers * current_frame.conversion_factor_length_to_m, np.ones(shape = (ref_3D_points_wrt_C_inliers.shape[0], 1))))  # Make homogeneous point coordinates
            # TODO: not resolving for the individual colors yet
            # frame_tracked_colors_for_matches = correspondences_pano_list[frame_tracked_idx].random_colors_RGB[matched_query_indices_pano_top]
            # rgb_points_inliers = frame_tracked_colors_for_matches[indices_inliers]

        return True, "tracking used %d inlier point correspondences" % (self.num_tracked_correspondences)

    def bootstrap_tracker(self):
        # Create non-central camera system with top and bottom cameras
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        t_top_wrt_C = self.omnistereo_model.top_model.T_model_wrt_C[:3, 3]
        t_bottom_wrt_C = self.omnistereo_model.bot_model.T_model_wrt_C[:3, 3]
        rot_top_wrt_C = self.omnistereo_model.top_model.T_model_wrt_C[:3, :3]
        rot_bottom_wrt_C = self.omnistereo_model.bot_model.T_model_wrt_C[:3, :3]
        # Reference to positions of the different cameras seen from the viewpoint (wrt [C] in our case).
        self.cam_offsets = np.array([t_top_wrt_C, t_bottom_wrt_C])
        # Reference to rotations from the different cameras back to the viewpoint (wrt [C] in our case).
        self.cam_rotations = np.array([rot_top_wrt_C, rot_bottom_wrt_C])
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        self.num_features_detection_for_static_stereo = 100  # For each bucket (only applies to GFT)
        self.omnistereo_model.feature_matcher_for_static_stereo = FeatureMatcher(method = self.detection_method, matcher_type = self.matching_type, k_best = self.k_best_matches, percentage_good_matches = self.percentage_good_matches, num_of_features = self.num_features_detection_for_static_stereo, use_radius_match = False)

        # FIXME: too arbitrary and it should become adaptive base on speed of motion
        self.max_horizontal_diff_f2f_matches = 0.0625 * self.max_horizontal_search_ratio * self.omnistereo_model.top_model.panorama.cols  # use -1 for not constraints!
        self.omnistereo_model.feature_matcher_for_motion = FeatureMatcher(method = self.detection_method, matcher_type = self.matching_type, k_best = self.k_best_matches, percentage_good_matches = self.percentage_good_matches, num_of_features = self.num_features_detection_for_motion, use_radius_match = self.use_descriptor_radius_match_for_motion)

        # Generate azimuthal masks on panoramas for "bucketting"
        show_azimuthal_masks = False
        azimuth_mask_degrees = 10
        overlap_degrees = 0  # Useful, so we don't miss features located near the edge of the mask.
        first_stand_azimuthal_location = 50  # degrees
        stand_masks_azimuth_coord_in_degrees_list = [first_stand_azimuthal_location, first_stand_azimuthal_location + 120, first_stand_azimuthal_location + 240]
        stand_masks_width_in_degrees = 3
        self.omni_mask_extra_padding = 10
        self.omnistereo_model.top_model.panorama.generate_azimuthal_masks(azimuth_mask_degrees = azimuth_mask_degrees, overlap_degrees = overlap_degrees, show = show_azimuthal_masks, elev_mask_padding = self.omni_mask_extra_padding, stand_masks_azimuth_coord_in_degrees_list = stand_masks_azimuth_coord_in_degrees_list, stand_masks_width_in_degrees = stand_masks_width_in_degrees)
        self.omnistereo_model.bot_model.panorama.generate_azimuthal_masks(azimuth_mask_degrees = azimuth_mask_degrees, overlap_degrees = overlap_degrees, show = show_azimuthal_masks, elev_mask_padding = self.omni_mask_extra_padding, stand_masks_azimuth_coord_in_degrees_list = stand_masks_azimuth_coord_in_degrees_list, stand_masks_width_in_degrees = stand_masks_width_in_degrees)

class TrackerRGBDSE3(TrackerSE3):

    def __init__(self, camera_model, show_3D_points = False, **kwargs):
        TrackerSE3.__init__(self, camera_model, show_3D_points, **kwargs)
        self.number_of_cams = 1  # We have a single RGB-D camera
#         self.T_C_wrt_S_init = kwargs.get("T_C_wrt_S_init", T_C_wrt_S_init_default)
        self.T_C_wrt_S_init = tr.identity_matrix()  # TEST!
        self.T_C_curr_frame_wrt_S_est = kwargs.get("T_C_wrt_S_init", self.T_C_wrt_S_init)
        self.bootstrap_tracker()

        if self.save_correspondence_images:
            self.correspondence_images = join(self.correspondence_images_path, "matches")
            make_sure_path_exists(self.correspondence_images)
            self.depth_images_vis = join(self.correspondence_images_path, "depth_visualization")
            make_sure_path_exists(self.depth_images_vis)

    def track_frame(self, reference_frame, current_frame):
        self.num_tracked_correspondences = 0
        self.inlier_tracked_correspondences_ratio = 0.
        # 1) Find matches between views at frames at time k-1 and k
        # Find matches from top between frames at time k-1 and k
        (matched_train_indices, matched_kpts_train, matched_desc_train), (matched_query_indices, matched_kpts_query, matched_desc_query), random_colors = match_features_frame_to_frame(cam_model = self.camera_model, train_kpts = reference_frame.keypoints_and_descriptors.keypoints, train_desc = reference_frame.keypoints_and_descriptors.descriptors, query_kpts = current_frame.keypoints_and_descriptors.keypoints, query_desc = current_frame.keypoints_and_descriptors.descriptors, random_colors_RGB = reference_frame.keypoints_and_descriptors.random_colors_RGB,
                                                                                                                                                                                                                                                                 keypts_as_points_train = reference_frame.keypoints_and_descriptors.pixel_coords, keypts_as_points_query = current_frame.keypoints_and_descriptors.pixel_coords,
                                                                                                                                                                                                                                                                 max_horizontal_diff = self.max_horizontal_diff_f2f_matches, max_descriptor_distance_radius = -1, pano_img_train = reference_frame.rgb_img,
                                                                                                                                                                                                                                                                 pano_img_query = current_frame.rgb_img, show_matches = self.show_f2f_correspondences_initial, win_name = 'All Initial Matches (F2F) t=%d' % (current_frame.frame_id))
        num_initial_matches = len(matched_train_indices)
        if num_initial_matches < 2 * self.n_points_for_RANSAC_model * self.number_of_cams:  # We penalize this threhold by the number of cameras (since points may be redundant)
            # For example, we need least 6 unique keypoint features used for both the generation and verification of the SE3 pose hypothesis via RANSAC and the P3P algorithm.
            return False, "Cannot track on only %d point correspondences" % (num_initial_matches)

        # 2) estimate a (rigid) transformation between camera poses (motion estimate) and minimize error metric
        bearing_vectors_of_tracked_frame = current_frame.bearing_vectors[matched_query_indices]
        ref_3D_points_wrt_C = reference_frame.keypoints_3D_points[matched_train_indices]

        # 3) Estimate some initial absolute pose using RANSAC
        ransac_transformation_wrt_C, indices_inliers = absolute_pose_ransac(bearing_vectors_of_tracked_frame[..., :3], ref_3D_points_wrt_C[..., :3], self.pose_est_algorithm, self.backprojection_score_threshold_3D_to_2D, self.max_ransac_iterations_3D_to_2D)
        # scale_decomposed_wrt_C, shear_decomposed_wrt_C, angles_decomposed_wrt_C, translation_decomposed_wrt_C, persp_decomposed_wrt_C = tr.decompose_matrix(ransac_transformation_wrt_C)

        if self.show_f2f_correspondences_final or self.save_correspondence_images:
            matchings_img = draw_matches_between_frames(pano_img_train = reference_frame.rgb_img, pano_img_query = current_frame.rgb_img, matched_kpts_train = matched_kpts_train[indices_inliers], matched_kpts_query = matched_kpts_query[indices_inliers], random_colors = random_colors[indices_inliers], win_name = 'Matches after RANSAC (F2F) t=%d' % (current_frame.frame_id), show_window = self.show_f2f_correspondences_final)
            if self.save_correspondence_images:
                num_of_zero_padding = 6
                current_frame_id_as_str = str(current_frame.frame_id).zfill(num_of_zero_padding)

                cv2.imwrite("%s/%s.png" % (self.correspondence_images, current_frame_id_as_str), matchings_img)
                # Also saving the depth images for visualization here (not in the camera)
                depth_img_vis = current_frame.get_depth_img_visualization()
                cv2.imwrite("%s/%s.png" % (self.depth_images_vis, current_frame_id_as_str), depth_img_vis)

        self.num_tracked_correspondences = len(indices_inliers)
        self.inlier_tracked_correspondences_ratio = float(self.num_tracked_correspondences) / float(num_initial_matches)

        # 4) Nonlinear optimization (BA pose refinement) using all the RANSAC inliers
        bearing_vectors_of_tracked_frame_inliers = bearing_vectors_of_tracked_frame[indices_inliers]
        ref_3D_points_wrt_C_inliers = ref_3D_points_wrt_C[indices_inliers]
        t_ransac = ransac_transformation_wrt_C[:3, 3]
        R_ransac = ransac_transformation_wrt_C[:3, :3]
        nonlinear_transformation = absolute_pose_optimize_nonlinear(bearing_vectors_of_tracked_frame_inliers, ref_3D_points_wrt_C_inliers, t_ransac, R_ransac)
        nonlinear_transformation_homo = np.identity(4)
        nonlinear_transformation_homo[:3] = nonlinear_transformation
        nonlinear_transformation_homo[:3, 3] = nonlinear_transformation[:3, 3] * current_frame.conversion_factor_length_to_m  # Converting translation to [m]
        current_frame.T_frame_wrt_tracking_ref_frame = nonlinear_transformation_homo
        # 5) use the rigid transformation to rotate/translate the source onto the target
        # The pose of current tracked frame wrt to the fixed [S]cene frame
        self.T_C_curr_frame_wrt_S_est = tr.concatenate_matrices(self.T_Ckey_wrt_S_est_list[-1], current_frame.T_frame_wrt_tracking_ref_frame)

        if self.show_3D_points:
            # Get the following from the "query" which is the current frame being tracked
            # Only using the inlier points
            self.xyz_homo_points_wrt_C_inliers = np.hstack((ref_3D_points_wrt_C_inliers * current_frame.conversion_factor_length_to_m, np.ones(shape = (ref_3D_points_wrt_C_inliers.shape[0], 1))))  # Make homogeneous point coordinates
            # TODO: not resolving for the individual colors yet
            # frame_tracked_colors_for_matches = correspondences_pano_list[frame_tracked_idx].random_colors_RGB[matched_query_indices]
            # rgb_points_inliers = frame_tracked_colors_for_matches[indices_inliers]

        return True, "tracking used %d inlier point correspondences" % (self.num_tracked_correspondences)

    def bootstrap_tracker(self):
        self.camera_model.feature_matcher_for_motion = FeatureMatcher(method = self.detection_method, matcher_type = self.matching_type, k_best = self.k_best_matches, percentage_good_matches = self.percentage_good_matches, num_of_features = self.num_features_detection_for_motion, use_radius_match = self.use_descriptor_radius_match_for_motion)
        self.max_horizontal_diff_f2f_matches = self.max_horizontal_search_ratio * (self.camera_model.center_x * 2.)  # use -1 for not constraints!

def run_VO(visualizer_3D_VO, camera_model, gt_poses_filename = None, est_poses_filename = "estimated_frame_poses_TUM.txt", img_filename_template = None, depth_filename_template = None, img_indices = [], use_multithreads_for_VO = False, results_path = "~/temp", thread_name = ""):
    from omnistereo.common_tools import get_poses_from_file, save_as_tum_poses_to_file
    from omnistereo.common_cv import get_images
    from omnistereo.transformations import transform44_from_TUM_entry
    from omnistereo.camera_models import OmniStereoModel, RGBDCamModel
    if len(thread_name) > 0:
        thread_name = thread_name + ": "
    msg_exit = thread_name + "\n"
    if isinstance(camera_model, OmniStereoModel):
        trackerClass = TrackerStereoSE3
        KeyFrameClass = StereoPanoramicKeyFrame
    elif isinstance(camera_model, RGBDCamModel):
        from omnistereo.common_cv import get_depthmap_float32_from_png
        trackerClass = TrackerRGBDSE3
        KeyFrameClass = RGBDKeyFrame

    results_path = realpath(expanduser(results_path))
    make_sure_path_exists(results_path)
    print("%sSaving VO results at %s" % (thread_name, results_path))
    messages_log_filename = join(results_path, "printed_messages.log")
    messages_log_file = open(messages_log_filename, 'w')

    # Let's use only meters to be consistent all throughout
    pose_output_file_units = "m"  # camera_model.units
    units_scale_factor = get_length_units_conversion_factor(input_units = camera_model.units, output_units = pose_output_file_units)
    do_tracking = True
    save_frame_poses = True and do_tracking
    show_gt_poses = True
    zero_up_gt_wrt_origin = True
    show_3D_points = False
    save_correspondence_images = True
    use_uniform_color = True
    is_outdoors = False  # <<<< SETME:
    if is_outdoors:
        pos_distance_keyframe_creation_threshold = 0.1  # in [m] Arbitrary threshold for keyframe creation based on displacement
        pos_max_distance_keyframe_creation_threshold = 0.20  # in [m]
        angle_keyframe_creation_threshold = np.deg2rad(1)  # Arbitrary threshold for keyframe creation based on rotation angle
        angle_max_keyframe_creation_threshold = np.deg2rad(4.)  # Arbitrary threshold for keyframe creation based on rotation angle
        threshold_tracked_correspondence_keyframe_creation = 0.10  # Arbitrary threshold for keyframe creation regarding the current number of tracked correspondences and the past average
    else:
        pos_distance_keyframe_creation_threshold = 0.01  # in [m] Arbitrary threshold for keyframe creation based on displacement
        pos_max_distance_keyframe_creation_threshold = 0.20  # in [m]
        angle_keyframe_creation_threshold = np.deg2rad(1.0)  # Arbitrary threshold for keyframe creation based on rotation angle
        angle_max_keyframe_creation_threshold = np.deg2rad(10.0)  # Arbitrary threshold for keyframe creation based on rotation angle
        threshold_tracked_correspondence_keyframe_creation = 0.10  # Arbitrary threshold for keyframe creation regarding the current number of tracked correspondences and the past average
    threshold_keypoints_keyframe_creation = 0.10  # Arbitrary threshold for keyframe creation regarding the number of kpts_top on the current frame and the Keyframe.
    axis_length = .5  # units are [m] for visualization context

    image_names_list = get_images(img_filename_template, indices_list = img_indices, show_images = False, return_names_only = True)
    if isinstance(camera_model, RGBDCamModel):
        depthfile_names_list = get_images(depth_filename_template, indices_list = img_indices, show_images = False, return_names_only = True)

    # TEST: just using the same image hack
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvv
#     gt_poses_filename = None
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    tracker = trackerClass(camera_model = camera_model, show_3D_points = show_3D_points, save_correspondence_images = save_correspondence_images, results_path = results_path)

    # Visual Odometry (Pose Estimation)
    T_Rgt_wrt_S = None
    # Make the GT appear more pastelish!
    axis_colors_gt = np.array([[1, 0.5, 0.25, 1],
                              [1, 0.5, 0.25, 1],
                              [0.25, 1, 0.25, 1],
                              [0.25, 1, 0.25, 1],
                              [0.25, 0.5, 1, 1],
                              [0.25, 0.5, 1, 1]])
    if gt_poses_filename is None:
        if img_indices is None or len(img_indices) == 0:
            from glob import glob
            img_names = glob(img_filename_template)
            l = len(img_names)
            img_indices = range(l)
        else:
            l = max(len(img_indices), img_indices[-1])  # Assuming indices are ordered increasingly

        pose_info_tum = 7 * [0.0]
        # Zero translation
        pose_info_tum[0] = 0.0
        pose_info_tum[1] = 0.0
        pose_info_tum[2] = 0.0
        # Identity rotation for Camera frame [C] pose wrt to Scene frame [S]
        [pose_info_tum[3], pose_info_tum[4], pose_info_tum[5], pose_info_tum[6]] = [0., 0., 0., 1.]
        # Fill up the result lists:
        # Use only the current image in the list by align to index? Not feasible because we don't know what index the image is from
        cam_poses_list_TUM_order = l * [pose_info_tum]
        T_gt_for_all = transform44_from_TUM_entry(pose_info_tum, has_timestamp = False)
        transform_matrices_gt_list = l * [T_gt_for_all]
    else:
        is_tum_format = "tum" in gt_poses_filename.lower()
        if is_tum_format:
            gt_input_units = "m"  # These are the position units for both the "synthetic" and "real VICON" ground truth data
            cam_poses_list_TUM_order, transform_matrices_gt_list = get_poses_from_file(poses_filename = gt_poses_filename, input_units = gt_input_units, output_working_units = pose_output_file_units, indices = [], pose_format = "tum", zero_up_wrt_origin = zero_up_gt_wrt_origin)
        else:
            # Testing POV-Ray format:
            cam_poses_list_TUM_order, transform_matrices_gt_list = get_poses_from_file(poses_filename = gt_poses_filename, input_units = "cm", output_working_units = pose_output_file_units, indices = [], pose_format = "povray", zero_up_wrt_origin = zero_up_gt_wrt_origin)
            # Save poses in TUM to a file for later testing:
            save_as_tum_poses_to_file(output_tum_filename = gt_poses_filename.replace(".txt", "-tum.txt"), poses_7_list = cam_poses_list_TUM_order, input_units = pose_output_file_units, input_format = "tum", output_units = pose_output_file_units)
        if camera_model.T_Cest_wrt_Rgt is not None:
            T_Cest_wrt_Rgt = camera_model.T_Cest_wrt_Rgt  # Computed via superimposition during the calibration evaluation
            T_Cest_wrt_Rgt[:3, 3] = units_scale_factor * T_Cest_wrt_Rgt[:3, 3]  # Scale up the translation
            T_Rgt_wrt_Cest = tr.inverse_matrix(T_Cest_wrt_Rgt)
            T_Rgt_wrt_S = tr.concatenate_matrices(tracker.T_C_wrt_S_init, T_Rgt_wrt_Cest)
            T_S_wrt_Rgt = tr.inverse_matrix(T_Rgt_wrt_S)
            assert img_indices[-1] - 1 <= len(transform_matrices_gt_list)

    T_C_all_frames_wrt_S_true_list = transform_matrices_gt_list

    if save_frame_poses:
        est_poses_filename = join(results_path, est_poses_filename)
        est_poses_file = open(est_poses_filename, 'w')
        associated_gt_for_est_poses_filename = est_poses_filename.replace("estimated", "gt_associated")
        associated_gt_for_est_poses_file = open(associated_gt_for_est_poses_filename, 'w')
        keyframe_ids_filename = "keyframe_ids.txt"
        keyframe_ids_filename = join(results_path, keyframe_ids_filename)
        keyframe_ids_file = open(keyframe_ids_filename, 'w')

    # Running multiple views (as visualizing all point clouds)
    pts_pos = np.empty((0, 3))  # For 3D Visualization
    if use_uniform_color:
        pts_colors = []  # For 3D Visualization using the same color for a subset of points
        # vvvvvvvvvvvvv COLORS vvvvvvvvvvvvvvvvvvvvvvvvvv
        # chosen_colors = ('blue', 'magenta', 'green', 'red', 'cyan', 'orange', 'gray', 'brown', 'purple', 'pink')
        # OR:
        from  matplotlib.pyplot import get_cmap
        cmap = get_cmap('rainbow')  # Example colormaps: 'jet' 'rainbow' 'gnuplot', See http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html for more
        num_of_colors_cmap = len(img_indices)
        chosen_colors = [cmap(i) for i in np.linspace(0, 1, num = num_of_colors_cmap)]
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    else:
        pts_colors = np.empty((0, 4))  # For 3D Visualization

    reference_frame = None
    keyframes_list = []
    current_frame = None
    create_keyframe = True
    current_keyframe_id = img_indices[0]
    number_of_keyframes = 0
    number_of_tracked_frames_wrt_keyframes = 0
    num_tracked_correspondences_previous_average = 0

    inliers_RANSAC_percentage_ratio = tracker.inlier_tracked_correspondences_ratio  # Statistics for the inlier correspondences from RANSAC computed it as a cumulative moving average

    acc_time_img_read = 0.
    acc_time_frame_setup = 0.
    acc_time_frame_tracking = 0.
    acc_time_keyframe_resolution = 0.
    acc_time_keyframe_creation = 0.
    acc_time_frame_vo_complete = 0.
    for img_index_number, idx in enumerate(img_indices):
        # TEST: just using the same image hack
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvv
#         original_idx = idx
#         idx = 0
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        start_time_frame_vo = process_time()
        # ---------------------------------------------------
        start_time_img_read = process_time()

        if isinstance(camera_model, OmniStereoModel):
            current_omni_img = cv2.imread(image_names_list[img_index_number])
        elif isinstance(camera_model, RGBDCamModel):
            current_bgr_img = cv2.imread(image_names_list[img_index_number])
            current_rgb_img = cv2.cvtColor(current_bgr_img, cv2.COLOR_BGR2RGB)
            current_depth_map = get_depthmap_float32_from_png(depth_img_filename = depthfile_names_list[img_index_number], conversion_factor = camera_model.scaling_factor)

        # TEST: just using the same image hack
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvv
#         idx = original_idx
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        end_time_img_read = process_time()
        if img_index_number > 0:  # Don't count the first frame's time (just to make if more comparable with the other statistics
            time_ellapsed_img_read = end_time_img_read - start_time_img_read
            acc_time_img_read = acc_time_img_read + time_ellapsed_img_read
            # ---------------------------------------------------

        if do_tracking:
            # ATTENTION: be careful if generating panoramas here, the masks will be destroyed.
            # ---------------------------------------------------
            start_time_frame_setup = process_time()
            if isinstance(camera_model, OmniStereoModel):
                camera_model.set_current_omni_image(current_omni_img, generate_panoramas = False, view = False, apply_mask = True, mask_RGB = (0, 0, 0))  # Using Black pano mask
                current_frame = StereoPanoramicFrame(stereo_camera_model = camera_model, frame_id = idx, parent_id = current_keyframe_id)
            elif isinstance(camera_model, RGBDCamModel):
                current_frame = RGBDFrame(rgbd_camera_model = camera_model, frame_id = idx, rgb_img = current_rgb_img, depth_map = current_depth_map, parent_id = current_keyframe_id)
                # ^^^^ If rgb image and depth map is passed to the constructed, the kpts_top will be established at construction ^^^^^^, and also these images will be saved as members
                # current_frame.establish_keypoints(rgb = current_rgb_img, depth = current_depth_map)
            end_time_frame_setup = process_time()
            if img_index_number > 0:  # Don't count the first frame's time (just to make if more comparable with the other statistics
                time_ellapsed_frame_setup = end_time_frame_setup - start_time_frame_setup
                acc_time_frame_setup = acc_time_frame_setup + time_ellapsed_frame_setup
            # ---------------------------------------------------

        # The ground-truth pose of the rig wrt to its World/Scene reference (such as the Vicon world)
        # NOTE: it will be additionally transform by any constant initial transformation for the camera model specified within the Tracker instance
        # T_R_wrt_S_gt = tr.concatenate_matrices(T_C_all_frames_wrt_S_true_list[idx], tracker.T_C_wrt_S_init) # NAH, already accounted in T_Rgt_wrt_S
        T_R_wrt_S_gt = T_C_all_frames_wrt_S_true_list[idx]
        #=======================================================================
        if T_Rgt_wrt_S is not None:
            # BEFORE:
            # T_R_wrt_S_gt = tr.concatenate_matrices(T_Rgt_wrt_Cest, T_R_wrt_S_gt, T_Cest_wrt_Rgt)
            # AFTER:
            T_R_wrt_S_gt = tr.concatenate_matrices(T_Rgt_wrt_S, T_R_wrt_S_gt, T_S_wrt_Rgt)
            # NOTE: We are visualizing the fixed rigid transformation of the camera in the scene [S] wrt to the ground-truth rig frame [Rgt]
        #=======================================================================
        if show_gt_poses:
            text_at_origin_gt = 'G%d' % (idx)
            if visualizer_3D_VO is not None:
                visualizer_3D_VO.current_frame_axis_gt, visualizer_3D_VO.current_frame_text_idx_gt = visualizer_3D_VO.draw_current_frame(frame_axis = visualizer_3D_VO.current_frame_axis_gt, frame_text_id = visualizer_3D_VO.current_frame_text_idx_gt, T_C_wrt_S = T_R_wrt_S_gt, text_at_origin = text_at_origin_gt, text_color = "black", axis_length = axis_length, units_scale_factor = 1.0, axis_thickness = 5, axis_colors = axis_colors_gt)
                # Also drawing the ground-truth trajectory
                visualizer_3D_VO.add_pose_to_gt_trajectory(T_Cgt_wrt_S = T_R_wrt_S_gt, units_scale_factor = 1.0, line_thickness = 2, trajectory_color = "black")

        if img_index_number > 0:
            if do_tracking:
                # ---------------------------------------------------
                start_time_frame_tracking = process_time()
                # TRACK frame:
                tracking_status, tracking_msg = tracker.track_frame(reference_frame = reference_frame, current_frame = current_frame)
                if tracking_status:
                    # NO NEED anymore: tracker.T_C_curr_frame_wrt_S_est[:3, 3] = tracker.T_C_curr_frame_wrt_S_est[:3, 3] * units_scale_factor
                    reference_frame.children_ids.append(current_frame.frame_id)
                    number_of_tracked_frames_wrt_keyframes += 1
                else:
                    # Tracking failed
                    print(tracking_msg, file = messages_log_file)
                    warn("%sWarning failed: %s" % (thread_name, tracking_msg))
                    break

                end_time_frame_tracking = process_time()
                time_ellapsed_frame_tracking = end_time_frame_tracking - start_time_frame_tracking
                acc_time_frame_tracking = acc_time_frame_tracking + time_ellapsed_frame_tracking
                # ---------------------------------------------------

                # ---------------------------------------------------
                start_time_keyframe_resolution = process_time()

                if visualizer_3D_VO is not None:
                    text_at_origin_est = 'F%d' % (idx)
                    visualizer_3D_VO.current_frame_axis_est, visualizer_3D_VO.current_frame_text_idx_est = visualizer_3D_VO.draw_current_frame(frame_axis = visualizer_3D_VO.current_frame_axis_est, frame_text_id = visualizer_3D_VO.current_frame_text_idx_est, T_C_wrt_S = tracker.T_C_curr_frame_wrt_S_est, text_at_origin = text_at_origin_est, text_color = "red", axis_length = axis_length, units_scale_factor = 1.0, axis_thickness = 5, axis_colors = None)

                # Check keyframe creation if L2 distance to current keyframe has exceeded threshold:
                current_frame_L2_dist_to_keyframe = tr.rpe_translation_metric(current_frame.T_frame_wrt_tracking_ref_frame)
                current_frame_rot_angle_to_keyframe = tr.rpe_rotation_metric(current_frame.T_frame_wrt_tracking_ref_frame)
                num_tracked_correspondences = tracker.num_tracked_correspondences / float(tracker.number_of_cams)  # N_i: we divide by No of cameras because we have those views combined and possible repeated keypoint matches
                num_valid_reference_keypoints = reference_frame.num_valid_keypoints  # M_K
                num_valid_tracking_keypoints = current_frame.num_valid_keypoints  # M_F

                if (pos_distance_keyframe_creation_threshold < current_frame_L2_dist_to_keyframe < pos_max_distance_keyframe_creation_threshold) or (angle_keyframe_creation_threshold < current_frame_rot_angle_to_keyframe < angle_max_keyframe_creation_threshold):
                    # or satisfying_candidates_keyframe_creation_correspondences_ratio:
                    # if the number of kpts_top for the candidate frame is at least 50% of the current keyframe's number of kpts_top
                    if num_tracked_correspondences > threshold_tracked_correspondence_keyframe_creation * num_tracked_correspondences_previous_average and num_valid_tracking_keypoints > threshold_keypoints_keyframe_creation * num_valid_reference_keypoints:
                        msg_reason_keyframe_creation = ""
                        if pos_distance_keyframe_creation_threshold < current_frame_L2_dist_to_keyframe < pos_max_distance_keyframe_creation_threshold:
                            # Make sure that rotation is not crazy:
                            if current_frame_rot_angle_to_keyframe < angle_max_keyframe_creation_threshold:
                                msg_reason_keyframe_creation = msg_reason_keyframe_creation + "translation %.2f < %.4f < %.2f [m]" % (pos_distance_keyframe_creation_threshold, current_frame_L2_dist_to_keyframe, pos_max_distance_keyframe_creation_threshold)
                                create_keyframe = True
                        elif angle_keyframe_creation_threshold < current_frame_rot_angle_to_keyframe:
                            # Make sure that translation is not crazy:
                            if current_frame_L2_dist_to_keyframe < pos_max_distance_keyframe_creation_threshold < angle_max_keyframe_creation_threshold:
                                msg_reason_keyframe_creation = msg_reason_keyframe_creation + " rotation: %.2f > %.2f [radians]" % (current_frame_rot_angle_to_keyframe, angle_keyframe_creation_threshold)
                                create_keyframe = True
                        if create_keyframe:
                            msg_keyframe_creation = "%sFrame [%d] as Keyframe due to %s... with %d tracked keypoint correspondences after tracking %d frames" % (thread_name, idx, msg_reason_keyframe_creation, num_tracked_correspondences, number_of_tracked_frames_wrt_keyframes)
                        else:
                            msg_keyframe_creation = "%sFrame [%d] as failed to create Keyframe due to some CRAZYNESS" % (thread_name, idx)
                    else:
                        msg_keyframe_creation = "%sFrame [%d] as Keyframe...doesn't satisfy %d > %.2f * %.2f and %d > %.2f * %.2f" % (thread_name, idx, num_tracked_correspondences, threshold_tracked_correspondence_keyframe_creation, num_tracked_correspondences_previous_average, num_valid_tracking_keypoints, threshold_keypoints_keyframe_creation, num_valid_reference_keypoints)
                    print(msg_keyframe_creation)
                    print(msg_keyframe_creation, file = messages_log_file)
                # Update current average:
                num_tracked_correspondences_previous_average = (num_tracked_correspondences + (float(number_of_tracked_frames_wrt_keyframes) - 1.) * num_tracked_correspondences_previous_average) / float(number_of_tracked_frames_wrt_keyframes)
                end_time_keyframe_resolution = process_time()
                time_ellapsed_keyframe_resolution = end_time_keyframe_resolution - start_time_keyframe_resolution
                acc_time_keyframe_resolution = acc_time_keyframe_resolution + time_ellapsed_keyframe_resolution
                # Inlier point correspondences from RANSAC for statistics:
                inliers_RANSAC_percentage_ratio = (tracker.inlier_tracked_correspondences_ratio + (float(img_index_number - 1)) * inliers_RANSAC_percentage_ratio) / float(img_index_number)
                # ---------------------------------------------------

        if do_tracking:
            if create_keyframe:
                number_of_tracked_frames_wrt_keyframes = 0
                num_tracked_correspondences_previous_average = 0.
                # ---------------------------------------------------
                start_time_keyframe_creation = process_time()

                reference_frame = KeyFrameClass(frame = current_frame)
                current_keyframe_id = reference_frame.frame_id
                print(current_keyframe_id, file = keyframe_ids_file)

                keyframes_list.append(reference_frame)  # TODO: we should use a graph structure instead (like g2o)
                if len(tracker.T_Ckey_wrt_S_est_list) > 0:
                    T_C_curr_keyframe_wrt_S_est = tr.concatenate_matrices(tracker.T_Ckey_wrt_S_est_list[-1], reference_frame.T_frame_wrt_tracking_ref_frame)
                else:
                    T_C_curr_keyframe_wrt_S_est = tracker.T_C_curr_frame_wrt_S_est
                tracker.T_Ckey_wrt_S_est_list.append(T_C_curr_keyframe_wrt_S_est)

                end_time_keyframe_creation = process_time()
                time_ellapsed_keyframe_creation = end_time_keyframe_creation - start_time_keyframe_creation
                acc_time_keyframe_creation = acc_time_keyframe_creation + time_ellapsed_keyframe_creation
                # ---------------------------------------------------

                if visualizer_3D_VO is not None:
                    # Twice longer axis and in "red" text
                    # Too busy:
                    #===========================================================
                    # text_at_origin = 'K%d' % (idx)
                    # visualizer_3D_VO.add_pose_keyframe(T_C_wrt_S = T_C_curr_keyframe_wrt_S_est, text_at_origin = text_at_origin, text_color = "red", axis_length = axis_length, units_scale_factor = 1.0, axis_thickness = 2)
                    #===========================================================
                    # just drawing the trajectory of the keyframes
                    visualizer_3D_VO.add_pose_to_trajectory(T_C_wrt_S = T_C_curr_keyframe_wrt_S_est, units_scale_factor = 1.0, line_thickness = 2, trajectory_color = "red")

                    if show_3D_points:  # Only showing the points of the keyframe
                        # Get the following from the "query" which is the current frame being tracked
                        # Only using the inlier points
                        xyz_points_wrt_C = tracker.xyz_homo_points_wrt_C_inliers
                        # TODO: not resolving for the individual colors yet
                        # frame_tracked_colors_for_matches = correspondences_pano_list[frame_tracked_idx].random_colors_RGB[matched_query_indices_pano_top]
                        # rgb_points = frame_tracked_colors_for_matches[indices_inliers]

                        if len(xyz_points_wrt_C) > 0:
                            points_wrt_S = np.einsum("ij, nj->ni", T_C_curr_keyframe_wrt_S_est, xyz_points_wrt_C)[..., :3]  # Only need the xyz coordinates to display in VisPy
                            pts_pos = np.vstack((pts_pos, points_wrt_S))
                            # fill in the point-cloud data (Colors are normalized from 0 to 1.0)
                            if use_uniform_color:
                                new_pts_colors = len(points_wrt_S) * [chosen_colors[img_index_number % len(chosen_colors)]]
                                pts_colors = pts_colors + new_pts_colors
                            else:
                                rgb_points = tracker.rgb_points_inliers  # TODO: This doesn't exist yet
                                new_pts_colors = np.hstack((rgb_points / 255., np.ones_like(rgb_points[..., 0, np.newaxis])))  # Adding alpha=1 channel
                                pts_colors = np.vstack((pts_colors, new_pts_colors))

                            visualizer_3D_VO.update_3D_cloud(pts_pos = pts_pos, pts_colors = pts_colors)

                create_keyframe = False
                number_of_keyframes = number_of_keyframes + 1

            if save_frame_poses:
                # Recall the order of the quaternion in the "transformation" module is [q_w, q_i, q_j, q_k]
                q_est = tr.quaternion_from_matrix(matrix = tracker.T_C_curr_frame_wrt_S_est, isprecise = False)
                t_est = tr.translation_from_matrix(matrix = tracker.T_C_curr_frame_wrt_S_est)
                print(idx, t_est[0], t_est[1], t_est[2], q_est[1], q_est[2], q_est[3], q_est[0], sep = ' ', end = '\n', file = est_poses_file)
                if np.any(np.isnan(T_R_wrt_S_gt)):
                    q_gt = 4 * [np.nan]
                    t_gt = 3 * [np.nan]
                else:
                    q_gt = tr.quaternion_from_matrix(matrix = T_R_wrt_S_gt, isprecise = False)
                    t_gt = tr.translation_from_matrix(matrix = T_R_wrt_S_gt)  # Assuming that scale for GT translation should be in [meters] already
                print(idx, t_gt[0], t_gt[1], t_gt[2], q_gt[1], q_gt[2], q_gt[3], q_gt[0], sep = ' ', end = '\n', file = associated_gt_for_est_poses_file)
                # print(idx, *cam_poses_list_TUM_order[idx], sep = ' ', end = '\n', file = associated_gt_for_est_poses_file)

        if img_index_number > 0:
            end_time_frame_vo = process_time()
            time_ellapsed_frame_vo = end_time_frame_vo - start_time_frame_vo
            acc_time_frame_vo_complete = acc_time_frame_vo_complete + time_ellapsed_frame_vo

        msg_tracking_frame_done = "%sDONE with F[%d] (Parent K[%d]). C.M.Avg. RANSAC inlier ratio = %.3f" % (thread_name, current_frame.frame_id, current_frame.parent_id, inliers_RANSAC_percentage_ratio)
        print(msg_tracking_frame_done)
        print(msg_tracking_frame_done, file = messages_log_file)

    msg_num_keyframes = "%sVO done with %d keyframes" % (thread_name, number_of_keyframes)
    print(msg_num_keyframes)
    msg_exit = msg_exit + "\n" + msg_num_keyframes

    avg_time_ellapsed_img_read = acc_time_img_read / float(img_index_number)
    msg_avg_time_ellapsed_img_read = "Image Read Avg Time: {time:.8f} seconds".format(time = avg_time_ellapsed_img_read)
    avg_time_ellapsed_frame_setup = acc_time_frame_setup / float(img_index_number)
    msg_avg_time_ellapsed_frame_setup = "Frame Setup Avg Time: {time:.8f} seconds".format(time = avg_time_ellapsed_frame_setup)
    avg_time_ellapsed_frame_tracking = acc_time_frame_tracking / float(img_index_number)
    msg_avg_time_ellapsed_frame_tracking = "Frame Tracking Avg Time: {time:.8f} seconds".format(time = avg_time_ellapsed_frame_tracking)
    avg_time_ellapsed_keyframe_resolution = acc_time_keyframe_resolution / float(img_index_number)
    msg_avg_time_ellapsed_keyframe_resolution = "KeyFrame Resolution Avg Time: {time:.8f} seconds".format(time = avg_time_ellapsed_keyframe_resolution)
    avg_time_ellapsed_keyframe_creation = acc_time_keyframe_creation / float(number_of_keyframes)
    msg_avg_time_ellapsed_keyframe_creation = "KeyFrame Creation Avg Time: {time:.8f} seconds".format(time = avg_time_ellapsed_keyframe_creation)
    avg_time_ellapsed_frame_vo_complete = acc_time_frame_vo_complete / float(img_index_number)
    msg_avg_time_ellapsed_frame_vo_complete = "Overall Frame VO Avg Time: {time:.8f} seconds".format(time = avg_time_ellapsed_frame_vo_complete)
    msg_avg_times = msg_avg_time_ellapsed_img_read + "\n" + \
                    msg_avg_time_ellapsed_frame_setup + "\n" + \
                    msg_avg_time_ellapsed_frame_tracking + "\n" + \
                    msg_avg_time_ellapsed_keyframe_resolution + "\n" + \
                    msg_avg_time_ellapsed_keyframe_creation + "\n" + \
                    msg_avg_time_ellapsed_frame_vo_complete + "\n"
    print(msg_avg_times)
    msg_exit = msg_exit + "\n" + msg_avg_times

    if save_frame_poses:
        est_poses_file.close()
        msg_est_poses_file_saved = "Estimated poses in TUM format SAVED as " + est_poses_filename
        print(msg_est_poses_file_saved)
        msg_exit = msg_exit + "\n" + msg_est_poses_file_saved
        keyframe_ids_file.close()
        msg_kf_file_saved = "Keyframes (indices) SAVED as " + keyframe_ids_filename
        print(msg_kf_file_saved)
        msg_exit = msg_exit + "\n" + msg_kf_file_saved
        associated_gt_for_est_poses_file.close()
        msg_associated_gt_for_est_poses_file_saved = "Associated ground-truth poses in TUM format SAVED as" + associated_gt_for_est_poses_filename
        print(msg_associated_gt_for_est_poses_file_saved)
        msg_exit = msg_exit + "\n" + msg_associated_gt_for_est_poses_file_saved + "\n"

    print(msg_exit, file = messages_log_file)
    messages_log_file.close()

def driver_VO(camera_model, scene_path, scene_path_vo_results, scene_img_filename_template, depth_filename_template, num_scene_images, visualize_VO = False, use_multithreads_for_VO = True, step_for_scene_images = 1, first_image_index = 0, last_image_index = -1, thread_name = ""):
    if use_multithreads_for_VO:  # Don't put time because assuming experimental batch
        est_poses_filename = "estimated_frame_poses_TUM.txt"
    else:
        from datetime import datetime
        now_info = datetime.now()
        current_time_str = "%d-%d-%d-%d-%d-%d" % (now_info.year, now_info.month, now_info.day, now_info.hour, now_info.minute, now_info.second)
        est_poses_filename = "estimated_frame_poses_TUM-%s.txt" % (current_time_str)
    #=======================================================================
    # if experiment_name == "VICON" and model_version == "new" and not is_synthetic:
    #     # Checking that we start the prefix with a particular context word
    #     if scene_prefix_filename.find("lab") == 0:
    #         vo_frame_indices = [0, 1]
    #     if scene_prefix_filename.find("home") == 0:
    #         vo_frame_indices = []
    #     if scene_prefix_filename.find("office") == 0:
    #         vo_frame_indices = []
    #=======================================================================
    # else:

    gt_poses_filename = None
    rig_is_static = "static" in scene_path.lower() or "GCT" in scene_path.upper() or "CCNY" in scene_path.upper() or "park" in scene_path.lower() or "grand" in scene_path.lower()
    if not rig_is_static:  # Or has no ground-truth pose
        gt_poses_filename = join(scene_path, "gt_TUM.txt")

    if last_image_index > 0:
        last_image_index = min(last_image_index, num_scene_images)
    else:
        last_image_index = num_scene_images
    vo_frame_indices = list(range(first_image_index, last_image_index, step_for_scene_images))

    from omnistereo.common_plot import DrawerVO

    # 3D Visualization (Setup)
    if visualize_VO:
        from vispy import app
    import threading
    visualization_thread_lock = threading.Lock()  # TODO: use it
    vis_VO = None
    if visualize_VO:
        vis_VO = DrawerVO(new_3D_entities_lock = visualization_thread_lock, title = "VO Visualization %s" % (thread_name), bgcolor = "white")

    # pt_cloud_args_dict = dict(visualizer_3D_VO=vis_VO, omnistereo_model=gums_calibrated, poses_filename=None, omni_img_filename_template=scene_img_filename_template, features_detected_filename_template=features_detected_filename_template, img_indices=vo_frame_indices, compute_new_3D_points=compute_new_3D_points, save_3D_points=save_3D_points, points_3D_path=points_3D_path, points_3D_filename_template=points_3D_filename_template, dense_cloud=dense_cloud, manual_point_selection=dense_manual_3D_point_selection, show_3D_reference_cyl=show_3D_reference_cyl, load_stereo_tuner_from_pickle=load_stereo_tuner_from_pickle, save_pcl=save_pcl, stereo_tuner_filename=stereo_tuner_filename, tune_live=tune_live, save_sparse_features=save_sparse_features, load_sparse_features_from_file=load_sparse_features_from_file, do_VO=do_VO, use_multithreads_for_VO=use_multithreads_for_VO)
    vo_args_dict = dict(visualizer_3D_VO = vis_VO, camera_model = camera_model, gt_poses_filename = gt_poses_filename, est_poses_filename = est_poses_filename, img_filename_template = scene_img_filename_template, depth_filename_template = depth_filename_template, img_indices = vo_frame_indices, use_multithreads_for_VO = use_multithreads_for_VO, results_path = scene_path_vo_results, thread_name = thread_name)

    if use_multithreads_for_VO:
        vo_thread = threading.Thread(target = run_VO, kwargs = vo_args_dict)
        vo_thread.start()
        if visualize_VO:
            app.run()
        vo_thread.join()
    else:
        run_VO(**vo_args_dict)
        if visualize_VO:
            app.run()

    #===========================================================================
    # from omnistereo.common_cv import clean_up
    # clean_up(wait_key_time = 1)
    #===========================================================================
    print("%s Done with VO for %s!" % (thread_name, scene_path))
    return "NOTHING"
