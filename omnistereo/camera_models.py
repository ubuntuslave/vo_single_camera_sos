# -*- coding: utf-8 -*-
# camera_models.py

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
Tools for omnidirectional stereo vision using catadioptrics

@author: Carlos Jaramillo
'''

from __future__ import division
from __future__ import print_function
import numpy as np
import cv2
from time import process_time

def intersect_line3_with_plane_vectorized(line, plane):
    '''
    @brief it can compute the intersection of an 3D line and Plane using numpy arrays in a vectorized approach
    @param line: Either an Euclid Line3 object or a Numpy ndarray of points in row-wise order,
            such as [p.x,p.y,p.z,v.x,v.y,v.z], where p is a point in the line, and v is the direction vector of the line.
    @param plane: Either a Euclid Plane object or a Numpy ndarray such as [a,b,c,d] where the plane is defined by a*x+b*y+c*z=d
    @return The ndarray of the intersection coordinates with one ndarray per intersection.
            When a line is parallel to the plane, there's no intersection, so resulting coordinates are indicated by a Not a Number (nan) values.
            For example, it could return [[i1.x, i1.y, i1.z], [nan, nan, nan], [i3.x, i3.y, i3.z]] for 3 lines, where the second line doesn't intersect the plane.
    '''
    import omnistereo.euclid as euclid

    if isinstance(line, euclid.Line3):
        line_dir_vec_np = np.array(line.v)
        line_point_np = np.array(line.p)
    elif isinstance(line, np.ndarray):
        line_point_np = line[..., 0:3]
        line_dir_vec_np = line[..., 3:]
    else:
        raise TypeError ("Line must be a Euclid Line3 or defined as a ndarray \n" +
                         "such as [p.x,p.y,p.z,v.x,v.y,v.z], not %s" % type(line))

    if isinstance(plane, euclid.Plane):
        plane_normal = np.array(plane.n)
        k = plane.k
    elif isinstance(plane, np.ndarray):
        plane_normal = plane[0:3]
        k = plane[3]
    else:
        raise TypeError ("Plane must be a Euclid Plane or defined as a ndarray \n" +
                         "such as [a,b,c,d] where the plane is a*x+b*y+c*z=d, not %s" % type(plane))

    # BECAREFUL: inner product is not the same of "dot" product for matrices
    d = np.inner(plane_normal, line_dir_vec_np)

    # d == 0 when vectors are perpendicular because the line is parallel to the plane,
    # so we indicate that with Not a Number (nan) values.
    # Note: It's preferred to use nan because None changes the dtype of the ndarray to "object"
    d = np.where(d != 0.0, d, np.nan)

    u = np.where(d is not np.nan, (k - np.inner(plane_normal, line_point_np)) / d, np.nan)
    # Give u another length-1 axis on the end, and broadcasting will handle this without needing to actually construct a tripled array.
    intersection = np.where(u is not np.nan, line_point_np + u[..., np.newaxis] * line_dir_vec_np, np.nan)
    return intersection

def ortho_project_point3_onto_plane(point, plane):
    '''
    @brief Projects 3D points onto a Plane using numpy arrays in a vectorized approach
    @param point: Either an Euclid Point3 object or a Numpy ndarray of points in row-wise order,
            such as [p.x,p.y,p.z]
    @param plane: Either a Euclid Plane object or a Numpy ndarray such as [a,b,c,d] where the plane is defined by a*x+b*y+c*z=d
    @return The ndarray of the intersection coordinates with one ndarray per intersection.
            When a line is parallel to the plane, there's no intersection, so resulting coordinates are indicated by a Not a Number (nan) values.
            For example, it could return [[i1.x, i1.y, i1.z], [nan, nan, nan], [i3.x, i3.y, i3.z]] for 3 lines, where the second line doesn't intersect the plane.
    '''
    import omnistereo.euclid as euclid

    if isinstance(point, euclid.Point3):
        p = np.array(point)
    elif isinstance(point, np.ndarray):
        p = point[..., :3]
    else:
        raise TypeError ("Line must be a Euclid Point3 or defined as a ndarray \n" +
                         "such as [p.x,p.y,p.z], not %s" % type(point))

    if isinstance(plane, euclid.Plane):
        n = np.array(plane.n)
        k = plane.k
    elif isinstance(plane, np.ndarray):
        n = plane[0:3]
        k = plane[3]
    else:
        raise TypeError ("Plane must be a Euclid Plane or defined as a ndarray \n" +
                         "such as [a,b,c,d] where the plane is a*x+b*y+c*z=d, not %s" % type(plane))

    d = p.dot(n) - k
    p_x, p_y, p_z = p[..., 0], p[..., 1], p[..., 2]
    n_x, n_y, n_z = n[..., 0], n[..., 1], n[..., 2]

#     if res_coords_wrt_plane:
#         point_on_plane_x = p_x - n_x * (d + k)
#         point_on_plane_y = p_y - n_y * (d + k)
#         point_on_plane_z = p_z - n_z * (d + k)
#     else:
    point_on_plane_x = p_x - n_x * d
    point_on_plane_y = p_y - n_y * d
    point_on_plane_z = p_z - n_z * d
    point_on_plane = np.dstack((point_on_plane_x, point_on_plane_y, point_on_plane_z))

    return point_on_plane

def intersect_line3_with_sphere_vectorized(line, sphere = None):
    '''
    @brief it can compute the intersection of an 3D line and Plane using numpy arrays in a vectorized approach
    @param line: Either an Euclid Line3 object or a Numpy ndarray of points in row-wise order,
            such as [p.x,p.y,p.z,v.x,v.y,v.z], where p is a point in the line, and v is the direction vector of the line.
    @param sphere: A Euclid Sphere object
    @return The ndarray of the intersection coordinates with one ndarray per intersection, such that you get a matrix shape as [m,n,6],
            where the first 3 values [row,col, :3] for each entry [row,col] are the coordinates of the first point intersection, and [row,col, 3:] is for the second point.
    '''
    import omnistereo.euclid as euclid

    if isinstance(line, euclid.Line3):
        line_point_np = np.array(line.p)
        line_dir_vec_np = np.array(line.v)
    elif isinstance(line, np.ndarray):
        line_point_np = line[..., 0:3]
        line_dir_vec_np = line[..., 3:]
    else:
        raise TypeError ("Line must be a Euclid Line3 or defined as a ndarray \n" +
                         "such as [p.x,p.y,p.z,v.x,v.y,v.z], not %s" % type(line))

    if sphere == None:
        sphere_center_np = np.array([0, 0, 0])
        sphere_radius = 1.
    else:
        sphere_center_np = np.array(sphere.c)
        sphere_radius = sphere.r

    # Method based on Euclid's implementation of _intersect_line3_sphere
    # Magnitude squared
    a = np.sum(line_dir_vec_np ** 2, axis = -1)
    sph_mag_2 = np.sum(sphere_center_np ** 2, axis = -1)
    line_p_mag_2 = np.sum(line_point_np ** 2, axis = -1)

    p_wrt_sph = line_point_np - sphere_center_np
    b = 2 * np.sum(line_dir_vec_np * p_wrt_sph, axis = -1)

    c = sph_mag_2 + line_p_mag_2 - \
        2 * np.dot(line_point_np, sphere_center_np) - \
        sphere_radius ** 2

    det = b ** 2 - 4 * a * c
    sq = np.sqrt(det)

    u1 = (-b + sq) / (2 * a)
    u2 = (-b - sq) / (2 * a)
#     if not L._u_in(u1):
#         u1 = max(min(u1, 1.0), 0.0)
#     if not L._u_in(u2):
#         u2 = max(min(u2, 1.0), 0.0)

    line_segment_np = np.dstack((line_point_np + u1[..., np.newaxis] * line_dir_vec_np, line_point_np + u2[..., np.newaxis] * line_dir_vec_np))
    return line_segment_np

def get_normalized_points_XYZ(x, y, z):
    # vector magnitude
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    x_norm = x / d
    y_norm = y / d
    z_norm = z / d

    point_as_np = np.zeros(x.shape + (3,))  # Create a multidimensional array based on the shape of x, y, or z
    point_as_np[:, :, 0] = x_norm
    point_as_np[:, :, 1] = y_norm
    point_as_np[:, :, 2] = z_norm

    return point_as_np

def get_normalized_points(points_wrt_M):
#     x = points_wrt_M[..., 0]
#     y = points_wrt_M[..., 1]
#     z = points_wrt_M[..., 2]
#     # Compute the normalization (projection to the sphere) of points
#     points_norms = np.sqrt(x ** 2 + y ** 2 + z ** 2)
#     points_on_sphere = points_wrt_M[..., :3] / points_norms[..., np.newaxis]
    # More efficiently:
    points_on_sphere = points_wrt_M[..., :3] / (np.linalg.norm(points_wrt_M[..., :3], axis = -1)[..., np.newaxis])
    return points_on_sphere

# FIXME: change the data to use column vectors
def get_lines_through_single_point3(point1, point2):
    '''
    Lines composed from a single or set of x, y, z coordinates (point1) and the single point (point2).
    @param point1: An Euclid Point3 or a ndarray of point1 written as row vectors, for example [[p1.x,p1.y,p1.z],[[p2.x,p2.y,p2.z]], ...]
    @param point2: An Euclid Point3 or a row-vector point2 defined as [p.x,p.y,p.z], which is crossed by the lines passing by point1 with coordinates x, y, z.
    @return Line(s) from (a single or several) point1 directed toward the point2.
    '''
    import omnistereo.euclid as euclid

    if isinstance(point1, euclid.Point3):
        point1_as_np = np.array(point1)
    elif isinstance(point1, np.ndarray):
        point1_as_np = point1
    else:
        raise TypeError ("point2 must be a Euclid Point3 or defined as a row-vector in a ndarray \n" +
             "such as [p.x,p.y,p.z], not %s" % type(point2))
    if isinstance(point2, euclid.Point3):
        point2_as_np = np.array(point2)
    elif isinstance(point2, np.ndarray):
        point2_as_np = point2
    else:
        raise TypeError ("point2 must be a Euclid Point3 or defined as a row-vector in a ndarray \n" +
             "such as [p.x,p.y,p.z], not %s" % type(point2))
    lines = np.zeros(point1.shape[:-1] + (6,))  # Create a multidimensional array based on the point1's shape
    p = point1_as_np
    v = point1_as_np - point2_as_np
    lines[..., 0] = p[..., 0]
    lines[..., 1] = p[..., 1]
    lines[..., 2] = p[..., 2]
    lines[..., 3] = v[..., 0]
    lines[..., 4] = v[..., 1]
    lines[..., 5] = v[..., 2]

    return lines

class KeyPointAndDescriptor(object):

    def __init__(self, kpts_list, desc_list, coords_array = None, random_colors_RGB_list = [], do_flattening = False, **kwargs):
        # Keypoints
        if do_flattening:
            from omnistereo.common_tools import flatten
            self.keypoints = flatten(kpts_list)
            # Feature Descriptors
            desc = []
            for d_list in desc_list:
                if d_list is not None:
                    for d in d_list:
                        desc.append(d)
            self.descriptors = np.array(desc)
        else:  # It's the caller's responsibility to assume data is not in multidimensional arrays (list of lists)
            self.keypoints = kpts_list
            # Feature Descriptors
            self.descriptors = desc_list  # Descriptors are put in a Numpy Array!
        l = len(self.keypoints)
        # Pixel coordinates
        if coords_array is None or len(coords_array) == 0:
            self.pixel_coords = np.ones((1, l, 3))
            for idx in range(l):
                self.pixel_coords[0, idx, 0] = self.keypoints[idx].pt[0]
                self.pixel_coords[0, idx, 1] = self.keypoints[idx].pt[1]
        else:
            self.pixel_coords = coords_array

        # Random matching colors for visualization
        if random_colors_RGB_list is None or len(random_colors_RGB_list) < l:
            #===================================================================
            # from random import randint
            # self.random_colors_RGB = l * [(0, 0, 0)]
            # for i in range(l):
            #     random_color_RGB = (randint(0, 255), randint(0, 255), randint(0, 255))
            #     self.random_colors_RGB[i] = random_color_RGB
            #===================================================================
            self.random_colors_RGB = np.random.randint(low = 0, high = 256, size = (l, 3), dtype = "uint8")
        else:
            self.random_colors_RGB = random_colors_RGB_list

class PanoramicCorrespondences(object):

    def __init__(self, kpts_top_list, desc_top_list, kpts_bot_list, desc_bot_list, points_3D = None, m_top_array = None, m_bot_array = None, random_colors_RGB_list = [], do_flattening = False, **kwargs):
        # Keypoints
        if do_flattening:
            from omnistereo.common_tools import flatten
            self.kpts_top = flatten(kpts_top_list)
            self.kpts_bot = flatten(kpts_bot_list)
            # Feature Descriptors
            desc_top = []
            for d_list in desc_top_list:
                if d_list is not None:
                    for d in d_list:
                        desc_top.append(d)
            self.desc_top = np.array(desc_top)

            desc_bot = []
            for d_list in desc_bot_list:
                if d_list is not None:
                    for d in d_list:
                        desc_bot.append(d)
            self.desc_bot = np.array(desc_bot)
        else:  # It's the caller's responsibility to assume data is not in multidimensional arrays (list of lists)
            self.kpts_top = kpts_top_list
            self.kpts_bot = kpts_bot_list
            # Feature Descriptors
            self.desc_top = desc_top_list
            self.desc_bot = desc_bot_list

        # 3D Point coordinates
        if points_3D is not None:
            l_p3D = len(points_3D)
            if l_p3D > 0:
                if points_3D.shape[-1] == 3:
                    self.points_3D_coords_homo = np.ones((l_p3D, 4))
                    self.points_3D_coords_homo[:, :3] = points_3D[:, :3]
                else:
                    self.points_3D_coords_homo = points_3D
            else:
                self.points_3D_coords_homo = np.empty((0, 4))
        else:
            self.points_3D_coords_homo = []

        # Pixel coordinates
        if m_top_array is None or len(m_top_array) == 0:
            if len(self.kpts_top) > 0:
                self.m_top = cv2.KeyPoint_convert(self.kpts_top).astype(np.float)
                # Add the ones for the point list in homogeneous coordinates
                self.m_top = np.hstack((self.m_top, np.ones_like(self.m_top[..., 0, np.newaxis])))  # Adding a 1
            else:
                self.m_top = np.empty((0, 3))
        else:
            self.m_top = m_top_array

        if m_bot_array is None or len(m_bot_array) == 0:
            if len(self.kpts_bot) > 0:
                self.m_bot = cv2.KeyPoint_convert(self.kpts_bot).astype(np.float)
                # Add the ones for the point list in homogeneous coordinates
                self.m_bot = np.hstack((self.m_bot, np.ones_like(self.m_bot[..., 0, np.newaxis])))  # Adding a 1
            else:
                self.m_bot = np.empty((0, 3))
        else:
            self.m_bot = m_bot_array

        # Random matching colors for visualization
        l_top = len(self.kpts_top)
        l_bot = len(self.kpts_bot)
        l = max(l_top, l_bot)
        if random_colors_RGB_list is None or len(random_colors_RGB_list) < l:
            self.random_colors_RGB = np.random.randint(low = 0, high = 256, size = (l, 3), dtype = "uint8")
        else:
            self.random_colors_RGB = random_colors_RGB_list

class FeatureMatcher(object):

    def __init__(self, method, matcher_type, k_best, *args, **kwargs):
        '''
        @param method: Currently immplemented methods are: "ORB", "SIFT"
        @param matcher_type: Either "BF" for brute-force, or "FLANN" for a FLANN-based matcher
        @param k_best: The number of matches to be considered among each point feature
        '''
        self.feature_detection_method = method
        self.matcher_type = matcher_type
        self.k_best = k_best

        self.FLANN_INDEX_KDTREE = 1
        self.FLANN_INDEX_LSH = 6
        self.MIN_MATCH_COUNT = 10
        self.percentage_good_matches = kwargs.get("percentage_good_matches", 1.0)
        self.num_of_features = kwargs.get('num_of_features', 100)  # Not useful for detector that use non-maximal suppression, such as FAST and AGAST
        # ATTENTION: The radius for the radiusMatch here uses as a distance threshold a descriptor metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured in Pixels)!
        self.use_radius_match = kwargs.get('use_radius_match', False)  # NOTE: when radius match is used, the knn matching is disabled

        if self.matcher_type == "FLANN":
            # FIXME: We need to specify the correct dictionary parameters for FLANN
            # search_params = dict(checks=50)
            search_params = dict(checks = 10)  # The lower the number of checks, the faster
            if self.feature_detection_method.upper() == "SIFT" or self.feature_detection_method.upper() == "SURF":
                index_params = dict(algorithm = self.FLANN_INDEX_KDTREE, trees = 5)
            else:
                index_params = dict(algorithm = self.FLANN_INDEX_LSH,
                                    table_number = 6,  # 12
                                    key_size = 12,  # 20
                                    multi_probe_level = 1)  # 2
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # FIXME: We need to specify the correct normType parameter for the feature used with the Brute Force matcher
            if self.feature_detection_method.upper() == "SIFT" or self.feature_detection_method.upper() == "SURF":
                self.matcher = cv2.BFMatcher()  # By default, it is cv2.NORM_L2. It is good for SIFT, SURF etc
            else:  # For binary feature descriptors, such as ORB, BRIEF, BRISK, etc.
                # self.matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
                self.matcher = cv2.BFMatcher(normType = cv2.NORM_HAMMING)

    def match(self, query_descriptors, train_descriptors, max_descriptor_distance_radius = -1):
        '''
        @param max_descriptor_distance_radius: (Optional) This maximum radius distance is only applicable if the "use_radius_match" member attribute is set in order to apply a radiusMatch
        @attention: the max_descriptor_distance_radius is a metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured in Pixels)!
        '''
        self.matcher.clear()  # Clear matcher
        # self.matcher.add([top_descriptors])

        if self.use_radius_match:
            from omnistereo.common_tools import flatten
            matches_by_radius = self.matcher.radiusMatch(queryDescriptors = query_descriptors, trainDescriptors = train_descriptors, maxDistance = max_descriptor_distance_radius)
            matches = flatten(matches_by_radius)
        else:
            if self.k_best > 1:
                from omnistereo.common_tools import flatten
                # Save the best k matches:
                knn_matches = self.matcher.knnMatch(queryDescriptors = query_descriptors, trainDescriptors = train_descriptors, k = self.k_best)
                if self.feature_detection_method.upper() == "SIFT" and self.k_best == 2:
                    # Store only the good matches as per Lowe's ratio test.
                    knn_2_best_matches_for_SIFT = [m[0] for m in knn_matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
                    #===============================================================
                    # # This makes more sense to Carlos, but this is not how Lowe's ratio test is applied here https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
                    # knn_2_best_matches_for_SIFT = []
                    # for m in knn_matches:
                    #     if len(m) == 2:
                    #         if m[0].distance < m[1].distance * 0.75:
                    #             knn_2_best_matches_for_SIFT.append(m[0])  # store only the first one
                    #         else:
                    #             knn_2_best_matches_for_SIFT.append(m)  # store both matches
                    #     else:
                    #         knn_2_best_matches_for_SIFT.append(m)  # store the possible single match
                    #===============================================================
                    matches = flatten(knn_2_best_matches_for_SIFT)
                else:
                    # Flatten all matches in list of k-lists
                    matches = flatten(knn_matches)
            else:
                # returns the best match only
                matches = self.matcher.match(queryDescriptors = query_descriptors, trainDescriptors = train_descriptors)
        # Sort good matches (filter by above criteria) in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        return matches

class CamParams(object):
    '''
    Super class to stores camera pre-calibration parameters from a pre-calibration step (e.g. done in some MATLAB toolbox)
    '''

    def __init__(self, *args, **kwargs):
        self.parent_model = None
        self.focal_length = None
        self.pixel_size = None
        self.sensor_size = None
        self.pixel_area = None
        self.image_size = None
        self.center_point = None

        # Distortion
        self.k1 = None
        self.k2 = None
        self.k3 = None
        self.p1 = None
        self.p2 = None

        self.gamma1 = None  # Focal length on u [px/mm or px/m] based on distance units of model
        self.gamma2 = None  # Focal length on v [px/mm or px/m] based on distance units of model
        self.f_u = self.gamma1
        self.f_v = self.gamma2
        self.u_center = None
        self.v_center = None
        self.alpha_c = None
        self.skew_factor = None

        # Only used for masking: # TODO: use actually in the masking process
        self.center_point_inner = None
        self.center_point_outer = None
        self.u_center_inner = None
        self.v_center_inner = None
        self.u_center_outer = None
        self.v_center_outer = None

        # ROI:
        self.roi_min_x = None
        self.roi_min_y = None
        self.roi_max_x = None
        self.roi_max_y = None

        # TODO: add distortion parameters
        # Inverse camera projection matrix parameters
        self.K_inv = None

    def print_params(self):
        try :
            distortion_params = "k1={0:.6f}, k2={1:.6f}, k3={2:.6f}, p1={3:.6f}, p2={4:.6f} ".format(self.k1, self.k2, self.k3, self.p1, self.p2)
            print(distortion_params)
        except :
            pass

        try :
            aberr_params = "gamma1={0:.6f}, gamma2={1:.6f}, alpha_c={2:.6f}".format(self.gamma1, self.gamma2, self.alpha_c)
            print(aberr_params)
        except :
            pass

        try :
            misc_params = "u_center={0:.6f}, v_center={1:.6f}".format(self.u_center, self.v_center)
            print(misc_params)
        except :
            pass

        try :
            roi_params = "roi_min_x={0:.6f}, roi_min_y={1:.6f}, roi_max_x={2:.6f}, roi_max_y={3:.6f}".format(self.roi_min_x, self.roi_min_y, self.roi_max_x, self.roi_max_y)
            print(roi_params)
        except :
            pass

        try :
            min_hor = np.rad2deg(self.min_useful_FOV_hor)
            max_hor = np.rad2deg(self.max_useful_FOV_hor)
            min_ver = np.rad2deg(self.min_useful_FOV_ver)
            max_ver = np.rad2deg(self.max_useful_FOV_ver)

            cam_useful_FOVs = "Useful Camera FOVs (in degrees): min_hor={0:.3f}, min_ver={1:.3f}, max_hor={2:.3f}, max_ver={3:.3f}".format(min_hor, min_ver, max_hor, max_ver)
            print(cam_useful_FOVs)
        except :
            pass

class PinholeCamera(CamParams):

    def __init__(self, parent, image_size_pixels, focal_length = 1, pixel_size = None, skew_factor = 0, sensor_size = None, camera_matrix = None, custom_center = None):
        '''
        @param image_size_pixels:  Image size must be given as (width, height)
        '''
        self.set_intrinsic_matrix(image_size_pixels, focal_length, pixel_size, skew_factor, sensor_size, camera_matrix, custom_center)
        self.parent_model = parent

    def set_intrinsic_matrix(self, image_size_pixels, focal_length = 1, pixel_size = None, skew_factor = 0, sensor_size = None, camera_matrix = None, custom_center = None):
        self.image_size = np.array(image_size_pixels)  # Pixels area: (width, height)
        self.sensor_size = sensor_size  # (width, height) in [mm]
        if camera_matrix is not None and sensor_size is not None:
            from cv2 import calibrationMatrixValues
            aperture_width, aperture_height = sensor_size  # in [mm]
            fovx, fovy, self.focal_length, principal_point, aspect_ratio = calibrationMatrixValues(cameraMatrix = camera_matrix, imageSize = (image_size_pixels[0], image_size_pixels[1]), apertureWidth = aperture_width, apertureHeight = aperture_height)
            self.FOV = (np.deg2rad(fovx), np.deg2rad(fovy))
            self.K = camera_matrix
            self.f_u = camera_matrix[0, 0]  # in [px]
            self.f_v = camera_matrix[1, 1]  # in [px]
            self.skew_factor = camera_matrix[0, 1]
            hx = focal_length / self.f_u  # [mm / px]
            hy = focal_length / self.f_v  # [mm / px]
            self.pixel_size = np.array([hx, hy])  # in [mm/px]
            if custom_center != None:  # Update matrix to use extracted center (omnidirectional camera case)
                camera_matrix[0, 2] = custom_center[0]
                camera_matrix[1, 2] = custom_center[1]
            self.u_center = camera_matrix[0, 2]
            self.v_center = camera_matrix[1, 2]
            self.center_point = np.array([self.u_center, self.v_center])
        else:
            self.focal_length = focal_length  # focal length
            if self.sensor_size is not None:
                self.pixel_size = self.sensor_size / self.image_size  # (h_x, h_y) [mm / px]
            else:
                # We know the pixel size:
                if pixel_size is not None:
                    self.pixel_size = pixel_size  # (h_x, h_y) [mm / px]
                    self.sensor_size = self.pixel_size * self.image_size
                else:
                    from sys import exit
                    import warnings
                    warnings.warn("Error...%s" % ("Insufficient information. Please, specify at least either pixel size [mm] or sensor size [mm, mm]"))
                    print("Exiting from", __name__)
                    exit(1)

            self.FOV = 2 * np.arctan(self.sensor_size / (2 * self.focal_length))  # A tuple (hor, ver) FOVs in [radians]
            self.skew_factor = skew_factor

            self.f_u, self.f_v = self.focal_length / self.pixel_size  # [mm]/[mm/px] --> [px]

            if custom_center is not None:  # Update matrix to use extracted center (omnidirectional camera case)
                self.center_point = custom_center
            else:
                self.center_point = (self.image_size / 2.0) - 1  # image size should be given as (width, height)

            # Needed for radial masks with theoretical model:
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            self.center_point_inner = self.center_point
            self.u_center_inner = self.center_point[0]
            self.v_center_inner = self.center_point[1]
            self.center_point_outer = self.center_point
            self.u_center_outer = self.center_point[0]
            self.v_center_outer = self.center_point[1]
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            self.u_center, self.v_center = self.center_point
            # Camera's intrinsic matrix
            self.K = np.zeros((3, 3))
            self.K[0, :] = [self.f_u, self.skew_factor, self.u_center]
            self.K[1, 1] = self.f_v
            self.K[1, 2] = self.v_center
            self.K[2, 2] = 1.0

        self.pixel_area = self.pixel_size[0] * self.pixel_size[1]  # in [mm^2/px^2]
        print("Image size: %s pixels" % (self.image_size))
        print("Pixel size: %s mm" % (self.pixel_size))
        print("Sensor size: %s mm" % (self.sensor_size))
        print("Focal length: %s mm" % (self.focal_length))
        print("FOVs: hor = %f degrees, vert = %f degrees" % (np.rad2deg(self.FOV[0]), np.rad2deg(self.FOV[1])))
        print("Intrinsic Matrix:", self.K)

        f_u, f_v = self.f_u, self.f_v
        self.gamma1, self.gamma2 = self.f_u, self.f_v
        s = self.skew_factor
        u_c, v_c = self.u_center, self.v_center
        # self.K_inv = np.linalg.inv(self.K) # Also valid
        self.K_inv = np.zeros_like(self.K)
        self.K_inv[0, :] = [1 / f_u, -s / (f_u * f_v), (s * v_c - f_v * u_c) / (f_u * f_v)]
        self.K_inv[1, 1] = 1 / f_v
        self.K_inv[1, 2] = -v_c / f_v
        self.K_inv[2, 2] = 1.0
        print("Inverse Intrinsic Matrix:", self.K_inv)

    def get_spatial_resolution_in_2D(self, p, in_pixels = False, in_radians = True, use_spatial_resolution = True):
        '''
        Spatial resolution for the perspective camera using 2D angles

        @param p: The ndarray of 3d point(s) of shape (rows, cols, vect_size) to compute the resolution at
                   Note: p must be given already transformed on the respective frame (e.g. when using mirror 2 it must be in the virtual camera frame)
        @param in_pixels: Set to True in order to provide the spatial resolution normalized in \f$\left[{st}/{px^2}\right]\f$. If False (default), it will be given as \f$\left[{st}/{area}\right]\f$
        @param in_radians: True (default) to provide the spatial resolution in \f$\left[{rad}/{px}\right]\f$. Otherwise it will be given as \f$\left[{deg}/{px}\right]\f$
        @param use_spatial_resolution: If True, it indicates to calculate the spatial resolution (as a ratio) in [length per radian] units instead of [radian per length].

        @return the ndarray of 2D resolutions for the respective profile points
        '''
        eta_cam = self.get_spatial_resolution(p)  # We get [st/mm^2]
        # Convert solid angle to 2D angle (as cross-section of cone) and get the square root of area
        # Thinking the infinitesimal surface is a square:
#         eta_cam_2D = np.sqrt(eta_cam) / convert_steradian_to_radian(1)
        # Thinking the infinitesimal surface is a circle:
        eta_cam_2D = 2 * np.sqrt(eta_cam / np.pi) / convert_steradian_to_radian(1)
        if use_spatial_resolution == False:
            eta_cam_2D = 1 / eta_cam_2D
        eta_cam_2D = convert_resolution_units(self.pixel_size[0], in_pixels, in_radians, use_spatial_resolution, eta_cam_2D, in_2D = True)

        return eta_cam_2D

    def get_spatial_resolution(self, p, in_pixels = False, in_steradians = True, use_spatial_resolution = True):
        '''
        Spatial resolution for the conventional perspective camera

        @param p: The ndarray of 3d point(s) of shape (rows, cols, vect_size) to compute the resolution at
                  Note: p must be given already transformed on the respective frame (e.g. when using mirror 2 it must be in the virtual camera frame)
        @param in_pixels: Set to True in order to provide the spatial resolution normalized in \f$\left[{st}/{px^2}\right]\f$. If False (default), it will be given as \f$\left[{st}/{area}\right]\f$
        @param in_steradians: True (default) to provide the spatial resolution in \f$\left[{st}/{px^2}\right]\f$. Otherwise it will be given as \f$\left[{px^2}/{area}\right]\f$
               To convert from steradian to square degrees multiply by 3282.810874
        @param use_spatial_resolution: If True, it indicates to calculate the spatial resolution (as a ratio) in [area per st] units instead of [st per area].

        @return the ndarray of resolutions for the respective input points
        '''
        f = self.focal_length
#         if in_pixels:
#             f = f / self.pixel_size[0]

        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        r_sq = get_r(x, y, give_r_squared = True)

        # Radius wise:
        cos_psi_cube = (z / (np.sqrt(z ** 2 + r_sq))) ** 3
        # Equation on the image plane (Used to verify camera resolution instead of above as to a flat mirror):
#         u_coord = np.arange(-self.image_size[0] / 2, self.image_size[0] / 2)
#         if in_pixels == False:
#             u_coord = u_coord * self.pixel_size[0]
#         cos_psi_cube = (f / (np.sqrt(f ** 2 + u_coord ** 2))) ** 3

        eta_cam = f ** 2 / cos_psi_cube
        if use_spatial_resolution == False:
            eta_cam = 1 / eta_cam

        return eta_cam

    def get_spatial_resolution_in_2D_as_BakerNayar(self, p, in_pixels = False, in_radians = True, use_spatial_resolution = True):
        '''
        Spatial resolution (in 2D) for the conventional perspective camera using Baker & Nayar's equations

        @param p: The ndarray of 3d point(s) of shape (rows, cols, vect_size) to compute the resolution at
                  Note: p must be given already transformed on the respective frame (e.g. when using mirror 2 it must be in the virtual camera frame)
        @param in_pixels: Set to True in order to provide the spatial resolution normalized in \f$\left[{st}/{px^2}\right]\f$. If False (default), it will be given as \f$\left[{st}/{area}\right]\f$
        @param in_radians: True (default) to provide the spatial resolution in \f$\left[{st}/{px^2}\right]\f$. Otherwise it will be given as \f$\left[{px^2}/{area}\right]\f$
               To convert from steradian to square degrees multiply by 3282.810874
        @return the ndarray of resolutions for the respective input points
        '''
        eta_cam = self.get_spatial_resolution_as_BakerNayar(p)
        # Convert solid angle to 2D angle (as cross-section of cone) and get the square root of area
        eta_cam_2D = np.sqrt(eta_cam) / convert_steradian_to_radian(1)
        if use_spatial_resolution == False:
            eta_cam_2D = 1 / eta_cam_2D

        eta_cam_2D = convert_resolution_units(self.pixel_size[0], in_pixels, in_radians, use_spatial_resolution, eta_cam_2D, in_2D = True)

        return eta_cam_2D

    def get_spatial_resolution_as_BakerNayar(self, p, in_pixels = False, in_steradians = True, use_spatial_resolution = True):
        '''
        Spatial resolution for the conventional perspective camera using Baker & Nayar's equations

        @param p: The ndarray of 3d point(s) of shape (rows, cols, vect_size) to compute the resolution at
                  Note: p must be given already transformed on the respective frame (e.g. when using mirror 2 it must be in the virtual camera frame)
        @param in_pixels: Set to True in order to provide the spatial resolution normalized in \f$\left[{st}/{px^2}\right]\f$. If False (default), it will be given as \f$\left[{st}/{area}\right]\f$
        @param in_steradians: True (default) to provide the spatial resolution in \f$\left[{st}/{px^2}\right]\f$. Otherwise it will be given as \f$\left[{px^2}/{area}\right]\f$
               To convert from steradian to square degrees multiply by 3282.810874
        @return the ndarray of resolutions for the respective input points
        '''
        u = self.focal_length
        c = self.parent_model.c
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        r_sq = get_r(x, y, give_r_squared = True)

        cos_psi_cube = ((c - z) / (np.sqrt((c - z) ** 2 + r_sq))) ** 3
        eta_cam = u ** 2 / cos_psi_cube

        if in_pixels:
            eta_cam = eta_cam / self.pixel_area  # [mm^2/st]/[mm^2/px^2] ==> [px^2/st]
        if in_steradians == False:
            # To convert from steradian to square degrees multiply by (180/pi)^2 or 3282.810874
            eta_cam = eta_cam / ((180 / np.pi) ** 2)  # [area/st]/[deg^2/st] ==> [area/deg^2]

        if use_spatial_resolution == False:
            eta_cam = 1 / eta_cam

        return eta_cam

class PerspectiveCamModel(object):

    def __init__(self, **kwargs):
        self.units = kwargs.get("units", "m")
        self.intrinsic_params_matrix = kwargs.get("intrinsic_params_matrix", None)
        self.distortion_coeffs = kwargs.get("distortion_coeffs", None)
        self.image_size = kwargs.get("image_size", None)
        self.T_model_wrt_C = np.identity(4)
        self.T_C_wrt_model = np.identity(4)
#         self.cam_params = CamParams(kwargs)

class RGBDCamModel(object):

    def __init__(self, **kwargs):
        self.fx = kwargs.get("fx", 525.0)
        self.fy = kwargs.get("fy", 525.0)
        self.center_x = kwargs.get("center_x", 319.5)
        self.center_y = kwargs.get("center_y", 239.5)
        self.focal_length_m = kwargs.get("focal_length_m", 1.0 / 1000.0)
        #===============================================================================
        # self.fx = 517.3
        # self.fy = 516.5
        # centerX = 318.6
        # centerY = 255.3
        #===============================================================================
        self.depth_is_Z = kwargs.get("depth_is_Z", True)
        # depth_is_Z: When True, the depth value is a radial distance. Otherwise, the depth encodes the Z-values.
        # NOTE: See http://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html for information about the transformations

        self.units = kwargs.get("units", "m")  # Depth units in meters by default
        self.scaling_factor = kwargs.get("scaling_factor", 1. / 1000.0)  # [mm] to [m] for the depth data
        self.do_undistortion = kwargs.get("do_undistortion", False)
        self.distortion_coeffs = kwargs.get("distortion_coeffs", np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633]))
        self.K = np.matrix([[self.fx, 0, self.center_x], [0, self.fy, self.center_y], [0, 0, 1]])
        self.undistorted_K = np.empty([3, 3])

        self.image_size = kwargs.get("image_size", None)
        self.T_model_wrt_C = np.identity(4)
        self.T_C_wrt_model = np.identity(4)
        self.T_Cest_wrt_Rgt = None
        self.feature_matcher_for_motion = None

    def get_depth_Z(self, depth, uv_coords = None, verbose = False):
        if not self.depth_is_Z:
            if uv_coords is None:
                uv_coords = np.transpose(np.indices(depth.shape[::-1]), (0, 2, 1))
            else:
                assert depth.shape[:2] == depth.shape
            focal_length = self.focal_length_m  # Camera Focal length in meters
            if verbose:
                print("Z-Depth evaluated with fx = %.6f [px], fy = %.6f [px], and a focal length = %.6f [m]" % (self.fx, self.fy, focal_length))
            pixel_size_horizontal = focal_length / self.fx
            pixel_size_vertical = focal_length / self.fy
            x_i = pixel_size_horizontal * (uv_coords[0] - self.center_x)
            y_i = pixel_size_vertical * (uv_coords[1] - self.center_y)
            z_i = np.ones_like(x_i) * focal_length
            xyz_img_plane_coords = np.dstack([x_i, y_i, z_i])
            d_to_img_plane = np.linalg.norm(xyz_img_plane_coords, axis = -1)
            depth = focal_length * depth / d_to_img_plane

        return depth

    def get_only_valid_RGB_XYZ(self, rgb, depth, uv_coords = None):
        '''
        A flattened result of the RGB and XYZ values will be returned. Here, the UV-coordinates mapping may be changed because those 0-depth values will be skipped.

        @param uv_coords: (Optional) The UV-coordinates associated with the RGB and Depth value arguments. This must a numpy array of shape 2xMxN, where M is a number of rows, and N is a number of columns. The number of rows and columns should match the corresponding shapes of the "rgb" and "depth" inputs.
        '''
        if (self.do_undistortion):
            # TODO: this should only happens once!
            rgb_undistorted = cv2.undistort(rgb, self.K, self.distortion_coeffs, None, self.undistorted_K)
        else:
            rgb_undistorted = rgb

        if uv_coords is None:
            uv_coords = np.transpose(np.indices(depth.shape[::-1]), (0, 2, 1))
        else:
            assert rgb.shape[:2] == depth.shape
            assert rgb.shape[:2] == uv_coords[0].shape

        depth = self.get_depth_Z(depth = depth, uv_coords = uv_coords)
        nonzero_depth_uv_pixel_coords = depth != 0
        valid_depth_Z_values = depth[nonzero_depth_uv_pixel_coords]
        valid_RGB = rgb_undistorted[nonzero_depth_uv_pixel_coords]

        u_coords = uv_coords[0]
        v_coords = uv_coords[1]
        u = u_coords[nonzero_depth_uv_pixel_coords]
        v = v_coords[nonzero_depth_uv_pixel_coords]
        Z = valid_depth_Z_values
        X = (u - self.center_x) * Z / self.fx
        Y = (v - self.center_y) * Z / self.fy
        valid_XYZ = np.hstack([X[..., np.newaxis], Y[..., np.newaxis], Z[..., np.newaxis]])

        return valid_RGB, valid_XYZ

    def get_XYZ(self, depth, u_coords = None, v_coords = None):
        '''
        The shape of the UV-coordinates will dictate preserved in the XYZ outputs. Those 0-depth values will get NANs for their XYZ values

        @param depth: The depth map (2-dimensional MxN array) holding the depth values (a.k.a. Z-values)
        @param u_coords: (Optional) The U-coordinates associated with the depth value arguments to be resolved for.
        @param v_coords: (Optional) The V-coordinates associated with the depth value arguments to be resolved for.

        @return: The XYZ coordinates. However, when u and v coordinates are passed, the resulting XYZ matrix will have flattened dimensions (e.g. 1xSIZEx3). Otherwise, the resulting shape will be MxNx3
        '''
        depth = self.get_depth_Z(depth = depth, uv_coords = None)
        Z = np.where(depth != 0, depth, np.nan)
        if u_coords is None or v_coords is None:
            uv_coords = np.transpose(np.indices(depth.shape[::-1]), (0, 2, 1))
            u_coords = uv_coords[0]
            v_coords = uv_coords[1]
        else:
            # We have to resolve for the depth value at the particular coordinates
            u_coords = u_coords.ravel()
            v_coords = v_coords.ravel()
            Z = Z[v_coords, u_coords]

        X = (u_coords - self.center_x) * Z / self.fx
        Y = (v_coords - self.center_y) * Z / self.fy
        XYZ = np.dstack((X, Y, Z))
        return XYZ

class OmniCamModel(object):
    '''
    A single omnidirectional camera model
    Mostly, a superclass template template to be implemented by a custom camera model
    '''

    def _init_default_values(self, **kwargs):
        self.is_calibrated = False
        self.mirror_number = kwargs.get("mirror_number", 0)
        self.F = kwargs.get("F", np.array([0., 0., 0., 1.]).reshape(4, 1))  # Focus (inner, real)
        self.Fv = kwargs.get("F_virt", np.array([0., 0., 0., 1.]).reshape(4, 1))  # Focus (outer, virtual)
        self.T_model_wrt_C = np.identity(4)
        self.T_C_wrt_model = np.identity(4)
        # NOTE: in this model configuration, the rotation is an identity. However, this will not always be the case for other configurations.
        self.T_model_wrt_C[:3, 3] = self.F[:3, 0]
        self.T_C_wrt_model[:3, 3] = -self.F[:3, 0]

        self.units = kwargs.get("units", "m")

        # Physical radii
        self.r_reflex = kwargs.get("r_reflex", None)
        self.r_min = kwargs.get("r_min", None)
        self.r_max = kwargs.get("r_max", None)

        if self.r_min is not None:
            # Compute elevations
            xw = self.r_min
            yw = 0
            zw = self.get_z_hyperbola(xw, yw)
            theta = np.arctan2(zw - self.F[2], xw)
            if self.mirror_number == 1:
                self.lowest_elevation_angle = theta  # The elevation determined by the reflex mirror
            else:
                self.highest_elevation_angle = theta  # The elevation determined by the camera hole
        else:
            if self.mirror_number == 1:
                self.lowest_elevation_angle = -np.pi / 2.
            else:
                self.highest_elevation_angle = np.pi / 2.

        if self.r_max != None:  # TODO: Not general enough! Remove this out of here
            # Compute elevations
            xw = self.r_max
            yw = 0
            zw = self.get_z_hyperbola(xw, yw)
            theta = np.arctan2(zw - self.F[2], xw)
            if self.mirror_number == 1:  # Highest elevation determined by the the mirror's radius
                self.highest_elevation_angle = theta
            else:  # Lowest elevation is determined by the minimum elevation from mirror radius or camera hole radius
                theta_due_to_r_sys = theta
                if self.r_reflex != None:
                    c2 = self.c
                    k2 = self.k
                    d = self.d
                    r_ref = self.r_reflex
                    # NOTE: We solved the system of equations for the reflection point (as depicted in pink line in Geometry Expressions)
                    # Solution of the outward projection instead from F2v to the surface mirror 2
                    lambda2_sln = (c2 * d * k2 + c2 * np.sqrt(k2 * (k2 - 2) * (d ** 2 + 4 * r_ref ** 2))) / (k2 * (d ** 2 - 2 * k2 * r_ref ** 2 + 4 * r_ref ** 2))
                    p2_of_reflex_edge = np.array([[[lambda2_sln * r_ref , 0, d - lambda2_sln * d / 2.0, 1.0]]])
                    x_p2 = p2_of_reflex_edge[0, 0, 0]
                    z_p2 = p2_of_reflex_edge[0, 0, 2]
                    theta_due_to_r_reflex = np.arctan2(z_p2 - self.F[2], x_p2)
                else:
                    theta_due_to_r_reflex = -np.inf

                self.lowest_elevation_angle = max(theta_due_to_r_sys, theta_due_to_r_reflex)
        else:
            if self.mirror_number == 1:
                self.highest_elevation_angle = np.pi / 2.
            else:
                self.lowest_elevation_angle = -np.pi / 2.

        self.vFOV = self.highest_elevation_angle - self.lowest_elevation_angle
        self.globally_highest_elevation_angle = self.highest_elevation_angle
        self.globally_lowest_elevation_angle = self.lowest_elevation_angle
        # WISH: add a method to set the calibration parameters from Calibration as a single GUM

        self.outer_img_radius = 0  # outer radius encircling the ROI for occlusion boundary
        self.inner_img_radius = 0  # inner radius encircling the ROI for occlusion boundary
        self.r_lowest_elevation = 0  # Radius in pixels
        self.r_highest_elevation = 0  # Radius in pixels
        # Pixel coordinates of the centers corresponding to the highest/lowest elevation angles computed using the mask's centers (not necessarily the optimal image center):
        self.center_highest_elevation = None
        self.center_lowest_elevation = None

        self.precalib_params = CamParams()
        self.current_omni_img = None
        self.panorama = None
        self.mask = None
        self.construct_new_mask = True
        self.mask_RGB_color = None
        self.mask_background_img = None

    def set_pose(self, translation, rotation_matrix):
        self.F[:3, 0] = translation[:3]
        self.T_model_wrt_C[:3, 3] = translation[:3]
        self.T_model_wrt_C[:3, :3] = rotation_matrix[:3, :3]
        # Inverse transformation:
        self.T_C_wrt_model = np.identity(4)
        self.T_C_wrt_model[:3, :3] = rotation_matrix.T
        self.T_C_wrt_model[:3, 3] = -(rotation_matrix.T).dot(translation[:3])

    def set_params(self, **kwargs):

        if "center_point" in kwargs:
            center_point = kwargs.get("center_point", self.precalib_params.center_point)
            self.precalib_params.center_point = center_point
            self.precalib_params.u_center = center_point[0]
            self.precalib_params.v_center = center_point[1]

        if "center_point_inner" in kwargs:
            center_point_inner = kwargs.get("center_point_inner", self.precalib_params.center_point)
            self.precalib_params.center_point_inner = center_point_inner
            self.precalib_params.u_center_inner = center_point_inner[0]
            self.precalib_params.v_center_inner = center_point_inner[1]
        else:
            self.precalib_params.center_point_inner = self.precalib_params.center_point
            self.precalib_params.u_center_inner = self.precalib_params.center_point[0]
            self.precalib_params.v_center_inner = self.precalib_params.center_point[1]

        if "center_point_outer" in kwargs:
            center_point_outer = kwargs.get("center_point_outer", self.precalib_params.center_point)
            self.precalib_params.center_point_outer = center_point_outer
            self.precalib_params.u_center_outer = center_point_outer[0]
            self.precalib_params.v_center_outer = center_point_outer[1]
        else:
            self.precalib_params.center_point_outer = self.precalib_params.center_point
            self.precalib_params.u_center_outer = self.precalib_params.center_point[0]
            self.precalib_params.v_center_outer = self.precalib_params.center_point[1]

        # In case radii haven't been passed,
        # infer occlusion boundaries automatically from pre-calibration ROI parameters:
        _, v_center = self.get_center()

        radius_outer = kwargs.get("radius_outer", self.outer_img_radius)
        if radius_outer == 0:
            try:
                radius_outer = v_center - self.precalib_params.roi_max_y
            except:
                print("radius_outer not set")
                pass

        radius_inner = kwargs.get("radius_inner", self.inner_img_radius)
        if radius_inner == 0:
            try:
                radius_inner = 0  # It's zero for now (not masking anything).
            except:
                print("radius_inner not set")
                pass

        self.set_radial_limits_in_pixels(inner_img_radius = radius_inner, outer_img_radius = radius_outer, **kwargs)

    def print_params(self, **kwargs):
        self.print_precalib_params()
        print(self.mirror_name + " camera parameters:")
        self.precalib_params.print_params()
        print("Radial image height: %f pixels" % self.h_radial_image)

        vFOV = kwargs.get("vFOV", self.vFOV)
        max_elevation = kwargs.get("max_elevation", self.highest_elevation_angle)
        min_elevation = kwargs.get("min_elevation", self.lowest_elevation_angle)

        print("vFOV: %f degrees" % np.rad2deg(vFOV))
        print("Highest elevation angle: %f degrees" % np.rad2deg(max_elevation))
        print("Lowest elevation angle: %f degrees" % np.rad2deg(min_elevation))

    def print_precalib_params(self):
        pass

    def map_angles_to_unit_sphere(self, theta, psi):
        '''
        Resolves the point (normalized on the unit sphere) from the given direction angles

        @param theta: Elevation angle to a world point from some origin (Usually, the mirror's focus)
        @param psi: Azimuth angle to a world point from some origin.
        @return: The homogeneous coordinates (row-vector) of the 3D point coordinates (as an ndarray) corresponding to the passed direction angles
        '''
        if isinstance(theta, np.ndarray):
            theta_valid_mask = np.logical_not(np.isnan(theta))  # Filters index that are NaNs
            theta_validation = theta_valid_mask.copy()
            # CHECKME: It makes debugging harder, so validation on elevation is disabled for now
            #===================================================================
            # #Checking within valid elevation:
            # if self.is_calibrated:
            #     theta_validation[theta_valid_mask] = np.logical_and(self.lowest_elevation_angle <= theta[theta_valid_mask], theta[theta_valid_mask] <= self.highest_elevation_angle)
            #===================================================================

            b = np.where(theta_validation, np.cos(theta), np.nan)
            z = np.where(theta_validation, np.sin(theta), np.nan)
        else:
            b = np.cos(theta)
            z = np.sin(theta)

        x = b * np.cos(psi)
        y = b * np.sin(psi)

        #=======================================================================
        # z = np.where(np.logical_and(self.lowest_elevation_angle <= theta, theta <= self.highest_elevation_angle),
        #              np.sin(theta), np.nan)
        #=======================================================================

        w = np.ones_like(x)
        P_on_sphere = np.dstack((x, y, z, w))
        return P_on_sphere

    def get_pixel_from_direction_angles(self, azimuth, elevation, visualize = False):
        '''
        @brief Given the elevation and azimuth angles (w.r.t. mirror focus), the projected pixel's coordinates on the warped omnidirectional image is found.

        @param azimuth: The azimuth angles as an ndarray
        @param elevation: The elevation angles as an ndarray
        @retval u: The u pixel coordinate (or ndarray of u coordinates)
        @retval v: The v pixel coordinate (or ndarray of v coordinates)
        @retval m_homo: the pixel as a homogeneous ndarray (in case is needed)
        '''
        Pw_wrt_F = self.get_3D_point_from_angles_wrt_focus(azimuth = azimuth, elevation = elevation)
        return self.get_pixel_from_3D_point_wrt_M(Pw_wrt_F, visualize = False)

    def get_3D_point_from_angles_wrt_focus(self, azimuth, elevation):
        '''
        Finds a world point using the given projection angles towards the focus of the mirror

        @return: The numpy ndarray of 3D points (in homogeneous coordinates) w.r.t. origin of coordinates (\f$O_C$\f)
        '''
        raise NotImplementedError

    def get_pixel_from_3D_point_wrt_C(self, Pw_wrt_C, visualize = False):
        '''
        @brief Project a three-dimensional numpy array (rows x cols x 4) of 3D homogeneous points (eg. [x, y, z, 1]) as row-vectors to the image plane in (\a u,\a v).
        This function is already vectorized for Numpy performance.

        @param Pw_wrt_C: the multidimensional array of homogeneous coordinates of the points (wrt the origin of the common frame [C], e.g. camera pinhole)
        @param visualize: To indicate if a 3D visualization will be shown

        @retval u: the resulting ndarray of u coordinates on the image plane
        @retval v: the resulting ndarray of v coordinates on the image plane
        @retval m_homo: The pixel point(s) as numpy array in homogeneous coordinates
        '''
        raise NotImplementedError

    def get_pixel_from_3D_point_wrt_M(self, Pw_homo, visualize = False):
        '''
        @brief Project a three-dimensional numpy array (rows x cols x 4) of 3D homogeneous points (eg. [x, y, z, 1]) as row-vectors to the image plane in (\a u,\a v).
        This function is already vectorized for Numpy performance.

        @param Pw_homo: the multidimensional array of homogeneous coordinates of the points (wrt the MIRROR focus frame)
        @param visualize: To indicate if a 3D visualization will be shown

        @retval u: the resulting ndarray of u coordinates on the image plane
        @retval v: the resulting ndarray of v coordinates on the image plane
        @retval m_homo: The pixel point(s) as numpy array in homogeneous coordinates
        '''
        raise NotImplementedError

    def get_pixel_from_XYZ(self, x, y, z, visualize = False):
        '''
        @brief Project a 3D points (\a x,\a y,\a z) to the image plane in (\a u,\a v)
               NOTE: This function is not vectorized (not using Numpy explicitly).

        @param x: 3D point x coordinate (wrt the center of the unit sphere)
        @param y: 3D point y coordinate (wrt the center of the unit sphere)
        @param z: 3D point z coordinate (wrt the center of the unit sphere)

        @retval u: contains the image point u coordinate
        @retval v: contains the image point v coordinate
        @retval m: the undistorted 3D point (of type euclid.Point3) in the normalized projection plane
        '''
        raise NotImplementedError

    def get_obj_pts_proj_error(self, img_points, obj_pts_homo, T_G_wrt_C):
        '''
        Compute the pixel errors between the points on the image and the projected 3D points on an object frame [G] with respect to the fixed frame [C]

        @param img_points: The corresponding points on the image
        @param obj_pts_homo: The coordinates of the corresponding points with respect to the object's own frame [G].
        @param T_G_wrt_C: The transform matrix of [G] wrt to [C].

        @return: The array of pixels errors (euclidedian distances a.k.a L2 norms)
        '''
        obj_pts_wrt_C = np.einsum("ij, mnj->mni", T_G_wrt_C, obj_pts_homo)

        # The detected (observed) pixels for chessboard points
        _, _, projected_pts = self.get_pixel_from_3D_point_wrt_C(obj_pts_wrt_C)
        error_vectors = projected_pts[..., :2] - img_points[..., :2]
        error_norms = np.linalg.norm(error_vectors, axis = -1).flatten()
        return error_norms

    def get_confidence_weight_from_pixel_RMSE(self, img_points, obj_pts_homo, T_G_wrt_C):
        '''
        We define a confidence weight as the inverse of the pixel projection RMSE

        @param img_points: The corresponding points on the image
        @param obj_pts_homo: The coordinates of the corresponding points with respect to the object's own frame [G].
        @param T_G_wrt_C: The transform matrix of [G] wrt to [C].
        '''
        from omnistereo.common_tools import rms
        all_pixel_errors = self.get_obj_pts_proj_error(img_points, obj_pts_homo, T_G_wrt_C)
        rmse = rms(all_pixel_errors)
        weight = 1.0 / rmse
        return weight

    def lift_pixel_to_unit_sphere_wrt_focus(self, m, visualize = False, debug = False):
        '''
        @brief Lifts a pixel point from the image plane to the unit sphere
        @param m: A ndarray of k image point coordinates [u, v] per cell (e.g. shape may be rows, cols, 2)
        @param visualize: Indicates if visualization will take place
        @param debug: Indicates to print debugging statements
        @retval Ps: The Euclidean coordinates (as a rows x cols x 3 ndarray) of the point(s) on the sphere.
        '''
        raise NotImplementedError

    def get_direction_vector_from_focus(self, m):
        '''
        @param m: A ndarray of image point coordinates [u, v] per cell (e.g. shape may be rows, cols, 2)

        @return the array of direction vectors in homogenous coordinates (4-vectors)
        '''
        v = self.lift_pixel_to_unit_sphere_wrt_focus(m)

        return v

    def get_direction_angles_from_pixel(self, m_omni):
        '''
        In order to determine the respective azimuth and elevation angles, it lifts the pixel to the unit sphere using lift_pixel_to_unit_sphere_wrt_focus.

        @param m_pano: A numpy array of k image point coordinates [u, v] as row vector. Thus, shape is (rows, cols, 2)
        @retval azimuth, elevation: angles in radians for the image pixel with coordinates (u,v) in the distorted image. Angles are w.r.t. the mirror's focus.
        '''
        # FIXME: lifting with high distortion parameters produces wrong results
        Ps = self.lift_pixel_to_unit_sphere_wrt_focus(m_omni)  # Find point on unit sphere
        azimuth = np.arctan2(Ps[..., 1], Ps[..., 0])
        elevation = np.arcsin(Ps[..., 2])
        return azimuth, elevation

    def get_direction_angles_from_pixel_using_forward_projection(self, m_omni, individual_opt = True, verbose_optimization = False):
        '''
        For consistency with the forward projection model, the direction angles associated to the projected m_omni are obtained iteratively via forward projection.
        The initial are still found by lifting the pixel using the back-projection parameters (usually involving some undistortion procedure). Recall the distortion and its inverse solutions are not a bidirectional mapping.

        @param m_omni: A numpy array of k image point coordinates [u, v] as row vector. Thus, shape is (rows, cols, 2)
        @param individual_opt: When True (default), each point will be forward-projected individually. Otherwise, the projection occurs as group (faster), but in that case there is no guarantee that each point is optimal

        @retval azimuth, elevation: angles in radians for the image pixel with coordinates (u,v) in the distorted image. Angles are w.r.t. the mirror's focus.
        '''
        # Step 1) Find initial guess for the angles
        # Step 2) Refine the solution iteratively as to converge to its coordinates on the original omnidirectional image
        # Minimization of radial distance due to projection
        from scipy.optimize import least_squares
        if verbose_optimization:
            verbose_opt_type = 1
        else:
            verbose_opt_type = 0
        f_scale_proj_angles_opt = 1  # CHECKME: not sure about the scale ralted to the residuals in pixels
        # The target projected points are the argument to the point_fwd_proj_error function
        m_omni_reshaped = m_omni.reshape(1, -1, m_omni.shape[-1])
        # Only do the optimization-based projection if radial distortion exists in the model
        if isinstance(self.precalib_params, PinholeCamera):
            has_distortion = False
        else:
            has_distortion = self.precalib_params.k1 != 0. or self.precalib_params.k2 != 0 or self.precalib_params.k3 != 0.
        if individual_opt:
            azimuths_list = []
            elevations_list = []
            min_bounds_for_angles = [-2.* np.pi, -np.pi]
            max_bounds_for_angles = [2.* np.pi, np.pi]
            bounds_for_angles = (min_bounds_for_angles, max_bounds_for_angles)
            for m in m_omni_reshaped[0]:
                azimuth_init, elevation_init = self.get_direction_angles_from_pixel(m)
                if has_distortion:
                    initial_angles = list(azimuth_init.ravel()) + list(elevation_init.ravel())
                    func_args = (m[np.newaxis, ...])
                    undistort_params_result = least_squares(fun = self.point_fwd_proj_error, x0 = initial_angles, args = func_args, f_scale = f_scale_proj_angles_opt, method = "trf", loss = "soft_l1", verbose = verbose_opt_type, max_nfev = 1000, bounds = bounds_for_angles)
                    azimuths_list.append(undistort_params_result.x[0])
                    elevations_list.append(undistort_params_result.x[1])
                else:
                    if np.ndim(azimuth_init) > 1:
                        azimuths_list.append(azimuth_init[0, 0])
                    else:
                        azimuths_list.append(azimuth_init)
                    if np.ndim(azimuth_init) > 1:
                        elevations_list.append(elevation_init[0, 0])
                    else:
                        elevations_list.append(elevation_init)

            azimuths = np.array(azimuths_list)
            elevations = np.array(elevations_list)
        else:
            azimuth_init, elevation_init = self.get_direction_angles_from_pixel(m_omni)
            if has_distortion:
                initial_angles = list(azimuth_init.ravel()) + list(elevation_init.ravel())
                num_pairs = int(len(initial_angles) / 2)  # NOTE: assuming x is always a set of even elements due to having elevations and azimuths as params to evaluate for
                min_bounds_for_angles = num_pairs * [-2.* np.pi] + num_pairs * [-np.pi]
                max_bounds_for_angles = num_pairs * [2.* np.pi] + num_pairs * [np.pi]
                bounds_for_angles = (min_bounds_for_angles, max_bounds_for_angles)
                func_args = (m_omni_reshaped)
                undistort_params_result = least_squares(fun = self.point_fwd_proj_error, x0 = initial_angles, args = func_args, f_scale = f_scale_proj_angles_opt, method = "trf", loss = "soft_l1", verbose = verbose_opt_type, max_nfev = 1000, bounds = bounds_for_angles)
                # The order is given as first half: azimuth angles, and the other half for elevation angles
                azimuths = np.array(undistort_params_result.x[:num_pairs])
                elevations = np.array(undistort_params_result.x[num_pairs:])
            else:
                azimuths = azimuth_init.ravel()
                elevations = elevation_init.ravel()

        return azimuths, elevations

    def point_fwd_proj_error(self, x, target_points_coords):
        num_pairs = int(len(x) / 2)  # NOTE: assuming x is always a set of even elements due to having elevations and azimuths as params to evaluate for
        azimuth = x[0:num_pairs]
        elevation = x[num_pairs:]
        u_omni, v_omni, m_omni = self.get_pixel_from_direction_angles(azimuth, elevation)

        # Compute cost using The total cost is the L2 norms between the undistorted control points obtained during forward projection and the undistorted parameters
        m_diffs = m_omni[..., :2] - target_points_coords[..., :2]
        # For Least-Squares optimization:
        residuals_as_matrix = np.linalg.norm(m_diffs, axis = -1)
        residuals = residuals_as_matrix.ravel()
        return residuals

        #=======================================================================
        # total_cost = np.sum(m_diffs ** 2)
        # return total_cost
        #=======================================================================

    def get_all_direction_angles_per_pixel_radially(self):
        px_all_u = np.arange(self.inner_img_radius, self.outer_img_radius + 1) + self.precalib_params.u_center
        px_all_v = np.zeros_like(px_all_u) + self.precalib_params.v_center
        px_all = np.dstack((px_all_u, px_all_v))
        azimuth_all, elev_all = self.get_direction_angles_from_pixel(px_all)
        return azimuth_all, elev_all

    def distortion(self, mx_u, my_u):
        '''
        @brief Apply distortion to input point (from the normalised plane)

        @param mx_u: undistorted x coordinate of point on the normalised projection plane
        @param my_u: undistorted y coordinate of point on the normalised projection plane

        @retval dx_u: distortion value that was added to the undistorted point \f$mx_u\f$ such that the distorted point is produced \f$ mx_d = mx_u+dx_u \f$
        @retval dy_u: distortion value that was added to the undistorted point \f$my_u\f$ such that the distorted point is produced \f$ my_d = my_u+dy_u \f$
        '''
        raise NotImplementedError

    def get_center(self):
        '''
        @return: a tuple (u,v) of the center coordinates
        '''
        u = self.precalib_params.u_center
        v = self.precalib_params.v_center
        return u, v

    def _compute_boundary_elevations(self):
        '''
        Private method to compute the highest and lowest elevation angles related to occlusion boundaries
        '''
        # Approximating the pixel resolution of azimuths to be 1 degree
        # NOTE: These phi angles on the image advance on a counterclockwise direction around the center.
        # However, their order is reversed ("clockwise") around the z-axis of the model in the world.
        # This doesn't really matter because we are lifting the pixel to its corresponding 3D point, and then get chosen.
        phi_on_img_array = np.linspace(0, 2 * np.pi, num = 360, endpoint = False)

        if self.center_lowest_elevation is None:
            u_low = self.precalib_params.u_center + self.r_lowest_elevation * np.cos(phi_on_img_array)
            v_low = self.precalib_params.v_center + self.r_lowest_elevation * np.sin(phi_on_img_array)
        else:
            u_low = self.center_lowest_elevation[0] + self.r_lowest_elevation * np.cos(phi_on_img_array)
            v_low = self.center_lowest_elevation[1] + self.r_lowest_elevation * np.sin(phi_on_img_array)

#         self.low_img_points = np.array([u_low, v_low]).transpose()
        self.low_img_points = np.dstack((u_low, v_low))

        # Older way computed the extreme elevation angle based on the lifted 3D points' Z coordinates
        #=======================================================================
        # self.low_3D_points = self.lift_pixel_to_unit_sphere_wrt_focus(self.low_img_points, visualize=False, debug=False)
        # # Find the 3D point with the minimum Z value
        # self.lowest_lifted_point_index = np.argmin(self.low_3D_points[..., 2])
        # self.lowest_lifted_point_on_sphere_wrt_F = self.low_3D_points[:, self.lowest_lifted_point_index]
        # self.lowest_img_point = self.low_img_points[:, self.lowest_lifted_point_index]
        # # _, lowest_elev = self.get_direction_angles_from_pixel(self.lowest_img_point[np.newaxis, :])
        # _, lowest_elev = self.get_direction_angles_from_pixel_using_forward_projection(m_omni=self.lowest_img_point[np.newaxis, :])
        #=======================================================================
        # New way now takes the extreme elevation after finding the elevations using forward projection (it may takes longer due to performing N optimizations!):
        low_azimuths, low_elevations = self.get_direction_angles_from_pixel_using_forward_projection(m_omni = self.low_img_points, individual_opt = True)
        self.lowest_lifted_point_index = np.argmin(low_elevations)
        self.low_3D_points = self.get_3D_point_from_angles_wrt_focus(azimuth = low_azimuths, elevation = low_elevations)
        self.lowest_lifted_point_on_sphere_wrt_F = self.low_3D_points[:, self.lowest_lifted_point_index]
        self.lowest_img_point = self.low_img_points[:, self.lowest_lifted_point_index]
        lowest_elev = low_elevations[self.lowest_lifted_point_index]

        self.lowest_elevation_angle = float(lowest_elev)
        print("LOWEST pixel %s for point %s with elevation: %f degrees" % (self.lowest_img_point, self.lowest_lifted_point_on_sphere_wrt_F, np.rad2deg(self.lowest_elevation_angle)))

        if self.center_highest_elevation is None:
            u_high = self.precalib_params.u_center + self.r_highest_elevation * np.cos(phi_on_img_array)
            v_high = self.precalib_params.v_center + self.r_highest_elevation * np.sin(phi_on_img_array)
        else:
            u_high = self.center_highest_elevation[0] + self.r_highest_elevation * np.cos(phi_on_img_array)
            v_high = self.center_highest_elevation[1] + self.r_highest_elevation * np.sin(phi_on_img_array)

        self.high_img_points = np.dstack((u_high, v_high))

        # Older way computed the extreme elevation angle based on the lifted 3D points' Z coordinates
        #=======================================================================
        # self.high_3D_points = self.lift_pixel_to_unit_sphere_wrt_focus(self.high_img_points, debug=False)
        # # Find the 3D point with the maximum Z value
        # self.highest_lifted_point_index = np.argmax(self.high_3D_points[..., 2])
        # self.highest_lifted_point_on_sphere_wrt_F = self.high_3D_points[:, self.highest_lifted_point_index]
        # self.highest_img_point = self.high_img_points[:, self.highest_lifted_point_index]
        # _, highest_elev = self.get_direction_angles_from_pixel_using_forward_projection(m_omni=self.highest_img_point[np.newaxis, :])
        #=======================================================================
        # New way now takes the extreme elevation after finding the elevations using forward projection (it may takes longer due to performing N optimizations!):
        high_azimuths, high_elevations = self.get_direction_angles_from_pixel_using_forward_projection(m_omni = self.high_img_points, individual_opt = True)
        self.highest_lifted_point_index = np.argmax(high_elevations)
        self.high_3D_points = self.get_3D_point_from_angles_wrt_focus(azimuth = high_azimuths, elevation = high_elevations)
        self.highest_lifted_point_on_sphere_wrt_F = self.high_3D_points[:, self.highest_lifted_point_index]
        self.highest_img_point = self.high_img_points[:, self.highest_lifted_point_index]
        highest_elev = high_elevations[self.highest_lifted_point_index]

        self.highest_elevation_angle = float(highest_elev)
        print("HIGHEST pixel %s for point %s with elevation: %f degrees" % (self.highest_img_point, self.highest_lifted_point_on_sphere_wrt_F, np.rad2deg(self.highest_elevation_angle)))
        print(80 * "-")
        self.vFOV = self.get_vFOV()

    def set_radial_limits(self, r_min, r_max):
        '''
        Uses the bounding radii to compute elevation angles according to the mirror's position

        @param r_min: physical radius of the small (inner) bounding circle on the image (the radius of the reflex planar mirror or the camera hole)
        @param r_max: physical radius of the large (outer) bounding circle on the image (Usually the system radius)
        '''
        self.r_min = r_min
        self.r_max = r_max

    def set_radial_limits_in_pixels(self, outer_img_radius, inner_img_radius, **kwargs):
        self.set_radial_limits_in_pixels_mono(inner_img_radius = inner_img_radius, outer_img_radius = outer_img_radius, **kwargs)
        # Set the highest/lowest elevation globally among single view
        global_high_elev = self.highest_elevation_angle
        global_low_elev = self.lowest_elevation_angle
        self.globally_highest_elevation_angle = global_high_elev
        self.globally_lowest_elevation_angle = global_low_elev

    def set_radial_limits_in_pixels_mono(self, inner_img_radius, outer_img_radius, center_point = None, preview = False, **kwargs):
        '''
        Determines which image radial boundary corresponds to the highest and lowest elevantion angles.
        '''
        # WISHME: center points for inner and outer should be passed and the corresponding radial meausements have to be taken accordingly
        # NOTE: This is ok for now when using the theoretical model since the inner and outer centers are the same point!
        if center_point is not None:
            self.precalib_params.center_point = center_point
            self.precalib_params.u_center = center_point[0]
            self.precalib_params.v_center = center_point[1]

        self.inner_img_radius = inner_img_radius
        self.outer_img_radius = outer_img_radius
        self.h_radial_image = self.outer_img_radius - self.inner_img_radius
        # Adjust max radial distance based on the difference between the radial bounds (given their centers used for the mask)
        if not isinstance(self.precalib_params, PinholeCamera):
            if self.precalib_params.center_point_inner is not None and self.precalib_params.center_point_outer is not None:
                offset_between_mask_centers = np.linalg.norm(self.precalib_params.center_point_inner - self.precalib_params.center_point_outer)
                self.h_radial_image += offset_between_mask_centers

        if self.mirror_name == "bottom" or self.mirror_number == 2:
            # Objects in image appear inverted (upside-down)
            # WISHME: must be adjusted with the offset between the center_point and the center_point_outer (or am I just complicating things)
            # NOTE: This is ok for the theoretical model since the center point for the inner and outer is the same
            self.r_lowest_elevation = outer_img_radius
            self.r_highest_elevation = inner_img_radius

            if isinstance(self.precalib_params, PinholeCamera) or self.precalib_params.center_point_outer is None:
                self.center_lowest_elevation = self.precalib_params.center_point
            else:
                self.center_lowest_elevation = self.precalib_params.center_point_outer

            if isinstance(self.precalib_params, PinholeCamera) or self.precalib_params.center_point_inner is None:
                self.center_highest_elevation = self.precalib_params.center_point
            else:
                self.center_highest_elevation = self.precalib_params.center_point_inner
        else:  # top
            self.r_lowest_elevation = inner_img_radius
            self.r_highest_elevation = outer_img_radius

            if isinstance(self.precalib_params, PinholeCamera) or self.precalib_params.center_point_outer is None:
                self.center_highest_elevation = self.precalib_params.center_point
            else:
                self.center_highest_elevation = self.precalib_params.center_point_outer

            if isinstance(self.precalib_params, PinholeCamera) or self.precalib_params.center_point_inner is None:
                self.center_lowest_elevation = self.precalib_params.center_point
            else:
                self.center_lowest_elevation = self.precalib_params.center_point_inner

        # Needs to set GUM variables so the Cp is set accordingly with new params if any.
        self.set_model_params(**kwargs)

        self._compute_boundary_elevations()
        self._set_camera_FOVs()

        if preview and self.current_omni_img != None:
            vis_img = self.current_omni_img.copy()
            if self.mirror_number == 1:
                circle_color = (0, 255, 255)  # yellow in BGR
            else:
                circle_color = (0, 255, 0)  # green in BGR

            if isinstance(self.precalib_params, PinholeCamera) or self.precalib_params.center_point_inner is None:
                inner_center_as_int = (int(self.precalib_params.u_center), int(self.precalib_params.v_center))
            else:
                inner_center_as_int = (int(self.precalib_params.u_center_inner), int(self.precalib_params.v_center_inner))
            cv2.circle(img = vis_img, center = inner_center_as_int, radius = inner_img_radius, color = circle_color, thickness = 1, lineType = cv2.LINE_AA)

            if isinstance(self.precalib_params, PinholeCamera) or self.precalib_params.center_point_outer is None:
                outer_center_as_int = (int(self.precalib_params.u_center), int(self.precalib_params.v_center))
            else:
                outer_center_as_int = (int(self.precalib_params.u_center_outer), int(self.precalib_params.v_center_outer))
            cv2.circle(img = vis_img, center = outer_center_as_int, radius = outer_img_radius, color = circle_color, thickness = 1, lineType = cv2.LINE_AA)

            radial_limits_win = 'radial limits %s mirror' % self.mirror_name
            cv2.imshow(radial_limits_win, vis_img)
            cv2.waitKey(1)

    def set_omni_image(self, img, pano_width_in_pixels = 1200, generate_panorama = False, idx = -1, view = True, apply_mask = False, mask_RGB = None):
        self.current_omni_img = img
        if hasattr(self, "theoretical_model"):
            if self.theoretical_model is not None:
                self.theoretical_model.set_omni_image(img = img, pano_width_in_pixels = pano_width_in_pixels, generate_panorama = generate_panorama, idx = idx, view = view, apply_mask = apply_mask, mask_RGB = mask_RGB)

        if apply_mask:
            img = self.get_masked_image(omni_img = img, view = view, color_RGB = mask_RGB)

        if generate_panorama:
            import omnistereo.panorama as panorama
            self.panorama = panorama.Panorama(self, width = pano_width_in_pixels)
            self.panorama.set_panoramic_image(img, idx, view)
            try:
                mirror_name = self.mirror_name
            except:
                mirror_name = ""

            print(mirror_name.upper() + " Panorama's Settings:")
            self.panorama.print_settings()
        else:
            if self.panorama is not None:
                self.panorama.set_panoramic_image(img, idx, view)

    def get_masked_image(self, omni_img = None, view = True, color_RGB = None):
        '''
        @param color_RGB: A tuple specifying the desired background as (Red,Green,Blue). If None, the background is black

        @return: masked_img
        '''
        from omnistereo.common_tools import convert_to_tuple

        if omni_img is None:
            omni_img = self.current_omni_img

        # Small performance improvement by only generating masks if needed. Now the masks are stored for reusability!
        if self.construct_new_mask or self.mask is None:
            self.mask = self.make_mask(mask_shape = omni_img.shape[0:2], radius_pixel_shrinking = 0)  # Save mask as property

        # Apply mask
        masked_img = np.zeros(omni_img.shape)
        masked_img = cv2.bitwise_and(src1 = omni_img, src2 = omni_img, dst = masked_img, mask = self.mask)

        if color_RGB is not None:  # Paint the masked area other than black
            if color_RGB != self.mask_RGB_color:
                self.mask_RGB_color = color_RGB
                color_BGR = (color_RGB[2], color_RGB[1], color_RGB[0])
                self.mask_background_img = np.zeros_like(omni_img)
                self.mask_background_img[:, :, :] += np.array(color_BGR, dtype = "uint8")  # Paint the B-G-R channels for OpenCV

            mask_inv = cv2.bitwise_not(src = self.mask)
            # Apply the background using the inverted mask
            masked_img = cv2.bitwise_and(src1 = self.mask_background_img, src2 = self.mask_background_img, dst = masked_img, mask = mask_inv)

        self.construct_new_mask = False  # Clear mask construction flag

        # Show radial boundaries
        if view:
            win_name = "Masked with Radial Bounds"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, masked_img)
            cv2.waitKey(1)

        return masked_img

    def make_mask(self, mask_shape, radius_pixel_shrinking = 0):
        '''
        @param color_RGB: A tuple specifying the desired background as (Red,Green,Blue). If None, the background is black

        @return: masked_img
        '''
        from omnistereo.common_tools import convert_to_tuple

        # Small performance improvement by only generating masks if needed. Now the masks are stored for reusability!
            # circle centers
        center_point_outer = convert_to_tuple(self.precalib_params.center_point_outer.astype("int"))
        center_point_inner = convert_to_tuple(self.precalib_params.center_point_inner.astype("int"))
        # circle radii
        r_inner_top = self.inner_img_radius + radius_pixel_shrinking
        r_outer_top = self.outer_img_radius - radius_pixel_shrinking

        mask_omni = np.zeros(mask_shape, dtype = np.uint8)  # Black, single channel mask
        # Paint outer perimeter:
        cv2.circle(mask_omni, center_point_outer, int(r_outer_top), (255, 255, 255), -1, 8, 0)
        # Paint inner bound
        if r_inner_top > 0:
            cv2.circle(mask_omni, center_point_inner, int(r_inner_top), (0, 0, 0), -1, 8, 0)

        return mask_omni

    def _set_camera_FOVs(self):
        r_min_img = self.inner_img_radius
        r_max_img = self.outer_img_radius
        # Useless so far:
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        self.precalib_params.roi_max_x = self.precalib_params.u_center + r_max_img
        self.precalib_params.roi_max_y = self.precalib_params.v_center + r_max_img
        self.precalib_params.roi_min_x = self.precalib_params.u_center - r_max_img
        self.precalib_params.roi_min_y = self.precalib_params.v_center - r_max_img
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        try:
            h_x, h_y = self.precalib_params.pixel_size
            f = self.precalib_params.focal_length
            self.precalib_params.max_useful_FOV_hor = 2 * np.arctan((h_x * r_max_img) / f)
            self.precalib_params.min_useful_FOV_hor = 2 * np.arctan((h_x * r_min_img) / f)
            self.precalib_params.max_useful_FOV_ver = 2 * np.arctan((h_y * r_max_img) / f)
            self.precalib_params.min_useful_FOV_ver = 2 * np.arctan((h_y * r_min_img) / f)
        except:
            print("Cannot compute FOVs")

    def get_vFOV(self, **kwargs):
        '''
        Computes the vertical Field of View based on the physical radial limits
        @return: the vertical field of view computed from their elevation angle limits
        '''
        theta_max = kwargs.get("theta_max", self.highest_elevation_angle)
        theta_min = kwargs.get("theta_min", self.lowest_elevation_angle)

        self.vFOV = theta_max - theta_min
        return self.vFOV

    def get_a_hyperbola(self, c, k):
        a = c / 2 * np.sqrt((k - 2.0) / k)
        return a

    def get_b_hyperbola(self, c, k):
        b = c / 2 * np.sqrt(2.0 / k)
        return b

    def detect_sparse_features_on_panorama(self, feature_detection_method = "ORB", num_of_features = 50, median_win_size = 0, show = True):
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

        start_creation_time = process_time()
        # WISHME: Cut some TIMING by making the detector and descriptor part of the Class!
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
#             descriptor = cv2.xfeatures2d.LATCH_create(rotationInvariance=False)  # we are setting this rotationInvariance parameter to False because GFT doesn't provide the patch orientation
            descriptor = cv2.ORB_create(nfeatures = num_of_features)
#             descriptor = cv2.BRISK_create()
#             descriptor = cv2.xfeatures2d.FREAK_create()
#             descriptor = cv2.xfeatures2d.LUCID_create()

        end_creation_time = process_time()
        creation_time = end_creation_time - start_creation_time

        # TODO: implement other detectors, such as:
# >>> (kps, descs) = sift.detectAndCompute(gray, None)
# >>> print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
# # kps: 274, descriptors: (274, 128)
# >>> surf = cv2.xfeatures2d.SURF_create()
# >>> (kps, descs) = surf.detectAndCompute(gray, None)
#
# >>> kaze = cv2.KAZE_create()
# >>> (kps, descs) = kaze.detectAndCompute(gray, None)
# >>> print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
# # kps: 359, descriptors: (359, 64)
# >>> akaze = cv2.AKAZE_create()
# >>> (kps, descs) = akaze.detectAndCompute(gray, None)
# >>> print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
# # kps: 192, descriptors: (192, 61)
# >>> brisk = cv2.BRISK_create()
# >>> (kps, descs) = brisk.detectAndCompute(gray, None)

        pano_img = self.panorama.panoramic_img.copy()
        if self.panorama.panoramic_img is not None:
            if median_win_size > 0:
                pano_img = cv2.medianBlur(pano_img, median_win_size)
            if feature_detection_method.upper() == "GFT":  # or feature_detection_method.upper() == "FAST":
                if pano_img.ndim == 3:  # GFT needs a grayscale image
                    pano_img = cv2.cvtColor(pano_img, cv2.COLOR_BGR2GRAY)
                else:
                    pano_img = self.panorama.panoramic_img.copy()

        #=======================================================================

        keypts_detected_list = []
        descriptors_list = []
        if len(self.panorama.azimuthal_masks) > 0:
            # When spliting panorama in masks for spreading info around the 360 degree view.
            if show:
                pano_with_keypts = pano_img.copy()
            #===================================================================
            # detection_time = 0.
            # description_time = 0.
            #===================================================================
            for m in self.panorama.azimuthal_masks:
                #=======================================================================
                #===============================================================
                # start_detection_time = process_time()
                #===============================================================
                if feature_detection_method.upper() == "GFT":
                    #===========================================================
                    # start_GFT_time = process_time()
                    #===========================================================
                    pts_GFT = cv2.goodFeaturesToTrack(image = pano_img, maxCorners = num_of_features, qualityLevel = 0.01, minDistance = 5, mask = m, useHarrisDetector = useHarrisDetector)
                    #===========================================================
                    # end_GFT_time = process_time()
                    # current_GFT_time = end_GFT_time - start_GFT_time
                    #===========================================================
                    #===========================================================
                    # pts = pts_GFT.reshape((pts_GFT.shape[0], pts_GFT.shape[2]))
                    # if show:
                    #     from omnistereo.common_cv import draw_points
                    #     draw_points(pano_with_keypts, pts)
                    #     cv2.imshow("Detected Keypoints - Panorama of Mirror %d" % (self.mirror_number), pano_with_keypts)
                    #     cv2.waitKey(1)
                    #===========================================================
                    keypts_detected_on_mask = cv2.KeyPoint_convert(pts_GFT)
                else:
    #                 keypts_detected_on_mask, descriptors_on_mask = detector.detectAndCompute(image=pano_img, mask=m)
                    keypts_detected_on_mask = detector.detect(image = pano_img, mask = m)
                #===============================================================
                # end_detection_time = process_time()
                # current_detection_time = end_detection_time - start_detection_time
                # detection_time = detection_time + current_detection_time
                #===============================================================

                #===============================================================
                # start_description_time = process_time()
                #===============================================================
                keypts_detected_on_mask_described, descriptors_on_mask = descriptor.compute(image = pano_img, keypoints = keypts_detected_on_mask)
                keypts_detected_list.append(keypts_detected_on_mask_described)
                descriptors_list.append(descriptors_on_mask)
                #===============================================================
                # end_description_time = process_time()
                # description_time = description_time + (end_description_time - start_description_time)
                #===============================================================
                if show:
                    pano_with_keypts = cv2.drawKeypoints(pano_with_keypts, keypts_detected_on_mask_described, outImage = pano_with_keypts)
                    cv2.imshow("Detected Keypoints - Panorama of Mirror %d" % (self.mirror_number), pano_with_keypts)
                    cv2.waitKey(1)
        else:
            if feature_detection_method.upper() == "GFT":
                pts_GFT = cv2.goodFeaturesToTrack(image = pano_img, maxCorners = num_of_features, qualityLevel = 0.01, minDistance = 5, mask = None, useHarrisDetector = useHarrisDetector)
                keypts_detected = cv2.KeyPoint_convert(pts_GFT)
            else:
                # keypts_detected_on_mask, descriptors_on_mask = detector.detectAndCompute(image=pano_img, mask=m)
                # OR
                keypts_detected = detector.detect(image = pano_img, mask = None)

            keypts_detected, descriptors = descriptor.compute(image = pano_img, keypoints = keypts_detected)
            keypts_detected_list.append(keypts_detected)
            descriptors_list.append(descriptors)
            # kpts_top are put in a list of length n
            # descriptor are ndarrays of n x desc_size

            if show:
                pano_with_keypts = pano_img.copy()
                pano_with_keypts = cv2.drawKeypoints(pano_img, keypts_detected, outImage = pano_with_keypts)
                cv2.imshow("Detected Keypoints - Panorama of Mirror %d" % (self.mirror_number), pano_with_keypts)
                cv2.waitKey(1)

        return keypts_detected_list, descriptors_list

#     def detect_sparse_features(self, feature_detection_method = "ORB", num_of_features = 50, mask = None, median_win_size = 0, show = True):
#         '''
#         @param median_win_size: Window size for median blur filter. If value is 0, there is no filtering.
#         @return: List of list of kpts_top and ndarray of descriptors. The reason we use a list of each result is for separating mask-wise results for matchin.
#         '''
#         # Possible Feature Detectors are:
#         # SIFT, SURF, DAISY
#         # ORB:  oriented BRIEF
#         # BRISK
#         # AKAZE
#         # KAZE
#         # AGAST: Uses non-maximal suppression (NMS) by default
#         # FAST: Uses non-maximal suppression (NMS) by default
#         # GFT: Good Features to Track
#         # MSER: http://docs.opencv.org/master/d3/d28/classcv_1_1MSER.html#gsc.tab=0
#
#         # Possible Feature Descriptors are:
#         # SIFT, SURF: work to describe FAST and AGAST kpts_top
#         # ORB:  oriented BRIEF
#         # BRISK
#         # AKAZE
#         # KAZE
#         # DAISY: works to describe FAST and AGAST kpts_top
#         # FREAK: is essentially a descriptor.
#
#         if feature_detection_method.upper() == "ORB":
#             detector = cv2.ORB_create(nfeatures = num_of_features)
#             descriptor = detector
# #             descriptor = cv2.xfeatures2d.FREAK_create()
#             # detector.setPatchSize(10)  # TODO: set according to mask's width
#         if feature_detection_method.upper() == "AKAZE":
#             detector = cv2.AKAZE_create()
#             descriptor = detector
#         if feature_detection_method.upper() == "KAZE":
#             detector = cv2.KAZE_create()
#             descriptor = detector
#         if feature_detection_method.upper() == "BRISK":
#             detector = cv2.BRISK_create()
#             descriptor = detector
#         if feature_detection_method.upper() == "SIFT":
#             detector = cv2.xfeatures2d.SIFT_create()
#             descriptor = detector
#         if feature_detection_method.upper() == "SURF":
#             detector = cv2.xfeatures2d.SURF_create()
#             descriptor = detector
#         if feature_detection_method.upper() == "FAST":
#             detector = cv2.FastFeatureDetector_create()  # Uses non-maximal suppression (NMS) by default
#             detector.setNonmaxSuppression(True)
#             descriptor = cv2.xfeatures2d.DAISY_create()
# #             descriptor = cv2.xfeatures2d.FREAK_create()
# #             descriptor = cv2.xfeatures2d.SIFT_create() # FIXME: ORB and BRISK don't seem to work
#         if feature_detection_method.upper() == "STAR":
#             detector = cv2.xfeatures2d.StarDetector_create()
#             descriptor = cv2.xfeatures2d.DAISY_create()  # TODO: not sure which descriptor works best
#         if feature_detection_method.upper() == "AGAST":
#             detector = cv2.AgastFeatureDetector_create()  # Uses non-maximal suppression (NMS) by default
#             detector.setNonmaxSuppression(True)
#             descriptor = cv2.BRISK_create()
# #             descriptor = cv2.xfeatures2d.DAISY_create()
# #             descriptor = cv2.xfeatures2d.FREAK_create()
# #             descriptor = cv2.xfeatures2d.SIFT_create()
#         if feature_detection_method.upper() == "GFT":
#             useHarrisDetector = False  # I prefer not to use Harris Corners because not many matches occur on them
# #             descriptor = cv2.xfeatures2d.DAISY_create()
# #             descriptor = cv2.xfeatures2d.LATCH_create(rotationInvariance = False)  # we are setting this rotationInvariance parameter to False because GFT doesn't provide the patch orientation
#             descriptor = cv2.ORB_create(nfeatures = num_of_features)
# #             descriptor = cv2.BRISK_create()
# #             descriptor = cv2.xfeatures2d.FREAK_create()
# #             descriptor = cv2.xfeatures2d.LUCID_create()
#
#         omni_img = self.current_omni_img.copy()
#         if omni_img is not None:
#             if median_win_size > 0:
#                 omni_img = cv2.medianBlur(omni_img, median_win_size)
#             if feature_detection_method.upper() == "GFT":
#                 if omni_img.ndim == 3:
#                     omni_img_detect_with_GFT = cv2.cvtColor(omni_img, cv2.COLOR_BGR2GRAY)  # GFT needs a grayscale image
#                 else:
#                     omni_img_detect_with_GFT = self.current_omni_img.copy()
#
#         keypts_detected_list = []
#         descriptors_list = []
#         if mask is None:
#             mask = self.mask
#
#         start_detection_time = process_time()
#         if feature_detection_method.upper() == "GFT":
#             pts_GFT = cv2.goodFeaturesToTrack(image = omni_img_detect_with_GFT, maxCorners = num_of_features, qualityLevel = 0.01, minDistance = 5, mask = mask, useHarrisDetector = useHarrisDetector)
#             keypts_detected_on_mask = cv2.KeyPoint_convert(pts_GFT)
#         else:
#             keypts_detected_on_mask = detector.detect(image = omni_img, mask = mask)
#         end_detection_time = process_time()
#         detection_time = end_detection_time - start_detection_time
#
#         keypts_detected, descriptors = descriptor.compute(image = omni_img, keypoints = keypts_detected_on_mask)
#
#         keypts_detected_list.append(keypts_detected)
#         descriptors_list.append(descriptors)
#         # kpts_top are put in a list of length n
#         # descriptor are ndarrays of n x desc_size
#
#         if show:
#             omni_with_keypts = omni_img.copy()
#             omni_with_keypts = cv2.drawKeypoints(omni_img, keypts_detected, outImage = omni_with_keypts)
#             detected_kpts_win_name = "Detected Keypoints - Omnidirectional Image of Mirror %d" % (self.mirror_number)
#             cv2.namedWindow(detected_kpts_win_name, cv2.WINDOW_NORMAL)
#             cv2.imshow(detected_kpts_win_name, omni_with_keypts)
#             cv2.waitKey(1)
#
#         return keypts_detected_list, descriptors_list

    def approximate_with_PnP(self, img_points, obj_pts_homo):
        '''
        Approximate the transformation (pose) between a set of points on the image plane and their corresponding points on another plane
        '''

        raise NotImplementedError

    def draw_radial_bounds(self, omni_img = None, is_reference = False, view = True):
        '''
        @param is_reference: When set to True, it will use dotted lines and cross hairs for the center
        '''
        from omnistereo.common_tools import convert_to_tuple

        if omni_img is None:
            omni_img = self.current_omni_img

        center_point = self.precalib_params.center_point
        r_min = self.inner_img_radius
        r_max = self.outer_img_radius

        if hasattr(self, "mirror_name"):
            mirror_name = self.mirror_name.upper()
        else:
            if self.mirror_number == 1:
                mirror_name = "TOP"
            else:
                mirror_name = "BOTTOM"

        if mirror_name == "TOP":
            color = (255, 255, 0)  # GBR for cyan (top)
            delta_theta = 4  # degrees
        else:
            color = (255, 0, 255)  # GBR for magenta (bottom)
            delta_theta = 6  # degrees

        img = omni_img.copy()
        # Draw:
        # circle center
        u_c, v_c = int(center_point[0]), int(center_point[1])
        center = (u_c, v_c)
        if is_reference:
            dot_radius = 5
            cv2.circle(img, center, dot_radius, color, -1, cv2.LINE_AA, 0)

            # circle outline dotted
            dot_radius_bounds = 3
            # Because dots of the smaller radius circular bound should be more sparse
            # WISH: Factorize this drawing of dotted circle as a general function for common_cv
            delta_theta_min = np.deg2rad(delta_theta)  # written in degrees
            delta_theta_max = np.deg2rad(delta_theta / 2)  # written in degrees
            thetas_min = np.arange(start = 0, stop = 2 * np.pi, step = delta_theta_min)
            thetas_max = np.arange(start = 0, stop = 2 * np.pi, step = delta_theta_max)
            if self.precalib_params.center_point_outer is not None:
                u_c, v_c = convert_to_tuple(self.precalib_params.center_point_outer)

            u_coords_max = u_c + r_max * np.cos(thetas_max)
            v_coords_max = v_c + r_max * np.sin(thetas_max)

            if self.precalib_params.center_point_inner is not None:
                u_c, v_c = convert_to_tuple(self.precalib_params.center_point_inner)
            u_coords_min = u_c + r_min * np.cos(thetas_min)
            v_coords_min = v_c + r_min * np.sin(thetas_min)

            for u_min, v_min in zip(u_coords_min, v_coords_min):
                cv2.circle(img, (int(u_min), int(v_min)), dot_radius_bounds, color, -1, cv2.LINE_AA, 0)
            for u_max, v_max in zip(u_coords_max, v_coords_max):
                cv2.circle(img, (int(u_max), int(v_max)), dot_radius_bounds, color, -1, cv2.LINE_AA, 0)
        else:
            # Use crosshair as center point
            thickness = 3
            cross_hair_length = 15
            cv2.line(img = img, pt1 = (u_c - cross_hair_length, v_c), pt2 = (u_c + cross_hair_length, v_c), color = color, thickness = thickness, lineType = cv2.LINE_AA)  # crosshair horizontal
            cv2.line(img = img, pt1 = (u_c, v_c - cross_hair_length), pt2 = (u_c, v_c + cross_hair_length), color = color, thickness = thickness, lineType = cv2.LINE_AA)  # crosshair vertical
            # circle outlines BUT located at the true center:
            #===================================================================
            # cv2.circle(img, center, int(r_min), color, thickness, cv2.LINE_AA, 0)
            # cv2.circle(img, center, int(r_max), color, thickness, cv2.LINE_AA, 0)
            #===================================================================
            # Instead, using mask circle centers
            if self.precalib_params.center_point_outer is None:
                cv2.circle(img, center, int(r_max), color, thickness, cv2.LINE_AA, 0)
            else:
                center_point_outer = convert_to_tuple(self.precalib_params.center_point_outer.astype("int"))
                cv2.circle(img, center_point_outer, int(r_max), color, thickness, cv2.LINE_AA, 0)

            if self.precalib_params.center_point_inner is None:
                cv2.circle(img, center, int(r_min), color, thickness, cv2.LINE_AA, 0)
            else:
                center_point_inner = convert_to_tuple(self.precalib_params.center_point_inner.astype("int"))
                cv2.circle(img, center_point_inner, int(r_min), color, thickness, cv2.LINE_AA, 0)

        # Show radial boundaries
        if view:
            win_name = mirror_name + " - Radial Bounds"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, img)
            cv2.waitKey(1)

        return img

    def generate_panorama(self, omni_img, width_in_pixels = 1200, idx = -1, view = True, win_name_modifier = "", use_mask = True, border_RGB_color = None):
        if self.panorama is None:
            self.set_omni_image(omni_img, pano_width_in_pixels = width_in_pixels, generate_panoramas = True, idx = idx, view = view)

        omni_img_masked = omni_img
        if use_mask:
            import warnings
            try:
                from omnistereo.common_cv import get_masked_images_mono
                omni_images_masked_list = get_masked_images_mono(unmasked_images = [omni_img], camera_model = self, show_images = True, color_RGB = border_RGB_color)
                omni_img_masked = omni_images_masked_list[0]
            except:
                warnings.warn("Panorama index %d problem in %s" % (idx, __name__))

        pano_img = self.panorama.set_panoramic_image(omni_img = omni_img_masked, idx = idx, view = view, win_name_modifier = win_name_modifier, border_RGB_color = border_RGB_color)
        return pano_img

    def draw_zero_degree_on_panorama(self, pano_img = None):
        '''
        NOTE: Image should not be cropped.
        '''
        if pano_img is None:
            pano_annotated = self.panorama.panoramic_img.copy()
        else:
            pano_annotated = pano_img.copy()

        pano_cols = int(self.panorama.cols)
        last_col = pano_cols - 1

        # Draw 0-degree line
        row_at_zero_degrees = self.panorama.get_panorama_row_from_elevation(0)
        if not np.isnan(row_at_zero_degrees):
            line_color = (255, 0, 255)  # magenta in BGR
            cv2.line(img = pano_annotated, pt1 = (0, row_at_zero_degrees), pt2 = (last_col, row_at_zero_degrees), color = line_color, thickness = 1 , lineType = cv2.LINE_AA)

        highest_angle_row = self.panorama.get_panorama_row_from_elevation(self.highest_elevation_angle)
        lowest_angle_row = self.panorama.get_panorama_row_from_elevation(self.lowest_elevation_angle)

        cropped_win = 'Annotated panorama'
        cv2.namedWindow(cropped_win, cv2.WINDOW_NORMAL)
        cv2.imshow(cropped_win, pano_annotated)

        pressed_key = cv2.waitKey(10)

        return pressed_key

    def draw_elevations_on_panorama(self, pano_img = None, draw_own_limits = False):
        '''
        NOTE: Image should not be cropped.
        '''
        from omnistereo.common_tools import convert_to_tuple

        if pano_img is None:
            pano_img_annotated = self.panorama.panoramic_img.copy()
        else:
            pano_img_annotated = pano_img.copy()

        pano_cols = int(self.panorama.cols)
        last_col = pano_cols - 1

        pano_win_name = 'Annotated panorama'
        cv2.namedWindow(pano_win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(pano_win_name, pano_img_annotated)

        # Draw 0-degree line
        row_at_zero_degrees = self.panorama.get_panorama_row_from_elevation(0)
        if not np.isnan(row_at_zero_degrees):
            line_color = (255, 0, 255)  # magenta in BGR
            cv2.line(img = pano_img_annotated, pt1 = (0, row_at_zero_degrees), pt2 = (last_col, row_at_zero_degrees), color = line_color, thickness = 1 , lineType = cv2.LINE_AA)

        if draw_own_limits:
            line_color = (0, 0, 255)  # red in BGR
            top_highest = self.panorama.get_panorama_row_from_elevation(self.highest_elevation_angle)
            top_lowest = self.panorama.get_panorama_row_from_elevation(self.lowest_elevation_angle)
            cv2.line(img = pano_img_annotated, pt1 = (0, top_highest), pt2 = (last_col, top_highest), color = line_color, thickness = 1 , lineType = cv2.LINE_AA)
            cv2.line(img = pano_img_annotated, pt1 = (0, top_lowest), pt2 = (last_col, top_lowest), color = line_color, thickness = 1 , lineType = cv2.LINE_AA)

            # Draw boundaries as selected by user:
            dot_radius = 1
            dot_color_high = (0, 0, 200)  # RGB for blue (highest)
            # All at once is should work because the problem is not this global optimization, but it's something else:
            validity, high_points_on_pano = self.panorama.get_panoramic_pixel_coords_from_omni_pixel(self.high_img_points, verbose_optimization = False)
            for pt_high in high_points_on_pano[0]:
                cv2.circle(pano_img_annotated, convert_to_tuple(pt_high[:2]), dot_radius, dot_color_high, -1, cv2.LINE_AA, 0)
            # Doing one point at a time (too slow!):
            #===================================================================
            # high_points_on_pano_list = []
            # for pt_high_omni in self.high_img_points[0]:
            #     # Only do one pixel at the time because it's an optimization of its own projection (not a Bundle adjustment)
            #     validity, pt_high = self.panorama.get_panoramic_pixel_coords_from_omni_pixel(pt_high_omni[np.newaxis, ...])
            #     cv2.circle(pano_img_annotated, convert_to_tuple(pt_high[0, 0, :2]), dot_radius, dot_color_high, -1, cv2.LINE_AA, 0)
            #     high_points_on_pano_list.append(pt_high[0, 0])
            # high_points_on_pano = np.array(high_points_on_pano_list)
            #===================================================================

            dot_color_low = (0, 255, 0)  # RGB for green (lowest)
            # All at once is should work because the problem is not this global optimization, but it's something else:
            validity, low_points_on_pano = self.panorama.get_panoramic_pixel_coords_from_omni_pixel(self.low_img_points)
            for pt_low in low_points_on_pano[0]:
                cv2.circle(pano_img_annotated, convert_to_tuple(pt_low[:2]), dot_radius, dot_color_low, -1, cv2.LINE_AA, 0)
            # Doing one point at a time (too slow!):
            #===================================================================
            # low_points_on_pano_list = []
            # for pt_low_omni in self.low_img_points[0]:
            #     validity, pt_low = self.panorama.get_panoramic_pixel_coords_from_omni_pixel(pt_low_omni[np.newaxis, ...])
            #     cv2.circle(pano_img_annotated, convert_to_tuple(pt_low[0, 0, :2]), dot_radius, dot_color_low, -1, cv2.LINE_AA, 0)
            #     low_points_on_pano_list.append(pt_low[0, 0])
            # low_points_on_pano = np.array(low_points_on_pano_list)
            #===================================================================

            # NOTE: We can only use the LUTs in order to reflect the same points mapped to the panorama. Otherwise, recall that forward projection is not a 1-to-1 mapping with the lifting process
            dot_radius = 2
            omni_img_annotated = self.current_omni_img.copy()
            u_omni_high, v_omni_high, high_points_on_omni_from_pano = self.panorama.get_omni_pixel_coords_from_panoramic_pixel(m_pano = high_points_on_pano, use_LUTs = True)
            for pt_high in high_points_on_omni_from_pano[0].astype("int"):
                cv2.circle(omni_img_annotated, convert_to_tuple(pt_high[:2]), dot_radius, dot_color_high, -1, cv2.LINE_AA, 0)
            u_omni_low, v_omni_low, low_points_on_omni_from_pano = self.panorama.get_omni_pixel_coords_from_panoramic_pixel(m_pano = low_points_on_pano, use_LUTs = True)
            for pt_low in low_points_on_omni_from_pano[0].astype("int"):
                cv2.circle(omni_img_annotated, convert_to_tuple(pt_low[:2]), dot_radius, dot_color_low, -1, cv2.LINE_AA, 0)

            # NOTE:Single point test: will always fail because the forward projection is not a 1-to-1 mapping with the lifting process
            #===================================================================
            # test_point_3D = self.lift_pixel_to_unit_sphere_wrt_focus(self.highest_img_point, debug=False)
            # _, _, test_point_reprojection = self.get_pixel_from_3D_point_wrt_M(test_point_3D, visualize=False)
            # reproj_test_success = np.allclose(self.highest_img_point[..., :2], test_point_reprojection[..., :2])
            #===================================================================

            # Also, draw the bounds on the omni image as if they come from the panorama
            omni_win_name = 'Annotated Omni Bounds from Panorama'
            cv2.namedWindow(omni_win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(omni_win_name, omni_img_annotated)

        cv2.imshow(pano_win_name, pano_img_annotated)

        pressed_key = cv2.waitKey(1)

        return pressed_key

    def view_all_panoramas(self, omni_images_filename_pattern, img_indices, win_name_modifier = "", use_mask = False, mask_color_RGB = None):
        import warnings
        from omnistereo.common_cv import get_images

        omni_images = get_images(omni_images_filename_pattern, indices_list = img_indices, show_images = not use_mask)

        if img_indices is None or len(img_indices) == 0:  # Default value
            # use all the images in the set:
            img_indices = range(len(omni_images))

        for i in img_indices:
            try:
                #===============================================================
                # from time import process_time  # , perf_counter
                # start_time = process_time()
                #===============================================================
                self.generate_panorama(omni_images[i], idx = i, view = True, win_name_modifier = win_name_modifier, use_mask = use_mask, border_RGB_color = mask_color_RGB)
                #===============================================================
                # end_time = process_time()
                # time_ellapsed_1 = end_time - start_time
                # print("Time elapsed: {time:.8f} seconds".format(time=time_ellapsed_1))
                #===============================================================
            except:
                warnings.warn("Image index %d not found at %s" % (i, __name__))

    def match_features_omni_frame_to_frame(self, feature_matcher, kpt_and_desc_train, kpt_and_desc_query, img_train = None, img_query = None, show_matches = False, draw_horizontally = False, radius_match_max_dist = 100):
        matches = feature_matcher.match(query_descriptors = kpt_and_desc_query.descriptors, train_descriptors = kpt_and_desc_train.descriptors, max_descriptor_distance_radius = radius_match_max_dist)

        matched_kpts_train = []
        matched_kpts_query = []
        matched_desc_train = []
        matched_desc_query = []
        random_colors = []
        if show_matches:
            from omnistereo.common_cv import rgb2bgr_color
            offset_dim = 0
            if draw_horizontally:
                offset_dim = 1
            if img_train is None:
                query_vis_pixel_offset = self.current_omni_img.shape[offset_dim]
            else:
                query_vis_pixel_offset = img_train.shape[offset_dim]

        if len(matches) > 0:
            if show_matches:
                if img_train is not None:
                    if img_train.ndim == 3:
                        img_gray_train = cv2.cvtColor(img_train, cv2.COLOR_BGR2GRAY)
                    else:
                        img_gray_train = img_train.copy()
                gray_vis_train = cv2.cvtColor(img_gray_train, cv2.COLOR_GRAY2BGR)

                if img_query is not None:
                    if img_query.ndim == 3:
                        img_gray_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)
                    else:
                        img_gray_query = img_query.copy()
                gray_vis_query = cv2.cvtColor(img_gray_query, cv2.COLOR_GRAY2BGR)

            num_of_good_matches = int(feature_matcher.percentage_good_matches * len(matches))

            if num_of_good_matches > 0:
                for m in matches[:num_of_good_matches]:
                    kpt_train = kpt_and_desc_train.kpts_top[m.trainIdx]
                    kpt_query = kpt_and_desc_query.kpts_top[m.queryIdx]

                    desc_train = kpt_and_desc_train.descriptors[m.trainIdx]
                    desc_query = kpt_and_desc_query.descriptors[m.queryIdx]
                    matched_kpts_train.append(kpt_train)
                    matched_kpts_query.append(kpt_query)
                    matched_desc_train.append(desc_train)
                    matched_desc_query.append(desc_query)
                    random_color_RGB = kpt_and_desc_query.random_colors_RGB[m.queryIdx]
                    random_colors.append(random_color_RGB)
                    if show_matches:
                        random_color_BGR = rgb2bgr_color(random_color_RGB)
                        gray_vis_train = cv2.drawKeypoints(gray_vis_train, [kpt_train], outImage = gray_vis_train, color = random_color_BGR)
                        gray_vis_query = cv2.drawKeypoints(gray_vis_query, [kpt_query], outImage = gray_vis_query, color = random_color_BGR)
                        if draw_horizontally:
                            matches_img = np.hstack((gray_vis_train, gray_vis_query))
                        else:
                            matches_img = np.vstack((gray_vis_train, gray_vis_query))
                        # Enable to draw matches one by one:
                        #=======================================================
                        # train_pt = (int(kpt_train.pt[0]), int(kpt_train.pt[1]))  # Recall, pt is given as (u,v)
                        # if draw_horizontally:
                        #     query_pt = (int(kpt_query.pt[0]) + query_vis_pixel_offset, int(kpt_query.pt[1]))
                        # else:
                        #     query_pt = (int(kpt_query.pt[0]), int(kpt_query.pt[1] + query_vis_pixel_offset))
                        # matches_img = cv2.line(matches_img, train_pt, query_pt, color=random_color_RGB, thickness=1, lineType=cv2.LINE_8)
                        # cv2.imshow('Matches', matches_img)
                        # cv2.waitKey(0)
                        #=======================================================
                # Update number of good matches based on disparity filtering above
                num_of_good_matches = len(matched_kpts_train)

                # Draw frame to frame in a single image
                if show_matches:
                    if draw_horizontally:
                        matches_img = np.hstack((gray_vis_train, gray_vis_query))
                    else:
                        matches_img = np.vstack((gray_vis_train, gray_vis_query))
                # Draw connecting lines for matches and compose pixel matrices
                matched_m_train = np.ones((1, num_of_good_matches, 3))
                matched_m_query = np.ones((1, num_of_good_matches, 3))
                idx = 0
                if show_matches:
                    win_name = 'Matches (from View # %d)' % (self.mirror_number)
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                # TODO: Only append those points far enough from the last point inserted.
                for train_kpt, query_kpt, random_RGB_color in zip(matched_kpts_train, matched_kpts_query, random_colors):
                    if show_matches:
                        train_pt = (int(train_kpt.pt[0]), int(train_kpt.pt[1]))  # Recall, pt is given as (u,v)
                        if draw_horizontally:
                            query_pt = (int(query_kpt.pt[0]) + query_vis_pixel_offset, int(query_kpt.pt[1]))
                        else:
                            query_pt = (int(query_kpt.pt[0]), int(query_kpt.pt[1]) + query_vis_pixel_offset)
                        matches_img = cv2.line(matches_img, train_pt, query_pt, color = rgb2bgr_color(random_RGB_color), thickness = 1, lineType = cv2.LINE_8)
                        cv2.imshow(win_name, matches_img)
                    # Getting the floating point coordinates instead of int, so we can use precision elevation without LUTs
                    matched_m_train[0, idx, 0] = train_kpt.pt[0]
                    matched_m_train[0, idx, 1] = train_kpt.pt[1]
                    matched_m_query[0, idx, 0] = query_kpt.pt[0]
                    matched_m_query[0, idx, 1] = query_kpt.pt[1]
                    idx += 1  # increment index
        if show_matches:
            cv2.waitKey(1)

        return (matched_m_train, matched_kpts_train, matched_desc_train), (matched_m_query, matched_kpts_query, matched_desc_query), random_colors

class OmniStereoModel(object):
    '''
    The vertically-folded omnistereo model using GUM
    '''

    def __init__(self, top_model, bottom_model, **kwargs):
        '''
        Constructor
        '''
        self.top_model = top_model
        self.bot_model = bottom_model
        self.units = top_model.units
        self.set_params(**kwargs)
        self.infer_additional_parameters_from_models()
#         self.calibrator = calibration.CalibratorStereo(self)
        self.baseline = self.get_baseline()
        self.current_omni_img = None
        self.construct_new_mask = True
        self.mask_RGB_color = None
        self.mask_background_img = None
        self.T_Cest_wrt_Rgt = None  # Hand-eye transformation

    def set_poses(self, T_top_wrt_C, T_bottom_wrt_C):
        from omnistereo.transformations import concatenate_matrices

        translation_top_wrt_C = T_top_wrt_C[:3, 3]
        rotation_top_wrt_C = T_top_wrt_C[:3, :3]
        self.top_model.set_pose(translation = translation_top_wrt_C, rotation_matrix = rotation_top_wrt_C)

        translation_bottom_wrt_C = T_bottom_wrt_C[:3, 3]
        rotation_bottom_wrt_C = T_bottom_wrt_C[:3, :3]
        self.bot_model.set_pose(translation = translation_bottom_wrt_C, rotation_matrix = rotation_bottom_wrt_C)

        self.T_bot_wrt_top = concatenate_matrices(self.top_model.T_C_wrt_model, self.bot_model.T_model_wrt_C)
        self.T_top_wrt_bot = concatenate_matrices(self.bot_model.T_C_wrt_model, self.top_model.T_model_wrt_C)

    def set_params(self, **kwargs):

        if "center_point_top" in kwargs:
            center_point_top = kwargs.get("center_point_top", self.top_model.precalib_params.center_point)
            self.top_model.precalib_params.center_point = center_point_top
            self.top_model.precalib_params.u_center = center_point_top[0]
            self.top_model.precalib_params.v_center = center_point_top[1]
        if "center_point_top_inner" in kwargs:
            center_point_top_inner = kwargs.get("center_point_top_inner", self.top_model.precalib_params.center_point)
            self.top_model.precalib_params.center_point_inner = center_point_top_inner
            self.top_model.precalib_params.u_center_inner = center_point_top_inner[0]
            self.top_model.precalib_params.v_center_inner = center_point_top_inner[1]
        if "center_point_top_outer" in kwargs:
            center_point_top_outer = kwargs.get("center_point_top_outer", self.top_model.precalib_params.center_point)
            self.top_model.precalib_params.center_point_outer = center_point_top_outer
            self.top_model.precalib_params.u_center_outer = center_point_top_outer[0]
            self.top_model.precalib_params.v_center_outer = center_point_top_outer[1]
        if "center_point_bottom" in kwargs:
            center_point_bottom = kwargs.get("center_point_bottom", self.bot_model.precalib_params.center_point)
            self.bot_model.precalib_params.center_point = center_point_bottom
            self.bot_model.precalib_params.u_center = center_point_bottom[0]
            self.bot_model.precalib_params.v_center = center_point_bottom[1]
        if "center_point_bottom_inner" in kwargs:
            center_point_bottom_inner = kwargs.get("center_point_bottom_inner", self.bot_model.precalib_params.center_point)
            self.bot_model.precalib_params.center_point_inner = center_point_bottom_inner
            self.bot_model.precalib_params.u_center_inner = center_point_bottom_inner[0]
            self.bot_model.precalib_params.v_center_inner = center_point_bottom_inner[1]
        if "center_point_bottom_outer" in kwargs:
            center_point_bottom_outer = kwargs.get("center_point_bottom_outer", self.bot_model.precalib_params.center_point)
            self.bot_model.precalib_params.center_point_outer = center_point_bottom_outer
            self.bot_model.precalib_params.u_center_outer = center_point_bottom_outer[0]
            self.bot_model.precalib_params.v_center_outer = center_point_bottom_outer[1]

        # In case radii haven't been passed,
        # infer occlusion boundaries automatically from pre-calibration ROI parameters:
        _, v_center_top = self.top_model.get_center()
        _, v_center_bottom = self.bot_model.get_center()

        radius_top_outer = kwargs.get("radius_top_outer", self.top_model.outer_img_radius)
        if radius_top_outer == 0:
            try:
                radius_top_outer = v_center_top - self.top_model.precalib_params.roi_max_y
            except:
                print("radius_top_outer not set")
                pass

        radius_top_inner = kwargs.get("radius_top_inner", self.top_model.inner_img_radius)
        if radius_top_inner == 0:
            try:
                radius_top_inner = v_center_top - self.bot_model.precalib_params.roi_max_y
            except:
                print("radius_top_inner not set")
                pass

        radius_bottom_outer = kwargs.get("radius_bottom_outer", self.bot_model.outer_img_radius)
        if radius_bottom_outer == 0:
            try:
                radius_bottom_outer = v_center_bottom - self.bot_model.precalib_params.roi_max_y
            except:
                print("radius_bottom_outer not set")
                pass

        radius_bottom_inner = kwargs.get("radius_bottom_inner", self.bot_model.inner_img_radius)
        if radius_bottom_inner == 0:
            try:
                radius_bottom_inner = 0  # It's zero for now (not masking anything). Also, the pre-calibration roi_min seems
            except:
                print("radius_bottom_inner not set")
                pass

        if radius_top_outer > 0 and radius_bottom_outer > 0:
            self.set_radial_limits_in_pixels(radius_top_outer, radius_top_inner, radius_bottom_outer, radius_bottom_inner, **kwargs)

        self.common_vFOV = self.set_common_vFOV()

    def infer_additional_parameters_from_models(self):
        raise NotImplementedError

    def get_baseline(self, **kwargs):
        raise NotImplementedError

    def get_triangulated_point_wrt_Oc(self, elev1, elev2, azimuth):
        '''
        @brief Compute triangulated point's cartesian coordinates w.r.t. origin of camera frame using the given direction angles (in radians)

        @param alpha1: The elevation angle (or ndarray of angles) to the point with respect to mirror 1's focus \f$F_1$\f
        @param alpha2: The elevation angle (or ndarray of angles) to the point with respect to mirror 2's focus \f$F_2$\f
        @param phi: The common azimuth angle (or ndarray of angles) of the triangulated point

        @return: If parameters are scalars (for one point), the returned point is given as a simple (x,y,z) tuple. Otherwise, the np.ndarray (rows, cols, 3) encoding the Cartessian (x,y,z) coordinates of the triangulated point
        '''
        # Horizontal range:
        # FIXME: handle division by 0.
        rho = (self.baseline * np.cos(elev1) * np.cos(elev2)) / (np.sin(elev1 - elev2))
        # Compute triangulated point's cartesian coordinates w.r.t. origin of camera frame
        # See Jaramillo's article in Sensors 2016
        x = -rho * np.cos(azimuth)
        y = -rho * np.sin(azimuth)
        z = self.top_model.F[2, 0] - rho * np.tan(elev1)

        Pw_wrt_C = np.dstack((x, y, z))

        return Pw_wrt_C

    # TODO: Test multiple points (matrix of points as rows, cols)
    # TODO: pass omnidirectional correspondence points directly instead of rays
    def get_triangulated_midpoint(self, dir_ray1, dir_ray2):
        '''
        Approximates the triangulated 3D point (for both cases: intersecting rays or skew rays)
        using the common perpendicular to both back-projection rays \f$(\vect{v}_1,\vect{v}_2)$\f

        @param dir_ray1: Ray leaving top focus point. It must be of shape [rows,cols,3]
        @param dir_ray2: Ray leaving bottom focus point. It must be of shape [rows,cols,3]
        @note: back-projection vectors \f$(\vect{v}_1,\vect{v}_2)$\f must be given wrt focus

        @reval: mid_Pw: The midpoint (in Eucledian coordinates, a.k.a 3-vector) of the common perpendicular between the 2 direction rays.
        @retval (lambda1, lambda2, lambda_perp): a tuple of the relevant line parameters
        @reval: (G1, G1): The coordinates (wrt to reference/camera frame) for the end points on the common perpendicular line segment
        @reval: (perp_vect_unit, perp_mag): A tuple containing the direction (unit vector) of the common perpendicular to both rays and its magnitude (closes distance among both rays)
        '''
        v1 = dir_ray1[..., :3]
        v2 = dir_ray2[..., :3]

        perp_vect = np.cross(v1, v2)
        perp_vect_mag = np.linalg.norm(perp_vect, axis = -1)[..., np.newaxis]

#         if perp_vect_mag > 0:
        perp_vect_unit = perp_vect / perp_vect_mag
        # Solve the system of linear equations
        # Given as M * t = b. Then, t = M^-1 * b
        v1 = v1[..., np.newaxis]  # >>> Gives shape (...,3,1)
        v2 = v2[..., np.newaxis]  # >>> Gives shape (...,3,1)
        perp_vect_unit = perp_vect_unit[..., np.newaxis]  # >>> Gives shape (...,3,1)
        M = np.concatenate((v1, -v2, perp_vect_unit), axis = -1).reshape(v1.shape[0:-2] + (3, 3))  # >>> Gives shape (...,3,3)
        # NOTE: Working with column vectors, so we reshape along the way... (watch out for misshapes)
        f1 = self.top_model.F[:-1]
        f2 = self.bot_model.F[:-1]

        b = np.zeros_like(v1) + (f2 - f1)  # >>> Gives shape (...,3,1)
        # Use equation for lenght of common perpendicular:
#         perp_mag = np.abs(np.inner(b[..., 0], perp_vect_unit[..., 0]))  # >>> Gives shape ()
        perp_mag = np.abs(np.einsum("mnt, mnt->mn", b[..., 0], perp_vect_unit[..., 0]))
#===============================================================================
#         perp_mag_approx = np.zeros_like(perp_mag)
#         np.around(perp_mag, 10, perp_mag_approx)
#===============================================================================
#             ans = np.where(perp_mag_approx > 0, self.triangulate_for_skew_rays(f1, v1, perp_vect_unit, M, b), some_other_func)
#             mid_Pw, lambda1, lambda2, lambda_perp = ans[..., 0, 0], ans[..., 0, 1], ans[..., 0, 2], ans[..., 0, 3]
        # Method works for all cases (intersecting rays or skew rays)
        try:
            ans = self.triangulate_for_skew_rays(f1, v1, perp_vect_unit, M, b)  # >>> Gives shape (...,4)
            # mid_Pw, lambda1, lambda2, lambda_perp = ans[..., 0], ans[..., 1], ans[..., 2], ans[..., 3]
            mid_Pw, lambda1, lambda2, lambda_perp, G1 = ans[0], ans[1], ans[2], ans[3], ans[4]
            G2 = f2 + lambda2[..., np.newaxis] * v2
        except:
            print("Problem")
#===============================================================================
#         else:
#             # FIXME: handle diverging rays so
#
#             # Backup method using triangulation via trigonometry and intersection assumption
#             pass
#             # TODO: add function to produce results using the regular triangulation function
#===============================================================================

        return mid_Pw[..., 0], (lambda1, lambda2, lambda_perp), (G1[..., 0], G2[..., 0]), (perp_vect_unit[..., 0], perp_mag)  # perp_mag[...][0, 0])

    def triangulate_for_skew_rays(self, f1, v1, perp_vect_unit, M, b):
        # TODO: Vectorize or loop across rows and cols, eg.  f1[row,col]
        lambda_solns = np.linalg.solve(M, b)  # Implying solution vector [lambda1, lambda2, lambda_perp]

#         lambda1, lambda2, lambda_perp = lambda_solns[..., 0, 0], lambda_solns[..., 1, 0], lambda_solns[..., 2, 0]
        lambda1, lambda2, lambda_perp = lambda_solns[..., 0, :], lambda_solns[..., 1, :], lambda_solns[..., 2, :]
        # Point G1 (end point on common perpendicular)
        G1 = f1 + lambda1[..., np.newaxis] * v1
        mid_Pw = G1 + lambda_perp[..., np.newaxis] / 2.0 * perp_vect_unit
        return mid_Pw, lambda1, lambda2, lambda_perp, G1

    def resolve_pano_correspondences_from_disparity_map(self, ref_points_uv_coords, min_disparity = 0, max_disparity = 0, verbose = False, roi_cols = None):
        '''
        @return: 2 correspondence lists of pixel coordinates (as separate ndarrays of n rows by 2 cols) regarding the (u,v) coordinates on the top and the bottom
        '''
        # NOTE: the disparity map is linked to the left/top image so it is the reference for the correspondences
        #       Thus, the coordinates of the correspondence on the right/bottom image is found such that m_right = (u_left, v_left - disp[u,v])
        # NOTE: Throughout, the order of coordinates as (u,v) need to be swapped to (row, col)

        if roi_cols is not None:  # Fill up zeroes to those disparities outside of the ROI columns
            disparity_map = np.zeros_like(self.disparity_map)
            disparity_map[:, roi_cols[0]:roi_cols[1]] = self.disparity_map[:, roi_cols[0]:roi_cols[1]]
        else:
            disparity_map = self.disparity_map

        if max_disparity == 0:
            max_disparity = disparity_map.max()

        # Non-zero disparities:
        nonzero_disp_indices = disparity_map[ref_points_uv_coords[..., 1], ref_points_uv_coords[..., 0]] != 0
        ref_pano_points_coords_nonzero = ref_points_uv_coords[nonzero_disp_indices]

        # Select only those coordinates with valid disparities (based on minimum threshold)
        is_equal_or_greater_than_min = min_disparity <= disparity_map[ref_pano_points_coords_nonzero[..., 1], ref_pano_points_coords_nonzero[..., 0]]
        is_less_than_max = disparity_map[ref_pano_points_coords_nonzero[..., 1], ref_pano_points_coords_nonzero[..., 0]] <= max_disparity
        valid_disp_indices = np.logical_and(is_equal_or_greater_than_min, is_less_than_max)

        ref_pano_points_coords_valid = ref_pano_points_coords_nonzero[valid_disp_indices]
        disparities = disparity_map[ref_pano_points_coords_valid[..., 1], ref_pano_points_coords_valid[..., 0]]

        # Filter within target bounds:
        # NOTE: Assuming rectified panoramic images
        lowest_reference_row = self.bot_model.panorama.get_panorama_row_from_elevation(self.bot_model.lowest_elevation_angle)
        disp_indices_within_bounds = ref_pano_points_coords_valid[..., 1] - disparities <= lowest_reference_row
        ref_pano_points_coords = ref_pano_points_coords_valid[disp_indices_within_bounds]

        disparities = disparity_map[ref_pano_points_coords[..., 1], ref_pano_points_coords[..., 0]]
        target_pano_points_coords = np.transpose((ref_pano_points_coords[..., 0], ref_pano_points_coords[..., 1] - disparities))
        if verbose and np.count_nonzero(ref_pano_points_coords) > 0:
            print("CORRESPONDENCE: %s <-> %s using DISP: %f" % (ref_pano_points_coords[-1], target_pano_points_coords[-1], disparities[-1]))

        # Add an extra axis to form a 3 dimensional table.
        if ref_pano_points_coords.ndim < 3:
            ref_pano_points_coords = ref_pano_points_coords[np.newaxis, ...]
        if target_pano_points_coords.ndim < 3:
            target_pano_points_coords = target_pano_points_coords[np.newaxis, ...]

        return ref_pano_points_coords, target_pano_points_coords, disparities

    def get_correspondences_from_clicked_points(self, min_disparity = 1, max_disparity = 0):
        from omnistereo.common_cv import PointClicker
        # Get point clicked on panoramic image and mark it (visualize it)
        click_window_name = 'Reference Points (To Click)'
        cv2.namedWindow(click_window_name, cv2.WINDOW_NORMAL)
        pt_clicker = PointClicker(click_window_name, max_clicks = 1000)
        top_pano_coords, bot_pano_coords, disparities = pt_clicker.get_clicks_uv_coords_for_stereo(stereo_model = self, show_correspondence_on_circular_img = True, min_disparity = min_disparity, max_disparity = max_disparity, verbose = True)

        #=======================================================================
        # _, _, omni_top_coords = self.top_model.panorama.get_omni_pixel_coords_from_panoramic_pixel(top_pano_coords, use_LUTs=False)
        # _, _, omni_bot_coords = self.bot_model.panorama.get_omni_pixel_coords_from_panoramic_pixel(bot_pano_coords, use_LUTs=False)
        # return omni_top_coords, omni_bot_coords, disparities
        #=======================================================================

        return top_pano_coords, bot_pano_coords, disparities

    def triangulate_from_clicked_points(self, min_disparity = 1, max_disparity = 0, use_PCL = False, export_to_pcd = False, cloud_path = "data", cloud_index = None, use_LUTs = False, use_midpoint_triangulation = False):
        '''
        @param cloud_index: used for saving identifiable point clouds.
        '''
        top_pano_points_coords, bot_pano_points_coords, pano_disparities = self.get_correspondences_from_clicked_points(min_disparity = min_disparity, max_disparity = max_disparity)  # For testing disparity matches purposes
        az1, el1 = self.top_model.panorama.get_direction_angles_from_pixel_pano(top_pano_points_coords, use_LUTs = use_LUTs)
        az2, el2 = self.bot_model.panorama.get_direction_angles_from_pixel_pano(bot_pano_points_coords, use_LUTs = use_LUTs)
        # Get XYZ from triangulation and put into some cloud
        points_3D_homo = self.get_triangulated_point_from_direction_angles(dir_angs_top = (az1, el1), dir_angs_bot = (az2, el2), use_midpoint_triangulation = use_midpoint_triangulation)
        return self.generate_point_clouds(points_3D_homo, top_pano_points_coords, use_PCL = use_PCL, export_to_pcd = export_to_pcd, cloud_path = cloud_path, cloud_index = cloud_index)

    def triangulate_from_depth_map(self, min_disparity = 1, max_disparity = 0, use_PCL = False, export_to_pcd = False, cloud_path = "data", use_LUTs = False, roi_cols = None, use_midpoint_triangulation = False, cloud_index = None, use_opengv_triangulation = True, profile_triang_opengv = False):
        '''
        @param cloud_index: used for saving identifiable point clouds.
        '''
        # Get matching pairs
        ref_points_uv_coords = np.transpose(np.indices(self.disparity_map.shape[::-1]), (1, 2, 0))
        top_pano_points_coords, bot_pano_points_coords, pano_disparities = self.resolve_pano_correspondences_from_disparity_map(ref_points_uv_coords, min_disparity = min_disparity, max_disparity = max_disparity, roi_cols = roi_cols)

        #=======================================================================
        # _, _, omni_top_coords = self.top_model.panorama.get_omni_pixel_coords_from_panoramic_pixel(top_pano_points_coords, use_LUTs=use_LUTs)
        # _, _, omni_bot_coords = self.bot_model.panorama.get_omni_pixel_coords_from_panoramic_pixel(bot_pano_points_coords, use_LUTs=use_LUTs)
        # # Get XYZ from triangulation and put into some cloud
        # xyz_points = self.get_triangulated_point_from_pixels(m1=omni_top_coords, m2=omni_bot_coords, use_midpoint_triangulation=False)
        #=======================================================================

        az1, el1 = self.top_model.panorama.get_direction_angles_from_pixel_pano(top_pano_points_coords, use_LUTs = use_LUTs)
        az2, el2 = self.bot_model.panorama.get_direction_angles_from_pixel_pano(bot_pano_points_coords, use_LUTs = use_LUTs)
        # Get XYZ from triangulation and put into some cloud
        if profile_triang_opengv:
            from time import process_time
            from pyopengv import triangulation_triangulate, triangulation_triangulate2
            num_time_samples = 10
            time_ellapsed_triang_mine_acc = 0.
            time_ellapsed_triang1_acc = 0.
            time_ellapsed_triang2_acc = 0.
            error_squared_triang1_acc = 0.
            error_squared_triang2_acc = 0.
            T_omnistereo_bot_wrt_top = np.identity(4)
            # NOTE: in this model configuration, the rotation is an identity. However, this will not always be the case for other configurations.
            T_omnistereo_bot_wrt_top_translation = -(self.top_model.F[:3] - self.bot_model.F[:3]).astype(np.float)
            T_omnistereo_bot_wrt_top[:3, 3] = T_omnistereo_bot_wrt_top_translation[:3, 0]
            bearing_vectors_top = self.top_model.get_3D_point_from_angles_wrt_focus(azimuth = az1, elevation = el1)
            bearing_vectors_bottom = self.bot_model.get_3D_point_from_angles_wrt_focus(azimuth = az2, elevation = el2)
            bearing_vectors_inliers_top = bearing_vectors_top[0, ..., :3]
            bearing_vectors_inliers_bottom = bearing_vectors_bottom[0, ..., :3]

            # WISHME: make this a member variable since it's constant
            T_omnistereo_top_wrt_C = np.identity(4)
            # NOTE: in this model configuration, the rotation is an identity. However, this will not always be the case for other configurations.
            T_omnistereo_top_wrt_C[:3, 3] = self.top_model.F[:3, 0]

            for time_idx in range(num_time_samples):
                # Get XYZ wrt [C] from triangulation and put into some point cloud
                start_time_triang_mine = process_time()
                points_3D_homo = self.get_triangulated_point_from_direction_angles(dir_angs_top = (az1, el1), dir_angs_bot = (az2, el2), use_midpoint_triangulation = use_midpoint_triangulation)
                end_time_triang_mine = process_time()
                time_ellapsed_triang_mine = end_time_triang_mine - start_time_triang_mine
                time_ellapsed_triang_mine_acc += time_ellapsed_triang_mine

                # OpenGV's Triangulation Method 1:  linear
                start_time_triang1 = process_time()
                xyz_points_initial_check_triang1_wrt_top = triangulation_triangulate(bearing_vectors_inliers_top, bearing_vectors_inliers_bottom, T_omnistereo_bot_wrt_top_translation, T_omnistereo_bot_wrt_top[:3, :3])
                # Extra transformations to give answer wrt [C]:
                # ..............................................
                # Append 1's for homogeneous coordinates
                ones_matrix = np.ones(xyz_points_initial_check_triang1_wrt_top.shape[:-1])[..., np.newaxis]
                xyz_points_initial_check_triang1_wrt_top_homo = np.dstack((xyz_points_initial_check_triang1_wrt_top[np.newaxis, ...], ones_matrix[np.newaxis, ...]))
                xyz_points_initial_check_triang1_wrt_C_homo = np.einsum("ij, nj->ni", T_omnistereo_top_wrt_C, xyz_points_initial_check_triang1_wrt_top_homo[0])
                # ..............................................
                end_time_triang1 = process_time()
                time_ellapsed_triang1 = end_time_triang1 - start_time_triang1
                time_ellapsed_triang1_acc += time_ellapsed_triang1

                # test_triang1 = np.allclose(points_3D_homo, xyz_points_initial_check_triang1_wrt_C_homo, rtol=1e-1)
                error_L2_distances_triang1 = np.linalg.norm(points_3D_homo[..., :3] - xyz_points_initial_check_triang1_wrt_C_homo[..., :3], axis = -1)
                error_squared_triang1 = np.sum(error_L2_distances_triang1 ** 2)
                error_squared_triang1_acc += error_squared_triang1

                # OpenGV's Triangulation Method 2:  a fast non-linear approximation (very similar to my "use_midpoint_triangulation" method), but it seems faster by a factor of 10
                start_time_triang2 = process_time()
                xyz_points_initial_check_triang2_wrt_top = triangulation_triangulate2(bearing_vectors_inliers_top, bearing_vectors_inliers_bottom, T_omnistereo_bot_wrt_top_translation, T_omnistereo_bot_wrt_top[:3, :3])
                # Extra transformations to give answer wrt [C]:
                # ..............................................
                # Append 1's for homogeneous coordinates
                ones_matrix = np.ones(xyz_points_initial_check_triang2_wrt_top.shape[:-1])[..., np.newaxis]
                xyz_points_initial_check_triang2_wrt_top_homo = np.dstack((xyz_points_initial_check_triang2_wrt_top[np.newaxis, ...], ones_matrix[np.newaxis, ...]))
                xyz_points_initial_check_triang2_wrt_C_homo = np.einsum("ij, nj->ni", T_omnistereo_top_wrt_C, xyz_points_initial_check_triang2_wrt_top_homo[0])
                # ..............................................
                end_time_triang2 = process_time()
                time_ellapsed_triang2 = end_time_triang2 - start_time_triang2
                time_ellapsed_triang2_acc += time_ellapsed_triang2

                # test_triang2 = np.allclose(points_3D_homo, xyz_points_initial_check_triang2_wrt_C_homo, rtol=1e-1)
                error_L2_distances_triang2 = np.linalg.norm(points_3D_homo[..., :3] - xyz_points_initial_check_triang2_wrt_C_homo[..., :3], axis = -1)
                error_squared_triang2 = np.sum(error_L2_distances_triang2 ** 2)
                error_squared_triang2_acc += error_squared_triang2

            print("Triangulation (my method) - Average time elapsed: {time:.8f} seconds".format(time = time_ellapsed_triang_mine_acc / num_time_samples))
            print("Triangulation (linear) - Average time elapsed: {time:.8f} seconds. SSE = {sse:.10f} [mm^2]".format(time = time_ellapsed_triang1_acc / num_time_samples, sse = error_squared_triang1_acc / num_time_samples))
            print("Triangulation (nonlinear / midpoint approach) - Average time elapsed: {time:.10f} seconds. SSE = {sse:.8f} [mm^2]".format(time = time_ellapsed_triang2_acc / num_time_samples, sse = error_squared_triang2_acc / num_time_samples))
        else:
            if use_opengv_triangulation:
                from pyopengv import triangulation_triangulate, triangulation_triangulate2
                # WISHME: Make this a member variable since it's constant
                T_omnistereo_top_wrt_C = np.identity(4)
                # NOTE: in this model configuration, the rotation is an identity. However, this will not always be the case for other configurations.
                T_omnistereo_top_wrt_C[:3, 3] = self.top_model.F[:3, 0]

                # WISHME: applying the transformation wrt [C] takes some time. The user should provide the "reference" frame preferred for the answer. Also, make it a constant member variable
                T_omnistereo_bot_wrt_top = np.identity(4)
                # NOTE: in this model configuration, the rotation is an identity. However, this will not always be the case for other configurations.
                T_omnistereo_bot_wrt_top_translation = -(self.top_model.F[:3] - self.bot_model.F[:3]).astype(np.float)
                T_omnistereo_bot_wrt_top[:3, 3] = T_omnistereo_bot_wrt_top_translation[:3, 0]

                bearing_vectors_top = self.top_model.get_3D_point_from_angles_wrt_focus(azimuth = az1, elevation = el1)
                bearing_vectors_bottom = self.bot_model.get_3D_point_from_angles_wrt_focus(azimuth = az2, elevation = el2)
                bearing_vectors_inliers_top = bearing_vectors_top[0, ..., :3]
                bearing_vectors_inliers_bottom = bearing_vectors_bottom[0, ..., :3]
                if use_midpoint_triangulation:
                    points_3D_wrt_top = triangulation_triangulate(bearing_vectors_inliers_top, bearing_vectors_inliers_bottom, T_omnistereo_bot_wrt_top_translation, T_omnistereo_bot_wrt_top[:3, :3])
                else:
                    points_3D_wrt_top = triangulation_triangulate2(bearing_vectors_inliers_top, bearing_vectors_inliers_bottom, T_omnistereo_bot_wrt_top_translation, T_omnistereo_bot_wrt_top[:3, :3])
                ones_matrix = np.ones(points_3D_wrt_top.shape[:-1])[..., np.newaxis]
                points_3D_wrt_top_homo = np.dstack((points_3D_wrt_top[np.newaxis, ...], ones_matrix[np.newaxis, ...]))
                points_3D_wrt_C_homo = np.einsum("ij, nj->ni", T_omnistereo_top_wrt_C, points_3D_wrt_top_homo[0])[np.newaxis, ...]
            else:
                points_3D_wrt_C_homo = self.get_triangulated_point_from_direction_angles(dir_angs_top = (az1, el1), dir_angs_bot = (az2, el2), use_midpoint_triangulation = use_midpoint_triangulation)

        return self.generate_point_clouds(points_3D_wrt_C_homo, top_pano_points_coords, use_PCL = use_PCL, export_to_pcd = export_to_pcd, cloud_path = cloud_path, cloud_index = cloud_index)

    def generate_point_clouds(self, xyz_points, pano_ref_uv_coords, rgb_colors = None, use_PCL = False, export_to_pcd = True, cloud_path = "data", cloud_index = None):
        '''
        Triangulates pixel correspondences using current stereo model.

        @param cloud_index: used for saving identifiable point clouds.
        @param rgb_colors: An numpy array of RGB colors can be optionally passed. This is useful in case of experimenting with spartse features where individual points must be distinguish from each other.
        '''
        # set RGB info to complimentary cloud
        #=======================================================================
        # Using OMNI image
        # channels = self.current_omni_img.ndim
        # # RGB color is taken from either panorama (top panorama for now) # TODO: maybe color should be averaged
        # if channels == 1:
        #     # GrayScale
        #     ref_color_img = cv2.cvtColor(self.current_omni_img, cv2.COLOR_GRAY2RGB).astype('float32')
        # else:
        #     ref_color_img = cv2.cvtColor(self.current_omni_img, cv2.COLOR_BGR2RGB).astype('float32')
        # # Recall that now we are giving rows and cols as coords in the omni image
        # rgb_points = ref_color_img[list(omni_top_coords[..., 1].flatten()), list(omni_top_coords[..., 0].flatten())]
        #=======================================================================

        if rgb_colors is None:  # Using PANORAMIC image
            channels = self.top_model.panorama.panoramic_img.ndim
            # RGB color is taken from either panorama (top panorama for now) # TODO: maybe color should be averaged
            if channels == 1:
                # GrayScale
                ref_color_img = cv2.cvtColor(self.top_model.panorama.panoramic_img, cv2.COLOR_GRAY2RGB).astype('float32')
            else:
                ref_color_img = cv2.cvtColor(self.top_model.panorama.panoramic_img, cv2.COLOR_BGR2RGB).astype('float32')
            # Recall that now we are giving rows and cols as coords in the omni image
            rgb_points = ref_color_img[list(pano_ref_uv_coords[..., 1].flatten()), list(pano_ref_uv_coords[..., 0].flatten())]
        else:
            rgb_points = rgb_colors

        # Generate clouds
        xyz_cloud = None
        rgb_cloud = None
        if use_PCL:
            import pcl
            xyz_cloud = pcl.PointCloud(xyz_points[..., :3].reshape(-1, 3).astype('float32'))
            rgb_cloud = pcl.PointCloud(rgb_points.reshape(-1, 3).astype('float32'))

            if export_to_pcd:
                from omnistereo.common_tools import make_sure_path_exists
                make_sure_path_exists(cloud_path)  # This creates the path if necessary
                if cloud_index is None:
                    cloud_id = ""
                else:
                    cloud_id = "-%d" % (cloud_index)
                pcd_xyz_filename = "%s/XYZ%s.pcd" % (cloud_path, cloud_id)
                pcd_rgb_filename = "%s/RGB%s.pcd" % (cloud_path, cloud_id)
                # Export the 2 pointclouds to PCD files
                pcl.save(cloud = xyz_cloud, path = pcd_xyz_filename, format = "pcd", binary = False)
                print("Saved XYZ cloud to %s" % pcd_xyz_filename)
                pcl.save(cloud = rgb_cloud, path = pcd_rgb_filename, format = "pcd", binary = False)
                print("Saved RGB cloud to %s" % pcd_rgb_filename)
#===============================================================================
#         else:  # TODO: implemet the visualization with Mayavi once it successfully installed (pain in the ars!)
#             pass
#             # Using Matplotlib 3D (Too Slow)
#             from mpl_toolkits.mplot3d import axes3d
#             import matplotlib.pyplot as plt
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d', aspect="equal")
#
#             xyz = xyz_points[...,:3].reshape(-1, 3)
#
#             # Define coordinates and points
#             x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]  # Assign x, y, z values to match color
#             ax.scatter(x, y, z, c=rgb_points.reshape(-1, 3) / 255., s=50)
#             plt.show()
#===============================================================================

#===============================================================================
#             # Imports
# #             from mayavi.mlab import quiver3d, draw
#             from mayavi.mlab import points3d, draw
#
#             rgb = rgb_points.reshape(-1, 3).astype(np.uint8)
#             xyz = xyz_points.reshape(-1, 3)
#
#             # Primitives
#             N = rgb.shape[0]  # Number of points
#             ones = np.ones(N)
#             scalars = np.arange(N)  # Key point: set an integer for each point
#
#             # Define color table (including alpha), which must be uint8 and [0,255]
#             colors = np.zeros((N, 4) , dtype=np.uint8)
#             colors[:, :3] = rgb  # RGB color channels
#             colors[:, -1] = 255  # No transparency
#
#             # Define coordinates and points
# #             x, y, z = colors[:, 0], colors[:, 1], colors[:, 2]  # Assign x, y, z values to match color
#             x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]  # Assign x, y, z values to match color
#             pts = points3d(x, y, z)  # , ones, scale_mode='none')
# #             pts = quiver3d(x, y, z, ones, ones, ones, scalars=scalars, mode='sphere')  # Create points
# #             pts.glyph.color_mode = 'color_by_scalar'  # Color by scalar
# #             # Set look-up table and redraw
# #             pts.module_manager.scalar_lut_manager.lut.table = colors
# #             draw()
#===============================================================================

        return (xyz_points, rgb_points), (xyz_cloud, rgb_cloud)

    def set_radial_limits_in_pixels(self, outer_radius_top, inner_radius_top, outer_radius_bottom, inner_radius_bottom, **kwargs):
        self.top_model.set_radial_limits_in_pixels_mono(inner_img_radius = inner_radius_top, outer_img_radius = outer_radius_top, **kwargs)
        self.bot_model.set_radial_limits_in_pixels_mono(inner_img_radius = inner_radius_bottom, outer_img_radius = outer_radius_bottom, **kwargs)
        # compute the highest/lowest elevation globally among both top vs bottom
        global_high_elev = max(self.top_model.highest_elevation_angle, self.bot_model.highest_elevation_angle)
        global_low_elev = min(self.top_model.lowest_elevation_angle, self.bot_model.lowest_elevation_angle)  # Reset new global highest and lowest elevation angles on each GUM
        self.top_model.globally_highest_elevation_angle = global_high_elev
        self.bot_model.globally_highest_elevation_angle = global_high_elev
        self.top_model.globally_lowest_elevation_angle = global_low_elev
        self.bot_model.globally_lowest_elevation_angle = global_low_elev

    def draw_radial_bounds_stereo(self, omni_img = None, is_reference = False, view = True):
        if omni_img is None:
            omni_img = self.current_omni_img

        img = self.top_model.draw_radial_bounds(omni_img = omni_img, is_reference = is_reference, view = False)
        img = self.bot_model.draw_radial_bounds(omni_img = img, is_reference = is_reference, view = False)
        if view:  # FIXME: For some reason is only showing the bottom image as masked
            win_name = "OMNISTEREO Radial Bounds"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, img)
            cv2.waitKey(1)

        return img

    def generate_panorama_pair(self, omni_img, width_in_pixels = 1200, idx = -1, view = True, win_name_modifier = "", use_mask = True, border_RGB_color = None):
        if self.top_model.panorama is None or self.bot_model.panorama is None:
            self.set_current_omni_image(omni_img, pano_width_in_pixels = width_in_pixels, generate_panoramas = True, idx = idx, view = view)
        # Same omni image by default
        omni_img_top = omni_img
        omni_img_bot = omni_img

        if use_mask:
            import warnings
            try:
                from omnistereo.common_cv import get_masked_images_as_pairs
                omni_images_as_pairs = get_masked_images_as_pairs(unmasked_images = [omni_img], omnistereo_model = self, show_images = True, color_RGB = border_RGB_color)
                omni_img_top, omni_img_bot = omni_images_as_pairs[0]
                self.top_model.panorama.set_panoramic_image(omni_img = omni_img_top, idx = idx, view = True, win_name_modifier = win_name_modifier, border_RGB_color = border_RGB_color)
                self.bot_model.panorama.set_panoramic_image(omni_img = omni_img_bot, idx = idx, view = True, win_name_modifier = win_name_modifier, border_RGB_color = border_RGB_color)
            except:
                warnings.warn("Panorama (Pair) index %d problem in %s" % (idx, __name__))

        pano_img_top = self.top_model.panorama.set_panoramic_image(omni_img_top, idx, view, win_name_modifier, border_RGB_color = border_RGB_color)
        pano_img_bot = self.bot_model.panorama.set_panoramic_image(omni_img_bot, idx, view, win_name_modifier, border_RGB_color = border_RGB_color)
        return pano_img_top, pano_img_bot

    def view_all_panoramas(self, omni_images_filename_pattern, img_indices, win_name_modifier = "", use_mask = False, mask_color_RGB = None):
        import warnings
        from omnistereo.common_cv import get_images

        omni_images = get_images(omni_images_filename_pattern, indices_list = img_indices, show_images = not use_mask)

        if img_indices is None or len(img_indices) == 0:  # Default value
            # use all the images in the set:
            img_indices = range(len(omni_images))

        for i in img_indices:
            try:
                #===============================================================
                # from time import process_time  # , perf_counter
                # start_time = process_time()
                #===============================================================
                pano_img_top, pano_img_bot = self.generate_panorama_pair(omni_images[i], idx = i, view = True, win_name_modifier = win_name_modifier, use_mask = use_mask, border_RGB_color = mask_color_RGB)
                #===============================================================
                # end_time = process_time()
                # time_ellapsed_1 = end_time - start_time
                # print("Time elapsed: {time:.8f} seconds".format(time=time_ellapsed_1))
                #===============================================================
                save_panos_to_file = False
                if save_panos_to_file:
                    num_of_zero_padding = 6
                    n = str(i)
                    cv2.imwrite("/tmp/panorama-%s-%s.png" % (self.top_model.mirror_name, n.zfill(num_of_zero_padding)), pano_img_top)
                    cv2.imwrite("/tmp/panorama-%s-%s.png" % (self.bot_model.mirror_name, n.zfill(num_of_zero_padding)), pano_img_bot)

            except:
                warnings.warn("Image index %d not found at %s" % (i, __name__))

    def draw_elevations_on_panoramas(self, left8 = None, right8 = None, draw_own_limits = False):
        '''
        NOTE: Image should not be cropped.
        '''
        if left8 is None:
            left8_annotated = self.top_model.panorama.panoramic_img.copy()
        else:
            left8_annotated = left8.copy()

        if right8 is None:
            right8_annotated = self.bot_model.panorama.panoramic_img.copy()
        else:
            right8_annotated = right8.copy()

        pano_cols = int(self.top_model.panorama.cols)
        last_col = pano_cols - 1
        row_common_highest = self.top_model.panorama.get_panorama_row_from_elevation(self.common_highest_elevation_angle)
        row_common_lowest = self.top_model.panorama.get_panorama_row_from_elevation(self.common_lowest_elevation_angle)

        # Draw 0-degree line
        row_at_zero_degrees = self.top_model.panorama.get_panorama_row_from_elevation(0)
        if not np.isnan(row_at_zero_degrees):
            line_color = (255, 0, 255)  # magenta in BGR
            cv2.line(img = left8_annotated, pt1 = (0, row_at_zero_degrees), pt2 = (last_col, row_at_zero_degrees), color = line_color, thickness = 1 , lineType = cv2.LINE_AA)
            cv2.line(img = right8_annotated, pt1 = (0, row_at_zero_degrees), pt2 = (last_col, row_at_zero_degrees), color = line_color, thickness = 1 , lineType = cv2.LINE_AA)

        # Draw line at common highest elevation
        if not np.isnan(row_common_highest):
            line_color = (255, 0, 0)  # blue in BGR
            cv2.line(img = left8_annotated, pt1 = (0, row_common_highest), pt2 = (last_col, row_common_highest), color = line_color, thickness = 1 , lineType = cv2.LINE_AA)
            cv2.line(img = right8_annotated, pt1 = (0, row_common_highest), pt2 = (last_col, row_common_highest), color = line_color, thickness = 1 , lineType = cv2.LINE_AA)

        # Draw line at common lowest elevation
        if not np.isnan(row_common_lowest):
            line_color = (0, 255, 0)  # green in BGR
            cv2.line(img = left8_annotated, pt1 = (0, row_common_lowest), pt2 = (last_col, row_common_lowest), color = line_color, thickness = 1 , lineType = cv2.LINE_AA)
            cv2.line(img = right8_annotated, pt1 = (0, row_common_lowest), pt2 = (last_col, row_common_lowest), color = line_color, thickness = 1 , lineType = cv2.LINE_AA)

        if draw_own_limits:
            line_color = (0, 0, 255)  # red in BGR
            # Top:
            top_highest = self.top_model.panorama.get_panorama_row_from_elevation(self.top_model.highest_elevation_angle)
            top_lowest = self.top_model.panorama.get_panorama_row_from_elevation(self.top_model.lowest_elevation_angle)
            cv2.line(img = left8_annotated, pt1 = (0, top_highest), pt2 = (last_col, top_highest), color = line_color, thickness = 1 , lineType = cv2.LINE_AA)
            cv2.line(img = right8_annotated, pt1 = (0, top_lowest), pt2 = (last_col, top_lowest), color = line_color, thickness = 1 , lineType = cv2.LINE_AA)
            # Bottom:
            bot_highest = self.bot_model.panorama.get_panorama_row_from_elevation(self.bot_model.highest_elevation_angle)
            bot_lowest = self.bot_model.panorama.get_panorama_row_from_elevation(self.bot_model.lowest_elevation_angle)
            cv2.line(img = left8_annotated, pt1 = (0, bot_highest), pt2 = (last_col, bot_highest), color = line_color, thickness = 1 , lineType = cv2.LINE_AA)
            cv2.line(img = right8_annotated, pt1 = (0, bot_lowest), pt2 = (last_col, bot_lowest), color = line_color, thickness = 1 , lineType = cv2.LINE_AA)

        cropped_top_win = 'Annotated panorama (top or LEFT)'
        cv2.namedWindow(cropped_top_win, cv2.WINDOW_NORMAL)
        cv2.imshow(cropped_top_win, left8_annotated)

        cropped_bot_win = 'Annotated panorama (bottom or RIGHT)'
        cv2.namedWindow(cropped_bot_win, cv2.WINDOW_NORMAL)
        cv2.imshow(cropped_bot_win, right8_annotated)
        pressed_key = cv2.waitKey(10)

        return pressed_key

    def get_fully_masked_images(self, omni_img = None, view = True, color_RGB = None):
        '''
        @param color_RGB: A tuple specifying the desired background as (Red,Green,Blue). If None, the background is black

        @return: masked_img_top, masked_img_bottom
        '''
        from omnistereo.common_tools import convert_to_tuple

        if omni_img is None:
            omni_img = self.current_omni_img

        # Small performance improvement by only generating masks if needed. Now the masks are stored for reusability!
        if self.construct_new_mask:
            # circle centers
            if isinstance(self.top_model.precalib_params, PinholeCamera):
                center_point_top_outer = convert_to_tuple(self.top_model.precalib_params.center_point.astype("int"))
                center_point_top_inner = convert_to_tuple(self.top_model.precalib_params.center_point.astype("int"))
            else:
                center_point_top_outer = convert_to_tuple(self.top_model.precalib_params.center_point_outer.astype("int"))
                center_point_top_inner = convert_to_tuple(self.top_model.precalib_params.center_point_inner.astype("int"))

            if isinstance(self.bot_model.precalib_params, PinholeCamera):
                center_point_bottom_outer = convert_to_tuple(self.bot_model.precalib_params.center_point.astype("int"))
                center_point_bottom_inner = convert_to_tuple(self.bot_model.precalib_params.center_point.astype("int"))
            else:
                center_point_bottom_outer = convert_to_tuple(self.bot_model.precalib_params.center_point_outer.astype("int"))
                center_point_bottom_inner = convert_to_tuple(self.bot_model.precalib_params.center_point_inner.astype("int"))

            # circle radii
            r_inner_top = self.top_model.inner_img_radius
            r_outer_top = self.top_model.outer_img_radius
            r_inner_bottom = self.bot_model.inner_img_radius
            r_outer_bottom = self.bot_model.outer_img_radius

            # TOP:
            mask_top = np.zeros(omni_img.shape[0:2], dtype = np.uint8)  # Black, single channel mask
            # Paint outer perimeter:
            cv2.circle(mask_top, center_point_top_outer, int(r_outer_top), (255, 255, 255), -1, 8, 0)
            # Paint inner bound for top (as the union of the two inner/outer masks)
            if r_inner_top > 0:
                cv2.circle(mask_top, center_point_top_inner, int(r_inner_top), (0, 0, 0), -1, 8, 0)
                if r_outer_bottom > 0:
                    cv2.circle(mask_top, center_point_bottom_outer, int(r_outer_bottom), (0, 0, 0), -1, 8, 0)
            self.top_model.mask = mask_top  # Save mask as property

            # BOTTOM:
            # Paint 2 black masks:
            mask_bottom_outer = np.zeros(omni_img.shape[0:2], dtype = np.uint8)  # Black, single channel mask
            mask_top_inner = np.zeros(omni_img.shape[0:2], dtype = np.uint8)  # Black, single channel mask
            cv2.circle(mask_bottom_outer, center_point_bottom_outer, int(r_outer_bottom), (255, 255, 255), -1, 8, 0)
            cv2.circle(mask_top_inner, center_point_bottom_inner, int(r_inner_top), (255, 255, 255), -1, 8, 0)
            # Paint the outer bound mask for the bottom (as the intersection of the two inner/outer masks)
            mask_bottom = np.zeros(omni_img.shape)
            mask_bottom = cv2.bitwise_and(src1 = mask_bottom_outer, src2 = mask_top_inner, dst = mask_bottom, mask = None)
            # Paint the black inner bound for the bottom
            cv2.circle(mask_bottom, center_point_bottom_inner, int(r_inner_bottom), (0, 0, 0), -1, 8, 0)
            self.bot_model.mask = mask_bottom  # Save mask as property

        # Apply TOP mask
        masked_img_top = np.zeros(omni_img.shape)
        masked_img_top = cv2.bitwise_and(src1 = omni_img, src2 = omni_img, dst = masked_img_top, mask = self.top_model.mask)
        # Apply BOTTOM mask
        masked_img_bottom = np.zeros(omni_img.shape)
        masked_img_bottom = cv2.bitwise_and(src1 = omni_img, src2 = omni_img, dst = masked_img_bottom, mask = self.bot_model.mask)

        if color_RGB is not None:  # Paint the masked area other than black
            if color_RGB != self.mask_RGB_color:
                self.mask_RGB_color = color_RGB
                color_BGR = (color_RGB[2], color_RGB[1], color_RGB[0])
                self.mask_background_img = np.zeros_like(omni_img)
                self.mask_background_img[:, :, :] += np.array(color_BGR, dtype = "uint8")  # Paint the B-G-R channels for OpenCV

            mask_top_inv = cv2.bitwise_not(src = self.top_model.mask)
            # Apply the background using the inverted mask
            masked_img_top = cv2.bitwise_and(src1 = self.mask_background_img, src2 = self.mask_background_img, dst = masked_img_top, mask = mask_top_inv)

            mask_bottom_inv = cv2.bitwise_not(src = self.bot_model.mask)
            # Apply the background using the inverted mask
            masked_img_bottom = cv2.bitwise_and(src1 = self.mask_background_img, src2 = self.mask_background_img, dst = masked_img_bottom, mask = mask_bottom_inv)

        self.construct_new_mask = False  # Clear mask construction flag

        # Show radial boundaries
        if view:
            win_name_top = "TOP Masked with ALL Radial Bounds"
            cv2.namedWindow(win_name_top, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name_top, masked_img_top)
            cv2.waitKey(1)
            win_name_bot = "BOTTOM Masked with ALL Radial Bounds"
            cv2.namedWindow(win_name_bot, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name_bot, masked_img_bottom)
            cv2.waitKey(1)

        return masked_img_top, masked_img_bottom

    def match_features_panoramic_top_bottom(self, keypts_list_top, desc_list_top, keypts_list_bot, desc_list_bot, min_rectified_disparity = 1, max_horizontal_diff = 1, show_matches = False, win_name = 'Matches'):
        '''
        @param min_rectified_disparity: Inclussive limit. This disparity helps to check for point correspondences with positive disparity (which should be the case for rectified stereo), such that (top_kpt.v_coord - bot_kpt.v_coord) >= min_rectified_disparity for the vertical stereo case
        @param max_horizontal_diff: Inclussive limit. This pixel distance on the u-axis. Ideally, this shouldn't be an issue for rectified panoramas, but it's useful to set when panoramas are not perfectly aligned.
        @return (matched_m_top_all, matched_kpts_top_list, matched_desc_top_list), (matched_m_bot_all, matched_kpts_bot_list, matched_desc_bot_list), random_RGB_colors
        '''
        matched_kpts_top_list = []
        matched_kpts_bot_list = []
        matched_desc_top_list = []
        matched_desc_bot_list = []

        for top_keypts, top_descriptors, bot_keypts, bot_descriptors in zip(keypts_list_top, desc_list_top, keypts_list_bot, desc_list_bot):
            if (len(top_keypts) == 0) or (len(bot_keypts) == 0):
                continue

            matches = self.feature_matcher_for_static_stereo.match(query_descriptors = bot_descriptors, train_descriptors = top_descriptors)

            if len(matches) > 0:
                num_of_good_matches = int(self.feature_matcher_for_static_stereo.percentage_good_matches * len(matches))

                if num_of_good_matches > 0:
                    train_indices = [m.trainIdx for m in matches[:num_of_good_matches]]
                    query_indices = [m.queryIdx for m in matches[:num_of_good_matches]]
                    matched_kpts_top_current = np.array(top_keypts)[train_indices]
                    matched_kpts_bot_current = np.array(bot_keypts)[query_indices]
                    matched_desc_top_current = np.array(top_descriptors)[train_indices]
                    matched_desc_bot_current = np.array(bot_descriptors)[query_indices]

                    matched_kpts_top_list.append(matched_kpts_top_current)
                    matched_kpts_bot_list.append(matched_kpts_bot_current)
                    matched_desc_top_list.append(matched_desc_top_current)
                    matched_desc_bot_list.append(matched_desc_bot_current)

        #=======================================================================
        # from omnistereo.common_cv import draw_flow, draw_keypoints
        # top_pts_color = (255, 0, 0)  # red because (R,G,B)
        # draw_points(top_pano_gray_vis, top_pts[..., :2].reshape(-1, 2), color=top_pts_color, thickness=2)
        # draw_keypoints(top_pano_gray_vis, top_keypts, color=top_pts_color)
        # cv2.imshow('Points (TOP)', top_pano_gray_vis)
        #=======================================================================

        #=======================================================================
        # bot_pts = np.copy(top_pts)
        # bot_pts, status, err = cv2.calcOpticalFlowPyrLK(prevImg=top_pano_gray, nextImg=bot_pano_gray, prevPts=top_pts, nextPts=bot_pts)
        # draw_points(bot_pano_gray_vis, bot_pts[..., :2].reshape(-1, 2), color=top_pts_color, thickness=2)
        # draw_keypoints(bot_pano_gray_vis, bot_keypts, color=top_pts_color)
        # cv2.imshow('Points (BOTTOM)', bot_pano_gray_vis)
        # cv2.imshow('flow', draw_flow(bot_pano_gray, flow))
        #=======================================================================

        matched_kpts_top_all = np.concatenate(matched_kpts_top_list)
        matched_kpts_bot_all = np.concatenate(matched_kpts_bot_list)
        matched_desc_top_all = np.concatenate(matched_desc_top_list)
        matched_desc_bot_all = np.concatenate(matched_desc_bot_list)

        # Filter them according to constraints
        keypts_as_points_top_all = cv2.KeyPoint_convert(matched_kpts_top_all).astype(np.float)
        keypts_as_points_bot_all = cv2.KeyPoint_convert(matched_kpts_bot_all).astype(np.float)
        from omnistereo.common_cv import filter_pixel_correspondences
        validation_list = filter_pixel_correspondences(matched_points_top = keypts_as_points_top_all, matched_points_bot = keypts_as_points_bot_all, min_rectified_disparity = min_rectified_disparity, max_horizontal_diff = max_horizontal_diff)

        matched_kpts_top_final = matched_kpts_top_all[validation_list]
        matched_kpts_bot_final = matched_kpts_bot_all[validation_list]
        matched_desc_top_final = matched_desc_top_all[validation_list]
        matched_desc_bot_final = matched_desc_bot_all[validation_list]
        random_colors_final = np.random.randint(low = 0, high = 256, size = (matched_kpts_top_final.shape[0], 3), dtype = "uint8")
        # Add the ones for the point list in homogeneous coordinates
        matched_m_top_final = np.hstack((keypts_as_points_top_all[validation_list], np.ones_like(keypts_as_points_top_all[validation_list][..., 0, np.newaxis])))  # Adding a 1
        matched_m_bot_final = np.hstack((keypts_as_points_bot_all[validation_list], np.ones_like(keypts_as_points_bot_all[validation_list][..., 0, np.newaxis])))  # Adding a 1

        if show_matches:
            from omnistereo.common_plot import draw_matches_between_frames
            draw_matches_between_frames(pano_img_train = self.top_model.panorama.panoramic_img, pano_img_query = self.bot_model.panorama.panoramic_img, matched_kpts_train = matched_kpts_top_final, matched_kpts_query = matched_kpts_bot_final, random_colors = random_colors_final, win_name = win_name)

        return (matched_m_top_final, matched_kpts_top_final, matched_desc_top_final), (matched_m_bot_final, matched_kpts_bot_final, matched_desc_bot_final), random_colors_final

    def set_panoramas(self, pano_top, pano_bottom):
        self.top_model.panorama = pano_top
        self.bot_model.panorama = pano_bottom

    def set_current_omni_image(self, img, pano_width_in_pixels = 1200, generate_panoramas = False, idx = -1, view = False, apply_mask = True, mask_RGB = None):
        self.current_omni_img = img
        if hasattr(self, "theoretical_model"):
            if self.theoretical_model is not None:
                self.theoretical_model.set_current_omni_image(img = img, pano_width_in_pixels = pano_width_in_pixels, generate_panoramas = generate_panoramas, idx = idx, view = False, apply_mask = apply_mask, mask_RGB = mask_RGB)

        if apply_mask:
            img_top, img_bot = self.get_fully_masked_images(omni_img = img, view = view, color_RGB = mask_RGB)
        else:
            img_top = img
            img_bot = img
        # Notice that the mask is not reapplied next because it was alread done in the previous step if apply_mask was True
        self.top_model.set_omni_image(img_top, pano_width_in_pixels = pano_width_in_pixels, generate_panorama = generate_panoramas, idx = idx, view = view, apply_mask = False)
        self.bot_model.set_omni_image(img_bot, pano_width_in_pixels = pano_width_in_pixels, generate_panorama = generate_panoramas, idx = idx, view = view, apply_mask = False)

    def get_system_vFOV(self, **kwargs):
        '''
        Computes the so-called system vFOV angle out of the total view covered by  the two mirrors' vFOVs

        @return: the total system's vertical field of view in radians
        '''
        theta1_max = kwargs.get("theta1_max", self.top_model.highest_elevation_angle)
        theta1_min = kwargs.get("theta1_min", self.top_model.lowest_elevation_angle)
        theta2_max = kwargs.get("theta2_max", self.bot_model.highest_elevation_angle)
        theta2_min = kwargs.get("theta2_min", self.bot_model.lowest_elevation_angle)

        max_elevation = max(theta1_max, theta2_max)
        min_elevation = min(theta1_min, theta2_min)

        alpha_sys = max_elevation - min_elevation

        return alpha_sys

    def get_imaging_ratio(self, print_info = False):
        _, _, m1_common_highest = self.top_model.get_pixel_from_direction_angles(0, self.common_highest_elevation_angle)
        _, _, m1_common_lowest = self.top_model.get_pixel_from_direction_angles(0, self.common_lowest_elevation_angle)
        _, _, m2_common_highest = self.bot_model.get_pixel_from_direction_angles(0, self.common_highest_elevation_angle)
        _, _, m2_common_lowest = self.bot_model.get_pixel_from_direction_angles(0, self.common_lowest_elevation_angle)
        h1 = np.linalg.norm(m1_common_highest - m1_common_lowest)  # FIXME:use the norm
        h2 = np.linalg.norm(m2_common_highest - m2_common_lowest)
        img_ratio = h1 / h2
        if print_info:
            print("Stereo ROI's imaging ratio = %f" % (img_ratio))
            print("using h1 = %f,  and h2=%f" % (h1, h2))

        return img_ratio

    def set_common_vFOV(self, **kwargs):
        '''
        Computes the so-called common vFOV angle out of the overlapping region of the two mirrors' vFOVs
        @note: We are assuming the bottom model's maximum elevation angle is always greater than or equal to the top's maximum elevation angle.

        @return: the common vertical field of view in radians
        '''
        verbose = kwargs.get("verbose", False)
        theta1_max = kwargs.get("theta1_max", self.top_model.highest_elevation_angle)
        theta1_min = kwargs.get("theta1_min", self.top_model.lowest_elevation_angle)
        theta2_max = kwargs.get("theta2_max", self.bot_model.highest_elevation_angle)
        theta2_min = kwargs.get("theta2_min", self.bot_model.lowest_elevation_angle)

        self.common_lowest_elevation_angle = max(theta1_min, theta2_min)
        self.common_highest_elevation_angle = min(theta1_max, theta2_max)

        # Generalized approach:
        self.common_vFOV = self.common_highest_elevation_angle - self.common_lowest_elevation_angle

        # Reset globally high/low elevation angles
        global_high_elev = max(self.top_model.globally_highest_elevation_angle, self.bot_model.globally_highest_elevation_angle)
        self.top_model.globally_highest_elevation_angle = global_high_elev
        self.bot_model.globally_highest_elevation_angle = global_high_elev
        global_low_elev = min(self.top_model.globally_lowest_elevation_angle, self.bot_model.globally_lowest_elevation_angle)
        self.top_model.globally_lowest_elevation_angle = global_low_elev
        self.bot_model.globally_lowest_elevation_angle = global_low_elev

        if verbose:
            print("Common vFOV: %.2f degrees" % (np.rad2deg(self.common_vFOV)))
            print("\tCommon highest elevation: %.2f degrees" % (np.rad2deg(self.common_highest_elevation_angle)))
            print("\tCommon lowest elevation: %.2f degrees" % (np.rad2deg(self.common_lowest_elevation_angle)))
            print("\tusing (min,max) elevations: Top(%.2f,%.2f) degrees, Bottom(%.2f,%.2f) degrees" % (np.rad2deg(theta1_min), np.rad2deg(theta1_max), np.rad2deg(theta2_min), np.rad2deg(theta2_max)))
            print("\tGlobal highest elevation: %.2f degrees" % (np.rad2deg(global_high_elev)))
            print("\tGlobal lowest elevation: %.2f degrees" % (np.rad2deg(global_low_elev)))

        return self.common_vFOV

#     FIXME: redundant definition?
#        def calibrate(self):
#         '''
#         Performs the omnidirectional stereo calibration parameters.
#         @note: Only doing extrinsic optimization at the moment.
#         '''
#         self.calibrator.calibrate()

    def print_params(self, header_message = ""):
        self.top_model.print_params(header_message = "%s mirror: " % (self.top_model.mirror_name) + header_message)
        self.bot_model.print_params(header_message = "%s mirror: " % (self.bot_model.mirror_name) + header_message)
        print("Baseline = %0.4f %s" % (self.baseline, self.units))
        print("Common vFOV %0.4f degrees" % (np.rad2deg(self.common_vFOV)))
        self.get_imaging_ratio(print_info = True)

    def get_triangulated_points_from_pixel_disp(self, disparity = 1, m1 = None):
        '''
        Use this method only for plotting because it's unrealistic to obtain disparities on the omnidirectional images

        @param disparity: The pixel disparity (on the omnidirectional image) to use while computing the depth
        @param m1: A ndarray of specific pixels to get the depth for
        @param m2: A ndarray of corresponding pixels to triangulate with
        @return: the ndarray of \f$ \rho_w$\f for all ray intersection with \f$\Delta m$\f pixel disparity
        @note: This is true way of computing max depth resolution from pixel disparity
        '''
        if m1 is None:
            azim1, elev1 = self.top_model.get_all_direction_angles_per_pixel_radially()
        else:
            azim1, elev1 = self.top_model.get_direction_angles_from_pixel(m1)

        pixels2_u, pixels2_v, _ = self.bot_model.get_pixel_from_direction_angles(azim1, elev1)
        px2_u_with_disp = pixels2_u - disparity  # np.floor(pixels2_u) - disparity

        if m1 is None:
            pixels2 = np.dstack((px2_u_with_disp, pixels2_v))
        else:
            pixels2_v_to_use = np.repeat(pixels2_v, px2_u_with_disp.size).reshape(px2_u_with_disp.shape)
            pixels2 = np.dstack((px2_u_with_disp, pixels2_v_to_use))

        azim2, elev2 = self.bot_model.get_direction_angles_from_pixel(pixels2)
        triangulated_points = self.get_triangulated_point_wrt_Oc(elev1, elev2, azim2)

        return triangulated_points

    def get_triangulated_point_from_pixels(self, m1, m2, use_midpoint_triangulation = False):
        # Using the common perpendicular midpoint method (vectorized)
        if use_midpoint_triangulation:
            # FIXME: not reliable direction vectors from lifting until the "undistortion" bug is fixed
            direction_vectors_top = self.top_model.get_direction_vector_from_focus(m1)
            direction_vectors_bottom = self.bot_model.get_direction_vector_from_focus(m2)
            triangulated_points, _, _, _ = self.get_triangulated_midpoint(direction_vectors_top, direction_vectors_bottom)
        else:
            # FIXME: not reliable direction vectors from lifting until the "undistortion" bug is fixed
            az1, el1 = self.top_model.get_direction_angles_from_pixel(m1)
            az2, el2 = self.bot_model.get_direction_angles_from_pixel(m2)
#             triangulated_points = self.get_triangulated_point_wrt_Oc(el1, el2, (az1 + az2) / 2.0)
            triangulated_points = self.get_triangulated_point_wrt_Oc(el1, el2, (az1 + az2) / 2.0)
#             triangulated_points = np.where(pano_disparities[..., np.newaxis] > 0, triangulated_points_original[0], -1.0 * triangulated_points_original[0])
#             triangulated_points = triangulated_points_original[0, pano_disparities > 0]

        return triangulated_points

    def get_confidence_weight_from_pixel_RMSE_stereo(self, img_points_top, img_points_bot, obj_pts_homo, T_G_wrt_C):
        '''
        We define a confidence weight as the inverse of the pixel projection RMSE

        @param img_points_top: The corresponding points on the top image
        @param img_points_bot: The corresponding points on the bottom image
        @param obj_pts_homo: The coordinates of the corresponding points with respect to the object's own frame [G].
        @param T_G_wrt_C: The transform matrix of [G] wrt to [C].
        '''
        from omnistereo.common_tools import rms
        all_pixel_errors_top = self.top_model.get_obj_pts_proj_error(img_points_top, obj_pts_homo, T_G_wrt_C)
        all_pixel_errors_bot = self.bot_model.get_obj_pts_proj_error(img_points_bot, obj_pts_homo, T_G_wrt_C)
        rmse = rms([np.nan_to_num(all_pixel_errors_top)] + [np.nan_to_num(all_pixel_errors_bot)])
        if rmse > 0:
            weight = 1.0 / rmse
        else:
            weight = 0.0
        return weight

    def filter_panoramic_points_due_to_reprojection_error(self, m_top, m_bot, xyz_points_wrt_C, pixel_error_threshold = 1):
        '''
        Filter outlier feature correspondences by projecting 3D points and measuring pixel norm to matched_m_top and matched_m_bot, so only pixels under a certain distance threshold remain.

        @param pixel_coords: Pixel coordinates on its panoramic image (for the top model)
        @param m_bot: Pixel coordinates on its panoramic image (for the bot model)
        @param xyz_points_wrt_C: The coordinateds of the estimated points wrt to the common frame [C]
        @param pixel_error_threshold: By default is 1 pixel of the pixel error computed of out the norm betwen the detected m pixel points and the reprojected pixels from the XYZ points
        @return: a Boolean list related to the validity of the indices of good points from set
        '''

        _, _, m_top_projected = self.top_model.get_pixel_from_3D_point_wrt_C(xyz_points_wrt_C)
        _, m_pano_top = self.top_model.panorama.get_panoramic_pixel_coords_from_omni_pixel(m_top_projected)
        p2p_distances_top = np.linalg.norm(m_pano_top - m_top, axis = -1)
        # WISH: Ignoring for now the Warning generated when comparing np.nan's and threshold
        valid_top = np.where(p2p_distances_top < pixel_error_threshold, True, False)

        _, _, m_bot_projected = self.bot_model.get_pixel_from_3D_point_wrt_C(xyz_points_wrt_C)
        _, m_pano_bot = self.bot_model.panorama.get_panoramic_pixel_coords_from_omni_pixel(m_bot_projected)
        p2p_distances_bot = np.linalg.norm(m_pano_bot - m_bot, axis = -1)
        # WISH: Ignoring for now the Warning generated when comparing np.nan's and threshold
        valid_bot = np.where(p2p_distances_bot < pixel_error_threshold, True, False)

        valid_indices = np.logical_and(valid_top, valid_bot)

        return valid_indices

    def filter_panoramic_points_due_to_range(self, xyz_points_wrt_C, min_3D_range = 0, max_3D_range = 0.):
        '''
        Filter outlier feature correspondences by projecting 3D points under a certain range threshold remain.

        @param xyz_points_wrt_C: The coordinateds of the estimated points wrt to the common frame [C]
        @param min_3D_range: The minimum euclidean norm to be considered a valid point. 0 by default
        @param max_3D_range: The maximum euclidean norm to be considered a valid point. If 0 (defaul), this filtering is bypassed.

        @return: a Boolean list related to the validity of the indices of good points from set
        '''
        valid_indices = np.ones(shape = (xyz_points_wrt_C.shape[:-1]), dtype = "bool")
        if min_3D_range > 0 or max_3D_range > 0:
            norm_of_3D_points = np.linalg.norm(xyz_points_wrt_C, axis = -1)

            if min_3D_range > 0:
                valid_min_ranges = np.where(norm_of_3D_points >= min_3D_range, True, False)
                valid_indices = np.logical_and(valid_indices, valid_min_ranges)

            if max_3D_range > 0:
                valid_max_ranges = np.where(norm_of_3D_points <= max_3D_range, True, False)
                valid_indices = np.logical_and(valid_indices, valid_max_ranges)

        return valid_indices

    def get_triangulated_point_from_direction_angles(self, dir_angs_top, dir_angs_bot, use_midpoint_triangulation = False):
        '''
        @return: the homogeneous coordinates of the triangulated points
        '''
        az1, el1 = dir_angs_top
        az2, el2 = dir_angs_bot
        # Using the common perpendicular midpoint method (vectorized)
        if use_midpoint_triangulation:
            # We need to extract direction vectors from the given direction angles:
            # Assuming unit cilinder
            x1 = np.cos(az1)
            y1 = np.sin(az1)
            z1 = np.tan(el1)
            direction_vectors_top = np.dstack((x1, y1, z1))
            x2 = np.cos(az2)
            y2 = np.sin(az2)
            z2 = np.tan(el2)
            direction_vectors_bottom = np.dstack((x2, y2, z2))
            #===================================================================
            # rows = direction_vectors_top.shape[0]
            # cols = direction_vectors_top.shape[1]
            # triangulated_points = np.ndarray((rows, cols, 3))
            # # PAST: Was using a loop!. It's now better implemented using the common perpendicular midpoint method (vectorized)
            # for row in range(rows):
            #     for col in range(cols):
            #         mid_Pw, _, _, _ = self.get_triangulated_midpoint(direction_vectors_top[row, col], direction_vectors_bottom[row, col])
            #         triangulated_points[row, col] = mid_Pw
            #===================================================================
            #===================================================================
            triangulated_points, _, _, _ = self.get_triangulated_midpoint(direction_vectors_top, direction_vectors_bottom)
            #===================================================================
        else:
#             triangulated_points = self.get_triangulated_point_wrt_Oc(el1, el2, (az1 + az2) / 2.0)
            triangulated_points = self.get_triangulated_point_wrt_Oc(el1, el2, (az1 + az2) / 2.0)
#             triangulated_points = np.where(pano_disparities[..., np.newaxis] > 0, triangulated_points_original[0], -1.0 * triangulated_points_original[0])
#             triangulated_points = triangulated_points_original[0, pano_disparities > 0]

        # Append 1's for homogeneous coordinates
        ones_matrix = np.ones(triangulated_points.shape[:-1])
        points_3D_wrt_C_homo = np.dstack((triangulated_points, ones_matrix))  # Put back the ones for the homogeneous coordinates

        return points_3D_wrt_C_homo

    def get_stereo_ROI_nearest_vertices(self):
        '''
        @return: The 3 near bounding points (namely, \f${P}_{ns_{low}},{P}_{ns_{mid}}, {P}_{ns_{high}}$\f) for the stereo ROI
        '''
        mirror1 = self.top_model
        mirror2 = self.bot_model

        Pns_low = self.get_triangulated_point_wrt_Oc(mirror1.lowest_elevation_angle, mirror2.lowest_elevation_angle, 0)
        Pns_mid = self.get_triangulated_point_wrt_Oc(mirror1.lowest_elevation_angle, mirror2.highest_elevation_angle, 0)
        Pns_high = self.get_triangulated_point_wrt_Oc(mirror1.highest_elevation_angle, mirror2.highest_elevation_angle, 0)

        return Pns_low, Pns_mid, Pns_high

    def get_far_phony_depth_from_panoramas(self, only_valid_points = True):
        '''
        NOT GOOD: Just dealing with rays that don't converge in the front (but in the back).
        @return: the ndarray of \f$ \rho_w$\f for all ray intersection with only 1 pixel disparity (almost parallel rays, so they are far apart)
        @note: This is not an ideal way to compute the possible way for true pixel disparity (since that should only happen within the warped omnidirectional images)
        '''
        elevations_bot = self.bot_model.panorama.get_all_elevations(validate = True)[..., :-1]  # Pick from index 0 but not the last
        elevations_top = self.top_model.panorama.get_all_elevations(validate = True)[..., 1:]  # Pick from index 1 to the last
        azimuths_null = np.zeros_like(elevations_top)
        triangulated_points = self.get_triangulated_point_wrt_Oc(elevations_top, elevations_bot, azimuths_null)

        if only_valid_points:
            return triangulated_points[0, self.top_model.panorama.get_row_limits()[0]:self.bot_model.panorama.get_row_limits()[1]]
        else:
            return triangulated_points

    def get_depth_map_from_panoramas(self, method = "sgbm", use_cropped_panoramas = False, rows_roi = [], cols_roi = [], show = True, load_stereo_tuner_from_pickle = False, stereo_tuner_filename = "stereo_tuner.pkl", tune_live = False, use_heat_map = True, stereo_matcher_tuner = None):
        '''
        @param tune_live: Allows continuous frames to come (as for movies) so tuning can be perform live. When False, the tuning is attempted (until Esc is pressed)
        '''

        import cv2
        from omnistereo.common_cv import StereoMatchTuner

        # Rotate images counter-clockwise to do horizontal stereo
        if method == "bm":
            # Top panorama is the reference image (or thought as left image)
            if self.top_model.panorama.panoramic_img.ndim == 3:
                left8 = cv2.cvtColor(self.top_model.panorama.panoramic_img, cv2.COLOR_BGR2GRAY)
            else:
                left8 = self.top_model.panorama.panoramic_img.copy()

            # Bottom panorama is the target image (or thought as left image)
            if self.bot_model.panorama.panoramic_img.ndim == 3:
                right8 = cv2.cvtColor(self.bot_model.panorama.panoramic_img, cv2.COLOR_BGR2GRAY)
            else:
                right8 = self.bot_model.panorama.panoramic_img.copy()
        else:
            left8 = self.top_model.panorama.panoramic_img.copy()
            right8 = self.bot_model.panorama.panoramic_img.copy()

        # Initialize zeroed disparity map
        pano_rows = int(self.top_model.panorama.rows)
        pano_cols = int(self.top_model.panorama.cols)

        # NOTE: we choose either panorama due to rectification assumption as row-to-elevation mapping used in the following function
        row_common_highest = self.top_model.panorama.get_panorama_row_from_elevation(self.common_highest_elevation_angle)
        row_common_lowest = self.top_model.panorama.get_panorama_row_from_elevation(self.common_lowest_elevation_angle)

        if len(cols_roi) == 1:
            # Ambiguous:
            left = left8[:, cols_roi[0]:]
            right = right8[:, cols_roi[0]:]
        elif len(cols_roi) == 2:
            left = left8[:, cols_roi[0]:cols_roi[1]]
            right = right8[:, cols_roi[0]:cols_roi[1]]
        else:
            left = left8
            right = right8

        if show:
            self.draw_elevations_on_panoramas(left, right)  # Uncropped

        if use_cropped_panoramas:
            # FIXME: crop it without destroying the left and right
            # mask = common_cv.create_rectangular_mask(img_input=right, points=[(0, row_common_highest), (self.top_model.panorama.width - 1, row_common_lowest)], preview=True)
            right = right[row_common_highest:row_common_lowest + 1]  # Recall that range doesn't include the last index (so we want to include it with +1)
            left = left[row_common_highest:row_common_lowest + 1]

        disp_img_win = 'Panoramic Disparity Map'
        # NOTE: the disparity map is linked to the left/top image so it is the reference for the correspondences
        #       Thus, the coordinates of the correspondence on the right/bottom image is found such that m_right = (u_left, v_left - disp[u,v])

        self.disparity_map = np.zeros(left8.shape[0:2], dtype = "float64")
        self.disparity_img = np.zeros(left8.shape[0:2], dtype = "uint8")

        if stereo_matcher_tuner is None:
            import os.path as osp
            file_exists = osp.isfile(stereo_tuner_filename)
            if (not file_exists) or (not load_stereo_tuner_from_pickle):
                stereo_matcher_tuner = StereoMatchTuner(left_img = left, right_img = right, rotate_images = True, method = method, win_name = disp_img_win, disp_first_valid_row = row_common_highest, disp_last_valid_row = row_common_lowest)
            else:
                from omnistereo.common_tools import load_obj_from_pickle
                stereo_matcher_tuner = load_obj_from_pickle(stereo_tuner_filename)
                stereo_matcher_tuner.reset_images(left_img = left, right_img = right, rotate_images = True, disp_first_valid_row = row_common_highest, disp_last_valid_row = row_common_lowest)
        else:
            stereo_matcher_tuner.reset_images(left_img = left, right_img = right, rotate_images = True, disp_first_valid_row = row_common_highest, disp_last_valid_row = row_common_lowest)

        # (my TRICK) Filter out of bound values by applying panoramic mask to the disparity image and depth map using radial bounds:
        white_blank_img = np.zeros_like(self.current_omni_img) + 255
        top_circ_mask, bot_circular_mask = self.get_fully_masked_images(omni_img = white_blank_img, view = False, color_RGB = None)
        # cv2.imshow("OMNI MASK (Reference)", top_circ_mask)
        # Recall, we need a single channel mask
        pano_mask = self.top_model.panorama.get_panoramic_image(top_circ_mask[..., 0], set_own = False)  # Using the left/top image as reference
        # cv2.imshow("PANORAMA MASK (Reference)", pano_mask)
        disp_map, disp_img = stereo_matcher_tuner.start_tuning(win_name = disp_img_win, save_file_name = stereo_tuner_filename, tune_live = tune_live, pano_mask = pano_mask, use_heat_map = use_heat_map)

        # Merge disparity results from plausible ROI maps
        if len(rows_roi) == 0:
            if len(cols_roi) == 1:
                # Ambiguous:
                self.disparity_map[:, cols_roi[0]:] = disp_map
                self.disparity_img[:, cols_roi[0]:] = disp_img
            elif len(cols_roi) == 2:
                self.disparity_map[:, cols_roi[0]:cols_roi[1]] = disp_map
                self.disparity_img[:, cols_roi[0]:cols_roi[1]] = disp_img
            else:
                self.disparity_map = disp_map
                self.disparity_img = disp_img
        elif len(rows_roi) == 1:  # Max row case
            if len(cols_roi) == 1:
                # Ambiguous:
                self.disparity_map[:rows_roi[0], cols_roi[0]:] = disp_map[:rows_roi[0]]
                self.disparity_img[:rows_roi[0], cols_roi[0]:] = disp_img[:rows_roi[0]]
            elif len(cols_roi) == 2:
                self.disparity_map[:rows_roi[0], cols_roi[0]:cols_roi[1]] = disp_map[:rows_roi[0]]
                self.disparity_img[:rows_roi[0], cols_roi[0]:cols_roi[1]] = disp_img[:rows_roi[0]]
            else:
                self.disparity_map[:rows_roi[0]] = disp_map[:rows_roi[0]]
                self.disparity_img[:rows_roi[0]] = disp_img[:rows_roi[0]]

