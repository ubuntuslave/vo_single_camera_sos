# -*- coding: utf-8 -*-
# panorama.py

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
Tools for generating panoramic images from an omnidirectional view

@author: Carlos Jaramillo
@contact: cjaramillo@gradcenter.cuny.edu
'''

from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
import sys
from time import process_time

class Panorama(object):
    '''
    Computes a panorama mapping by lifting points from a distorted omnidirectional image.
    '''

    def __init__(self, projection_model, **kwargs):
        '''
        Constructor for generating a look-up table (LUT) that will be employed for mapping a corresponding omnidrectional image
        to a cylindric panorama.
        @param projection_model: The omnistereo projection used, such as the Generalized Uniform Model (GUM) obtained from the pre-calibration stage.
        '''

        self.model = projection_model
        self.name = projection_model.mirror_name + " " + __name__

        # The current distorted omnidirectional image at a given time
        self.omni_img = None
        self.panoramic_img = None  # The associated Panoramic image to the omni_img
        self.psi_LUT = None  # LUT for azimuthal angles as a 1D numpy array (it wraps around 2*pi when computed)
        self.theta_LUT = None  # LUT for elevation angles (as 1D numpy array)
        self.theta_LUT_validated = None  # LUT for elevation angles (as 1D numpy array) with np.nan for those elavations out of view bounds
        self.valid_omni_pixel_coords = None
        self.world2cam_LUT_map_y = None  # Used when remapping operates under float-point precision
        self.world2cam_LUT_map_x = None  # Used when remapping operates under float-point precision
        self.world2cam_LUT_map_xy_fixed = None  # Used when remapping operates under fixed-point precision
        self.world2cam_LUT_interpolation = None  # Used when remapping operates under fixed-point precision
        self.height = None  # Exact float number
        self.rows = None  # A whole number (the ceiling approximation of self.height
        self.width = None  # Exact float number
        self.cols = None  # A whole number (the ceiling approximation of self.width
        self.width_is_correct = False  # Initially, width dimension is assumed to be incorrect
        self.height_is_correct = False  # Initially, height dimension is assumed to be incorrect
        self.cyl_radius = 1.0
        self.globally_highest_elevation_angle = self.model.globally_highest_elevation_angle
        self.globally_lowest_elevation_angle = self.model.globally_lowest_elevation_angle
        self.cyl_height_max = None
        self.z_height_min = None
        self.cyl_height = None
        self.pixel_size = 1.0  # To be changed if width is passed as an argument
        self._set_cylinder_height()  # Initial height setup (important when height is given a priori)
        self.azimuthal_masks = []  # A list of masks used during feature detection and matching
        self.set_panorama_dimensions(**kwargs)

    def set_panorama_dimensions(self, **kwargs):
        '''
        Sets other relevant parameters used during Panorama generation via remapping
        @param kwargs: Dictionary arguments for
                       height, width, interpolation, border_method, azimuthal_shift = 0.):
        '''
        # Arbitrary elevation limits
        self.interpolation_method = kwargs.get("interpolation", cv2.INTER_LINEAR)
        self.border_method = kwargs.get("border_method", cv2.BORDER_CONSTANT)
        # Angular shift on psi (azimuthal)
        self.azimuthal_shift = kwargs.get("azimuthal_shift", 0)

        # Make a more educated guess as the correct aspect ratio based on square pattern points if calibration data exists in the GUM
        # if kwargs.has_key("width"):  # Give priority to width dimension on the setup
        # In fact has_key() was removed in Python 3.x.
        if "width" in kwargs.keys():
            self.width_is_correct = True
            self.height_is_correct = False
        # elif kwargs.has_key("height"):
        elif "height" in kwargs.keys():
            self.width_is_correct = False
            self.height_is_correct = True

        if self.width_is_correct == False or self.height_is_correct == False:
            self.width = kwargs.get("width", 600)
            self.height = kwargs.get("height", 100)
            self.rows = int(np.ceil(self.height))
            self.cols = int(np.ceil(self.width))
            size_resolution_method = kwargs.get("size_resolution_method", "")
            self.regenerate_LUTs(size_resolution_method)

    def print_settings(self):
        if (self.cols is not None) and (self.rows is not None):
            print("Panorama size: %d x %d. With pixel size = %f m" % (self.cols, self.rows, self.pixel_size))
        else:
            print("Panorama size not set yet!")

        print("Globally highest elevation angle: %f degrees" % np.rad2deg(self.globally_highest_elevation_angle))
        print("Globally lowest elevation angle: %f degrees" % np.rad2deg(self.globally_lowest_elevation_angle))
        print("Azimuthal shift = %f degrees" % (np.rad2deg(self.azimuthal_shift)))
        print("Z' max = %f,  Z' min = %f" % (self.cyl_height_max, self.z_height_min))
        # print("Interpolation method: %s" % self.interpolation_method)
        # print("Border method: %s" % self.border_method)

    def set_panoramic_image(self, omni_img, idx = -1, view = True, win_name_modifier = "", border_RGB_color = None):
        pano_img = self.get_panoramic_image(omni_img, set_own = True, border_RGB_color = border_RGB_color)
        if view:
            pano_win_name = win_name_modifier + "%s [%d]" % (self.name, idx)
            cv2.namedWindow(pano_win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(pano_win_name, pano_img)
            cv2.waitKey(1)
        return pano_img

    def _set_cylinder_height(self):
        # Set interval sizes:
        self.cyl_height_max = self.cyl_radius * np.tan(self.globally_highest_elevation_angle)
        self.z_height_min = self.cyl_radius * np.tan(self.globally_lowest_elevation_angle)
        self.cyl_height = self.cyl_height_max - self.z_height_min

    def _resolve_dimensions_pixel_sizing(self):
        '''
        @brief A custom method for resolving the panoramic image dimensions using the information on the physical pixel size of the panoramic image
        '''
        self.cyl_circumference = 2 * np.pi * self.cyl_radius
        if self.height_is_correct == False:
            # 1) Find the pixel size on the cylindrical panorama based on cylinder radius and number of columns (width of the panoramic image)
            self.pixel_size = self.cyl_circumference / float(self.cols)
            # 2) We set square pixels, so the same pixel size applies along the height.
            # 3) Use the global minimum and maximum elevations to infer the physical height (and rows) of the cylinder
            self.height = self.cyl_height / self.pixel_size
            self.rows = int(np.ceil(self.height))
            self.height_is_correct = True
        if self.width_is_correct == False:
            # 1) Find the pixel size on the cylindrical panorama based on cylinder physical height and desired number rows (height of panoramic image)
            self.pixel_size = self.cyl_height / float(self.rows)
            # 2) We set square pixels, so the same pixel size applies along the height.
            # 3) Use the desired circumference for the cylinder to compute the number of columns (or image width)
            self.width = self.cyl_circumference / self.pixel_size
            self.cols = int(np.ceil(self.width))
            self.height_is_correct = True

            self.width_is_correct = True

        self.aspect_ratio = float(self.cols) / float(self.rows)

    def _resolve_dimensions_linear_interpolation(self):
        '''
        @brief Simple method for resolving the panoramic image dimensions using linear interpolation (based on Spacek's 2003 paper)
        '''
        self.aspect_ratio = 2 * np.pi
        if self.height_is_correct == False:
            self.height = self.width / self.aspect_ratio
            self.height_is_correct = True
        if self.width_is_correct == False:
            self.width = self.height * self.aspect_ratio
            self.width_is_correct = True

#===============================================================================
#     def _resolve_dimensions_square_guided(self):
#         '''
#         @brief Resolve the correct aspect ratio for the panorama dimensions
#         Computes the most appropriate panorama's aspect ratio from existing corners during calibration (if any).
#         Otherwise, the user is prompt to manually draw an square around a know a square object.
#         The \f$L_2\f$ distance between some adjacent corners on the \f$x\f$ and \f$y\f$ directions on the panorama.
#         - If the width, \f$w\f$, of the panorama is given in pixels, then the corresponding scale factor \f$\lambda_h\f$ for an arbitrary initial height, \f$h\f$, such
#         \f$d_x = \lambda_h d_y\f$, where \f$d_x\f$ and \f$d_y\f$ are the corresponding pixel dimensions of the square. Hence, \f$\lambda_h = \frac{d_x}{d_y}\f$,
#         so the corrected height is computed by \f$w_{correct} = \lambda_w * w_{incorrect}\f$
#         - Similarly, if the height, \f$h\f$, is given, we can solve for \f$\lambda_w\f$ on an arbitrary initial width, \f$w\f$, from \f$d_y = \lambda_w d_x\f$.
#         Thus, \f$\lambda_w = \frac{d_y}{d_x}\f$, so the corrected width is computed by \f$w_{correct} = \lambda_w * w_{incorrect}\f$
#
#         @note: If object is not perfectly square, it may generate squashed looking aspect ratios
#         '''
#         # Get dx and dy from any set of adjacent chessboard corners
#         # Resolve the correct aspect ratio for the panorama's dimensions
#
#         if self.model.calibrator is not None and len(self.model.calibrator.omni_monos) > 0:
#             # Grab points from the first pattern
#             img_pts = self.model.calibrator.omni_monos[0].image_points
#             # Note that the shape of img_pts is (rows, cols, 2 to store (u, v))
#             p1_hor = euclid.Point2(img_pts[0, 0, 0], img_pts[0, 0, 1])
#             print("p1_hor: %s" % p1_hor)
#             p2_hor = euclid.Point2(img_pts[0, 1, 0], img_pts[0, 1, 1])
#             print("p2_hor: %s" % p2_hor)
#             p1_ver = euclid.Point2(img_pts[0, 0, 0], img_pts[0, 0, 1])
#             print("p1_ver: %s" % p1_ver)
#             p2_ver = euclid.Point2(img_pts[1, 0, 0], img_pts[1, 0, 1])
#             print("p2_ver: %s" % p2_ver)
#
#             dx, dy = 1, 1  # Initial values to produce an aspect ratio of 1 in case of invalid points
#             m1_hor = np.dstack((p1_hor.x, p1_hor.y))
#             is_valid_p1_hor, u_pano, v_pano = self.get_panoramic_pixel_coords_from_omni_pixel(m1_hor)
#             if is_valid_p1_hor:
#                 p1_pano_hor = euclid.Point2(u_pano, v_pano)
#
#             m2_hor = np.dstack((p2_hor.x, p2_hor.y))
#             is_valid_p2_hor, u_pano, v_pano = self.get_panoramic_pixel_coords_from_omni_pixel(m2_hor)
#             if is_valid_p2_hor:
#                 p2_pano_hor = euclid.Point2(u_pano, v_pano)
#
#             if is_valid_p1_hor and is_valid_p2_hor:
#                 dx = p1_pano_hor.distance(p2_pano_hor)
#
#             m1_ver = np.dstack((p1_ver.x, p1_ver.y))
#             is_valid_p1_ver, u_pano, v_pano = self.get_panoramic_pixel_coords_from_omni_pixel(m1_ver)
#             if is_valid_p1_ver:
#                 p1_pano_ver = euclid.Point2(u_pano, v_pano)
#
#             m2_ver = np.dstack((p2_ver.x, p2_ver.y))
#             is_valid_p2_ver, u_pano, v_pano = self.get_panoramic_pixel_coords_from_omni_pixel(m2_ver)
#             if is_valid_p2_ver:
#                 p2_pano_ver = euclid.Point2(u_pano, v_pano)
#
#             if is_valid_p1_ver and is_valid_p2_ver:
#                 dy = p1_pano_ver.distance(p2_pano_ver)
#
#             if self.height_is_correct == False:
#                 lambda_h = dx / dy
#                 self.height = lambda_h * self.height  # Uses the previous height value
#                 self.height_is_correct = True
#             if self.width_is_correct == False:
#                 lambda_w = dy / dx
#                 self.width = lambda_w * self.width  # Uses the previous width value
#                 self.width_is_correct = True
#
#         else:
#             print("Not square points from calibration found. Select 2 orthogonal line segments manually")
#             # WISH: allow user to enclose a rectangle
#===============================================================================

    def get_panoramic_image(self, input_omni_img, set_own = True, crop_out_bottom = False, border_RGB_color = None, use_floating_point_prec = True):
        '''
        @param input_omni_img: The distorted omnidirectional input image.
        @param crop_out_bottom: Indicates whether cropping according to computed ROI should be done.

        @return The panoramic image representation of the input_omni_img
        '''
        if border_RGB_color is None:
            border_color = (0, 0, 0)  # BLACK background
        else:
            from omnistereo.common_cv import rgb2bgr_color
            # Swap the passed color from RGB into BGR
            # border_color = (border_RGB_color[2], border_RGB_color[1], border_RGB_color[0])
            border_color = rgb2bgr_color(border_RGB_color)
        try:
            # WISH: Use a fixed-point map made by using convertMaps()
            # The reason you might want to convert from floating to fixed-point representations of a map is that they can yield much faster (~2x) remapping operations. In the converted case,  contains pairs (cvFloor(x), cvFloor(y)) and  contains indices in a table of interpolation coefficients.
            if set_own:
                self.omni_img = input_omni_img
            self.color_channels = input_omni_img.ndim
            if self.color_channels < 3:
                panoramic_img = np.zeros((self.rows, self.cols))  # Initialize image with zeros
            else:
                panoramic_img = np.ones((self.rows, self.cols, self.color_channels), input_omni_img.dtype) * np.array(border_color, dtype = input_omni_img.dtype)  # Apply background color
                # FIXME2: Is green a good color to be ignored (used as non existing background?) Smarter way would be to scan those pixels on the rim and use a color not in that set.
#                 self.panoramic_img[:, :, 1] += 192  # Paint Green channel: Initialize image with green background
#                 self.panoramic_img[:, :, :] += 255  # Paint WHITE background

            #===================================================================
            # start_time_remap = process_time()
            #===================================================================
            if use_floating_point_prec:
                prec_str = "Floating-point"
                map_x_32 = self.world2cam_LUT_map_x.astype('float32')  # NOTE: the float32 type
                map_y_32 = self.world2cam_LUT_map_y.astype('float32')  # NOTE: the float32 type
                panoramic_img = cv2.remap(input_omni_img, map_x_32, map_y_32,
                                               self.interpolation_method,
                                               self.panoramic_img,  # dst
                                               self.border_method,  # the pixels in the destination image that corresponds to the "outliers" in the source image are not modified by the function
                                               border_color  # borderValue
                                               )
            else:
                prec_str = "Fixed-point"
                panoramic_img = cv2.remap(input_omni_img, self.world2cam_LUT_map_xy_fixed, self.world2cam_LUT_interpolation,
                                               self.interpolation_method,
                                               self.panoramic_img,  # dst
                                               self.border_method,  # the pixels in the destination image that corresponds to the "outliers" in the source image are not modified by the function
                                               border_color  # borderValue
                                               )
            #===================================================================
            # end_time_remap = process_time()
            # time_ellapsed_remap = end_time_remap - start_time_remap
            # print("Panorama remapping ({prec}) - Time elapsed: {time:.8f} seconds".format(time=time_ellapsed_remap, prec=prec_str))
            #===================================================================

            if crop_out_bottom:
                # WISH: Crop panorama by slicing it as row (or y):y+height_offset, col (or x), x+width_offset
                panoramic_img = panoramic_img[0:self.width, 0:self.height, :]
            if set_own:
                self.panoramic_img = panoramic_img
            return panoramic_img
        except:
            print("Problem! Exiting from" , __name__)
            sys.exit(1)

    def get_point_on_cylinder_from_omni_pixel(self, m_omni, radius_cyl_pan = 1.0):
        '''
        Computes the 3D position in the panoramic cylinder for the given pixel (u,v) coordinates on the distorted (warped) image.
        @param m_omni: A numpy array of k image omnidirectionsl pixel coordinates [u, v] as row vector. Thus, shape is (rows, cols, 2)
        @param radius_cyl_pan: The radius of the cylinder, whose default radius is the unit.
        @retval: X, Y, Z: The corresponding 3D coordinates for the lifted pixed (u,v) to the cylindrical panorama. Coordinates are w.r.t. the mirror's focus.

        '''
        # NOTE: the commented out function lifts the pixel
        # psi, theta = self.model.get_direction_angles_from_pixel(m_omni)
        psi, theta = self.model.get_direction_angles_from_pixel_using_forward_projection(m_omni)
        X = radius_cyl_pan * np.cos(psi)
        Y = radius_cyl_pan * np.sin(psi)
        Z = radius_cyl_pan * np.tan(theta)
        return X, Y, Z

    def get_point_on_cylinder_from_pano_pixel(self, row, col, radius_cyl_pan = 1.0):
        '''
        Computes the 3D position in the panoramic cylinder for the given panoramic pixel (u,v) or (row, col) coordinates on the distorted (warped) image.
        @param row: The row (or u-coordinate) of the pixel on the panoramic image
        @param col: The col (or v-coordinate) of the pixel on the panoramic image
        @param radius_cyl_pan: The radius of the cylinder, whose default radius is the unit.
        @retval: X, Y, Z: The corresponding 3D coordinates on the cylindrical panorama. Coordinates are w.r.t. the mirror's focus.

        '''
        theta = self.get_elevation_from_panorama_row(row)
        psi = self.get_azimuth_from_panorama_col(col)

        X = radius_cyl_pan * np.cos(psi)
        Y = radius_cyl_pan * np.sin(psi)
        Z = radius_cyl_pan * np.tan(theta)
        return X, Y, Z

    def get_points_on_cylinder(self, radius_cyl_pan = 1.0):
        row_indices = np.arange(self.rows, dtype = np.int64)
        col_indices = np.arange(self.cols, dtype = np.int64)
        theta = self.get_elevation_from_panorama_row(row_indices)
        psi = self.get_azimuth_from_panorama_col(col_indices)
        x_cyl = radius_cyl_pan * np.outer(np.ones(self.rows), np.cos(psi))
        y_cyl = radius_cyl_pan * np.outer(np.ones(self.rows), np.sin(psi))
        z_cyl = np.outer(np.tan(theta), np.ones(self.cols))

        cyl_points = np.dstack((x_cyl, y_cyl, z_cyl))
        return cyl_points

    def get_panoramic_pixel_coords_from_omni_pixel(self, m_omni, verbose_optimization = False):
        '''
        @brief Computes the pixel coordinates on the panoramic image that maps to the given pixel (u,v) coordinates on the distorted image.

        @param m_omni: A numpy array of k image point coordinates [u, v] as row vector. Thus, shape is (rows, cols, 2)
        @retval: is_valid, u_pano, v_pano: The corresponding pixel on the panoramic image for the lifted pixel coordinates (u,v).
        '''
        # NOTE: the commented out function lifts the pixel
        # azimuth, elevation = self.model.get_direction_angles_from_pixel(m_omni)
        azimuth, elevation = self.model.get_direction_angles_from_pixel_using_forward_projection(m_omni, individual_opt = True, verbose_optimization = verbose_optimization)

        u_pano, v_pano, m_pano = self.get_panorama_pixel_coords_from_direction_angles(theta = elevation, psi = azimuth)
        # u_pano = self.get_panorama_col_from_azimuth(azimuth)
        # v_pano = self.get_panorama_row_from_elevation(elevation)
        # if np.isnan(u_pano) or np.isnan(v_pano):
        #    is_valid = False
        # else:
        #    is_valid = True

        is_valid = np.where(np.isnan(m_pano), False, True)

        return is_valid, m_pano

    def get_omni_pixel_coords_from_panoramic_pixel(self, m_pano, use_LUTs = False):
        '''
        @brief Computes the corresponding pixel coordinates on the omnidirectional image that maps to the given pixel (u,v) coordinates on the panoramic image.

        @param m_pano: A numpy array of k image point coordinates [u, v] as row vector. Thus, shape is (rows, cols, 2)
        @param use_LUTs: boolean flag that indicates whether to use LUTs to obtain angular information (Warning: LUTs can introduce discretization error)

        @retval: u_omni: The numpy array of u-coordinates for the pixel on the omnidirectional image
        @retval: v_omni: The numpy array of v-coordinates for the pixel on the omnidirectional image
        @retval: m_omni: The corresponding table of pixels (u,v) coordinates on the omnidirectional image
        '''
        if use_LUTs:
            u_omni = self.world2cam_LUT_map_x[m_pano[..., 1].astype('uint'), m_pano[..., 0].astype('uint')]
            v_omni = self.world2cam_LUT_map_y[m_pano[..., 1].astype('uint'), m_pano[..., 0].astype('uint')]
            m_omni = np.ones(shape = (m_pano.shape[:2] + (3,)), dtype = m_pano.dtype)
            m_omni[..., 0] = u_omni
            m_omni[..., 1] = v_omni
        else:
            azimuth, elevation = self.get_direction_angles_from_pixel_pano(m_pano, use_LUTs = use_LUTs)
            u_omni, v_omni, m_omni = self.model.get_pixel_from_direction_angles(azimuth, elevation)

        return u_omni, v_omni, m_omni

    def _generate_LUTs(self, lift_points_from_omni_image = False):
        '''
        Compute the x and y coordinates Look-up tables (LUTs) for mapping from the desired panorama (world) to the omnidirectional (distorted) image

        @param lift_points_from_omni_image: If specified, the LUTs will be generated by lifting points from the omnidirectional images. Otherwise, a forward-projective mapping to fill each panoramic pixel will happen.
        '''
        print("Please, wait. Generating LUTs ...", end = "")

        # Find the rows limits without using LUTs yet
        #=======================================================================
        # row_for_highest_elevation = self.get_panorama_row_from_elevation(self.model.highest_elevation_angle, use_LUT=False)
        # row_for_lowest_elevation = self.get_panorama_row_from_elevation(self.model.lowest_elevation_angle, use_LUT=False)
        #=======================================================================
        from omnistereo.common_tools import reverse_axis_elems
        # Note: the angles from the omnidirectional image should grow on clockwise rotation
        self.psi_LUT = reverse_axis_elems(np.linspace(0, 2 * np.pi, num = self.cols, endpoint = False))  # WISH: + (self.azimuthal_shift % np.pi)
        # Convert the above 1D array to a 2D array (so it looks like an image or table)
        self.psi_LUT_2D = np.zeros((self.rows, self.cols), dtype = 'float32') + self.psi_LUT.astype('float32')
        cyl_height_LUT = np.linspace(self.cyl_height_max, self.z_height_min, num = self.rows, endpoint = False)
        # when elevation is out of bounds, a "nan" value is assigned in the LUT
        # thetas = np.arctan2(cyl_height_LUT, self.cyl_radius)
        # self.theta_LUT = np.where(np.logical_and(self.model.lowest_elevation_angle <= thetas, thetas <= self.model.highest_elevation_angle), thetas, np.nan)
        # Instead: fill all elevations:
        self.theta_LUT = np.arctan2(cyl_height_LUT, self.cyl_radius)
        self.theta_LUT_validated = np.where(np.logical_and(self.model.lowest_elevation_angle <= self.theta_LUT, self.theta_LUT <= self.model.highest_elevation_angle), self.theta_LUT, np.nan)

        # Convert the above 1D array to a 2D array (so it looks like an image or table)
        # self.theta_LUT_2D = np.transpose(np.zeros((self.cols, self.rows), dtype='float32') + self.theta_LUT.astype('float32')) # Without validation
        self.theta_LUT_2D = np.zeros((self.rows, self.cols), dtype = 'float32') + self.theta_LUT_validated[..., np.newaxis].astype('float32')  # Validated elevations

        if lift_points_from_omni_image:
            # TODO: implement
            if self.model.mask is not None:
                #===============================================================
                # cv2.namedWindow("Viz Omni Mask", cv2.WINDOW_NORMAL)
                # cv2.imshow("Viz Omni Mask", self.model.mask)
                # cv2.waitKey(0)
                #===============================================================
                v_omni, u_omni = np.nonzero(self.model.mask)  # v <-- row, u <-- col
                self.valid_omni_pixel_coords = np.dstack((u_omni, v_omni))  # , np.ones_like(u)))
                validity, valid_points_on_pano = self.get_panoramic_pixel_coords_from_omni_pixel(self.valid_omni_pixel_coords, verbose_optimization = True)
                self.world2cam_LUT_map_x = np.zeros_like(self.theta_LUT_2D) + np.nan  # Make all nan at first
                self.world2cam_LUT_map_y = np.zeros_like(self.theta_LUT_2D) + np.nan  # Make all nan at first
                # We are going to compute an average coordinate, so we need to keep track of the counter
                world2cam_LUT_sum_counter = np.zeros_like(self.theta_LUT_2D) + np.nan  # Make all nan at first
                # WISHME: Improve the performance by not using a loop. The "unique" function doesn't apply because its uniqueness is only considered on the first element
                for idx in range(len(u_omni)):
                    pano_row_candidate = valid_points_on_pano[0, idx, 1]
                    pano_col_candidate = valid_points_on_pano[0, idx, 0]
                    if np.isnan(self.world2cam_LUT_map_x[pano_row_candidate, pano_col_candidate]):
                        # assign the first time if writes to this panoramic coordinates:
                        self.world2cam_LUT_map_x[pano_row_candidate, pano_col_candidate] = u_omni[idx]  # put omni u-coord
                        self.world2cam_LUT_map_y[pano_row_candidate, pano_col_candidate] = v_omni[idx]  # put omni u-coord
                        world2cam_LUT_sum_counter[pano_row_candidate, pano_col_candidate] = 1
                    else:  # accumulate
                        self.world2cam_LUT_map_x[pano_row_candidate, pano_col_candidate] += u_omni[idx]  # add omni u-coord
                        self.world2cam_LUT_map_y[pano_row_candidate, pano_col_candidate] += v_omni[idx]  # add omni u-coord
                        world2cam_LUT_sum_counter[pano_row_candidate, pano_col_candidate] += 1
                # Compute averages:
                self.world2cam_LUT_map_x = self.world2cam_LUT_map_x / world2cam_LUT_sum_counter
                self.world2cam_LUT_map_y = self.world2cam_LUT_map_y / world2cam_LUT_sum_counter
            else:
                pass  # TODO: For now, we assume mask always exists
        else:
            self.world2cam_LUT_map_x, self.world2cam_LUT_map_y, _ = self.model.get_pixel_from_direction_angles(self.psi_LUT_2D, self.theta_LUT_2D)

        # Compute this for fixed-point precision (speed up).
        # NOTE: I measured the time difference without noticing much of an advantage!
        self.world2cam_LUT_map_xy_fixed, self.world2cam_LUT_interpolation = cv2.convertMaps(self.world2cam_LUT_map_x.astype(np.float32), self.world2cam_LUT_map_y.astype(np.float32), cv2.CV_16SC2)

        print("done!")
        # np.count_nonzero(~np.isnan(self.world2cam_LUT_map_x))
        #=======================================================================
        # test = np.where(np.isnan(self.world2cam_LUT_map_x), 0, 255)
        # test = test.astype("uint8")
        # cv2.namedWindow("TEST")
        # cv2.imshow("TEST", test)
        # cv2.waitKey(0)
        #=======================================================================

    def regenerate_LUTs(self, method = "pixel sizing"):
        '''
        Will regenerate the LUTs. Useful when dimensions have changed or for debugging.

        @param method: The method used for resolving the panorama aspect ratio.
               Currently, implemented methods are: "pixel sizing" (DEFAULT)  | "square guided" | "linear interpolation"
        '''
        self._generate_LUTs()  # Generate some initial LUTs to attempt the correction next.
        # Resolve the correct aspect ratio for the panorama's dimensions
        if self.width_is_correct or self.height_is_correct:
            if method == "pixel sizing" or method == "":
                self._resolve_dimensions_pixel_sizing()
            #===================================================================
            # # Not longer needed because it uses Euclid and also the signature for get_panoramic_pixel_coords_from_omni_pixel changed
            # elif method == "square guided":
            #     self._resolve_dimensions_square_guided()
            #===================================================================
            elif method == "linear interpolation":
                self._resolve_dimensions_linear_interpolation()

        if self.height_is_correct and self.width_is_correct:
            self.rows = int(np.ceil(self.height))
            self.cols = int(np.ceil(self.width))
            print("Panorama size corrected to: %d x %d" % (self.cols, self.rows))
            self._generate_LUTs()  # Re-generate the LUTs using the corrected dimensions.

    def generate_azimuthal_masks(self, azimuth_mask_degrees, overlap_degrees = 0, mask_also_on_elev = True, elev_mask_padding = 0, stand_masks_azimuth_coord_in_degrees_list = [], stand_masks_width_in_degrees = 1, show = False):
        '''
        @param stand_masks_width_in_degrees: This value will be applied half-and-half around the coordinates list provides (also in degrees)
        '''
        azimuth_mask_radians = np.deg2rad(azimuth_mask_degrees)
        overlap_radians = np.deg2rad(overlap_degrees)
        self.azimuthal_masks = []  # Clear list
        if show:
            win_names_list = []
        if mask_also_on_elev:
            if elev_mask_padding > 0:
                omni_img_mask = self.model.make_mask(mask_shape = self.model.current_omni_img.shape[0:2], radius_pixel_shrinking = elev_mask_padding)
            else:
                omni_img_mask = self.model.mask
            pano_img_mask = self.get_panoramic_image(input_omni_img = omni_img_mask, set_own = False, border_RGB_color = (0, 0, 0))

        if len(stand_masks_azimuth_coord_in_degrees_list) > 0:
            mask_for_stands = np.zeros((self.rows, self.cols), dtype = np.uint8) + 255  # White, single channel mask for the entire image as valid initially
            stand_masks_azimuth_coord_in_radians_list = np.deg2rad(stand_masks_azimuth_coord_in_degrees_list)
            stand_mask_width_half_in_radians = np.deg2rad(stand_masks_width_in_degrees / 2.0)
            for azim_coord in stand_masks_azimuth_coord_in_radians_list:
                init_angle_stand_mask = azim_coord - stand_mask_width_half_in_radians
                if init_angle_stand_mask <= 0:
                    pt1_u_coord = self.get_panorama_col_from_azimuth(azimuth = 0)
                else:
                    pt1_u_coord = self.get_panorama_col_from_azimuth(azimuth = init_angle_stand_mask)

                end_angle_stand_mask = azim_coord + stand_mask_width_half_in_radians
                if end_angle_stand_mask >= 2.0 * np.pi:
                    pt2_u_coord = 0
                else:
                    pt2_u_coord = self.get_panorama_col_from_azimuth(azimuth = (end_angle_stand_mask))

                # Paint roi to be ignored in black
                cv2.rectangle(mask_for_stands, pt1 = (pt1_u_coord, 0), pt2 = (pt2_u_coord, int(self.cols - 1)), color = (0, 0, 0), thickness = -1, lineType = 8, shift = 0)
            if show:
                cv2.imshow("Mask for stands", mask_for_stands)

        # WISH: speed up this loop
        for d in np.arange(start = 0, stop = 2 * np.pi - azimuth_mask_radians / 2., step = azimuth_mask_radians):  # WISH: add padding to masks so there is overlap on the cutting boundaries
            init_angle = d - overlap_radians
            if init_angle <= 0:
                pt1_u_coord = self.get_panorama_col_from_azimuth(azimuth = 0)
            else:
                pt1_u_coord = self.get_panorama_col_from_azimuth(azimuth = init_angle)

            end_angle = d + azimuth_mask_radians + overlap_radians
            if end_angle >= 2.0 * np.pi:
                pt2_u_coord = 0
            else:
                pt2_u_coord = self.get_panorama_col_from_azimuth(azimuth = (end_angle))
            mask = np.zeros((self.rows, self.cols), dtype = np.uint8)  # Black, single channel mask
            # Paint roi white
            cv2.rectangle(mask, pt1 = (pt1_u_coord, 0), pt2 = (pt2_u_coord, int(self.cols - 1)), color = (255, 255, 255), thickness = -1, lineType = 8, shift = 0)
            if mask_also_on_elev:
                mask = cv2.bitwise_and(src1 = mask, src2 = pano_img_mask, dst = mask, mask = mask)
            if len(stand_masks_azimuth_coord_in_degrees_list) > 0:
                mask = cv2.bitwise_and(src1 = mask, src2 = mask_for_stands, dst = mask, mask = mask)

            self.azimuthal_masks.append(mask)
            if show:
                mask_name = "Pano Mask Azimutal at {angle_deg:.2f} [degrees]".format(angle_deg = np.rad2deg(d))
                win_names_list.append(mask_name)
                cv2.namedWindow(mask_name, cv2.WINDOW_NORMAL)
                cv2.imshow(mask_name, mask)

        if show:
            cv2.waitKey(0)
            for w in win_names_list:  # Destroy these recent windows
                cv2.destroyWindow(w)

    def get_elevation_from_panorama_row(self, row):
        elevation = None  # Initialize to a certainly invalid elevation
        if isinstance(row, np.ndarray):
            # Not giving back the right shape:
            # vvvvvvvv only keeps the valid answers vvvvvvvvvv
            # row_flattened = row.flatten()
            # validation_list = np.logical_and(0 <= row_flattened, row_flattened < self.rows)
            # ans = row_flattened[validation_list]
            # elevation = self.theta_LUT.take(ans)
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            # This is keeping the shape, with nan's for invalid answers
            # FIXME: It's not keeping the right answers, but bradcasting the answer at row 0
            validation_list = np.logical_and(0 <= row, row < self.rows)
            elevation = np.zeros_like(validation_list) + np.nan  # Make all nan at first
            # and just update the value elements of the answer
            elevation[validation_list] = self.theta_LUT.take(row[validation_list].astype("int"))
            # FIXME: beware that the "astype" is lowering precission, so it's advisable to test without LUTs readings as integers
            # and actual floating point precision values instead
        else:
            if 0 <= row < self.rows:
                elevation = self.theta_LUT[int(row)]

        return elevation

    def get_elevation_from_panorama_row_without_LUT(self, row):
        elevation = None  # Initialize to a certainly invalid elevation
        if isinstance(row, np.ndarray):
            elevation = np.where(np.logical_and(0. <= row, row < self.rows), np.arctan2(self.cyl_height_max - self.pixel_size * row, self.cyl_radius), np.nan)
        else:
            if 0 <= row < self.rows:
                elevation = np.arctan2(self.cyl_height_max - self.pixel_size * row, self.cyl_radius)

        return elevation

    def get_azimuth_from_panorama_col(self, col):
        azimuth = None  # Initialize to a certainly invalid azimuth
        if isinstance(col, np.ndarray):
            azimuth = np.where(np.logical_and(0. <= col, col < self.cols), self.psi_LUT.take(col.flatten().astype("int")).reshape(col.shape), np.nan)
        else:
            if 0 <= col < self.cols:
                azimuth = self.psi_LUT[int(col)]
        return azimuth

    def get_azimuth_from_panorama_col_without_LUT(self, col):
        azimuth = None  # Initialize to a certainly invalid azimuth
        if isinstance(col, np.ndarray):
            azimuth = np.where(np.logical_and(0. <= col, col < self.cols), self.cyl_circumference - self.pixel_size * col, np.nan)
        else:
            if 0 <= col < self.cols:
                azimuth = self.cyl_circumference - self.pixel_size * col
        return azimuth

    def get_all_elevations(self, validate = True):
        if validate:
            return self.theta_LUT_validated
        else:
            return self.theta_LUT

    def get_direction_angles_from_pixel_pano(self, m_pano, use_LUTs = False):
        '''
        Provides the respective azimuth and elevation angles that a given pixel \f$m$\f on the panorama resolves to.

        @param m_pano: A numpy ndarray of k (rows x columns) image point coordinates [u, v] as row vector. Thus, shape is (rows, cols, 2)
        @param use_LUTs: boolean flag that indicates whether to use LUTs to obtain angular information (Warning: LUTs can introduce discretization error)

        @retval azimuth, elevation: angles for the panorama pixel(s) . Angles are w.r.t. the mirror's focus and given in radians.
        '''
        if self.world2cam_LUT_map_x is None or self.world2cam_LUT_map_y is None or use_LUTs == False:
            azimuth = self.get_azimuth_from_panorama_col_without_LUT(m_pano[..., 0])  # from u-coordinate
            elevation = self.get_elevation_from_panorama_row_without_LUT(m_pano[..., 1])  # from v-coordinate
        else:  # Use the function that uses the LUTs
            azimuth = self.get_azimuth_from_panorama_col(m_pano[..., 0])  # from u-coordinate
            elevation = self.get_elevation_from_panorama_row(m_pano[..., 1])  # from v-coordinate

        return azimuth, elevation

    def get_row_limits(self):
        '''
        @return: the tuple of rows associated to the maximum and minimum elevation angles for this model
        '''
        row_highest = self.get_panorama_row_from_elevation(elevation = self.model.highest_elevation_angle)
        row_lowest = self.get_panorama_row_from_elevation(elevation = self.model.lowest_elevation_angle)
        return row_highest, row_lowest

    def get_panorama_row_from_elevation(self, elevation):
        '''
        @param elevation: The target elevation angle (in radians) for which the corresponding row in the panoramic image is to be found.
        @retval is_valid: False indicates that the angle is outside of the allowed elevation range.
        @retval row: The valid row number (first row is index 0 and last row is height-1) in the panorama where the elevation maps to.
        '''
        elevation_validated = np.where(np.logical_and(self.model.lowest_elevation_angle <= elevation, elevation <= self.model.highest_elevation_angle), elevation, np.nan)
        h = np.tan(elevation_validated)
        s = self.cyl_height_max - h
        row = np.where(np.isnan(s), np.nan, np.uint((s / self.pixel_size))).astype("uint")  # This helps maintaining non-nan values as uint (trust me!)
        # CHECKME: the following purge should not be necessary
        # Purge out of bounds rows (negative values) due to the uint conversion
        # row = np.where(row >= self.rows, np.nan, row).astype("uint")  # There shouldn't exist!
        return row

    def get_panorama_col_from_azimuth(self, azimuth):
        '''
        @param azimuth: The target azimuth angle (in radians) for which the corresponding col in the panoramic image is to be found.
        @retval col: The valid col number (first col is index 0 and last is width-1) in the panorama where the azimuth maps to.
        '''
        azimuth_filtered = np.mod(azimuth, 2.0 * np.pi)  # Filter input azimuth so values are only positive angles between 0 and 2PI
        arc_length = self.cyl_radius * azimuth_filtered
        col = np.uint(self.cols - 1 - np.uint(arc_length / self.pixel_size))
        return col

    def get_panorama_pixel_coords_from_direction_angles(self, theta, psi):
        '''
        @brief Given the elevation and azimuth angles (w.r.t. mirror focus), the projected pixel on the panoramic image is found.

        @param theta: The elevation angles as an ndarray
        @param psi: The azimuth angles as an ndarray
        @retval u: The u pixel coordinate (or ndarray of u coordinates)
        @retval v: The v pixel coordinate (or ndarray of v coordinates)
        @retval m_homo: the pixel as a homogeneous ndarray (in case is needed)
        '''
        u = self.get_panorama_col_from_azimuth(psi)
        v = self.get_panorama_row_from_elevation(theta)
        m_homo = np.dstack((u, v, np.ones_like(u)))
        return u, v, m_homo
