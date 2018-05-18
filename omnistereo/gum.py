# -*- coding: utf-8 -*-
# gum.py

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
@package omnistereo
The Generalized Unified Model for Stereo (GUMS) Omnidirectional Systems

@author: Carlos Jaramillo
'''

from __future__ import print_function
from __future__ import division

import struct
# import euclid
import numpy as np
from omnistereo import camera_models
from omnistereo import common_tools
from omnistereo.camera_models import OmniCamModel, OmniStereoModel, CamParams

class Parameters(CamParams):
    '''
    Stores parameters from a pre-calibration step (e.g. done in some MATLAB toolbox)
    '''

    def __init__(self, precalib_filename, new_method = True, cam_model = None, **kwargs):
        '''
        Constructor for retrieving parameters from pre-calibration file
        \f$ \mbox{LaTeX} \f$  formulas examples.
        The distance between \f$(x_1,y_1)\f$ and \f$(x_2,y_2)\f$ is
        \f$\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}\f$.
        Unnumbered
        \f[
            C(u,v) = \frac
                {\sum{\left\{T(x,y) * I(x-u,y-v)\right\}}}
                {\sqrt{ \sum{I(x-u,y-v)^2}}}
        \f]
        \f{eqnarray*}{
            g &=& \frac{Gm_2}{r^2} \\
              &=& \frac{(6.673 \times 10^{-11}\,\mbox{m}^3\,\mbox{kg}^{-1}\,
                  \mbox{s}^{-2})(5.9736 \times 10^{24}\,\mbox{kg})}{(6371.01\,\mbox{km})^2} \\
              &=& 9.82066032\,\mbox{m/s}^2
        \f}
        @param precalib_filename: the complete filename where pre-calibration parameters were saved (from MATLAB)
        @param new_method: Indicates if the new model by Zhang will be used as opposed to the old from Mei.
        '''
        self.new_method = new_method
        self.precalib_filename = precalib_filename

        # 3D projection point Cp
        # Xi's are the coordinates of Cp wrt [M]
        self.xi1 = 0.  # x coordinate
        self.xi2 = 0.  # y coordinate
        self.xi3 = 1. * cam_model.z_axis  # z coordinate

        # Distortion
        self.k1 = 0.
        self.k2 = 0.
        self.k3 = 0.
        self.p1 = 0.
        self.p2 = 0.
        self.use_distortion = True  # Always use distortion parameters

        # Radial Undistortion (NEW)
        self.l1 = 0.
        self.l2 = 0.
        self.l3 = 0.

        # Image size must be as (width, height)
        if "image_size_pixels" in kwargs.keys():
            self.image_size = np.array(kwargs.get("image_size_pixels"))

        if "center_uv_point" in kwargs.keys():
            self.center_point = np.array(kwargs.get("center_uv_point"))
            self.u_center, self.v_center = self.center_point
        else:
            self.center_point = (self.image_size / 2.0) - 1
            self.u_center, self.v_center = self.center_point

        # An appropriate default values will be found accordingly during initialization and calibration
        self.gamma1 = 300.  # TODO: Find the proper value to be used as a dummy camera for the ChArUco detection
        self.gamma2 = 300.  # TODO: Find the proper value to be used as a dummy camera for the ChArUco detection
        self.alpha_c = 0.

        # ROI:
        self.roi_min_x = None
        self.roi_min_y = None
        self.roi_max_x = None
        self.roi_max_y = None

#         self.set_camera_params(**kwargs)
        self.read_params(cam_model)

    def set_generalized_cam_params(self, **kwargs):
        # Set "available" intrinsic camera parameters:
        if "alpha_c" in kwargs.keys():
            self.alpha_c = kwargs.get("alpha_c")
        if "gamma1" in kwargs.keys():
            self.gamma1 = kwargs.get("gamma1")
        if "gamma2" in kwargs.keys():
            self.gamma2 = kwargs.get("gamma2")
        if "u_center" in kwargs.keys():
            self.u_center = kwargs.get("u_center")
        if "v_center" in kwargs.keys():
            self.v_center = kwargs.get("v_center")
        self.center_point = np.array([self.u_center, self.v_center])

        # Set inverse camera matrix components
        self.inv_K11 = 1 / self.gamma1
        self.inv_K12 = -self.alpha_c / self.gamma2
        self.inv_K13 = self.alpha_c * self.v_center / self.gamma2 - self.u_center / self.gamma1
        self.inv_K22 = 1 / self.gamma2
        self.inv_K23 = -self.v_center / self.gamma2

    def set_pinhole_cam_params(self, **kwargs):
        if "focal_length" in kwargs.keys():
            self.focal_length = kwargs.get("focal_length")
        else:
            self.focal_length = 1.0

        if "image_size" in kwargs.keys():
            self.image_size = np.array(kwargs.get("image_size"))

        if "pixel_size" in kwargs.keys():
            self.pixel_size = kwargs.get("pixel_size")
        else:
            self.pixel_size = None

        if "sensor_size" in kwargs.keys():
            self.sensor_size = kwargs.get("sensor_size")
        else:
            self.sensor_size = None

        if self.sensor_size is not None:
            self.pixel_size = self.sensor_size / self.image_size  # (h_x, h_y) [mm / px]
        elif self.pixel_size is not None:
            # We know the pixel size:
            self.sensor_size = self.pixel_size * self.image_size

        self.FOV = 2 * np.arctan(self.sensor_size / (2 * self.focal_length))  # A tuple (hor, ver) FOVs in [radians]

    def set_gum_params(self, **kwargs):
#         self.set_camera_params(**kwargs)
        if "center_uv_point" in kwargs.keys():
            self.center_point = np.array(kwargs.get("center_uv_point"))
            self.u_center, self.v_center = self.center_point

        # TODO: implement/complete all possible settings of params
        if "xi1" in kwargs.keys():
            self.xi1 = kwargs.get("xi1")
        if "xi2" in kwargs.keys():
            self.xi2 = kwargs.get("xi2")
        if "xi3" in kwargs.keys():
            self.xi3 = kwargs.get("xi3")

        # Mirror Cp position based on real-mirror's location
        # Cp = (-xi1, -xi2, -xi3)  according to the literature because we don't want images to be inverted
        # Maybe? ^^^^^^^^^^^, it's just the coordinates of Cp wrt [M]
        # Recall the pinhole projection model where the image plane can be put in front of the pinhole,
        # so projection is not inverted (fliped on x and y)
        #=======================================================================
        # self.xi1 = -self.xi1
        # # Rotation around the x-axis of model
        # # TODO: remove hack, hopefully by implementing my own calibration model
        # self.xi2 = -self.xi2 * z_axis
        # self.xi3 = -self.xi3 * z_axis
        #=======================================================================

        # self.Cp = euclid.Point3(self.xi1, self.xi2, self.xi3)
        self.Cp = np.array([self.xi1, self.xi2, self.xi3])

        # Inverse camera projection matrix parameters
        # Correct! (checked with Wolfram Mathematica)
        # without alpha K12 term:
        #=======================================================================
        # self.inv_K11 = 1 / self.gamma1
        # self.inv_K13 = -self.u_center / self.gamma1
        # self.inv_K22 = 1 / self.gamma2
        # self.inv_K23 = -self.v_center / self.gamma2
        #=======================================================================
        self.set_generalized_cam_params(**kwargs)

#         if self.k1 == 0 and self.k2 == 0 and self.p1 == 0 and self.p2 == 0:
#             print("No distortion!!!")
#             self.use_distortion = False
#         else:
        self.use_distortion = True  # Always use distortion, especially during calibration

    def read_params(self, cam_model):

        # WISH: read a .mat file instead (easier) with mlab
        # or, even better, save as an Options file from MATLAB to be read as a dictionary or text file (CSV)
        try:
            f = open(self.precalib_filename, "rb")
            # The new_cv's parameters .bin file has the following:
            # xi1, xi2, xi3, kc, gamma1, gamma2, u_center, v_center, alpha_c, roi_min, roi_max

            first_bit = 0
            bit_size = 8

            if self.new_method:
                bytes_read = 3
            else:
                bytes_read = 1

            bits_read = first_bit + bytes_read * bit_size
            block = f.read(bits_read)

            if self.new_method:
                (self.xi1, self.xi2, self.xi3,) = struct.unpack('ddd', block[first_bit:bits_read])
            else:
                (self.xi3,) = struct.unpack('d', block[first_bit:bits_read])
                self.xi1, self.xi2 = 0.0, 0.0

            bytes_read = 5
            bits_read = first_bit + bytes_read * bit_size
            block = f.read(bits_read)  # 5 doubles
            (self.k1, self.k2, self.p1, self.p2, self.k3,) = struct.unpack('ddddd', block[first_bit:bits_read])

            bytes_read = 5
            bits_read = first_bit + bytes_read * bit_size
            block = f.read(bits_read)
            (self.gamma1, self.gamma2, self.u_center, self.v_center, self.alpha_c,) = struct.unpack('ddddd', block[first_bit:bits_read])
            # NOTE: gamma_i = f_i * eta ...because f and eta cannot be estimated independently.
            # Where focal lengths: f_1 = f_x and f_2 = f_y
            # Basically, gamma_1 is the K11 term in the camera intrinsic matrix
            #            gamma_2 is the K22 term
            # See equation (1) in Mei's 2007 paper

            # Recall u_center and v_center need -1 because Matlab's result start counting at 1
            self.u_center -= 1
            self.v_center -= 1
            self.center_point = np.array([self.u_center, self.v_center])

            bytes_read = 4
            bits_read = first_bit + bytes_read * bit_size
            block = f.read(bits_read)
            (self.roi_min_x, self.roi_min_y, self.roi_max_x, self.roi_max_y) = struct.unpack('dddd', block[first_bit:bits_read])
            # Recall we have to offset by -1 because Matlab's result start counting at 1
            self.roi_min_x -= 1
            self.roi_min_y -= 1
            self.roi_max_x -= 1
            self.roi_max_y -= 1

            f.close()
        except:
            # When reading the precalibration file fails:
            # Using default params:
            print("No file found for precalib. It will use what it currently has or its theoretical params (if not calibrated yet)!")

        finally:
            if not cam_model.is_calibrated:
                if cam_model.use_theoretical and cam_model.theoretical_model is not None:
                    # self.u_center,  self.v_center = (self.image_size / 2.0) - 1
                    self.xi1, self.xi2 = 0., 0.
                    c = cam_model.theoretical_model.c
                    k = cam_model.theoretical_model.k  # Unitless
                    d = cam_model.theoretical_model.d

                    self.f_u, self.f_v = cam_model.theoretical_model.precalib_params.focal_length / cam_model.theoretical_model.precalib_params.pixel_size  # [mm]/[mm/px] --> [px]

                    xi3_theor, gamma_theor = cam_model.get_theoretical_xi(c, k, self.f_u)
                    self.xi3 = cam_model.z_axis * xi3_theor
                    self.gamma1 = abs(gamma_theor)
                    self.gamma2 = abs(gamma_theor)
                    self.alpha_c = 0.
                    self.k1 = 0.
                    self.k2 = 0.
                    self.k3 = 0.  # Just there in case
                    self.p1 = 0.
                    self.p2 = 0.
                    self.l1 = 0.  # Radial Undistortion 1
                    self.l2 = 0.  # Radial Undistortion 2
                    self.l3 = 0.  # Radial Undistortion 3
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def print_params(self, header_message = ""):
        print(header_message)
        xis = "xi1={0:.6f}, xi2={1:.6f}, xi3={2:.6f}".format(self.xi1, self.xi2, self.xi3)
        print(xis)
        distortion_params = "Distortion: k1={0:.7f}, k2={1:.7f}, k3={2:.7f}, p1={3:.7f}, p2={4:.7f}".format(self.k1, self.k2, self.k3, self.p1, self.p2)
        print(distortion_params)
        undistortion_params = "Radial undistortion: l1={0:.7f}, l2={1:.7f}, l3={2:.7f}".format(self.l1, self.l2, self.l3)
        print(undistortion_params)
        misc_params = "alpha_c={4:.6f}, gamma1={0:.3f}, gamma2={1:.3f}, u_center={2:.4f}, v_center={3:.4f}".format(self.gamma1, self.gamma2, self.u_center, self.v_center, self.alpha_c)
        print(misc_params)
        #=======================================================================
        # roi_params = "roi_min_x={0:.6f}, roi_min_y={1:.6f}, roi_max_x={2:.6f}, roi_max_y={3:.6f}".format(self.roi_min_x, self.roi_min_y, self.roi_max_x, self.roi_max_y)
        # print(roi_params)
        #=======================================================================

class GUM(OmniCamModel):
    '''
    A single-mirror Generalized Uniform Model
    '''

    def __init__(self, precalib_filename, new_method = True, z_axis = 1.0, use_theoretical_xi_and_gamma = False, **kwargs):
        '''
        @brief GUM constructor. It will use the pre-calibration parameters to form the single mirror GUM
        @param precalib_filename: the complete filename where pre-calibration parameters were saved (from MATLAB) to be read via Parameters
        @param new_method: Indicates if the new model by Zhang will be used as opposed to the old from Mei.
        @param z_axis: Indicates if it parameters are for the top or bottom mirror (-1) based on actual mirror orientation that need to be flipped in the model
        '''
        self.radial_bounds_filename = None  # Convenient to remember this file name in case we need to refine the radial masks
        self.theoretical_model = None
        self.use_theoretical = use_theoretical_xi_and_gamma
        self.new_method = new_method
        self.z_axis = z_axis  # to indicate if it's the top (-1) or bottom mirror based on actual mirror orientation
        if self.z_axis > 0:
            self.mirror_name = "top"
            self.mirror_number = 1
        else:
            self.mirror_name = "bottom"
            self.mirror_number = 2
        self.precalib_filename = precalib_filename
        self._init_default_values(mirror_number = self.mirror_number)
        #=======================================================================
        # # FIXME: Too specific, make it general and actually just part of the precalib params object
        # self.image_size_pixels = image_size_pixels
        # self.focal_length = focal_length
        # self.pixel_size = pixel_size
        #=======================================================================
        self.precalib_params = None

        self.set_model_params(**kwargs)

    def init_theoretical_model(self, theoretical_model):
        self.theoretical_model = theoretical_model
        self.theoretical_model.current_omni_img = self.current_omni_img

        # Focus location of top omnidirectional camera given as a 4x1 homogeneous position vector (numpy array)
        self.F = theoretical_model.F

    def set_model_params(self, **kwargs):
        '''
        Sets the essential properties of the model, such as the position of projection point, Cp.
        '''
        # FIXME: setup new precalib_params
        if self.precalib_params is None:
            self.precalib_params = Parameters(self.precalib_filename, cam_model = self, **kwargs)
        else:
            self.precalib_params.read_params(cam_model = self)

        self.precalib_params.set_gum_params(**kwargs)

        # self.Pm = euclid.Point3(0, 0, 0)  # Model origin point (center of unit sphere)'
        self.Pm = np.array([0., 0., 0.])  # Model origin point (center of unit sphere)
        self.Cp_wrt_M = self.precalib_params.Cp

        # The face of the projection plane is opposite to the model's direction
#         vector_of_normalized_plane = euclid.Vector3(0, 0, -1.0 * self.z_axis)
        self.plane_n = np.array([0, 0, 1.0 * self.z_axis])  # Plane's Normal vector
        # Recall the Z-distance from Cp to the normalized projection plane is 1 unit away
#         point_on_plane = euclid.Point3(0, 0, self.Cp_wrt_M.z + self.z_axis)
        self.plane_k = self.Cp_wrt_M[2] - self.plane_n[2]  # signed distance (z-position) of normalized projection plane
#         self.normalized_projection_plane = euclid.Plane(point_on_plane, vector_of_normalized_plane)

    def update_optimized_params(self, params, only_intrinsics = False, only_extrinsics = False, suppress_tz_optimization = False, final_update = False):
        '''
        Plays an important role during calibration as the new parameter values have to be updated
        '''
        if only_intrinsics:
            suppress_tz_optimization = True
            only_extrinsics = False

        tz_param_idx_offset = 0
        # TODO: assuming coaxial constraint for now, so only the z-translation component is being updated
        if suppress_tz_optimization:
            tz_param_idx_offset = 1
        else:
            M_wrt_C_tz = params[0]
            # Update focal position:
            self.F[2] = M_wrt_C_tz  # F_z is M_wrt_C (be careful that the calibration parameters are actually computing it as such)

        if only_extrinsics == False:
            # Related to xi's:
            xis = params[1 - tz_param_idx_offset:4 - tz_param_idx_offset]
            self.precalib_params.xi1, self.precalib_params.xi2, self.precalib_params.xi3 = xis
            self.precalib_params.Cp = np.array([self.precalib_params.xi1, self.precalib_params.xi2, self.precalib_params.xi3])
            self.Cp_wrt_M = self.precalib_params.Cp
            self.plane_k = self.Cp_wrt_M[2] - self.plane_n[2]  # signed distance (z-position) of normalized projection plane

            k_dist = params[4 - tz_param_idx_offset:7 - tz_param_idx_offset]
            self.precalib_params.k1, self.precalib_params.k2, self.precalib_params.k3 = k_dist

            # Set intrinsic camera parameters:
            self.precalib_params.set_generalized_cam_params(alpha_c = params[7 - tz_param_idx_offset], gamma1 = params[8 - tz_param_idx_offset], gamma2 = params[9 - tz_param_idx_offset], u_center = params[10 - tz_param_idx_offset], v_center = params[11 - tz_param_idx_offset])

        self.set_pose(translation = self.F[:3, 0], rotation_matrix = np.identity(3))  # TODO: assuming coaxial constraint for now all the time

        if final_update:
            self.set_params()
            # NOTE: we are assuming only translation vector in effect for the model. No rotation!
            self.print_precalib_params(header_message = "Intrinsic parameters:", plot_distortions = False)

    def update_undistortion_params(self, l1, l2, l3):
        '''
        Plays an important role during lifting a pixel using radial distortion
        '''
        self.precalib_params.l1 = l1
        self.precalib_params.l2 = l2
        self.precalib_params.l3 = l3
        self.set_params()  # This takes care of updating the elevation angles by using the undistortion parameters

    def print_precalib_params(self, header_message = "", plot_distortions = False):
        print(header_message)
        title_name = self.mirror_name + " Mirror's pre-calibration parameters:"
        print(title_name.capitalize())
        print("z-axis = %d" % (self.z_axis))
        print("Cp_wrt_M: %s" % self.Cp_wrt_M)
        self.precalib_params.print_params(header_message = "with")
        if plot_distortions:
            from omnistereo.common_plot import plot_GUM_distortion
            plot_GUM_distortion(model = self, distortion_params = [self.precalib_params.k1, self.precalib_params.k2, self.precalib_params.k3], undistortion_params = [self.precalib_params.l1, self.precalib_params.l2, self.precalib_params.l3])

    def get_theoretical_xi(self, c, k, f):
        '''
        @param c: Distance between focal points
        @param k: the k parameter associated with the profile's curvature
        @param f: Focal length in pixels
        '''
        d = c  # distance between focal points
        # The line segment connecting the two intersection points of this line with the hyperbola is known as the latus rectum
        a = self.get_a_hyperbola(c, k)
        b = self.get_b_hyperbola(c, k)
        latus = (2 * b ** 2) / a
        # Therefore,
        p = latus / 4
        xi, eta, gamma = self.get_unified_model_params(p, d, f)
        return xi, gamma

    def get_unified_model_params(self, p, d, f):
        '''
        Hyperbola's unified model parameters

        @param p: Related to 4p latus rectum
        @param d: distance between focal points
        @param f: focal length of camera
        '''
        xi = d / np.sqrt(d ** 2 + 4 * p ** 2)
        eta = -2 * p / np.sqrt(d ** 2 + 4 * p ** 2)
        # gamma is the focal length of the generalized perspective projection
        # gamma = f * eta  in [px/mm or px/m] based on units of p and d
        gamma = eta * f
        return xi, eta, gamma

    def estimate_gamma(self, collinear_points_from_omni_img, center_point):
        '''
        Assuming xi=1 (for a parabolic mirror profile), we compute the gamma (generalized focal length) parameter

        @param collinear_points_from_omni_img: Image point coordinates that lie on a non-radial 3D line. At least four points are needed.
        @param center_point: The pixel coordinates for the principal point

        @return: gamma : estimate of the extended focal, N : line normal, and a success flag indicating if the extraction succeeded, and the normal_xy_squared
        '''
        gamma = np.nan
        N = np.ones(shape = (3, 1)) * np.nan
        success = False
        normal_xy_squared = np.nan

        if len(collinear_points_from_omni_img) >= 4:
            # NOTE: Following page 27 from Mei's thesis:
            # Change coordinates of points to be wrt the center
            points_wrt_Ic = collinear_points_from_omni_img - np.array(center_point)
            # Compute nx4 matrix P
            #===================================================================
            # For xi=1: Ps_wrt_M := [u_wrt_Ic, v_wrt_Ic, 0.5 * gamma - 0.5(u_wrt_Ic^2 + v_wrt_Ic^2)/gamma]
            a_portion = 0.5 * np.ones((points_wrt_Ic.shape[0], 1))  # Recall 1/2*a
            b_portion = -0.5 * np.sum(points_wrt_Ic ** 2, axis = -1)[..., np.newaxis]  # Recall -(u_c^2 + v_c^2)/2
            #===================================================================
            # For any xi:
            # xi = np.abs(self.precalib_params.xi3)
            # u2_plus_v2 = np.sum(points_wrt_Ic ** 2, axis=-1)[..., np.newaxis]
            # beta = np.sqrt(1 + (1 - xi ** 2) * u2_plus_v2)  # beta = sqrt(1+(1-xi^2)(u^2+v^2))
            # a_portion = beta / (xi + beta)
            # b_portion = -xi * u2_plus_v2 / (xi + beta)
            #===================================================================
            # if self.z_axis < 0:
            #     gamma_factor_a_portion = 1.0
            #     gamma_factor_b_portion = 1.0
            # else:
            #     gamma_factor_a_portion = 3.0
            #     gamma_factor_b_portion = -1.0
            # a_portion = gamma_factor_a_portion * 0.5 * np.ones((points_wrt_Ic.shape[0], 1))  # Recall 1/2*a
            # b_portion = -0.5 * np.sum(points_wrt_Ic ** 2, axis=-1)[..., np.newaxis]  # Recall -(u_c^2 + v_c^2)/2
            #===================================================================
            P_nx4 = np.hstack((points_wrt_Ic, a_portion, b_portion))
            # Solve for c = [nx; ny; gamma*nz; nz/gamma] using SVD
            U, S, V = np.linalg.svd(P_nx4)  # or with U_scipy, s_scipy, Vh_scipy = scipy.linalg.svd(P_nx4)
            c_soln = V[-1]  # Choose last row associated with smallest singular value
            t = c_soln[0] ** 2 + c_soln[1] ** 2 + c_soln[2] * c_soln[3]
            if t > 0:  # Avoid division by 0
                d = 1 / np.sqrt(t)  # just the inverse of the normal vector's magnitude
                n_x_normalized = c_soln[0] * d
                n_y_normalized = c_soln[1] * d
                # CHECKME: this makes sense, but I think it's wrong!!!
                # check that nx^2 + ny^2 > threshold (for example threshold = 0.95) to be sure the line image is not radial:
                threshold = 0.95  # In other words, a normal vector (of the Hyperplane) perpendicular to the xy-plane should be avoided when n_z is close to 0???? then the normal vector is almost on the xy-plane
                normal_xy_squared = n_x_normalized ** 2 + n_y_normalized ** 2
                if normal_xy_squared <= threshold:
                    n_z_normalized = np.sqrt(1 - normal_xy_squared)
                    N = np.array([n_x_normalized, n_y_normalized, n_z_normalized])
                    gamma = abs(c_soln[2] * d / n_z_normalized)
                    success = True
                    # TEST points' z-coords normalized: p_z_coords_normalized = gamma/2 - (points_wrt_Ic[...,0]**2 + points_wrt_Ic[...,1]**2)/(2*gamma)
                    # TEST points' z-coords: p_z_coords = 2*gamma**2 / (points_wrt_Ic[...,0]**2 + points_wrt_Ic[...,1]**2 + gamma**2) - 1
                    # TEST points' norms: p_norms = np.sqrt(p_z_coords**2+points_wrt_Ic[...,0]**2 + points_wrt_Ic[...,1]**2)
        return gamma, N, success, normal_xy_squared

    def approximate_homography_like_MATLAB_BUT_NOPE(self, img_points, obj_points_homo, visualize = False, T_G_wrt_C = None, T_G_wrt_F = None):
        '''
        Approximate the transformation (pose) between a set of points on the image plane and their corresponding points in a plane
        @note: Assuming the theoretical model has ben provided!!
        '''
        # Step 1) Back projection to mirror using known model parameters
        obj_pts_2D, T_obj_2D = common_tools.get_2D_points_normalized(obj_points_homo[..., [0, 2, 3]])
        mirror_points_3D_homo = self.theoretical_model.lift_pixel_to_mirror_surface(img_points)
        mirror_pts_2D, T_mirror_2D = common_tools.get_2D_points_normalized(mirror_points_3D_homo)

        # HOMOGRAPHY implementation:
        x1 = obj_pts_2D.reshape(-1, 3)
        x2 = mirror_pts_2D.reshape(-1, 3)
        n_pts = len(x1)
        A = np.zeros((3 * n_pts, 9))

        for n in range(n_pts):
            X = x1[n]
            x, y, w = x2[n]
            A[3 * n, 3:6], A[3 * n, 6:] = -w * X, y * X
            A[3 * n + 1, :3], A[3 * n + 1, 6:] = w * X, -x * X
            A[3 * n + 2, :3], A[3 * n + 2, 3:6] = -y * X, x * X

        U, D, V = np.linalg.svd(A, full_matrices = False)

        # Extract homography
        H_normalized = V[-1].reshape(3, 3).T  # Grab the last row, reshape to a 3x3 matrix, and transpose

        # Denormalise
        # H = T2\H_normalized*T1 in Matlab, which means solving the system Ax = B, as x = A\B
        H_by_T1 = np.dot(H_normalized, T_obj_2D)
        H, resid, rank, sing_vals = np.linalg.lstsq(T_mirror_2D, H_by_T1)

        test_transf = np.dot(H, obj_points_homo[0, 0, [0, 2, 3]])
        if np.sign(test_transf[0]) != np.sign(mirror_points_3D_homo[0, 0, 0]):
            H = -H

        # TODO: Object points now lay into an xy-plane, so this has transformed the plane with a -90 degree rotation on the x-axis
        # TEST:
        # >>>> pts_back = np.einsum("df,mnf->mnd", H, obj_points_homo[..., [0, 2,3]])
        # >>>> pts_back_normalized = pts_back/np.abs(pts_back[...,2,np.newaxis]) --> should give pts_dst
        # Obviously, answer needs to find nonzero constant c, such that c * pts_dst = H * pts_source

        R = np.zeros_like(H)
        lamb = 1 / np.linalg.norm(H[:, 0])
        R[:, 0] = lamb * H[:, 0]
        R[:, 1] = lamb * H[:, 1]
        R[:, 2] = np.cross(R[:, 0], R[:, 1])
        t = lamb * H[:, 2]

        # enforce orthonormality of the rotation matrix
        U, S, V = np.linalg.svd(R)
        rot_matrix = np.dot(U, V)
        T = np.identity(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = t

        import omnistereo.transformations as tr
        R4 = np.identity(4)
        R4[:3, :3] = rot_matrix
        angles = tr.euler_from_matrix(R4, axes = "sxyz")
        print("Euler angles (degrees):", np.rad2deg(angles))
        print("Translation (mm):", t)

        # TEST: Not sure, it's not working
        # np.dot(tr.inverse_matrix(T)???, obj_points_homo[0,0,[0,2,1,3]])
        # b_offset??? + np.dot(T, obj_points_homo[0,0])[...,:3]

        return T

    def approximate_transformation(self, img_points, obj_pts_homo, use_PnP = True, visualize = False, T_G_wrt_C = None, app = None, finish_drawing = True):
        '''
        Approximate the transformation (pose) between an object frame [G] with respect to the fixed frame [C]
        @note: Assuming the theoretical model has ben provided!!

        @param img_points: The corresponding points on the image
        @param obj_pts_homo: The coordinates of the corresponding points with respect to the object's own frame. set of points on an object (Needs to be planar if using Homography)
        @param use_PnP: To choose the method: PnP (Perspective from N Points) or Planar Homography (only for coplanar points on the object frame)
        @param T_G_wrt_C: The ground truth (if known) transform matrix of [G] wrt to [C]. This is only used for visualization.
        @return: The estimated transformation matrix of the object frame [G] with respect to the fixed frame [C]
        '''
        import omnistereo.transformations as tr

        # TEST: A simple test using forward projection. Get the object points wrt [C]:
        # obj_pts_wrt_C = np.einsum("ij, mnj->mni", T_G_wrt_C, obj_pts_homo)
        # mirror_points_3D_homo = self.theoretical_model.get_primer_point_for_projection(obj_pts_wrt_C)

        mirror_focus_point = self.theoretical_model.F
        # Back projection to mirror using known model parameters
        mirror_points_3D_homo = self.theoretical_model.lift_pixel_to_mirror_surface(img_points)  # Homogeneous

        # Approximate an image (reference) plane from the projections of the mirror points
        # Find the orientation of this plane using the 2 edges of the pattern on the mirror
        b_orig = mirror_points_3D_homo[-1, 0, :3]  # For fake virtual camera as seen through an image (This is the upper-left corner)
        v_hor_on_fake_cam = mirror_points_3D_homo[-1, -1, :3] - b_orig  # like the x-axis
        v_ver_on_fake_cam = mirror_points_3D_homo[0, 0, :3] - b_orig  # like the y-axis (but not orthogonal)
        b_normal_cross = np.cross(v_hor_on_fake_cam, v_ver_on_fake_cam)
        b_normal = b_normal_cross / np.linalg.norm(b_normal_cross)  # Unit normal vector of the camera plane
        # orthogonal projection
        M = tr.projection_matrix(point = b_orig, normal = b_normal, direction = None)
        proj_points_on_plane_wrt_C = np.einsum("ij, mnj->mni", M, mirror_points_3D_homo)
        # Find centroid point (a.k.a. optical center point) via an orthographic projecting of the focus onto the plane:
        b_centroid = np.dot(M, mirror_focus_point)[:3, 0]
        # Make plane orthogonal to focus of mirror and centroid point
        F_wrt_C = mirror_focus_point[:3, 0]
        b_cent_to_F = b_centroid - F_wrt_C  # Normal to plane
        b_cent_to_F_mag = np.linalg.norm(b_cent_to_F)  # Norm

        cam_C_x_axis, cam_C_y_axis, cam_C_z_axis = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])  # for [C] frame
        fake_cam_x_axis = proj_points_on_plane_wrt_C[-1, -1, :3] - b_orig
        fake_cam_x_axis = fake_cam_x_axis / tr.vector_norm(fake_cam_x_axis)
        fake_cam_z_axis = b_normal
        # Resolve for the y-axis (orthogonally)
        fake_cam_y_axis = np.cross(fake_cam_z_axis, fake_cam_x_axis)
        fake_cam_y_axis = fake_cam_y_axis / tr.vector_norm(fake_cam_y_axis)

        # Attempt to do translation first (move the centroid to the origin), then rotation!
        proj_points_on_plane_wrt_C_offset = proj_points_on_plane_wrt_C[..., :3] - b_centroid
        # See http://www.starlino.com/dcm_tutorial.html to learn more about the DCM (Direction Cosine Matrix)
        b_DCM = np.array([[np.dot(cam_C_x_axis, fake_cam_x_axis), np.dot(cam_C_y_axis, fake_cam_x_axis), np.dot(cam_C_z_axis, fake_cam_x_axis)],
                          [np.dot(cam_C_x_axis, fake_cam_y_axis), np.dot(cam_C_y_axis, fake_cam_y_axis), np.dot(cam_C_z_axis, fake_cam_y_axis)],
                          [np.dot(cam_C_x_axis, fake_cam_z_axis), np.dot(cam_C_y_axis, fake_cam_z_axis), np.dot(cam_C_z_axis, fake_cam_z_axis)]
                          ])

        # De-rotate points on plane so coordinates are with respect to the fake camera plane
        # NOTE: rotation seems fine now with DCM, but points are offset on their z-coordinates by the orthogonal distance "mirror_proj_plane.k"
        # This is correct since the plane frame itself is given by normalized versors
        pts_fake_cam = np.einsum("ij, mnj->mni", b_DCM, proj_points_on_plane_wrt_C_offset)[..., :2]  # Remove the z-coordinate
        pt_origin = pts_fake_cam[-1, 0, :2]

        px_offset = -pt_origin[:2]
        pts_fake_cam = pts_fake_cam + px_offset  # Offset the optical center point
        # pts_fake_cam = pts_fake_cam - px_origin  # Offset to origin
        px_centroid = -pt_origin
        # Fake camera matrix for solving PnP
        f = b_cent_to_F_mag  # [mm] Fake camera's focal length. Recall f_u = f / hx
        h_px = 1  # 1 [mm]/[px]
        c_u, c_v = px_centroid
        K_cam = np.array([[f / h_px, 0, c_u], [0, f / h_px, c_v], [0, 0, 1]])

        if use_PnP:
            from cv2 import solvePnP
            retval, rot_vec_G_wrt_fake_cam, tr_vec_G_wrt_fake_cam = solvePnP(obj_pts_homo[..., :3].reshape(-1, 3).astype(np.float32), pts_fake_cam.reshape(-1, 2).astype(np.float32), K_cam, distCoeffs = None)
            # from cv2 import solvePnPRansac
            # rot_vec_G_wrt_fake_cam, tr_vec_G_wrt_fake_cam, inliers = solvePnPRansac(obj_pts_homo[..., :3].reshape(-1, 3).astype(np.float32), pts_fake_cam.reshape(-1, 2).astype(np.float32), K_cam, distCoeffs=None)
            # Convert Rodrigues rotation vector to rotation matrix
            from cv2 import Rodrigues
            R_G_wrt_fake_cam, rot_jacobian = Rodrigues(rot_vec_G_wrt_fake_cam)
            T_G_wrt_fake_cam = np.identity(4)  # 4x4 transformation matrix
            # Note: PnP follows a homogeneous transformation: first doing rotation first and then translation
            T_G_wrt_fake_cam[:3, :3] = R_G_wrt_fake_cam  # Rotation
            T_G_wrt_fake_cam[:3, 3] = tr_vec_G_wrt_fake_cam[:, 0]  # translation
        else:
            # vvvvvvvvvvvvvvv PLANAR HOMOGRAPHY vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            K_cam_inv = tr.inverse_matrix(K_cam)
            T_obj_pts_3D_to_2D = tr.rotation_matrix(np.deg2rad(-90), cam_C_x_axis)
            obj_pts = np.einsum("ij, mnj->mni", T_obj_pts_3D_to_2D, obj_pts_homo)
            obj_pts_2D = obj_pts[..., :2]
            from cv2 import findHomography
            # NOTE: method 0 is the quickest, just based on computing homography using DLT (I think). Results seem precise enough
            H, mask = findHomography(obj_pts_2D.reshape(-1, 2).astype(np.float32), pts_fake_cam.reshape(-1, 2).astype(np.float32), method = 0)

            # The scaling parameter is determined from the orthonormality condition
            lamb = 1 / np.linalg.norm(np.dot(K_cam_inv, H[:, 0]))
            R = np.zeros_like(H)
            R[:, 0] = lamb * np.dot(K_cam_inv, H[:, 0])
            R[:, 1] = lamb * np.dot(K_cam_inv, H[:, 1])
            R[:, 2] = np.cross(R[:, 0], R[:, 1])
            # We need to enforce orthonormality of the rotation matrix? Results are then decomposable without scale!
            U, S, V = np.linalg.svd(R)
            R_G_wrt_fake_cam = np.dot(U, V)

            tr_G_wrt_fake_cam = lamb * np.dot(K_cam_inv, H[:, 2])  # Translation vector

            T_G_wrt_fake_cam = np.identity(4)  # 4x4 transformation matrix
            # Note: PnP follows a homogeneous transformation: first doing rotation first and then translation
            T_G_wrt_fake_cam[:3, :3] = R_G_wrt_fake_cam  # Rotation
            T_G_wrt_fake_cam[:3, 3] = tr_G_wrt_fake_cam  # translation
            # Include the initial object rotation:
            T_G_wrt_fake_cam = tr.concatenate_matrices(T_G_wrt_fake_cam, T_obj_pts_3D_to_2D)
            # ^^^^^^^^^^^^^^^ PLANAR HOMOGRAPHY ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # Recall that the camera frame uses y-axis pointing down and  z-axis pointing outwards.
        rot_C_wrt_fake_cam = np.identity(4)  # 4x4 transformation matrix
        rot_C_wrt_fake_cam[:3, :3] = b_DCM  # Rotation
        # TEST: rotation angles are with respect to the "static" reference frame and are given as (ang_x, ang_y, ang_z)
        #       where R = Rz Ry Rx, so rotation around x-axis happens first, then around y, end then the fixed z-axis
        # >>>> scale, shear, angles, trans, persp = tr.decompose_matrix(rot_C_wrt_fake_cam)
        # >>>> rot_test = tr.euler_matrix(*angles, axes="sxyz")
        # >>>> np.allclose(rot_test, rot_C_wrt_fake_cam)
        rot_fake_cam_wrt_C = tr.inverse_matrix(rot_C_wrt_fake_cam)
        trans_fake_cam_wrt_C = tr.translation_matrix(F_wrt_C)
        # We must do rotation first and then translation:
        T_fake_cam_wrt_C = tr.concatenate_matrices(trans_fake_cam_wrt_C, rot_fake_cam_wrt_C)

        # Transformation of [G] wrt [C]
        T_G_wrt_C_est = tr.concatenate_matrices(T_fake_cam_wrt_C, T_G_wrt_fake_cam)
        # TEST: >>> scale, shear, angles, trans, persp = tr.decompose_matrix(T_G_wrt_C_est)

        # Visualization
        if visualize:
            if self.mirror_number == 1:
                grid_color = 'c'
            else:
                grid_color = 'm'

            use_visvis = True
            try:
                import visvis as vv
            except:
                use_visvis = False

            if hasattr(self, "theoretical_model"):
                if use_visvis:
                    from omnistereo.common_plot import draw_model_mono_visvis
                    app = draw_model_mono_visvis(self.theoretical_model, finish_drawing = False, mirror_transparency = 0.5, show_labels = False, show_only_real_focii = True, show_reference_frame = True, show_grid_box = False, busy_grid = False)
                else:
                    from omnistereo.common_plot import draw_omnistereo_model_vispy
                    draw_omnistereo_model_vispy(self.theoretical_model, show_grid = True, backend = 'PyQt5')

            a = vv.gca()

            # TODO: use z_offset for point's coordinates

            norm_length_vis = 10

            # Draw Fake optical-axis
            #===================================================================
            # pp_o = vv.Pointset(3)
            # pp_o.append(b_centroid); pp_o.append(mirror_focus_point[:3, 0]);
            # pp_o.append(b_centroid + norm_length_vis * 1.5 * fake_cam_z_axis);
            # line_n = vv.Line(a, pp_o)
            # line_n.ls = '--'  # line_style
            # line_n.lw = 2  # line_thickness
            # line_n.lc = grid_color
            #===================================================================

            # Draw fake cam's x-axis
            pp_cam_x = vv.Pointset(3)
            pp_cam_x.append(F_wrt_C); pp_cam_x.append(F_wrt_C + norm_length_vis * fake_cam_x_axis);
            line_cam_x = vv.Line(a, pp_cam_x)
            line_cam_x.ls = '-'  # line_style
            line_cam_x.lw = 1  # line_thickness
            line_cam_x.lc = "r"

            # Draw fake cam's y-axis
            pp_cam_y = vv.Pointset(3)
            pp_cam_y.append(F_wrt_C); pp_cam_y.append(F_wrt_C + norm_length_vis * fake_cam_y_axis);
            line_cam_y = vv.Line(a, pp_cam_y)
            line_cam_y.ls = '-'  # line_style
            line_cam_y.lw = 1  # line_thickness
            line_cam_y.lc = "g"

            # Draw fake image plane's x-axis
            pp_x = vv.Pointset(3)
            pp_x.append(b_orig); pp_x.append(b_orig + norm_length_vis * fake_cam_x_axis);
            line_x = vv.Line(a, pp_x)
            line_x.ls = '-'  # line_style
            line_x.lw = 1  # line_thickness
            line_x.lc = "r"

            # Draw fake image plane's y-axis
            pp_y = vv.Pointset(3)
            pp_y.append(b_orig); pp_y.append(b_orig + norm_length_vis * fake_cam_y_axis);
            line_y = vv.Line(a, pp_y)
            line_y.ls = '-'  # line_style
            line_y.lw = 1  # line_thickness
            line_y.lc = "g"

            # Draw fake image plane's z-axis
            pp_n = vv.Pointset(3)
            pp_n.append(b_orig); pp_n.append(b_orig + norm_length_vis * fake_cam_z_axis);
            line_n = vv.Line(a, pp_n)
            line_n.ls = '-'  # line_style
            line_n.lw = 1  # line_thickness
            line_n.lc = "b"

            #===================================================================
            # # Draw Fake camera grid
            # xx = proj_points_on_plane_wrt_C[..., 0]
            # yy = proj_points_on_plane_wrt_C[..., 1]
            # zz = proj_points_on_plane_wrt_C[..., 2]
            # plane_grid = vv.grid(xx, yy, zz, axesAdjust=True, axes=a)
            # plane_grid.edgeColor = grid_color  # (1., 1., 0., 1.0)  # yellow
            # plane_grid.edgeShading = "plain"  # possible shaders: None, plain, flat, gouraud, smooth
            # plane_grid.diffuse = 0.0
            # # Lower right point (just because!)
            # plane_lr_pt = vv.Point(proj_points_on_plane_wrt_C[0, -1, 0], proj_points_on_plane_wrt_C[0, -1, 1], proj_points_on_plane_wrt_C[0, -1, 2])
            # vv.plot(plane_lr_pt, ms='.', mc=grid_color, mw=5, ls='', mew=0, axesAdjust=False)
            # # First point (lower left in grid board)
            # plane_ll_pt = vv.Point(proj_points_on_plane_wrt_C[0, 0, 0], proj_points_on_plane_wrt_C[0, 0, 1], proj_points_on_plane_wrt_C[0, 0, 2])
            # vv.plot(plane_ll_pt, ms='.', mc='r', mw=5, ls='', mew=0, axesAdjust=False)
            #===================================================================

            # Draw real calibration grid in the world
            #===================================================================
            # try:
            #     obj_pts_wrt_C = np.einsum("df,mnf->mnd", T_G_wrt_C, obj_pts_homo)  # points in actual position
            #     xx_obj_pts_wrt_C = obj_pts_wrt_C[..., 0]
            #     yy_obj_pts_wrt_C = obj_pts_wrt_C[..., 1]
            #     zz_obj_pts_wrt_C = obj_pts_wrt_C[..., 2]
            #     obj_pts_wrt_C_grid = vv.grid(xx_obj_pts_wrt_C, yy_obj_pts_wrt_C, zz_obj_pts_wrt_C, axesAdjust=True, axes=a)
            #     obj_pts_wrt_C_grid.edgeColor = 'g'  # Green for ground truth
            #     obj_pts_wrt_C_grid.edgeShading = "plain"  # possible shaders: None, plain, flat, gouraud, smooth
            #     obj_pts_wrt_C_grid.diffuse = 0.0
            # except:
            #     print("Warning: No Ground Truth T_G_wrt_C available in %s" % (__name__))
            #===================================================================

            # Draw enboxing projection to Ground Truth origin
            #===================================================================
            # pp_g_ll = vv.Pointset(3)
            # pp_g_ll.append(obj_pts_wrt_C[0, 0, :3]); pp_g_ll.append(self.theoretical_model.F[:3, 0]);
            # line_g_ll = vv.Line(a, pp_g_ll)
            # line_g_ll.ls = '--'  # line_style
            # line_g_ll.lw = 1  # line_thickness
            # line_g_ll.lc = (0.8, .5, 0)
            #===================================================================

            # Draw estimated calibration grid in the world
            obj_pts_est = np.einsum("ij, mnj->mni", T_G_wrt_C_est, obj_pts_homo)
            xx_obj_pts_est = obj_pts_est[..., 0]
            yy_obj_pts_est = obj_pts_est[..., 1]
            zz_obj_pts_est = obj_pts_est[..., 2]
            obj_pts_est_grid = vv.grid(xx_obj_pts_est, yy_obj_pts_est, zz_obj_pts_est, axesAdjust = True, axes = a)
            obj_pts_est_grid.edgeColor = grid_color  # color?
            obj_pts_est_grid.edgeShading = "plain"  # possible shaders: None, plain, flat, gouraud, smooth
            obj_pts_est_grid.diffuse = 0.0
            Og_pt = vv.Point(obj_pts_est[0, 0, 0], obj_pts_est[0, 0, 1], obj_pts_est[0, 0, 2])
            vv.plot(Og_pt, ms = '.', mc = "k", mw = 5, ls = '', mew = 0, axesAdjust = False)

            # Draw 4 corners of points on mirror surface (color encoded)
            xx = mirror_points_3D_homo[..., 0]
            yy = mirror_points_3D_homo[..., 1]
            zz = mirror_points_3D_homo[..., 2]
            plane_grid = vv.grid(xx, yy, zz, axesAdjust = True, axes = a)
            plane_grid.edgeColor = grid_color  # (1., 1., 0., 1.0)  # yellow
            plane_grid.edgeShading = "plain"  # possible shaders: None, plain, flat, gouraud, smooth
            plane_grid.diffuse = 0.0
            # 00 point (origin) to estimated calibration grid pose
            pt_00 = mirror_points_3D_homo[0, 0, :3]
            plane_lr_pt = vv.Point(pt_00[0], pt_00[1], pt_00[2])
            vv.plot(plane_lr_pt, ms = '.', mc = 'k', mw = 6, ls = '', mew = 0, axesAdjust = False)
            pp_00 = vv.Pointset(3)
            pp_00.append(obj_pts_est[0, 0, :3]); pp_00.append(self.theoretical_model.F[:3, 0]);
            line_00 = vv.Line(a, pp_00)
            line_00.ls = '--'  # line_style
            line_00.lw = 1  # line_thickness
            line_00.lc = grid_color

            # row=0,col=-1 point
            #===================================================================
            # pt_0l = mirror_points_3D_homo[0, -1, :3]
            # plane_0l_pt = vv.Point(pt_0l[0], pt_0l[1], pt_0l[2])
            # vv.plot(plane_0l_pt, ms='.', mc='r', mw=6, ls='', mew=0, axesAdjust=False)
            # pp_0l = vv.Pointset(3)
            # pp_0l.append(obj_pts_est[0, -1, :3]); pp_0l.append(self.theoretical_model.F[:3, 0]);
            # line_0l = vv.Line(a, pp_0l)
            # line_0l.ls = '--'  # line_style
            # line_0l.lw = 1  # line_thickness
            # line_0l.lc = grid_color
            #===================================================================

            # row=-1,col=-1 point
            #===================================================================
            # pt_ll = mirror_points_3D_homo[-1, -1, :3]
            # plane_ll_pt = vv.Point(pt_ll[0], pt_ll[1], pt_ll[2])
            # vv.plot(plane_ll_pt, ms='.', mc='g', mw=6, ls='', mew=0, axesAdjust=False)
            # pp_ll = vv.Pointset(3)
            # pp_ll.append(obj_pts_est[-1, -1, :3]); pp_ll.append(self.theoretical_model.F[:3, 0]);
            # line_ll = vv.Line(a, pp_ll)
            # line_ll.ls = '--'  # line_style
            # line_ll.lw = 1  # line_thickness
            # line_ll.lc = grid_color
            #===================================================================

            if finish_drawing:
                # Start app
                vv.title('Approximated Grid Pose')
                app.Run()

        return T_G_wrt_C_est, app

    def estimate_RT(self, img_points, obj_pts_homo, use_PnP = True, visualize = False, T_G_wrt_C = None, app = None, finish_drawing = True, z_offset = 0.0):
        '''
        Approximate the transformation (pose) between an object frame [G] with respect to the GUM frame [M]

        @param img_points: The corresponding points on the image
        @param obj_pts_homo: The coordinates of the corresponding points with respect to the object's own frame. set of points on an object (Needs to be planar if using Homography)
        @param use_PnP: To choose the method: PnP (Perspective from N Points) or Planar Homography (only for coplanar points on the object frame)
        @param T_G_wrt_C: The ground truth (if known) transform matrix of [G] wrt to [C]. This is only used for visualization.
        @param z_offset: the translation on z (related to baseline for the folded configuration) to use as separation between the drawings

        @return: The estimated transformation matrix of the object frame [G] with respect to the GUM sphere frame [M], and the app (if any 3D visualization with visvis)
        '''
        import omnistereo.transformations as tr

        # TEST: A simple test using forward projection. Get the object points wrt [C]:
        # obj_pts_wrt_C = np.einsum("ij, mnj->mni", T_G_wrt_C, obj_pts_homo)
        # mirror_points_3D_homo = self.theoretical_model.get_primer_point_for_projection(obj_pts_wrt_C)

        # Back projection to unit sphere using initial model parameters
        # FIXME: for the spherical rig, the distortion should be big, so this lifting doesn't work well
        sphere_points_3D = self.lift_pixel_to_unit_sphere_wrt_focus(img_points)  # Homogeneous
        # Append 1's for homogeneous coordinates
        ones_matrix = np.ones(sphere_points_3D.shape[:-1])
        sphere_points_3D_homo = np.dstack((sphere_points_3D, ones_matrix))

        # Approximate an image (reference) plane from the projections of the mirror points
        # Find the orientation of this plane using the 2 edges of the pattern on the mirror
#         b_orig = sphere_points_3D_homo[-1, 0, :3]  # For fake VIRTUAL camera as seen through an image (This is the upper-left corner)
        # Get the upper leftmost valid point
        # Scan along x and y for the first not NAN
        num_rows, num_cols = img_points.shape[:2]
        col_offset = 0
        num_max_diag = np.min([num_rows, num_cols])
        b_is_valid = False
        while not b_is_valid:  # Searching across the diagonal (see slide for clarification in this counter-intuitive arrangement)
            for i in range(num_max_diag):
                virtual_origin_pt_row = num_max_diag - 1 - i
                virtual_origin_pt_col = i + col_offset
                b_is_valid = not np.any(np.isnan(img_points[virtual_origin_pt_row, virtual_origin_pt_col]))
                if b_is_valid:
                    break
            # Don't exceed the search
            if not b_is_valid and col_offset < num_max_diag - 1:
                col_offset = col_offset + 1
            else:
                break

        if b_is_valid:
            b_orig = sphere_points_3D_homo[virtual_origin_pt_row, virtual_origin_pt_col, :3]  # For fake VIRTUAL camera as seen through an image (This is the upper-left corner)
        else:
            pass  # TODO: try search for any valid point

        # Removing NANs points for versor point candidates, and then selecting the farthest point wrt the b_orig on the respective direction
        v_hor_points = sphere_points_3D_homo[virtual_origin_pt_row, :, :3]
        v_hor_valid_point = [pt for pt in v_hor_points if not np.any(np.isnan(pt))][-1]  # Pick furthest valid point from b_orig (on same row)
        v_ver_points = sphere_points_3D_homo[:, virtual_origin_pt_col, :3]
        v_ver_valid_point = [pt for pt in v_ver_points if not np.any(np.isnan(pt))][0]  # Pick furthest valid point from b_orig (on same column)

        v_hor_on_fake_cam = v_hor_valid_point - b_orig  # like the x-axis (Previously, I was doing: sphere_points_3D_homo[-1, -1, :3] - b_orig)
        v_ver_on_fake_cam = v_ver_valid_point - b_orig  # like the y-axis (but not orthogonal) (Previously, I was doing: sphere_points_3D_homo[0, 0, :3] - b_orig
        b_normal_cross = np.cross(v_hor_on_fake_cam, v_ver_on_fake_cam)
        b_normal = b_normal_cross / np.linalg.norm(b_normal_cross)  # Unit normal vector of the camera plane
        # orthogonal projection
        M = tr.projection_matrix(point = b_orig, normal = b_normal, direction = None)
        proj_points_on_plane_wrt_C = np.einsum("ij, mnj->mni", M, sphere_points_3D_homo)
        # Find centroid point (a.k.a. optical center point) via an orthographic projecting of the focus onto the plane:
        # Trying to use the actual focal length
        # Pm = np.array([0, 0, 0, 1]).reshape(4, 1)
        Pm = self.F.reshape(4, 1)  # We don't know this yet (unless correctly initialized, but was getting the wrong estimations anyway)
        b_centroid = np.dot(M, Pm)[:3, 0]
        # Make plane orthogonal to focus of mirror and centroid point
        F_wrt_M = Pm[:3, 0]
        b_cent_to_F = b_centroid - F_wrt_M  # Normal to plane
        b_cent_to_F_mag = np.linalg.norm(b_cent_to_F)  # Norm

        cam_M_x_axis, cam_M_y_axis, cam_M_z_axis = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])  # for [C] frame
        fake_cam_hor_versor_points = proj_points_on_plane_wrt_C[virtual_origin_pt_row, :, :3]
        fake_cam_hor_versor_valid_point = [pt for pt in fake_cam_hor_versor_points if not np.any(np.isnan(pt))][-1]
        fake_cam_x_axis = fake_cam_hor_versor_valid_point - b_orig
        fake_cam_x_axis = fake_cam_x_axis / tr.vector_norm(fake_cam_x_axis)
        fake_cam_z_axis = b_normal
        # Resolve for the y-axis (orthogonally)
        fake_cam_y_axis = np.cross(fake_cam_z_axis, fake_cam_x_axis)
        fake_cam_y_axis = fake_cam_y_axis / tr.vector_norm(fake_cam_y_axis)

        # Attempt to do translation first (move the centroid to the origin), then rotation!
        proj_points_on_plane_wrt_C_offset = proj_points_on_plane_wrt_C[..., :3] - b_centroid
        # See http://www.starlino.com/dcm_tutorial.html to learn more about the DCM (Direction Cosine Matrix)
        b_DCM = np.array([[np.dot(cam_M_x_axis, fake_cam_x_axis), np.dot(cam_M_y_axis, fake_cam_x_axis), np.dot(cam_M_z_axis, fake_cam_x_axis)],
                          [np.dot(cam_M_x_axis, fake_cam_y_axis), np.dot(cam_M_y_axis, fake_cam_y_axis), np.dot(cam_M_z_axis, fake_cam_y_axis)],
                          [np.dot(cam_M_x_axis, fake_cam_z_axis), np.dot(cam_M_y_axis, fake_cam_z_axis), np.dot(cam_M_z_axis, fake_cam_z_axis)]
                          ])

        # De-rotate points on plane so coordinates are with respect to the fake camera plane
        # NOTE: rotation seems fine now with DCM, but points are offset on their z-coordinates by the orthogonal distance "mirror_proj_plane.k"
        # This is correct since the plane frame itself is given by normalized versors
        pts_fake_cam = np.einsum("ij, mnj->mni", b_DCM, proj_points_on_plane_wrt_C_offset)[..., :2]  # Remove the z-coordinate
        pt_origin = pts_fake_cam[virtual_origin_pt_row, virtual_origin_pt_col, :2]

        px_offset = -pt_origin[:2]
        pts_fake_cam = pts_fake_cam + px_offset  # Offset the optical center point
        # pts_fake_cam = pts_fake_cam - pt_origin  # Offset to origin
        px_centroid = -pt_origin
        # Fake camera matrix for solving PnP
        f = b_cent_to_F_mag  # [mm] Fake camera's focal length. Recall f_u = f / hx
        h_px = 1  # 1 [mm]/[px]
        c_u, c_v = px_centroid
        K_cam = np.array([[f / h_px, 0, c_u], [0, f / h_px, c_v], [0, 0, 1]])

        pts_fake_cam_single_row = pts_fake_cam.reshape(-1, 2)
        valid_pts_fake_cam = np.array([pt for pt in pts_fake_cam_single_row if not np.any(np.isnan(pt))])
        if use_PnP:
            from cv2 import solvePnP
            obj_pts_as_single_row = obj_pts_homo[..., :3].reshape(-1, 3)
            valid_obj_pts = np.array([pt for pt in obj_pts_as_single_row if not np.any(np.isnan(pt))])
            retval, rot_vec_G_wrt_fake_cam, tr_vec_G_wrt_fake_cam = solvePnP(valid_obj_pts.astype(np.float32), valid_pts_fake_cam.astype(np.float32), K_cam, distCoeffs = None)
            # from cv2 import solvePnPRansac
            # rot_vec_G_wrt_fake_cam, tr_vec_G_wrt_fake_cam, inliers = solvePnPRansac(obj_pts_homo[..., :3].reshape(-1, 3).astype(np.float32), pts_fake_cam.reshape(-1, 2).astype(np.float32), K_cam, distCoeffs=None)
            # Convert Rodrigues rotation vector to rotation matrix
            from cv2 import Rodrigues
            R_G_wrt_fake_cam, rot_jacobian = Rodrigues(rot_vec_G_wrt_fake_cam)
            T_G_wrt_fake_cam = np.identity(4)  # 4x4 transformation matrix
            # Note: PnP follows a homogeneous transformation: first doing rotation first and then translation
            T_G_wrt_fake_cam[:3, :3] = R_G_wrt_fake_cam  # Rotation
            T_G_wrt_fake_cam[:3, 3] = tr_vec_G_wrt_fake_cam[:, 0]  # translation
        else:
            # vvvvvvvvvvvvvvv PLANAR HOMOGRAPHY vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            K_cam_inv = tr.inverse_matrix(K_cam)
            T_obj_pts_3D_to_2D = tr.rotation_matrix(np.deg2rad(-90), cam_M_x_axis)
            obj_pts = np.einsum("ij, mnj->mni", T_obj_pts_3D_to_2D, obj_pts_homo)
            obj_pts_2D = obj_pts[..., :2]
            # Reshape and remove NANs from data set
            obj_2D_pts_as_single_row = obj_pts_2D.reshape(-1, 2)
            valid_obj_2D_pts = np.array([pt for pt in obj_2D_pts_as_single_row if not np.any(np.isnan(pt))])
            from cv2 import findHomography
            # NOTE: method 0 is the quickest, just based on computing homography using DLT (I think). Results seem precise enough
            H, mask = findHomography(valid_obj_2D_pts.astype(np.float32), valid_pts_fake_cam.astype(np.float32), method = 0)

            # The scaling parameter is determined from the orthonormality condition
            lamb = 1 / np.linalg.norm(np.dot(K_cam_inv, H[:, 0]))
            R = np.zeros_like(H)
            R[:, 0] = lamb * np.dot(K_cam_inv, H[:, 0])
            R[:, 1] = lamb * np.dot(K_cam_inv, H[:, 1])
            R[:, 2] = np.cross(R[:, 0], R[:, 1])
            # We need to enforce orthonormality of the rotation matrix? Results are then decomposable without scale!
            U, S, V = np.linalg.svd(R)
            R_G_wrt_fake_cam = np.dot(U, V)

            tr_G_wrt_fake_cam = lamb * np.dot(K_cam_inv, H[:, 2])  # Translation vector

            T_G_wrt_fake_cam = np.identity(4)  # 4x4 transformation matrix
            # Note: PnP follows a homogeneous transformation: first doing rotation first and then translation
            T_G_wrt_fake_cam[:3, :3] = R_G_wrt_fake_cam  # Rotation
            T_G_wrt_fake_cam[:3, 3] = tr_G_wrt_fake_cam  # translation
            # Include the initial object rotation:
            T_G_wrt_fake_cam = tr.concatenate_matrices(T_G_wrt_fake_cam, T_obj_pts_3D_to_2D)
            # ^^^^^^^^^^^^^^^ PLANAR HOMOGRAPHY ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # Recall that the camera frame uses y-axis pointing down and  z-axis pointing outwards.
        rot_M_wrt_fake_cam = np.identity(4)  # 4x4 transformation matrix
        rot_M_wrt_fake_cam[:3, :3] = b_DCM  # Rotation
        # TEST: rotation angles are with respect to the "static" reference frame and are given as (ang_x, ang_y, ang_z)
        #       where R = Rz Ry Rx, so rotation around x-axis happens first, then around y, end then the fixed z-axis
        # >>>> scale, shear, angles, trans, persp = tr.decompose_matrix(rot_M_wrt_fake_cam)
        # >>>> rot_test = tr.euler_matrix(*angles, axes="sxyz")
        # >>>> np.allclose(rot_test, rot_M_wrt_fake_cam)
        rot_fake_cam_wrt_M = tr.inverse_matrix(rot_M_wrt_fake_cam)
        trans_fake_cam_wrt_M = tr.translation_matrix(F_wrt_M)
        # We must do rotation first and then translation:
        T_fake_cam_wrt_M = tr.concatenate_matrices(trans_fake_cam_wrt_M, rot_fake_cam_wrt_M)

        # Transformation of [G] wrt [M]
        T_G_wrt_M_est = tr.concatenate_matrices(T_fake_cam_wrt_M, T_G_wrt_fake_cam)
        # TEST: >>> scale, shear, angles, trans, persp = tr.decompose_matrix(T_G_wrt_M_est)
        # Put wrt to [C] frame
        T_G_wrt_C_est = np.identity(4)  # 4x4 transformation matrix
        T_M_wrt_C = tr.translation_matrix(self.F[:3, 0])
        T_G_wrt_C_est = tr.concatenate_matrices(T_M_wrt_C, T_G_wrt_M_est)
        (real_origin_pt_row, real_origin_pt_col) = (0, 0)

        # TODO: offset the flipping of coordinates on the vertical length of the grid
        # Testing for now just by offsetting the difference among the actual grid origin and the fake origin
        #=======================================================================
        # diff_flipped_for_grid = obj_pts_homo[virtual_origin_pt_row, virtual_origin_pt_col, :3] - obj_pts_homo[real_origin_pt_row, real_origin_pt_col, :3]
        # trans_diff_flipped_for_grid = tr.translation_matrix(diff_flipped_for_grid)
        # T_G_wrt_C_est = tr.concatenate_matrices(T_G_wrt_C_est, trans_diff_flipped_for_grid)
        #=======================================================================

        # Visualization
        if visualize:
            if self.mirror_name == "top":
                grid_color = 'c'
            else:
                grid_color = 'm'

            import visvis as vv
            from omnistereo.common_plot import draw_GUM_visvis
            gum_viz_scale = 1. / 100.
            app = draw_GUM_visvis(app = app, model = self, finish_drawing = False, mirror_transparency = 0.5, show_labels = True, show_only_real_focii = False, show_reference_frame = True, show_grid_box = False, busy_grid = False, z_offset = z_offset, scale = gum_viz_scale)
            a = vv.gca()
            # TODO: use z_offset for point's coordinates

            norm_length_vis = 1.0 * gum_viz_scale
            b_orig_vis = b_orig * gum_viz_scale
            # Draw Fake optical-axis
            #===================================================================
            # pp_o = vv.Pointset(3)
            # pp_o.append(b_centroid); pp_o.append(mirror_focus_point[:3, 0]);
            # pp_o.append(b_centroid + norm_length_vis * 1.5 * fake_cam_z_axis);
            # line_n = vv.Line(a, pp_o)
            # line_n.ls = '--'  # line_style
            # line_n.lw = 2  # line_thickness
            # line_n.lc = grid_color
            #===================================================================

            # TOO BUSY:
            #===================================================================
            # # Draw fake cam's x-axis
            # pp_cam_x = vv.Pointset(3)
            # pp_cam_x.append(F_wrt_C); pp_cam_x.append(F_wrt_C + norm_length_vis * fake_cam_x_axis);
            # line_cam_x = vv.Line(a, pp_cam_x)
            # line_cam_x.ls = '-'  # line_style
            # line_cam_x.lw = 1  # line_thickness
            # line_cam_x.lc = "r"
            #
            # # Draw fake cam's y-axis
            # pp_cam_y = vv.Pointset(3)
            # pp_cam_y.append(F_wrt_C); pp_cam_y.append(F_wrt_C + norm_length_vis * fake_cam_y_axis);
            # line_cam_y = vv.Line(a, pp_cam_y)
            # line_cam_y.ls = '-'  # line_style
            # line_cam_y.lw = 1  # line_thickness
            # line_cam_y.lc = "g"
            #===================================================================

            # Draw fake image plane's x-axis
            pp_x = vv.Pointset(3)
            pp_x.append(b_orig_vis); pp_x.append(b_orig_vis + norm_length_vis * fake_cam_x_axis);
            line_x = vv.Line(a, pp_x)
            line_x.ls = '-'  # line_style
            line_x.lw = 1  # line_thickness
            line_x.lc = "r"

            # Draw fake image plane's y-axis
            pp_y = vv.Pointset(3)
            pp_y.append(b_orig_vis); pp_y.append(b_orig_vis + norm_length_vis * fake_cam_y_axis);
            line_y = vv.Line(a, pp_y)
            line_y.ls = '-'  # line_style
            line_y.lw = 1  # line_thickness
            line_y.lc = "g"

            # Draw fake image plane's z-axis
            pp_n = vv.Pointset(3)
            pp_n.append(b_orig_vis); pp_n.append(b_orig_vis + norm_length_vis * fake_cam_z_axis);
            line_n = vv.Line(a, pp_n)
            line_n.ls = '-'  # line_style
            line_n.lw = 1  # line_thickness
            line_n.lc = "b"

            #===================================================================
            # # Draw Fake camera grid
            # xx = proj_points_on_plane_wrt_C[..., 0]
            # yy = proj_points_on_plane_wrt_C[..., 1]
            # zz = proj_points_on_plane_wrt_C[..., 2]
            # plane_grid = vv.grid(xx, yy, zz, axesAdjust=True, axes=a)
            # plane_grid.edgeColor = grid_color  # (1., 1., 0., 1.0)  # yellow
            # plane_grid.edgeShading = "plain"  # possible shaders: None, plain, flat, gouraud, smooth
            # plane_grid.diffuse = 0.0
            # # Lower right point (just because!)
            # plane_lr_pt = vv.Point(proj_points_on_plane_wrt_C[0, -1, 0], proj_points_on_plane_wrt_C[0, -1, 1], proj_points_on_plane_wrt_C[0, -1, 2])
            # vv.plot(plane_lr_pt, ms='.', mc=grid_color, mw=5, ls='', mew=0, axesAdjust=False)
            # # First point (lower left in grid board)
            # plane_ll_pt = vv.Point(proj_points_on_plane_wrt_C[0, 0, 0], proj_points_on_plane_wrt_C[0, 0, 1], proj_points_on_plane_wrt_C[0, 0, 2])
            # vv.plot(plane_ll_pt, ms='.', mc='r', mw=5, ls='', mew=0, axesAdjust=False)
            #===================================================================

            # Draw real calibration grid in the world
            #===================================================================
            # try:
            #     obj_pts_wrt_C = np.einsum("df,mnf->mnd", T_G_wrt_C, obj_pts_homo)  # points in actual position
            #     xx_obj_pts_wrt_C = obj_pts_wrt_C[..., 0]
            #     yy_obj_pts_wrt_C = obj_pts_wrt_C[..., 1]
            #     zz_obj_pts_wrt_C = obj_pts_wrt_C[..., 2]
            #     obj_pts_wrt_C_grid = vv.grid(xx_obj_pts_wrt_C, yy_obj_pts_wrt_C, zz_obj_pts_wrt_C, axesAdjust=True, axes=a)
            #     obj_pts_wrt_C_grid.edgeColor = 'g'  # Green for ground truth
            #     obj_pts_wrt_C_grid.edgeShading = "plain"  # possible shaders: None, plain, flat, gouraud, smooth
            #     obj_pts_wrt_C_grid.diffuse = 0.0
            # except:
            #     print("Warning: No Ground Truth T_G_wrt_C available in %s" % (__name__))
            #===================================================================

            # Draw enboxing projection to Ground Truth origin
            #===================================================================
            # pp_g_ll = vv.Pointset(3)
            # pp_g_ll.append(obj_pts_wrt_C[0, 0, :3]); pp_g_ll.append(self.theoretical_model.F[:3, 0]);
            # line_g_ll = vv.Line(a, pp_g_ll)
            # line_g_ll.ls = '--'  # line_style
            # line_g_ll.lw = 1  # line_thickness
            # line_g_ll.lc = (0.8, .5, 0)
            #===================================================================

            # Draw estimated calibration grid in the world
            obj_pts_est = np.einsum("ij, mnj->mni", T_G_wrt_C_est, obj_pts_homo)
            xx_obj_pts_est = obj_pts_est[..., 0]
            yy_obj_pts_est = obj_pts_est[..., 1]
            zz_obj_pts_est = obj_pts_est[..., 2]
            obj_pts_est_grid = vv.grid(xx_obj_pts_est, yy_obj_pts_est, zz_obj_pts_est, axesAdjust = True, axes = a)
            obj_pts_est_grid.edgeColor = grid_color  # color?
            obj_pts_est_grid.edgeShading = "plain"  # possible shaders: None, plain, flat, gouraud, smooth
            obj_pts_est_grid.diffuse = 0.0
            Og_pt = vv.Point(obj_pts_est[real_origin_pt_row, real_origin_pt_col, 0], obj_pts_est[real_origin_pt_row, real_origin_pt_col, 1], obj_pts_est[real_origin_pt_row, real_origin_pt_col, 2])
            vv.plot(Og_pt, ms = '.', mc = "k", mw = 5, ls = '', mew = 0, axesAdjust = False)

            # Draw 4 corners of points on mirror surface (color encoded)
            xx = sphere_points_3D_homo[..., 0] * gum_viz_scale
            yy = sphere_points_3D_homo[..., 1] * gum_viz_scale
            zz = sphere_points_3D_homo[..., 2] * gum_viz_scale
            plane_grid = vv.grid(xx, yy, zz, axesAdjust = True, axes = a)
            plane_grid.edgeColor = grid_color  # (1., 1., 0., 1.0)  # yellow
            plane_grid.edgeShading = "plain"  # possible shaders: None, plain, flat, gouraud, smooth
            plane_grid.diffuse = 0.0
            # 00 point (origin) to estimated calibration grid pose
            pt_00 = sphere_points_3D_homo[real_origin_pt_row, real_origin_pt_col, :3] * gum_viz_scale
            plane_lr_pt = vv.Point(pt_00[0], pt_00[1], pt_00[2])
            vv.plot(plane_lr_pt, ms = '.', mc = 'k', mw = 6, ls = '', mew = 0, axesAdjust = False)
            pp_00 = vv.Pointset(3)
            pp_00.append(obj_pts_est[real_origin_pt_row, real_origin_pt_col, :3]); pp_00.append(Pm[:3, 0] * gum_viz_scale);
            line_00 = vv.Line(a, pp_00)
            line_00.ls = '--'  # line_style
            line_00.lw = 1  # line_thickness
            line_00.lc = grid_color

            # row=0,col=-1 point
            #===================================================================
            # pt_0l = sphere_points_3D_homo[0, -1, :3]
            # plane_0l_pt = vv.Point(pt_0l[0], pt_0l[1], pt_0l[2])
            # vv.plot(plane_0l_pt, ms='.', mc='r', mw=6, ls='', mew=0, axesAdjust=False)
            # pp_0l = vv.Pointset(3)
            # pp_0l.append(obj_pts_est[0, -1, :3]); pp_0l.append(Pm[:3, 0]);
            # line_0l = vv.Line(a, pp_0l)
            # line_0l.ls = '--'  # line_style
            # line_0l.lw = 1  # line_thickness
            # line_0l.lc = grid_color
            #===================================================================

            # row=-1,col=-1 point
            #===================================================================
            # pt_ll = sphere_points_3D_homo[-1, -1, :3]
            # plane_ll_pt = vv.Point(pt_ll[0], pt_ll[1], pt_ll[2])
            # vv.plot(plane_ll_pt, ms='.', mc='g', mw=6, ls='', mew=0, axesAdjust=False)
            # pp_ll = vv.Pointset(3)
            # pp_ll.append(obj_pts_est[-1, -1, :3]); pp_ll.append(Pm[:3, 0]);
            # line_ll = vv.Line(a, pp_ll)
            # line_ll.ls = '--'  # line_style
            # line_ll.lw = 1  # line_thickness
            # line_ll.lc = grid_color
            #===================================================================

            if finish_drawing:
                # Start app
                vv.title('Approximated Grid Pose')
                app.Run()

        return T_G_wrt_C_est, app

    def estimate_inverse_distortion_params(self, n_random_points = 1000, visualize = False):
        from scipy.optimize import least_squares

        # Step 1) Generate control points as a geodesic sample space of points on the unit sphere
        # Generates N events on the sphere in three dimensions:
        n = 3  # or any positive integer
        x = np.random.normal(size = (n_random_points, n))
        x /= np.linalg.norm(x, axis = 1)[:, np.newaxis]
        points_on_sphere_wrt_M = x[np.newaxis, ...]

        # Step 2) Project points to normalized undistorted plane
        #     a.k.a. self.p_undistorted_geo_sample

        # Top:
        # ..................................................................................
        points_on_sphere_wrt_Cp = self.get_points_on_sphere_wrt_Cp(points_on_sphere_wrt_M)
        p_undistorted_geo_sample_all = self.get_undistorted_points(points_on_sphere_wrt_Cp)

        # Step 3) Distort control points using existing distortion parameters k1, k2
        #     a.k.a. self.p_distorted_geo_sample
        p_distorted_geo_sample_all = self.get_distorted_points(p_undistorted_geo_sample_all)
        _, _, m_sample_homo = self.get_pixel_from_point_on_image_plane(p_distorted_geo_sample_all)
        # Validate points to be found in the ROI (radial bounds)
        radial_distances_outer = np.linalg.norm(m_sample_homo[..., :2] - self.precalib_params.center_point_outer, axis = -1)
        radial_distances_inner = np.linalg.norm(m_sample_homo[..., :2] - self.precalib_params.center_point_inner, axis = -1)
        valid_outer = np.where(radial_distances_outer < self.outer_img_radius, True, False)
        valid_inner = np.where(radial_distances_inner > self.inner_img_radius, True, False)
        valid_indices = np.logical_and(valid_inner, valid_outer)
        m_valid_sample_homo = m_sample_homo[valid_indices][np.newaxis, ...]
        self.p_undistorted_geo_sample = p_undistorted_geo_sample_all[valid_indices][np.newaxis, ...]
        self.p_distorted_geo_sample = p_distorted_geo_sample_all[valid_indices][np.newaxis, ...]

        # visualize sample on image
        if visualize:
            from omnistereo.common_cv import draw_points
            import cv2
            img_sample_vis = self.current_omni_img.copy()
            sample_projected_color = (255, 0, 0)  # red because (R,G,B)
            draw_points(img_sample_vis, m_valid_sample_homo[..., :2].reshape(-1, 2), color = sample_projected_color, thickness = 3)
            sample_proj_win = "Sphere Points Sample - %s mirror" % (self.mirror_name)
            cv2.namedWindow(sample_proj_win, cv2.WINDOW_NORMAL)
            cv2.imshow(sample_proj_win, img_sample_vis)
            cv2.waitKey(1)
            # cv2.destroyWindow(sample_proj_win)

        initial_undistortion_parameters = [0., 0., 0.]
        bounds_for_undistortion = ([-10, -10, -10], [10, 10, 10])
        f_scale_undistort_opt = 0.001
        undistort_params_result = least_squares(fun = self.point_undistortion_error, x0 = initial_undistortion_parameters, f_scale = f_scale_undistort_opt, method = "trf", loss = "soft_l1", args = (), verbose = 2, max_nfev = 10000, bounds = bounds_for_undistortion)
        print("MONO (%s mirror) radial undistortion results:\n" % (self.mirror_name), undistort_params_result)
        l1_opt = undistort_params_result.x[0]
        l2_opt = undistort_params_result.x[1]
        l3_opt = undistort_params_result.x[2]
        # from scipy.optimize import minimize
        # undistort_params_result = minimize(fun=self.point_undistortion_error, x0=initial_undistortion_parameters, args=(), method=self.opt_method, jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options={'maxiter' : 20000, 'disp' : True, })
        # Update undistort params to model
        self.update_undistortion_params(l1 = l1_opt, l2 = l2_opt, l3 = l3_opt)
        self.print_precalib_params(header_message = "FINAL Intrinsic parameters:", plot_distortions = True)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def point_undistortion_error(self, x):
        # Step 1) Generate control points as a geodesic sample space of points on the unit sphere
        # Step 2) Project points to normalized undistorted plane
        #     a.k.a. self.p_undistorted_geo_sample
        # Step 3) Distort control points using existing distortion parameters k1, k2
        #     a.k.a. self.p_distorted_geo_sample
        # ^^^^^^^^^^^^^^^^^^
        # The first three steps happen only once outside the optimization iterations
        #
        # Step 4) Undistort back distorted control points using the current "undistortion" parameters l1,l2
        p_undistorted = self.get_radially_distorted_points(p = self.p_distorted_geo_sample, dist_params = x)

        # Step 5) The total cost is the L2 norms between the undistorted control points obtained during forward projection and the undistorted parameters
        p_diffs = p_undistorted - self.p_undistorted_geo_sample
        # For Least-Squares optimization:
        residuals_as_matrix = np.linalg.norm(p_diffs, axis = -1)
        residuals = residuals_as_matrix.ravel()
        return residuals

        #=======================================================================
        # total_cost = np.sum(p_diffs ** 2)
        # return total_cost
        #=======================================================================

    def get_points_on_sphere_wrt_Cp(self, points_on_sphere_wrt_M):
        points_on_sphere_wrt_Cp = points_on_sphere_wrt_M - np.array(self.Cp_wrt_M)

        return points_on_sphere_wrt_Cp

    def get_points_wrt_M(self, points_wrt_C_homo):
        T_C_wrt_M = self.T_C_wrt_model
        points_wrt_M = np.einsum("ij, klj->kli", T_C_wrt_M, points_wrt_C_homo)
        return points_wrt_M

    def get_undistorted_points(self, points_on_sphere_wrt_Cp):
        # p_und = points_on_sphere_wrt_Cp / (-self.z_axis * points_on_sphere_wrt_Cp[..., 2][..., np.newaxis ])  # Noticed the z_axis sign as to avoid flipped projections
        p_und = points_on_sphere_wrt_Cp / np.abs(points_on_sphere_wrt_Cp[..., 2][..., np.newaxis ])  # CHECKME: Dividing by the absolute value of the z-component instead as to avoid possible xy-coords inversions onto image plane
        return p_und[..., :2]  # Drop the z dimension

    def get_distorted_points(self, p_und):
        p_distorted = self.get_radially_distorted_points(p_und, dist_params = [self.precalib_params.k1, self.precalib_params.k2, self.precalib_params.k3])
        return p_distorted

    # TODO: get rid of means and scale_factors, since they are not needed really.
#     def compute_jacobian_projection(self, extrinsic_parameters, means, scale_factors, points_wrt_G, calib_idx, do_only_grids_extrinsics=False, only_C_wrt_M_tz=False, only_translation_params=False, T_G_wrt_C_for_testing=None, return_jacobian=True, pts_scale_factor=1.):
    def compute_jacobian_projection(self, extrinsic_parameters, points_wrt_G, calib_idx, only_intrinsics = False, only_extrinsics = False, do_only_grids_extrinsics = False, only_C_wrt_M_tz = False, only_translation_params = False, T_G_wrt_C_for_testing = None, compute_jacobian = True, pts_scale_factor = 1.):
        '''
        The Jacobian of the projection function (using partial derivative composition), we get a 2 x |params| matrix
        @param extrinsic_parameters: the vector of ALL extrinsic parameters (grid poses and tz for mirror) wrt to the [C] frame to evaluate the jacobian with
        @param points_wrt_G: A matrix of 3D points with respect to the grid (calibration pattern) frame.
        @param calib_idx: The index of the pattern so the extrinsic Jacobians are filled appropriately
        @param only_intrinsics: It specifies whether to compute only the set of intrinsic parameters. When set to True, the extrinsic parameters won't be computed.

        @return The matrix of projection function jacobians for all points, and the matrix of projected points as well.
        '''

        if only_intrinsics:
            only_extrinsics = False
            do_only_grids_extrinsics = False
            only_C_wrt_M_tz = False
            only_translation_params = False

#         if do_only_grids_extrinsics == False:
#             self.F[:3, 0] = extrinsic_parameters[-3:]
            # TODO: set rotation as well!

#===============================================================================
#         # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#         # TEST Jacobian implementation d_of_fw_wrt_e:
#         # Ground truth is given as g:
#         # Use any arbitrary g to test
#         from scipy.optimize import check_grad
#         import transformations as tr
#
#         q_true = tr.quaternion_from_euler(np.deg2rad(0.00), np.deg2rad(0.00), np.deg2rad(-90.0000), axes="sxyz")  # [1, 0, 0, 0]
#         t_true = [1000, 0, 0]
#         g_true = list(q_true) + t_true
#         T_G_wrt_C_true = common_tools.get_transformation_matrix(g_true)
#
#         q_estimated = tr.quaternion_from_euler(np.deg2rad(0.005), np.deg2rad(2.005), np.deg2rad(-91.002), axes="sxyz")  # [1, 0, 0, 0]
#         t_estimated = [900, 10, 10]
#         g_estimated = list(q_estimated) + t_estimated  # 1 grid only
#         # GUM's pose
#         tz_estimated = [-self.theoretical_model.F[2, 0] + 0.5]
#         params_estimated = g_estimated + tz_estimated
#
#         def func_fw_wrt_e_est(p_est, pts_wrt_G, c_idx, only_grids_extrinsics, only_gum_extrinsic, ret_jac):
#             _, points_wrt_M_est, _ = self.der_of_fw_wrt_e(p_est, means, scale_factors, pts_wrt_G, c_idx, only_grids_extrinsics, only_gum_extrinsic, T_G_wrt_C_true, compute_jacobian=False)  # TODO: pasing only g params for now
#             # Transform between G and M
#             T_C_wrt_M_true = common_tools.get_transformation_matrix([1, 0, 0, 0, 0, 0, -self.theoretical_model.F[2, 0]])
#             T_G_wrt_M_true = tr.concatenate_matrices(T_C_wrt_M_true, T_G_wrt_C_true)
#             points_wrt_M_true = np.einsum("ij, klj->kli", T_G_wrt_M_true, pts_wrt_G)
#
#             points_diff_norm = np.linalg.norm(points_wrt_M_est - points_wrt_M_true, axis=-1)
#             sum_diffs = np.sum(points_diff_norm)
#             return sum_diffs
#
#         def jac_fw_wrt_e_est(p_est, pts_wrt_G, c_idx, only_grids_extrinsics, only_gum_extrinsic, ret_jac):
#             d_of_fw_wrt_e_est, points_wrt_M_est, _ = self.der_of_fw_wrt_e(p_est, means, scale_factors, pts_wrt_G, c_idx, only_grids_extrinsics, only_gum_extrinsic, T_G_wrt_C_true, ret_jac)
#             # Transform between G and M
#             T_C_wrt_M_true = common_tools.get_transformation_matrix([1, 0, 0, 0, 0, 0, -self.theoretical_model.F[2, 0]])
#             T_G_wrt_M_true = tr.concatenate_matrices(T_C_wrt_M_true, T_G_wrt_C_true)
#             points_wrt_M_true = np.einsum("ij, klj->kli", T_G_wrt_M_true, pts_wrt_G)
#             points_diff_norm = np.linalg.norm(points_wrt_M_est - points_wrt_M_true, axis=-1)
#             x_in_C_est = points_wrt_M_est[..., 0]
#             x_in_C_true = points_wrt_M_true[..., 0]
#             y_in_C_est = points_wrt_M_est[..., 1]
#             y_in_C_true = points_wrt_M_true[..., 1]
#             z_in_C_est = points_wrt_M_est[..., 2]
#             z_in_C_true = points_wrt_M_true[..., 2]
#             d_of_fdiff_wrt_x_in_M = (x_in_C_est - x_in_C_true) / points_diff_norm
#             d_of_fdiff_wrt_y_in_M = (y_in_C_est - y_in_C_true) / points_diff_norm
#             d_of_fdiff_wrt_z_in_M = (z_in_C_est - z_in_C_true) / points_diff_norm
#             d_of_fdiff_wrt_P_in_M = np.append(d_of_fdiff_wrt_x_in_M[..., np.newaxis], d_of_fdiff_wrt_y_in_M[..., np.newaxis], axis=-1)
#             d_of_fdiff_wrt_P_in_M = np.append(d_of_fdiff_wrt_P_in_M, d_of_fdiff_wrt_z_in_M[..., np.newaxis], axis=-1)
#
#             # The difference partial derivative
#             # Simply, multiply each difference by the rotation jacobian
#             d_fw_diffs = np.einsum("ijt, ijtf->ijf", d_of_fdiff_wrt_P_in_M, d_of_fw_wrt_e_est)
#
#             # Tested among all elements (using my HACK):
#             # >>>> np.all([np.allclose(np.dot(points_diffs[i,j], d_of_fH_wrt_e_est[i,j]),d_fH_diffs[i,j]) for (i,j), x in np.ndenumerate(d_fH_diffs[...,0])])
#
#             gradient_of_all_diffs_sum = np.einsum("ijk -> k", d_fw_diffs)
#             return gradient_of_all_diffs_sum
#
#         err_fw_jacs = check_grad(func_fw_wrt_e_est, jac_fw_wrt_e_est, params_estimated, points_wrt_G, calib_idx, do_only_grids_extrinsics, only_C_wrt_M_tz, compute_jacobian)
#         # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#===============================================================================

        # Using composition of partial derivatives (See paper/presentation where I explain this):
#         d_of_fw_wrt_e, points_wrt_M, T_G_wrt_M = self.der_of_fw_wrt_e(extrinsic_parameters, means, scale_factors, points_wrt_G, calib_idx, do_only_grids_extrinsics, only_C_wrt_M_tz, only_translation_params, T_G_wrt_C_for_testing, compute_jacobian, pts_scale_factor)
        d_of_fw_wrt_e, points_wrt_M, T_G_wrt_M = self.der_of_fw_wrt_e(e = extrinsic_parameters,
                                                                      points_wrt_G = points_wrt_G,
                                                                      calib_idx = calib_idx,
                                                                      only_for_intrinsics = only_intrinsics,
                                                                      do_only_grids_extrinsics = do_only_grids_extrinsics,
                                                                      only_C_wrt_M_tz = only_C_wrt_M_tz,
                                                                      only_translation_params = only_translation_params,
                                                                      T_G_wrt_C_ground_truth_for_testing = T_G_wrt_C_for_testing,
                                                                      compute_jacobian = compute_jacobian,
                                                                      pts_scale_factor = pts_scale_factor)

        points_on_sphere_wrt_M = camera_models.get_normalized_points(points_wrt_M)
        points_on_sphere_wrt_Cp = self.get_points_on_sphere_wrt_Cp(points_on_sphere_wrt_M)
        projected_points_undistorted = self.get_undistorted_points(points_on_sphere_wrt_Cp)

        if compute_jacobian:
            import omnistereo.transformations as tr
            d_of_fH_wrt_xi = self.der_of_fH_wrt_xi(points_wrt_M * pts_scale_factor)

            if only_intrinsics:
                d_of_fH_wrt_e_AND_xi = d_of_fH_wrt_xi
            else:
                # Normalize points
    #             t = tr.translation_from_matrix(T_G_wrt_M)[:3]
    #             t_orig_norm = np.linalg.norm(t)
    #             points_wrt_M_normalized = points_wrt_M / t_orig_norm
    #             d_of_fH_wrt_fw = self.der_of_fH_wrt_fw(points_wrt_M_normalized)
                d_of_fH_wrt_fw = self.der_of_fH_wrt_fw(points_wrt_M * pts_scale_factor)
                # Using Einstein sum to multiply all the derivatives
                d_of_fH_wrt_e = np.einsum("ijkl, ijlm->ijkm", d_of_fH_wrt_fw, d_of_fw_wrt_e)
                # Tested among all elements (using my HACK):
                # >>>> np.all([np.allclose(np.dot(d_of_fH_wrt_fw[i,j], d_of_fw_wrt_e[i,j]),d_of_fH_wrt_e[i,j]) for (i,j), x in np.ndenumerate(d_of_fH_wrt_e[...,0,0])])

                # Concatenate both partials (extrinsic + projection point intrinsic parameter) derivative parts:
                d_of_fH_wrt_e_AND_xi = np.concatenate((d_of_fH_wrt_e, d_of_fH_wrt_xi), axis = -1)

#===============================================================================
#             # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#             # TEST Jacobian implementation d_of_fH_wrt_fw:
#             # Ground truth is given as g:
#             # Use any arbitrary g to test
#             from scipy.optimize import check_grad
#             import transformations as tr
#
#             q_true = tr.quaternion_from_euler(np.deg2rad(0.00), np.deg2rad(0.00), np.deg2rad(-90.0000), axes="sxyz")  # [1, 0, 0, 0]
#             t_true = [1000, 0, 0]
#             g_true = list(q_true) + t_true
#             T_G_wrt_C_true = common_tools.get_transformation_matrix(g_true)
#
#             q_estimated = tr.quaternion_from_euler(np.deg2rad(0.005), np.deg2rad(2.005), np.deg2rad(-91.002), axes="sxyz")  # [1, 0, 0, 0]
#             t_estimated = [900, 10, 10]
#             g_estimated = list(q_estimated) + t_estimated  # 1 grid only
#             # GUM's pose
#             tz_estimated = [-self.theoretical_model.F[2, 0] + 0.5]
#             params_estimated = g_estimated + tz_estimated
#
#             def func_fH_wrt_g_est(p_est, pts_wrt_G):
#                 _, points_wrt_M_est, _ = self.der_of_fw_wrt_e(p_est, means, scale_factors, pts_wrt_G, calib_idx, do_only_grids_extrinsics, only_C_wrt_M_tz, only_translation_params, T_G_wrt_C_true, compute_jacobian=False)
#                 points_on_sphere_wrt_M_est = camera_models.get_normalized_points(points_wrt_M_est)
#                 points_on_sphere_wrt_Cp_est = self.get_points_on_sphere_wrt_Cp(points_on_sphere_wrt_M_est)
#                 projected_points_undistorted_est = self.get_undistorted_points(points_on_sphere_wrt_Cp_est)
#
#                 T_C_wrt_M_true = common_tools.get_transformation_matrix([1, 0, 0, 0, 0, 0, -self.theoretical_model.F[2, 0]])
#                 T_G_wrt_M_true = tr.concatenate_matrices(T_C_wrt_M_true, T_G_wrt_C_true)
#                 points_wrt_M_true = np.einsum("ij, klj->kli", T_G_wrt_M_true, pts_wrt_G)
#                 points_on_sphere_wrt_M_true = camera_models.get_normalized_points(points_wrt_M_true)
#                 points_on_sphere_wrt_Cp_true = self.get_points_on_sphere_wrt_Cp(points_on_sphere_wrt_M_true)
#                 projected_points_undistorted_true = self.get_undistorted_points(points_on_sphere_wrt_Cp_true)
#
#                 points_diff_norm = np.linalg.norm(projected_points_undistorted_est - projected_points_undistorted_true, axis=-1)
#                 sum_diffs = np.sum(points_diff_norm)
#                 return sum_diffs
#
#             def jac_fH_wrt_g_est(p_est, pts_wrt_G):
#                 d_of_fw_wrt_e_est, points_wrt_M_est, _ = self.der_of_fw_wrt_e(p_est, means, scale_factors, pts_wrt_G)
#                 points_on_sphere_wrt_M_est = camera_models.get_normalized_points(points_wrt_M_est)
#                 points_on_sphere_wrt_Cp_est = self.get_points_on_sphere_wrt_Cp(points_on_sphere_wrt_M_est)
#                 projected_points_undistorted_est = self.get_undistorted_points(points_on_sphere_wrt_Cp_est)
#
#                 T_C_wrt_M_true = common_tools.get_transformation_matrix([1, 0, 0, 0, 0, 0, -self.theoretical_model.F[2, 0]])
#                 T_G_wrt_M_true = tr.concatenate_matrices(T_C_wrt_M_true, T_G_wrt_C_true)
#                 points_wrt_M_true = np.einsum("ij, klj->kli", T_G_wrt_M_true, pts_wrt_G)
#                 points_on_sphere_wrt_M_true = camera_models.get_normalized_points(points_wrt_M_true)
#                 points_on_sphere_wrt_Cp_true = self.get_points_on_sphere_wrt_Cp(points_on_sphere_wrt_M_true)
#                 projected_points_undistorted_true = self.get_undistorted_points(points_on_sphere_wrt_Cp_true)
#
#                 points_diff_norm = np.linalg.norm(projected_points_undistorted_est - projected_points_undistorted_true, axis=-1)
#                 x_und_est = projected_points_undistorted_est[..., 0]
#                 x_und_true = projected_points_undistorted_true[..., 0]
#                 y_und_est = projected_points_undistorted_est[..., 1]
#                 y_und_true = projected_points_undistorted_true[..., 1]
#                 d_of_fdiff_wrt_x_und_est = (x_und_est - x_und_true) / points_diff_norm
#                 d_of_fdiff_wrt_y_und_est = (y_und_est - y_und_true) / points_diff_norm
#                 d_of_fdiff_wrt_p_und_est = np.append(d_of_fdiff_wrt_x_und_est[..., np.newaxis], d_of_fdiff_wrt_y_und_est[..., np.newaxis], axis=-1)
#
#                 d_of_fH_wrt_fw_est = self.der_of_fH_wrt_fw(points_wrt_M_est)
#                 d_of_fH_wrt_e_est = np.einsum("ijkl, ijlm->ijkm", d_of_fH_wrt_fw_est, d_of_fw_wrt_e_est)
#
#                 # Projected points differences
#                 points_on_sphere_wrt_M_est = camera_models.get_normalized_points(points_wrt_M_est)
#                 points_on_sphere_wrt_Cp_est = self.get_points_on_sphere_wrt_Cp(points_on_sphere_wrt_M_est)
#                 projected_points_undistorted_est = self.get_undistorted_points(points_on_sphere_wrt_Cp_est)
#
#                 # The difference partial derivative
#                 # Simply, multiply each difference by the rotation jacobian
#                 d_fH_diffs = np.einsum("ijt, ijtf->ijf", d_of_fdiff_wrt_p_und_est, d_of_fH_wrt_e_est)
#                 # Tested among all elements (using my HACK):
#                 # >>>> np.all([np.allclose(np.dot(points_diffs[i,j], d_of_fH_wrt_e_est[i,j]),d_fH_diffs[i,j]) for (i,j), x in np.ndenumerate(d_fH_diffs[...,0])])
#
#                 gradient_of_all_diffs_sum = np.einsum("ijk -> k", d_fH_diffs)
#                 return gradient_of_all_diffs_sum
#
#             err_fH_jacs = check_grad(func_fH_wrt_g_est, jac_fH_wrt_g_est, params_estimated, (points_wrt_G))
#             # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#===============================================================================

        if self.precalib_params.use_distortion:
            p_distorted = self.get_distorted_points(projected_points_undistorted)
            if compute_jacobian:
                d_of_fd_wrt_fH = self.der_of_fd_wrt_fH(projected_points_undistorted)
#                 d_of_fd_wrt_e = np.einsum("ijkl, ijlm->ijkm", d_of_fd_wrt_fH, d_of_fH_wrt_e)
                d_of_fd_wrt_e_AND_xi = np.einsum("ijkl, ijlm->ijkm", d_of_fd_wrt_fH, d_of_fH_wrt_e_AND_xi)
                d_of_fd_wrt_d = self.der_of_fd_wrt_d(projected_points_undistorted)
                # Concatenate d_of_fd_wrt_e_AND_xi with to d_of_fd_wrt_d
                d_of_fd_wrt_e_AND_xi_AND_d = np.concatenate((d_of_fd_wrt_e_AND_xi, d_of_fd_wrt_d), axis = -1)

                # TEST:>>>> np.all([np.allclose(np.dot(d_of_fd_wrt_fH[i,j], d_of_fH_wrt_e[i,j]),d_of_fd_wrt_e[i,j]) for (i,j), x in np.ndenumerate(d_of_fd_wrt_e[...,0,0])])
                # Apply distortion to undistorted projected points
                d_of_fp_wrt_fd = self.der_of_fp_wrt_fd(p_distorted)

#                 d_of_fp_wrt_e = np.einsum("ijkl, ijlm->ijkm", d_of_fp_wrt_fd, d_of_fd_wrt_e)
                d_of_fp_wrt_e_AND_xi_AND_d = np.einsum("ijkl, ijlm->ijkm", d_of_fp_wrt_fd, d_of_fd_wrt_e_AND_xi_AND_d)
                # TEST:>>>> np.all([np.allclose(np.dot(d_of_fp_wrt_fd[i,j], d_of_fd_wrt_e[i,j]),d_of_fp_wrt_e[i,j]) for (i,j), x in np.ndenumerate(d_of_fp_wrt_e[...,0,0])])
                d_of_fp_wrt_c = self.der_of_fp_wrt_c(p_distorted)
                d_of_fp_wrt_e_AND_xi_AND_d_AND_c = np.concatenate((d_of_fp_wrt_e_AND_xi_AND_d, d_of_fp_wrt_c), axis = -1)

                d_of_fp_wrt_params = d_of_fp_wrt_e_AND_xi_AND_d_AND_c
        else:  # Without distortion:
            p_distorted = projected_points_undistorted
            if compute_jacobian:
                d_of_fp_wrt_fH = self.der_of_fp_wrt_fd(p_distorted)
#                 d_of_fp_wrt_e = np.einsum("ijkl, ijlm->ijkm", d_of_fp_wrt_fH, d_of_fH_wrt_e)

                # Make a bunch of zeros for d_of_fd_wrt_d
                result_shape_d = projected_points_undistorted.shape[:-1] + (2, 3)
                d_of_fd_wrt_d = np.repeat(np.zeros((1, 2, 3)), projected_points_undistorted[..., 0].size, axis = 0).reshape(result_shape_d)

                d_of_fd_wrt_e_AND_xi_AND_d = np.concatenate((d_of_fH_wrt_e_AND_xi, d_of_fd_wrt_d), axis = -1)
                d_of_fp_wrt_e_AND_xi_AND_d = np.einsum("ijkl, ijlm->ijkm", d_of_fp_wrt_fH, d_of_fd_wrt_e_AND_xi_AND_d)
                # TEST:>>>> np.all([np.allclose(np.dot(d_of_fp_wrt_fH[i,j], d_of_fH_wrt_e[i,j]),d_of_fp_wrt_e[i,j]) for (i,j), x in np.ndenumerate(d_of_fp_wrt_e[...,0,0])])

                # Make a bunch of zeros for d_of_fp_wrt_c
                result_shape_c = projected_points_undistorted.shape[:-1] + (2, 5)
                d_of_fp_wrt_c = np.repeat(np.zeros((1, 2, 5)), projected_points_undistorted[..., 0].size, axis = 0).reshape(result_shape_c)
                d_of_fp_wrt_e_AND_xi_AND_d_AND_c = np.concatenate((d_of_fp_wrt_e_AND_xi_AND_d, d_of_fp_wrt_c), axis = -1)

                d_of_fp_wrt_params = d_of_fp_wrt_e_AND_xi_AND_d_AND_c

            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    #         # TEST Jacobian implementation d_of_fH_wrt_fw:
    #         # Ground truth is given as g:
    #         # Use any arbitrary g to test
    #         import transformations as tr
    #         q_estimated = tr.quaternion_from_euler(np.deg2rad(3.005), np.deg2rad(2.005), np.deg2rad(100.002), axes="sxyz")  # [1, 0, 0, 0]
    #         t_estimated = [900, 10, 10]
    #         params_est = list(q_estimated) + t_estimated
    #
    #         def func_fd_wrt_g_est(p_est, pts_wrt_G):
    #             _, points_wrt_M_est = self.der_of_fw_wrt_e(p_est[0:7], pts_wrt_G)  # TODO: pasing only g params for now
    #             # Transform between G and C
    #             points_on_sphere_wrt_M_est = camera_models.get_normalized_points(points_wrt_M_est)
    #             points_on_sphere_wrt_Cp_est = self.get_points_on_sphere_wrt_Cp(points_on_sphere_wrt_M_est)
    #             projected_points_undistorted_est = self.get_undistorted_points(points_on_sphere_wrt_Cp_est)
    #             if self.precalib_params.use_distortion:
    #                 projected_points_distorted_est = self.get_distorted_points(projected_points_undistorted_est)
    #             else:  # Without distortion:
    #                 projected_points_distorted_est = projected_points_undistorted_est
    #
    #             points_diffs = projected_points_distorted_est - p_distorted
    #             sum_sq_diffs = 0.5 * np.sum(points_diffs ** 2)
    #             return sum_sq_diffs
    #
    #         def jac_fd_wrt_g_est(p_est, pts_wrt_G):
    #             d_of_fw_wrt_e_est, points_wrt_M_est = self.der_of_fw_wrt_e(p_est[0:7], pts_wrt_G)
    #
    #             # Projected points differences
    #             points_on_sphere_wrt_M_est = camera_models.get_normalized_points(points_wrt_M_est)
    #             points_on_sphere_wrt_Cp_est = self.get_points_on_sphere_wrt_Cp(points_on_sphere_wrt_M_est)
    #             projected_points_undistorted_est = self.get_undistorted_points(points_on_sphere_wrt_Cp_est)
    #             # Apply distortion to undistorted projected points
    #             projected_points_distorted_est = self.get_distorted_points(projected_points_undistorted)
    #
    #             d_of_fH_wrt_fw_est = self.der_of_fH_wrt_fw(points_wrt_M_est)
    #             d_of_fH_wrt_e_est = np.einsum("ijkl, ijlm->ijkm", d_of_fH_wrt_fw_est, d_of_fw_wrt_e_est)
    #             d_of_fd_wrt_fH_est = self.der_of_fd_wrt_fH(projected_points_undistorted_est)
    #             d_of_fd_wrt_e_est = np.einsum("ijkl, ijlm->ijkm", d_of_fd_wrt_fH_est, d_of_fH_wrt_e_est)
    #
    #             points_diffs = projected_points_distorted_est - p_distorted
    #
    #             # The difference partial derivative
    #             d_fd_diffs = np.einsum("ijk, ijkl->ijl", points_diffs[..., :3], d_of_fd_wrt_e_est)
    #             # Tested among all elements (using my HACK):
    #             # >>>> np.all([np.allclose(np.dot(points_diffs[i,j], d_of_fH_wrt_e_est[i,j]),d_fH_diffs[i,j]) for (i,j), x in np.ndenumerate(d_fH_diffs[...,0])])
    #
    #             gradient_of_all_diffs_sum = np.einsum("ijk -> k", d_fd_diffs)
    #             return gradient_of_all_diffs_sum
    #
    #         if self.precalib_params.use_distortion:
    #             err_fd_jacs = check_grad(func_fd_wrt_g_est, jac_fd_wrt_g_est, params_est, (points_wrt_G))
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # Step 6: Get the pixel in the image:
        # Finally, apply the generalized projection matrix
        u, v, m_homo = self.get_pixel_from_point_on_image_plane(p_distorted)

        if compute_jacobian:
            if only_extrinsics:
                # FIXME: For now, just extracting the extrinsic parameters on the Jacobian matrix
                d_of_fp_wrt_e = d_of_fp_wrt_params[..., :, :len(extrinsic_parameters)]
                return d_of_fp_wrt_e, m_homo
            else:
                return d_of_fp_wrt_params, m_homo
        else:
            return None, m_homo

    def der_of_fp_wrt_fd(self, projected_points_distorted):
        '''
        Partial derivative of f_proj wrt f_D

        @param projected_points_distorted: The projected points with distortion (used only for gathering the shape of the final matrix)

        @return: the multidimensional partial derivatives matrix for the points
        '''
        d_single_instance = np.array([[self.precalib_params.gamma1, self.precalib_params.gamma1 * self.precalib_params.alpha_c], [0, self.precalib_params.gamma2]])
        d_single_instance_normalized = d_single_instance  # np.identity(2)  # FIXME: use appropriate factor from input (when we get to this point in the future)
        result_shape = projected_points_distorted.shape[:-1] + d_single_instance_normalized.shape  # Get (data grid shape + (2,2))
        d_of_fp_wrt_fd = np.repeat(d_single_instance_normalized[np.newaxis, ...], projected_points_distorted[..., 0].size, axis = 0).reshape(result_shape)
        return d_of_fp_wrt_fd

    def der_of_fp_wrt_c(self, projected_points_distorted):
        '''
        Partial derivative of f_proj wrt c, where c is the camara projection matrix parameters: alpha, gamma1, gamma2, u_center, v_center

        @param projected_points_distorted: The projected points with distortion (used only for gathering the shape of the final matrix)

        @return: the multidimensional partial derivatives matrix for the points
        '''
        x_distorted = projected_points_distorted[..., 0]
        y_distorted = projected_points_distorted[..., 1]

        init_single_instance = np.array([[0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]])
        result_shape = projected_points_distorted.shape[:-1] + init_single_instance.shape  # Get (data grid shape + (2,5))
        d_of_fp_wrt_c = np.repeat(init_single_instance[np.newaxis, ...], projected_points_distorted[..., 0].size, axis = 0).reshape(result_shape)
        d_of_fp_wrt_c[..., 0, 0] = y_distorted * self.precalib_params.gamma1
        d_of_fp_wrt_c[..., 0, 1] = x_distorted + y_distorted * self.precalib_params.alpha_c
        d_of_fp_wrt_c[..., 1, 2] = y_distorted
        return d_of_fp_wrt_c

    def der_of_fd_wrt_fH(self, projected_points_undistorted):
        '''
        a.k.a. der_of_fd_wrt_fpi: Partial derivative of f_d (the distortion function) wrt f_H (the composite projection to the undistorted normal plane)

        @param projected_points_undistorted: The projected points onto the undistorted plane

        @return: the multidimensional partial derivatives matrix for the points
        '''
        # WISH: put this function inside the projection function so the points transformation is not redundant!
        x = projected_points_undistorted[..., 0]
        y = projected_points_undistorted[..., 1]

        rho_und_sq = x ** 2 + y ** 2
        factor = 1 + self.precalib_params.k1 * rho_und_sq + self.precalib_params.k2 * rho_und_sq ** 2 + self.precalib_params.k3 * rho_und_sq ** 4

        identity_2 = np.identity(2)[np.newaxis, ...]
        result_shape = x.shape + identity_2.shape[1:]  # Get (data grid shape + (2,2))
        d_of_fd_wrt_fpi = np.repeat(identity_2, x.size, axis = 0).reshape(result_shape)

        d_of_fd_wrt_fpi = factor[..., np.newaxis, np.newaxis] * d_of_fd_wrt_fpi
        return d_of_fd_wrt_fpi

    def der_of_fd_wrt_d(self, projected_points_undistorted):
        '''
        Partial derivative of f_d (the distortion function) wrt distortion parameters (k_dist_1, k_dist_2)

        @param projected_points_undistorted: The projected points onto the undistorted plane
        @return: the multidimensional partial derivatives matrix for the points
        '''
        # WISH: put this function inside the projection function so the points transformation is not redundant!
        x = projected_points_undistorted[..., 0]
        y = projected_points_undistorted[..., 1]

        rho_und_sq = x ** 2 + y ** 2

        # Column 0, der fd wrt k1
        r0c0 = x * rho_und_sq
        r1c0 = y * rho_und_sq
        d_of_fd_wrt_k_dist_1 = np.dstack((r0c0, r1c0))

        # Column 1, der fd wrt k2
        r0c1 = x * (rho_und_sq ** 2)
        r1c1 = y * (rho_und_sq ** 2)
        d_of_fd_wrt_k_dist_2 = np.dstack((r0c1, r1c1))

        d_of_fd_wrt_d = np.append(d_of_fd_wrt_k_dist_1[..., np.newaxis], d_of_fd_wrt_k_dist_2[..., np.newaxis], axis = -1)

        # Column 2, der fd wrt k3
        r0c2 = x * (rho_und_sq ** 4)
        r1c2 = y * (rho_und_sq ** 4)
        d_of_fd_wrt_k_dist_3 = np.dstack((r0c2, r1c2))

        d_of_fd_wrt_d = np.append(d_of_fd_wrt_d, d_of_fd_wrt_k_dist_3[..., np.newaxis], axis = -1)

        return d_of_fd_wrt_d

    def der_of_fH_wrt_fw(self, points_wrt_M):
        '''
        Partial derivative of fH wrt fw

        @param points_wrt_M: The points with coordinates with respect to M (the mirror or GUM frame)

        @return: the multidimensional partial derivatives matrix for the points
        '''
        # WISH: put this function inside the projection function so the points transformation is not redundant!
        x = points_wrt_M[..., 0]
        y = points_wrt_M[..., 1]
        z = points_wrt_M[..., 2]

        xi_x = self.Cp_wrt_M[0]
        xi_y = self.Cp_wrt_M[1]
        xi_z = self.Cp_wrt_M[2]

        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)  # Points' norms
        factor = -self.z_axis / (r * (z - r * xi_z) ** 2)
        # TEST:>>>> np.all([np.allclose(self.z_axis / (r[i,j] * (z[i,j] - r[i,j] * xi_z) ** 2),factor[i,j]) for (i,j), x in np.ndenumerate(factor)])

        # Column 0
        r0c0 = r * z - xi_x * x * z - xi_z * (y ** 2 + z ** 2)
        r1c0 = x * (y * xi_z - z * xi_y)
        d_of_fH_wrt_x = np.dstack((r0c0, r1c0))

        # Column 1
        r0c1 = y * (x * xi_z - z * xi_x)
        r1c1 = r * z - xi_y * y * z - xi_z * (x ** 2 + z ** 2)
        d_of_fH_wrt_y = np.dstack((r0c1, r1c1))

        # Column 2
        r0c2 = -r * x + xi_z * x * z + xi_x * (x ** 2 + y ** 2)
        r1c2 = -r * y + xi_z * y * z + xi_y * (x ** 2 + y ** 2)
        d_of_fH_wrt_z = np.dstack((r0c2, r1c2))

        d_of_fH_wrt_fw = np.append(d_of_fH_wrt_x[..., np.newaxis], d_of_fH_wrt_y[..., np.newaxis], axis = -1)
        d_of_fH_wrt_fw = np.append(d_of_fH_wrt_fw, d_of_fH_wrt_z[..., np.newaxis], axis = -1)
        d_of_fH_wrt_fw = factor[..., np.newaxis, np.newaxis] * d_of_fH_wrt_fw
        # TEST:
        # >>>> temp = np.append(d_of_fH_wrt_x[..., np.newaxis], d_of_fH_wrt_y[..., np.newaxis], axis=-1)
        # >>>> temp = np.append(temp, d_of_fH_wrt_z[..., np.newaxis], axis=-1)
        # >>>> np.all([np.allclose(factor[i,j]*temp[i,j],d_of_fH_wrt_fw[i,j]) for (i,j), x in np.ndenumerate(d_of_fH_wrt_fw[...,0,0])])
        return d_of_fH_wrt_fw

    def der_of_fH_wrt_xi(self, points_wrt_M):
        '''
        Partial derivative of fH wrt xi vector [xi_x, xi_y, xi_z]

        @param points_wrt_M: The points with coordinates with respect to M (the mirror or GUM frame)

        @return: the multidimensional partial derivatives matrix for the points
        '''
        x = points_wrt_M[..., 0]
        y = points_wrt_M[..., 1]
        z = points_wrt_M[..., 2]

        xi_x = self.Cp_wrt_M[0]
        xi_y = self.Cp_wrt_M[1]
        xi_z = self.Cp_wrt_M[2]

        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)  # Points' norms
        denominator = r * xi_z - z
        factor = r * (-1 * self.z_axis) / denominator
        # TEST:>>>> np.all([np.allclose(r[i,j] *self.z_axis / (r[i,j] * xi_z - z[i,j]), factor[i,j]) for (i,j), x in np.ndenumerate(factor)])

        identity_3 = np.identity(3)[np.newaxis, ...]
        result_shape = x.shape + identity_3.shape[1:]  # Get (data grid shape + (3,3))
        temp_d_of_fH_wrt_xi = np.repeat(identity_3, x.size, axis = 0).reshape(result_shape)

        # Implementing equation where values appear only in the xi_z column
        temp_d_of_fH_wrt_xi[..., 0, -1] = (x - r * xi_x) / denominator
        temp_d_of_fH_wrt_xi[..., 1, -1] = (y - r * xi_y) / denominator

        d_of_fH_wrt_xi = factor[..., np.newaxis, np.newaxis] * temp_d_of_fH_wrt_xi[..., :2, :]
        return d_of_fH_wrt_xi

#     def der_of_fw_wrt_e(self, e, means, scale_factors, points_wrt_G, calib_idx=0, do_only_grids_extrinsics=False, only_C_wrt_M_tz=False, only_translation_params=False, T_G_wrt_C_for_testing=None, return_jacobian=True, pts_scale_factor=1.):
    def der_of_fw_wrt_e(self, e, points_wrt_G, calib_idx = 0, only_for_intrinsics = False, do_only_grids_extrinsics = False, only_C_wrt_M_tz = False, only_translation_params = False, T_G_wrt_C_ground_truth_for_testing = None, compute_jacobian = True, pts_scale_factor = 1.):
        '''
        Partial derivative of fw wrt e = [g_g, s], where each pose is given as [q0, q1, q2, q3, tx, ty, tz]

        @param only_for_intrinsics: Helps for passing the external need to only compute intrinsic parameters, so here only the theoretical values will be used and returned

        @return: a (Nx3x14) matrix but + 7L???
        @return: the points wrt to the GUM frame
        '''
        # WISH: put this function inside the projection function so it's points transformation is not redundant!
        import omnistereo.transformations as tr

        if only_translation_params:
            param_offset = 3
        else:
            param_offset = 7

        if do_only_grids_extrinsics or only_for_intrinsics:
            # TODO: When using only the grid parameters (no s was given), we set this manually
            #===================================================================
            # s = 7 * [0]
            # tr_M_wrt_C = tr.translation_matrix(self.F[:3].reshape(3))  # Creates a 4x4 homogeneous matrix
            # tr_C_wrt_M = tr.inverse_matrix(tr_M_wrt_C)  # Invert
            # s[4:] = tr.translation_from_matrix(tr_C_wrt_M)
            # rot_C_wrt_M = tr.rotation_from_matrix(tr_C_wrt_M)
            # rot_mat_M_wrt_C = tr.rotation_matrix(rot_C_wrt_M[0], rot_C_wrt_M[1], rot_C_wrt_M[2])
            # s[:4] = tr.quaternion_from_matrix(rot_mat_M_wrt_C, isprecise=True)
            #===================================================================
            s = self.F[2, 0]
        else:
            s = e[-1]  # * scale_factors[-1] + means[-1]  # The pose of [M] wrt to [C]
            # s = e[-7:]  # The pose of [M] wrt to [C]

        # FIXME: even if no extrinsics are to estimated, is the Jacobian related to the "e" still needed?
        if only_C_wrt_M_tz or only_for_intrinsics:
            T_G_wrt_C = T_G_wrt_C_ground_truth_for_testing
        else:
            g = e[calib_idx * param_offset:calib_idx * param_offset + param_offset]
            # m = means[calib_idx * param_offset:calib_idx * param_offset + param_offset]
            # f = scale_factors[calib_idx * param_offset:calib_idx * param_offset + param_offset]
            g_scaled_up = g  # * f + m
            if compute_jacobian:
#===============================================================================
#                 # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#                 # TEST Jacobian implementation:
#                 # Ground truth is given as g:
#                 # Use any arbitrary g to test
#                 from scipy.optimize import check_grad
#                 import transformations as tr
#
#                 q_true = tr.quaternion_from_euler(np.deg2rad(0.00), np.deg2rad(0.00), np.deg2rad(-90.0000), axes="sxyz")  # [1, 0, 0, 0]
#                 t_true = [1000, 0, 0]
#                 g_true = list(q_true) + t_true
#                 q_estimated = tr.quaternion_from_euler(np.deg2rad(0.001), np.deg2rad(0.001), np.deg2rad(-90.01), axes="sxyz")  # [1, 0, 0, 0]
#                 t_estimated = [1000.001, 0.01, 0.01]
#                 g_estimated = list(q_estimated) + t_estimated
#                 def func_fg_wrt_g(g_est, pts_wrt_G):
#                     # Transform between G and C
#                     T_G_wrt_C_true = common_tools.get_transformation_matrix(g_true)
#                     pts_wrt_C_true = np.einsum("ij, klj->kli", T_G_wrt_C_true, pts_wrt_G)
#
#                     T_G_wrt_C_est = common_tools.get_transformation_matrix(g_est)
#                     pts_wrt_C_est = np.einsum("ij, klj->kli", T_G_wrt_C_est, pts_wrt_G)
#
#                     # pts_rot_est = np.dot(rot_matrix_est, pts)
#                     points_diff_norm = np.linalg.norm(pts_wrt_C_est - pts_wrt_C_true, axis=-1)
#                     sum_diffs = np.sum(points_diff_norm)
#                     return sum_diffs
#
#                 def jac_fg_wrt_g(g_est, pts_wrt_G):
#                     # Transform between G and C
#                     T_G_wrt_C_true = common_tools.get_transformation_matrix(g_true)
#                     pts_wrt_C_true = np.einsum("ij, klj->kli", T_G_wrt_C_true, pts_wrt_G)
#
#                     T_G_wrt_C_est = common_tools.get_transformation_matrix(g_est)
#                     pts_wrt_C_est = np.einsum("ij, klj->kli", T_G_wrt_C_est, pts_wrt_G)
#                     points_diff_norm = np.linalg.norm(pts_wrt_C_est - pts_wrt_C_true, axis=-1)
#                     x_in_C_est = pts_wrt_C_est[..., 0]
#                     x_in_C_true = pts_wrt_C_true[..., 0]
#                     y_in_C_est = pts_wrt_C_est[..., 1]
#                     y_in_C_true = pts_wrt_C_true[..., 1]
#                     z_in_C_est = pts_wrt_C_est[..., 2]
#                     z_in_C_true = pts_wrt_C_true[..., 2]
#                     d_of_fdiff_wrt_x_in_C = (x_in_C_est - x_in_C_true) / points_diff_norm
#                     d_of_fdiff_wrt_y_in_C = (y_in_C_est - y_in_C_true) / points_diff_norm
#                     d_of_fdiff_wrt_z_in_C = (z_in_C_est - z_in_C_true) / points_diff_norm
#                     d_of_fdiff_wrt_P_in_C = np.append(d_of_fdiff_wrt_x_in_C[..., np.newaxis], d_of_fdiff_wrt_y_in_C[..., np.newaxis], axis=-1)
#                     d_of_fdiff_wrt_P_in_C = np.append(d_of_fdiff_wrt_P_in_C, d_of_fdiff_wrt_z_in_C[..., np.newaxis], axis=-1)
#
#                     d_fg_wrt_g = self.der_of_fg_wrt_g(g_est, m, f, pts_wrt_G, pts_scale_factor)
#                     # The difference partial derivative
#                     # Simply, multiply each difference by the rotation jacobian
#                     d_fg_diffs = np.einsum("ijt, ijtf->ijf", d_of_fdiff_wrt_P_in_C, d_fg_wrt_g)
#                     # Tested among all elements (using my HACK):
#                     # >>>> np.all([np.allclose(np.dot(points_diffs[i,j], d_fg_wrt_g[i,j]),d_fg_diffs[i,j]) for (i,j), x in np.ndenumerate(d_fg_diffs[...,0])])
#                     gradient_of_all_diffs_sum = np.einsum("ijk -> k", d_fg_diffs)
#                     return gradient_of_all_diffs_sum
#
#                 err_fg_jacs = check_grad(func_fg_wrt_g, jac_fg_wrt_g, g_estimated, (points_wrt_G))
#                 # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#===============================================================================

#                 d_of_fg_wrt_g = self.der_of_fg_wrt_g(g_scaled_up, m, f, points_wrt_G, pts_scale_factor, only_translation_params)
                d_of_fg_wrt_g = self.der_of_fg_wrt_g(g_scaled_up, points_wrt_G, pts_scale_factor, only_translation_params)
                # The rotation of q_normlized_of_C_wrt_M
                # d_of_fw_wrt_fg = self.get_rot_matrix_from_quat(s[:4])
                # d_of_fw_wrt_g = np.einsum("ij, kljm->klim", d_of_fw_wrt_fg, d_of_fg_wrt_g)  # Einstein sum
                d_of_fw_wrt_g = d_of_fg_wrt_g  # Using coaxial alignment constraint of GUM

                # We need a 4x8x3x7
                # It can be checked manually (one by one), for example:
                # >>> U = np.random.rand(4,8,3,7)
                # >>> R = np.random.rand(3,3)
                # >>> result = np.einsum( "ij, kljm->klim", R, U)
                # >>> np.allclose(result[1,7],np.dot(R, U[1,7]))
                # >>> True
                # Or with my Hacky loop:
                # >>> np.all([np.allclose(np.dot(d_of_fw_wrt_fg, d_of_fg_wrt_g[i,j]),d_of_fw_wrt_g[i,j]) for (i,j), x in np.ndenumerate(d_of_fw_wrt_g[...,0,0])])

                # We fill zeroes for the grids indices that are not being treated.!!!!
                d_of_fw_wrt_e = np.zeros((d_of_fw_wrt_g.shape[:-1] + (len(e),)))
                d_of_fw_wrt_e[..., calib_idx * param_offset:calib_idx * param_offset + param_offset] = d_of_fw_wrt_g  # grid pose jacobian

            # Normalize quaternions part:
            # CHECKME: Maybe normalization is unnecessary, but just in case!
#             g[:4] = g[:4] / np.linalg.norm(g[:4])
#             # Denormalize points
#             g_scaled_up = g * f + m
            if only_translation_params:
                q_theoretical = tr.quaternion_from_matrix(T_G_wrt_C_ground_truth_for_testing, isprecise = False)
                T_G_wrt_C = common_tools.get_transformation_matrix(list(q_theoretical) + list(g_scaled_up))
            else:
                T_G_wrt_C = common_tools.get_transformation_matrix(g_scaled_up)

        # Transform between G and camera frame [C]
        points_wrt_C = np.einsum("ij, klj->kli", T_G_wrt_C, points_wrt_G)
        # TEST: check that transformation it is correct for all points:
        # >>> chess_coords_wrt_C_homo = points_wrt_G.dot(T_G_wrt_C.T)
        # >>> np.allclose(chess_coords_wrt_C_homo, points_wrt_C)
        # >>> True

        if do_only_grids_extrinsics == False and compute_jacobian:
            # TODO: the further a grid is, its points are more sensitive to error
            # That's why we need to make the tz translation also sensitive (scaled)
            d_of_fw_wrt_s = self.der_of_fw_wrt_s(s, points_wrt_C)  # / scale_factors[-1]
            if only_C_wrt_M_tz:
                d_of_fw_wrt_e = d_of_fw_wrt_s
            elif only_for_intrinsics:
                d_of_fw_wrt_e = []
            else:
#                 d_of_fw_wrt_e[..., -7:] = d_of_fw_wrt_s  # mirror pose jacobian
                d_of_fw_wrt_e[..., -1:] = d_of_fw_wrt_s  # mirror pose jacobian

            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#             # TEST Jacobian implementation:
#             from scipy.optimize import check_grad
#             # Ground truth is given as g:
#             # Use any arbitrary g to test
#             q_estimated = tr.quaternion_from_euler(np.deg2rad(0.005), np.deg2rad(0.005), np.deg2rad(1.002), axes="sxyz")  # [1, 0, 0, 0]
#             t_estimated = [0, 0, -123]
#             g_true = g
#             def func_fw_wrt_s(s_est, pts_wrt_G):
#                 # Transform between C wrt M
#                 T_C_wrt_M_true = common_tools.get_transformation_matrix(s)
#                 T_G_wrt_C_true = common_tools.get_transformation_matrix(g_true)
#                 # Transform between [G] and mirror [M]
#                 T_G_wrt_M_true = tr.concatenate_matrices(T_C_wrt_M_true, T_G_wrt_C_true)
#                 pts_wrt_M_true = np.einsum("ij, klj->kli", T_G_wrt_M_true, pts_wrt_G)
#
#                 T_C_wrt_M_est = common_tools.get_transformation_matrix(s_est)
#                 T_G_wrt_M_est = tr.concatenate_matrices(T_C_wrt_M_est, T_G_wrt_C_true)
#                 pts_wrt_M_est = np.einsum("ij, klj->kli", T_G_wrt_M_est, pts_wrt_G)
#
#                 points_diffs = pts_wrt_M_est - pts_wrt_M_true
#                 sum_sq_diffs = 0.5 * np.sum(points_diffs ** 2)
#                 return sum_sq_diffs
#
#             def jac_fw_wrt_s(s_est, pts_wrt_G):
#                 # Transform between C wrt M
#                 T_C_wrt_M_true = common_tools.get_transformation_matrix(s)
#                 T_G_wrt_C_true = common_tools.get_transformation_matrix(g_true)
#                 # Transform between [G] and mirror [M]
#                 T_G_wrt_M_true = tr.concatenate_matrices(T_C_wrt_M_true, T_G_wrt_C_true)
#                 pts_wrt_M_true = np.einsum("ij, klj->kli", T_G_wrt_M_true, pts_wrt_G)
#
#                 T_C_wrt_M_est = common_tools.get_transformation_matrix(s_est)
#                 T_G_wrt_M_est = tr.concatenate_matrices(T_C_wrt_M_est, T_G_wrt_C_true)
#                 pts_wrt_M_est = np.einsum("ij, klj->kli", T_G_wrt_M_est, pts_wrt_G)
#
#                 points_diffs = pts_wrt_M_est - pts_wrt_M_true
#
#                 # Transform between G and camera frame [C]
#                 points_wrt_C = np.einsum("ij, klj->kli", T_G_wrt_C_true, points_wrt_G)
#
#                 d_fw_wrt_s = self.der_of_fw_wrt_s(s_est, points_wrt_C)
#                 # The difference partial derivative
#                 # Simply, multiply each difference by the rotation jacobian
#                 d_fw_diffs = np.einsum("ijk, ijkl->ijl", points_diffs[..., :3], d_fw_wrt_s)
#                 # Tested among all elements (using my HACK):
#                 # >>>> np.all([np.allclose(np.dot(points_diffs[i,j], d_fg_wrt_g[i,j]),d_fg_diffs[i,j]) for (i,j), x in np.ndenumerate(d_fg_diffs[...,0])])
#
#                 gradient_of_all_diffs_sum = np.einsum("ijk -> k", d_fw_diffs)
#                 return gradient_of_all_diffs_sum
#
#             err_fw_wrt_s_jacs = check_grad(func_fw_wrt_s, jac_fw_wrt_s, list(q_estimated) + t_estimated, (points_wrt_G))
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#         s[:4] = s[:4] / np.linalg.norm(s[:4])
#         T_C_wrt_M = common_tools.get_transformation_matrix(s)
#         T_C_wrt_M = tr.translation_matrix([0, 0, s])
        T_C_wrt_M = self.T_C_wrt_model
        # Transform between [G] and mirror [M]
        T_G_wrt_M = tr.concatenate_matrices(T_C_wrt_M, T_G_wrt_C)
#         points_wrt_M = np.einsum("ij, klj->kli", T_G_wrt_M, points_wrt_G)
        # Or just using the intermediate results:
        points_wrt_M = np.einsum("ij, klj->kli", T_C_wrt_M, points_wrt_C)

        # Check that transformation it is correct for all points:
        # >>> chess_coords_wrt_M_homo = points_wrt_G.dot(T_G_wrt_M.T)
        # >>> np.allclose(chess_coords_wrt_M_homo, points_wrt_M)
        # >>> True

        if compute_jacobian:
            return d_of_fw_wrt_e, points_wrt_M, T_G_wrt_M
        else:
            return None, points_wrt_M, T_G_wrt_M

#     def der_of_fg_wrt_g(self, g, m, f, points_wrt_G, pts_scale_factor=1., only_translation_params=False):
    def der_of_fg_wrt_g(self, g, points_wrt_G, pts_scale_factor = 1., only_translation_params = False):
        '''
        Partial derivative of fg wrt g vector [q0, q1, q2, q3, tx, ty, tz]

        @return: a (Nx3x7) matrix
        '''

        if only_translation_params:
            t = g[:3]
        else:
            # Rotation part
            q = g[:4]  # * f[:4] + m[:4]  # Derotate
            # Normalize points
            t = g[4:]  # * f[4:] + m[4:]  # Scaled up (Denormalized)
    #         t_orig_norm = np.linalg.norm(t)
    #         points_wrt_G_normalized = points_wrt_G / t_orig_norm

    #         pts_norms = np.linalg.norm(points_wrt_G, axis=-1)
    #         max_norm = np.max(pts_norms)
    #         points_wrt_G_normalized = points_wrt_G / max_norm

    #         d_rot = self.der_of_f_wrt_q(q, points_wrt_G_normalized)
            d_rot = self.der_of_f_wrt_q(q, points_wrt_G)

        # translation part
        d_trans = self.der_of_f_wrt_t(t, points_wrt_G)
        # Normalize translation jacobian:
        d_trans_normalized = d_trans

#===============================================================================
#         # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#         # TEST Jacobian implementation:
#         # Ground truth is given as g:
#         # Use any arbitrary g to test
#         from scipy.optimize import check_grad
#         import transformations as tr
#
#         q_true = tr.quaternion_from_euler(np.deg2rad(0.00), np.deg2rad(0.00), np.deg2rad(0.0), axes="sxyz")  # [1, 0, 0, 0]
#         t_true = [1000, 0, 0]
#         g_true = list(q_true) + t_true
#         q_estimated = tr.quaternion_from_euler(np.deg2rad(0.00), np.deg2rad(0.00), np.deg2rad(0.0), axes="sxyz")  # [1, 0, 0, 0]
#         t_estimated = [1500.01, 6.01, -19.01]
#         def func_fg_wrt_t(t_est, pts_wrt_G):
#             # Transform between G and C
#             T_G_wrt_C_true = common_tools.get_transformation_matrix(g_true)
#             pts_wrt_C_true = np.einsum("ij, klj->kli", T_G_wrt_C_true, pts_wrt_G)
#
#             g_est = list(q_estimated) + list(t_est)
#             T_G_wrt_C_est = common_tools.get_transformation_matrix(g_est)
#             pts_wrt_C_est = np.einsum("ij, klj->kli", T_G_wrt_C_est, pts_wrt_G)
#
#             # pts_rot_est = np.dot(rot_matrix_est, pts)
#             points_diff_norm = np.linalg.norm(pts_wrt_C_est - pts_wrt_C_true, axis=-1)
#             sum_diffs = np.sum(points_diff_norm)
#             return sum_diffs
#
#         def jac_fg_wrt_t(t_est, pts_wrt_G):
#             # Transform between G and C
#             T_G_wrt_C_true = common_tools.get_transformation_matrix(g_true)
#             pts_wrt_C_true = np.einsum("ij, klj->kli", T_G_wrt_C_true, pts_wrt_G)
#
#             g_est = list(q_estimated) + t_est
#             T_G_wrt_C_est = common_tools.get_transformation_matrix(g_est)
#             pts_wrt_C_est = np.einsum("ij, klj->kli", T_G_wrt_C_est, pts_wrt_G)
#
#             points_diff_norm = np.linalg.norm(pts_wrt_C_est - pts_wrt_C_true, axis=-1)
#             x_in_C_est = pts_wrt_C_est[..., 0]
#             x_in_C_true = pts_wrt_C_true[..., 0]
#             y_in_C_est = pts_wrt_C_est[..., 1]
#             y_in_C_true = pts_wrt_C_true[..., 1]
#             z_in_C_est = pts_wrt_C_est[..., 2]
#             z_in_C_true = pts_wrt_C_true[..., 2]
#             d_of_fdiff_wrt_x_in_C = (x_in_C_est - x_in_C_true) / points_diff_norm
#             d_of_fdiff_wrt_y_in_C = (y_in_C_est - y_in_C_true) / points_diff_norm
#             d_of_fdiff_wrt_z_in_C = (z_in_C_est - z_in_C_true) / points_diff_norm
#             d_of_fdiff_wrt_P_in_C = np.append(d_of_fdiff_wrt_x_in_C[..., np.newaxis], d_of_fdiff_wrt_y_in_C[..., np.newaxis], axis=-1)
#             d_of_fdiff_wrt_P_in_C = np.append(d_of_fdiff_wrt_P_in_C, d_of_fdiff_wrt_z_in_C[..., np.newaxis], axis=-1)
#
#             d_fg_wrt_t = self.der_of_f_wrt_t(t, pts_wrt_G)
#             # The difference partial derivative
#             # Simply, multiply each difference by the rotation jacobian
#             d_fg_diffs = np.einsum("ijt, ijtf->ijf", d_of_fdiff_wrt_P_in_C, d_fg_wrt_t)
#             # Tested among all elements (using my HACK):
#             # >>>> np.all([np.allclose(np.dot(points_diffs[i,j], d_fg_wrt_g[i,j]),d_fg_diffs[i,j]) for (i,j), x in np.ndenumerate(d_fg_diffs[...,0])])
#
#             gradient_of_all_diffs_sum = np.einsum("ijk -> k", d_fg_diffs)
#             return gradient_of_all_diffs_sum
#
#         err_fg_wrt_t_jacs = check_grad(func_fg_wrt_t, jac_fg_wrt_t, t_estimated, (points_wrt_G))
#         # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#===============================================================================

        # Append partial derivatives
        if only_translation_params:
            d_of_fg_wrt_g = d_trans_normalized
        else:
            d_of_fg_wrt_g = np.append(d_rot, d_trans_normalized, axis = -1)

        return d_of_fg_wrt_g

    def der_of_f_wrt_t(self, t, points):
        '''
        Partial derivative of a homogeneous transformation function f wrt its t translation vector [tx, ty, tz]

        @return: a (rows x cols x 3 x 3) matrix
        '''
        # 3x3 Identity (with an extra axis/dimension to be used in the repetition broadcast)
        identity_trans = np.identity(len(t))[np.newaxis, ...]
        result_shape = points[..., 0].shape + identity_trans.shape[1:]  # Get (data grid shape + (3,3))
        trans_jacs = np.repeat(identity_trans, points[..., 0].size, axis = 0).reshape(result_shape)
        return trans_jacs

    def der_of_f_wrt_q(self, q, points):
        '''
        Partial derivative of a homogeneous transformation function f wrt q rotation quaternion vector [q0, q1, q2, q3]

        @return: (rows x cols x 3 x 4)
        '''
#===============================================================================
#         # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#         # TEST Jacobian implementation of d_of_f_wrt_q_normalized
#         # Ground truth is given as q:
#         # Use any arbitrary q to test
#         from scipy.optimize import check_grad
#         import transformations as tr
#         q_true = tr.quaternion_from_euler(np.deg2rad(0.00), np.deg2rad(0.00), np.deg2rad(-90.0000), axes="sxyz")  # [1, 0, 0, 0]
#         q_estimated = tr.quaternion_from_euler(np.deg2rad(-2.01), np.deg2rad(1.1), np.deg2rad(-91.00001), axes="sxyz")  # [1, 0, 0, 0]
#         def func_f_wrt_qn(q_est, pts):
#             # TODO: Rotate between G and C (without translation)
#             rot_matrix_true = tr.quaternion_matrix(q_true / np.linalg.norm(q_true))
# #             rot_matrix_true = tr.quaternion_matrix(q_true)
#             pts_rot_true = np.einsum("ij, klj->kli", rot_matrix_true, pts)
#             # pts_rot_true = np.dot(rot_matrix_true, pts)
#             rot_matrix_est = tr.quaternion_matrix(q_est / np.linalg.norm(q_est))
# #             rot_matrix_est = tr.quaternion_matrix(q_est)
#             pts_rot_est = np.einsum("ij, klj->kli", rot_matrix_est, pts)
#             # pts_rot_est = np.dot(rot_matrix_est, pts)
#             points_diff_norm = np.linalg.norm(pts_rot_est - pts_rot_true, axis=-1)
#             sum_sq_diffs = np.sum(points_diff_norm)
#             return sum_sq_diffs
#
#         def jac_f_wrt_qn(q_est, pts):
#             # WISH: redundante, but ok, just for testing
#             # Again, compute the transformation: Rotate between G and C (without translation)
#             rot_matrix_true = tr.quaternion_matrix(q_true / np.linalg.norm(q_true))
# #             rot_matrix_true = tr.quaternion_matrix(q_true)
#             pts_rot_true = np.einsum("ij, klj->kli", rot_matrix_true, pts)
#             # pts_rot_true = np.dot(rot_matrix_true, pts)
#             rot_matrix_est = tr.quaternion_matrix(q_est / np.linalg.norm(q_est))
# #             rot_matrix_est = tr.quaternion_matrix(q_est)
#             pts_rot_est = np.einsum("ij, klj->kli", rot_matrix_est, pts)
#
#             points_diff_norm = np.linalg.norm(pts_rot_est - pts_rot_true, axis=-1)
#             x_in_C_est = pts_rot_est[..., 0]
#             x_in_C_true = pts_rot_true[..., 0]
#             y_in_C_est = pts_rot_est[..., 1]
#             y_in_C_true = pts_rot_true[..., 1]
#             z_in_C_est = pts_rot_est[..., 2]
#             z_in_C_true = pts_rot_true[..., 2]
#             d_of_fdiff_wrt_x_in_C = (x_in_C_est - x_in_C_true) / points_diff_norm
#             d_of_fdiff_wrt_y_in_C = (y_in_C_est - y_in_C_true) / points_diff_norm
#             d_of_fdiff_wrt_z_in_C = (z_in_C_est - z_in_C_true) / points_diff_norm
#             d_of_fdiff_wrt_P_in_C = np.append(d_of_fdiff_wrt_x_in_C[..., np.newaxis], d_of_fdiff_wrt_y_in_C[..., np.newaxis], axis=-1)
#             d_of_fdiff_wrt_P_in_C = np.append(d_of_fdiff_wrt_P_in_C, d_of_fdiff_wrt_z_in_C[..., np.newaxis], axis=-1)
#
# #             d_of_rot_wrt_q = self.der_of_f_wrt_q_normalized(q_est, pts)
#             d_of_rot_wrt_q = self.der_of_f_wrt_q_generalized(q_est, pts)
#             d_of_q_normalized_wrt_q = self.der_of_q_normalized_wrt_q(q_est)[0, 0]
#             d_of_f_wrt_q = np.einsum("ijkl, lm->ijkm", d_of_rot_wrt_q, d_of_q_normalized_wrt_q)
# #             d_of_f_wrt_q = d_of_rot_wrt_q
#             d_of_f_diffs_wrt_q = np.einsum("ijt, ijtf->ijf", d_of_fdiff_wrt_P_in_C, d_of_f_wrt_q)
#
#             gradient_of_all_diffs_sum = np.einsum("ijk -> k", d_of_f_diffs_wrt_q)
#             return gradient_of_all_diffs_sum
#
#         err_rot_jacs_f_wrt_qn = check_grad(func_f_wrt_qn, jac_f_wrt_qn, q_estimated, (points))
#         # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#===============================================================================

        #=======================================================================
        # # If using the rotation matrix formula for normalized quaternions
        # d_of_f_wrt_q_normalized = self.der_of_f_wrt_q_normalized(q, points)
        # d_of_q_normalized_wrt_q = self.der_of_q_normalized_wrt_q(q)[0, 0]
        # d_of_f_wrt_q = np.einsum("ijkl, lm->ijkm", d_of_f_wrt_q_normalized, d_of_q_normalized_wrt_q)
        # # TEST:
        # # >>>> np.all([np.allclose(np.dot(d_of_f_wrt_q_normalized[i,j], d_of_q_normalized_wrt_q),d_of_f_wrt_q[i,j]) for (i,j), x in np.ndenumerate(d_of_f_wrt_q[...,0,0])])
        # return d_of_f_wrt_q
        #=======================================================================

        d_of_f_wrt_q_gen = self.der_of_f_wrt_q_generalized(q, points)
        d_of_q_normalized_wrt_q = self.der_of_q_normalized_wrt_q(q)[0, 0]
        d_of_f_wrt_q = np.einsum("ijkl, lm->ijkm", d_of_f_wrt_q_gen, d_of_q_normalized_wrt_q)
#         d_of_f_wrt_q = d_of_f_wrt_q_gen
        return d_of_f_wrt_q

    def der_of_fw_wrt_s(self, s, points_wrt_C):
        '''
        Partial derivative of fw wrt s vector [q0, q1, q2, q3, tx, ty, tz] for the pose of [M] wrt [C]

        @param s: Now, it's supposed to only be the z-component of the translation of [M] wrt [C] or just F_z
        @return: a (Nx3x7) matrix
        '''
#         # TODO: Generalize the rotation part (specially only for a azimuth angle required in the 2-camera rig (RemoteReality) case
#         q = s[:4]  # Note: it may be unormalized
#         t = s[4:]
#         d_rot = self.der_of_f_wrt_q(q, points_wrt_C)
#         # translation part
#         d_trans = self.der_of_f_wrt_t(t, points_wrt_C)
#         # Append partial derivatives
#         d_of_fw_wrt_s = np.append(d_rot, d_trans, axis=-1)
        # Simplified (coaxially-constrained) solution:
        d_of_tz = -np.array([[0.], [0.], [1.]])[np.newaxis, ...]
        result_shape = points_wrt_C[..., 0].shape + d_of_tz.shape[1:]  # Get (data grid shape + (3,3))
        trans_jacs = np.repeat(d_of_tz, points_wrt_C[..., 0].size, axis = 0).reshape(result_shape)
        d_of_fw_wrt_s = trans_jacs
        return d_of_fw_wrt_s

    def get_rot_matrix_from_quat(self, q):
        '''
        Compute rotation rotation matrix for a quaternion vector [q0, q1, q2, q3]

        @return: The (3 x 3) equivalent rotation matrix
        '''
        import omnistereo.transformations as tr
        # By implementing the equation:
        # vvvvvvvvvvvvvvvvvvvvvvv
        q_normalized = q / np.linalg.norm(q)
#         q0 = q_normalized[0]
#         q1 = q_normalized[1]
#         q2 = q_normalized[2]
#         q3 = q_normalized[3]
#
#         # Column 0
#         r0c0 = q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2
#         r1c0 = 2 * (q0 * q3 + q1 * q2)
#         r2c0 = 2 * (-q0 * q2 + q1 * q3)
#         rot_c0 = np.dstack((r0c0, r1c0, r2c0))
#
#         # Column 1
#         r0c1 = 2 * (-q0 * q3 + q1 * q2)
#         r1c1 = q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2
#         r2c1 = 2 * (q0 * q1 + q2 * q3)
#         rot_c1 = np.dstack((r0c1, r1c1, r2c1))
#
#         # Column 2
#         r0c2 = 2 * (q0 * q2 + q1 * q3)
#         r1c2 = 2 * (-q0 * q1 + q2 * q3)
#         r2c2 = q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2
#         rot_c2 = np.dstack((r0c2, r1c2, r2c2))
#
#         rot = np.append(rot_c0[..., np.newaxis], rot_c1[..., np.newaxis], axis=-1)
#         rot = np.append(rot, rot_c2[..., np.newaxis], axis=-1)
        # ^^^^^^^^^^^^^^^^^^^^^^

        # More efficiently:
        rot = tr.quaternion_matrix(q_normalized)
        return rot[:-1, :-1]

    def der_of_f_wrt_q_generalized(self, q, points):
        '''
        Partial derivative of any rotation function f wrt any q rotation quaternion vector [q0, q1, q2, q3]
        @note: This generalized form of the function to convert quaternion seems to work fine even for normalized quaternions
        @return: (rows x cols x 3 x 4)
        '''
#         q0 = q[0]
#         q1 = q[1]
#         q2 = q[2]
#         q3 = q[3]
        q_normalized = q / np.linalg.norm(q)
        q0 = q_normalized[0]
        q1 = q_normalized[1]
        q2 = q_normalized[2]
        q3 = q_normalized[3]

        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]

        # Column 0
        r0c0 = q0 * x - q3 * y + q2 * z
        r1c0 = q3 * x + q0 * y - q1 * z
        r2c0 = -q2 * x + q1 * y + q0 * z
        d_of_fg_wrt_q0_normalized = np.dstack((r0c0, r1c0, r2c0))

        # Column 1
        r0c1 = q1 * x + q2 * y + q3 * z
        r1c1 = q2 * x - q1 * y - q0 * z
        r2c1 = q3 * x + q0 * y - q1 * z
        d_of_fg_wrt_q1_normalized = np.dstack((r0c1, r1c1, r2c1))

        # Column 2
        r0c2 = -q2 * x + q1 * y + q0 * z
        r1c2 = q1 * x + q2 * y + q3 * z
        r2c2 = -q0 * x + q3 * y - q2 * z
        d_of_fg_wrt_q2_normalized = np.dstack((r0c2, r1c2, r2c2))

        # Column 3
        r0c3 = -q3 * x - q0 * y + q1 * z
        r1c3 = q0 * x - q3 * y + q2 * z
        r2c3 = q1 * x + q2 * y + q3 * z
        d_of_fg_wrt_q3_normalized = np.dstack((r0c3, r1c3, r2c3))

        d_of_fg_wrt_q_normalized = np.append(d_of_fg_wrt_q0_normalized[..., np.newaxis], d_of_fg_wrt_q1_normalized[..., np.newaxis], axis = -1)
        d_of_fg_wrt_q_normalized = np.append(d_of_fg_wrt_q_normalized, d_of_fg_wrt_q2_normalized[..., np.newaxis], axis = -1)
        d_of_fg_wrt_q_normalized = np.append(d_of_fg_wrt_q_normalized, d_of_fg_wrt_q3_normalized[..., np.newaxis], axis = -1)
        d_of_fg_wrt_q_normalized = 2 * d_of_fg_wrt_q_normalized

        return d_of_fg_wrt_q_normalized

    def der_of_f_wrt_q_normalized(self, q, points):
        '''
        Partial derivative of any rotation function f wrt normalized q rotation quaternion vector [q0, q1, q2, q3]

        @return: (rows x cols x 3 x 4)
        '''
        q_normalized = q / np.linalg.norm(q)
        q0 = q_normalized[0]
        q1 = q_normalized[1]
        q2 = q_normalized[2]
        q3 = q_normalized[3]

        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]

        # Column 0
        r0c0 = q3 * y - q2 * z
        r1c0 = -q3 * x - 2 * q0 * y + q1 * z
        r2c0 = q2 * x - q1 * y
        d_of_fg_wrt_q0_normalized = np.dstack((r0c0, r1c0, r2c0))

        # Column 1
        r0c1 = q2 * y + q3 * z
        r1c1 = q2 * x + q0 * z
        r2c1 = q3 * x - q0 * y - 2 * q1 * z
        d_of_fg_wrt_q1_normalized = np.dstack((r0c1, r1c1, r2c1))

        # Column 2
        r0c2 = -2 * q2 * x + q1 * y - q0 * z
        r1c2 = q1 * x + q3 * z
        r2c2 = q0 * x + q3 * y - 2 * q2 * z
        d_of_fg_wrt_q2_normalized = np.dstack((r0c2, r1c2, r2c2))

        # Column 3
        r0c3 = -2 * q3 * x + q0 * y + q1 * z
        r1c3 = -q0 * x - 2 * q3 * y + q2 * z
        r2c3 = q1 * x + q2 * y
        d_of_fg_wrt_q3_normalized = np.dstack((r0c3, r1c3, r2c3))

        d_of_fg_wrt_q_normalized = np.append(d_of_fg_wrt_q0_normalized[..., np.newaxis], d_of_fg_wrt_q1_normalized[..., np.newaxis], axis = -1)
        d_of_fg_wrt_q_normalized = np.append(d_of_fg_wrt_q_normalized, d_of_fg_wrt_q2_normalized[..., np.newaxis], axis = -1)
        d_of_fg_wrt_q_normalized = np.append(d_of_fg_wrt_q_normalized, d_of_fg_wrt_q3_normalized[..., np.newaxis], axis = -1)
        d_of_fg_wrt_q_normalized = 2 * d_of_fg_wrt_q_normalized

        return d_of_fg_wrt_q_normalized

    def der_of_q_normalized_wrt_q(self, q):
        '''
        Partial derivative of fg wrt q rotation quaternion vector [q0, q1, q2, q3]
        @param q: The unormalized quaternion

        @return: The (3 x 4) jacobian matrix
        '''
        q_norm_cube_factor = 1 / (np.linalg.norm(q) ** 3)

        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]

        # Column 0
        r0c0 = q1 ** 2 + q2 ** 2 + q3 ** 2
        r1c0 = -q0 * q1
        r2c0 = -q0 * q2
        r3c0 = -q0 * q3
        der_of_q_normalized_wrt_q0 = np.dstack((r0c0, r1c0, r2c0, r3c0))

        # Column 1
        r0c1 = -q0 * q1
        r1c1 = q0 ** 2 + q2 ** 2 + q3 ** 2
        r2c1 = -q1 * q2
        r3c1 = -q1 * q3
        der_of_q_normalized_wrt_q1 = np.dstack((r0c1, r1c1, r2c1, r3c1))

        # Column 2
        r0c2 = -q0 * q2
        r1c2 = -q1 * q2
        r2c2 = q0 ** 2 + q1 ** 2 + q3 ** 2
        r3c2 = -q2 * q3
        der_of_q_normalized_wrt_q2 = np.dstack((r0c2, r1c2, r2c2, r3c2))

        # Column 3
        r0c3 = -q0 * q3
        r1c3 = -q1 * q3
        r2c3 = -q2 * q3
        r3c3 = q0 ** 2 + q1 ** 2 + q2 ** 2
        der_of_q_normalized_wrt_q3 = np.dstack((r0c3, r1c3, r2c3, r3c3))

        d_of_q_normalized_wrt_q = np.append(der_of_q_normalized_wrt_q0[..., np.newaxis], der_of_q_normalized_wrt_q1[..., np.newaxis], axis = -1)
        d_of_q_normalized_wrt_q = np.append(d_of_q_normalized_wrt_q, der_of_q_normalized_wrt_q2[..., np.newaxis], axis = -1)
        d_of_q_normalized_wrt_q = np.append(d_of_q_normalized_wrt_q, der_of_q_normalized_wrt_q3[..., np.newaxis], axis = -1)
        d_of_q_normalized_wrt_q = q_norm_cube_factor * d_of_q_normalized_wrt_q

        return d_of_q_normalized_wrt_q

    def get_pixel_from_3D_point_wrt_C(self, Pw_wrt_C, visualize = False):
        '''
        @brief Project a three-dimensional numpy array (rows x cols x 4) of 3D homogeneous points (eg. [x, y, z, 1]) as row-vectors to the image plane in (\a u,\a v).
        This function is already vectorized for Numpy performance.

        @param Pw_wrt_C: the multidimensional array of homogeneous coordinates of the points (wrt the origin of the common frame [C], e.g. camera pinhole)
        '''
        points_wrt_M_top = self.get_points_wrt_M(Pw_wrt_C)
        u, v, m_homo = self.get_pixel_from_3D_point_wrt_M(points_wrt_M_top)
        return u, v, m_homo

    def get_pixel_from_3D_point_wrt_M(self, Pw_wrt_M, visualize = False):
        '''
        @brief Project a three-dimensional numpy array (rows x cols x 4) of 3D homogeneous points (eg. [x, y, z, 1]) as row-vectors to the image plane in (\a u,\a v).
        This function is already vectorized for Numpy performance.

        @param Pw_wrt_M: the multidimensional array of homogeneous coordinates of the points (wrt the origin of the GUM frame [M])
        @param visualize: To indicate if a 3D visualization will be shown

        @retval u: the resulting ndarray of u coordinates on the image plane
        @retval v: the resulting ndarray of v coordinates on the image plane
        @retval m_homo: The pixel point(s) as numpy array in homogeneous coordinates
        '''
        # Step 1: Assumed to have pose of point wrt GUM frame [M]
        # Step 2: Normalize point (GUM is a unit sphere)
        Ps_wrt_M = camera_models.get_normalized_points(Pw_wrt_M)  # Directly, project points onto the unit sphere

        # CORRECT but UNNECESSARY:
        # line_CpPs = camera_models.get_lines_through_single_point3(Ps_wrt_M, self.Cp_wrt_M)
        # Find intersecting point, p_und, between line_CpPs with normalized projection plane
        # p_und = camera_models.intersect_line3_with_plane_vectorized(line_CpPs, self.normalized_projection_plane)

        # Instead, simply follow equations in paper
        # Step 3: Get normalized point wrt to Center of Projection [Cp]
        Ps_wrt_Cp = self.get_points_on_sphere_wrt_Cp(Ps_wrt_M)

        # Step 4: Project to normalized plane (undistorted)
        p_und = self.get_undistorted_points(Ps_wrt_Cp)

        # Step 5: Distort the projected point
        if self.precalib_params.use_distortion:
            # Apply distortion to undistorted projected points
            p_distorted = self.get_distorted_points(p_und)
        else:
            p_distorted = p_und

        # Step 6: Get the pixel in the image:
        # Finally, apply the generalized projection matrix
        u, v, m_homo = self.get_pixel_from_point_on_image_plane(p_distorted)

        return u, v, m_homo
        # TODO: implement visualization

    def get_pixel_from_point_on_image_plane(self, p_distorted):
        p_dist_x, p_dist_y = p_distorted[..., 0], p_distorted[..., 1]
        # Recall gamma1 = K11, and gamma2 = K22
        # Get the pixel in the image:
        u = self.precalib_params.gamma1 * p_dist_x + self.precalib_params.gamma1 * self.precalib_params.alpha_c * p_dist_y + self.precalib_params.u_center
        # u = self.precalib_params.gamma1 * p_dist_x + self.precalib_params.u_center
        v = self.precalib_params.gamma2 * p_dist_y + self.precalib_params.v_center
        m_homo = np.dstack((u, v, np.ones_like(u)))
        return u, v, m_homo

    def get_3D_point_from_angles_wrt_focus(self, azimuth, elevation):
        '''
        Finds a world point using the given projection angles towards the focus of the mirror (the GUM center)

        @return: The numpy ndarray of 3D points (in homogeneous coordinates) w.r.t. origin of coordinates (\f$O_C$\f)
        '''
        P_on_sphere = self.map_angles_to_unit_sphere(elevation, azimuth)
#         # The points in the sphere is with respect to F, but we need the points w.r.t the system origin Oc
#         # Also, we arbitrarily scale up the position of the point so it is not inside the mirror
#         Pw = (P_on_sphere[..., :3] * 1000) + np.array(self.Pm)  # TODO: choose the appropriate position for the mirror's center (SVP)
#         Pw_homo = np.dstack((Pw, P_on_sphere[..., -1]))  # Put back the ones for the homogeneous coordinates
        return P_on_sphere

#===============================================================================
# COMMENTED out because not longer using "Euclid" as it causes trouble while pickling and copying
#     def get_pixel_from_XYZ(self, x, y, z, visualize=False):
#         '''
#         @brief Project a single 3D point (\a x,\a y,\a z) to the image plane in (\a u,\a v).
#              NOTE: This function is not vectorized (not using Numpy explicitly).
#
#         @param x: 3D point x coordinate (wrt the center of the unit sphere)
#         @param y: 3D point y coordinate (wrt the center of the unit sphere)
#         @param z: 3D point z coordinate (wrt the center of the unit sphere)
#
#         @retval u: contains the image point u coordinate
#         @retval v: contains the image point v coordinate
#         @retval m: the undistorted 3D point (of type euclid.Point3) in the normalized projection plane
#         '''
#         if self.new_method:
#             Ps_wrt_M = euclid.Point3(x, y, z).normalize()  # Normalized in-place
#             # Check with Euclid:
#             line_CpPs = euclid.Line3(self.Cp_wrt_M, Ps_wrt_M)
#             # Find intersecting point, m, between line_CpPs with normalized projection plane
#             m = self.normalized_projection_plane.intersect(line_CpPs)
#             mx_undistorted, my_undistorted = m.x, m.y
#         else:
#             # Project points to the normalized plane
#             z = z + self.precalib_params.xi3 * np.sqrt(x * x + y * y + z * z)  # FIXME2: sign of xi3 may be wrong
#             if z == 0:  # Avoid division by 0
#                 z = 1
#             mx_undistorted = x / (self.z_axis * z)
#             my_undistorted = y / (self.z_axis * z)
#
#         if self.precalib_params.use_distortion:
#             # Apply distortion
#             distortion_x, distortion_y = self.distortion_OLD(mx_undistorted, my_undistorted)
#             mx_distorted = mx_undistorted + distortion_x;
#             my_distorted = my_undistorted + distortion_y;
#         else:
#             mx_distorted = mx_undistorted;
#             my_distorted = my_undistorted;
#
#         # Finally, apply generalised projection matrix
#         u = self.precalib_params.gamma1 * mx_distorted + self.precalib_params.u_center
#         v = self.precalib_params.gamma2 * my_distorted + self.precalib_params.v_center
#
#         if visualize:
#             print("Point projected to normalized plane: %s and pixel (%f,%f)" % (m, u, v))
#             import matplotlib.pyplot as plt
#             from mplot3d import axes3d  # To be run from eclipse
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             # Draw GUM skeleton
#             ax = self.draw_fwd_projection(ax)
#             # Draw point m on normalized projection plane
#             ax.scatter(m.x, m.y, m.z, color="red", s=20)
#             ax.text(m.x + 0.1, m.y, m.z, "m", color="red")
#
#             # Draw point Ps on unit sphere
#             ax.scatter(Ps_wrt_M.x, Ps_wrt_M.y, Ps_wrt_M.z, color="purple", s=20)
#             ax.text(Ps_wrt_M.x + 0.1, Ps_wrt_M.y, Ps_wrt_M.z, "Ps", color="purple")
#             # Plot the m-Cp-Ps line segment
#             line_mCp_xs = [m.x, self.Cp_wrt_M.x, Ps_wrt_M.x]
#             line_mCp_ys = [m.y, self.Cp_wrt_M.y, Ps_wrt_M.y]
#             line_mCp_zs = [m.z, self.Cp_wrt_M.z, Ps_wrt_M.z]
#             ax.plot3D(line_mCp_xs, line_mCp_ys, line_mCp_zs, color="purple", linestyle='--', linewidth=1.5, alpha=1.0)
#
#             # Plot the space point P and it's projection to M
#             ax.scatter(x, y, z, color="magenta", s=20)
#             ax.text(x + 0.1, y, z, "P", color="magenta")
#             line_PM_xs = [x, self.Pm.x]
#             line_PM_ys = [y, self.Pm.y]
#             line_PM_zs = [z, self.Pm.z]
#             ax.plot3D(line_PM_xs, line_PM_ys, line_PM_zs, color="magenta", linestyle='--', linewidth=2.0, alpha=1.0)
#             plt.show()  # Show both figures as subplots (same window)
#
#         return u, v, m
#===============================================================================

    def lift_pixel_to_normalized_projection_plane(self, m):
        '''
        @brief Lifts a point (pixel) from the image plane to the undistorted normalized projection plane
        @param m: A numpy array of k image point coordinates [u, v]. Thus, shape is (1,k,2)

        @retval p_d: The Euclidean coordinates (as a [1xkx2] numpy array) of the point(s) on the normalized projection plane
        '''
        # Lift points to normalized plane
        u = m[..., 0]
        v = m[..., 1]
        # Apply inverse projection:
        x_distorted = self.precalib_params.inv_K11 * u + self.precalib_params.inv_K12 * v + self.precalib_params.inv_K13
        y_distorted = self.precalib_params.inv_K22 * v + self.precalib_params.inv_K23
        # Equivalently
        # x_distorted = (u - self.precalib_params.u_center) * self.precalib_params.inv_K11
        # y_distorted = (v - self.precalib_params.v_center) * self.precalib_params.inv_K22
        p_distorted = np.dstack((x_distorted, y_distorted))

        return p_distorted

    def lift_pixel_to_unit_sphere_wrt_focus(self, m, visualize = False, debug = False):
        '''
        @brief Lifts a point from the image plane to the unit sphere
        @param m: A numpy array of k image point coordinates [u, v]. Thus, shape is (1,k,2)
        @param visualize: Indicates if visualization will take place
        @param debug: Indicates to print debugging statements
        @retval Ps: The Euclidean coordinates (as a[1xkx3 numpy array) of the point(s) on the sphere.
        '''

        x_undistorted, y_undistorted = None, None  # The projected point on the normalized projection plane

        # Lift points to normalized plane and apply undistortion:
        if isinstance(m, np.ndarray):
            p_distorted = self.lift_pixel_to_normalized_projection_plane(m)

            if self.precalib_params.use_distortion:
                if self.precalib_params.l1 != 0:
                # if False:  # TODO: use this only for some kind of threshold or when the mirror is "spherical"
                    # Apply my optimal inverse distortion model
                    p_undistorted = self.get_radially_distorted_points(p_distorted, dist_params = [self.precalib_params.l1, self.precalib_params.l2, self.precalib_params.l3])
                    x_undistorted = p_undistorted[..., 0]
                    y_undistorted = p_undistorted[..., 1]
                else:  # Use Mei's approach
                    x_distorted = p_distorted[..., 0]
                    y_distorted = p_distorted[..., 1]
                    # Apply inverse distortion model
                    k1 = self.precalib_params.k1
                    k2 = self.precalib_params.k2
                    k3 = self.precalib_params.k3  # FIXME: not being used by this iverse distortion model
                    p1 = self.precalib_params.p1
                    p2 = self.precalib_params.p2
                    #===============================================================
                    # self.distortion_value_threshold = 0.001  # FIXME: for selection of undistorition method. Very arbitrary for now
                    # if self.distortion_value_threshold > np.max(np.abs([k1, k2, p1, p2])):  # If distotion parameters are small
                    #===============================================================
                    if True:
                        # CHECKME:
                        # Inverse distortion model (onl
                        # Closed-form solution proposed by Heikkila in 2000
                        # WARNING: Under the assumption that the distortion values and the jacobian values are SMALL!!!!
                        x2_d = x_distorted * x_distorted
                        y2_d = y_distorted * y_distorted
                        mxy_d = x_distorted * y_distorted
                        rho2_d = x2_d + y2_d
                        rho4_d = rho2_d ** 2

                        radDist_d = k1 * rho2_d + k2 * rho4_d
                        Dx_d = x_distorted * radDist_d + p2 * (rho2_d + 2 * x2_d) + 2 * p1 * mxy_d
                        Dy_d = y_distorted * radDist_d + p1 * (rho2_d + 2 * y2_d) + 2 * p2 * mxy_d
                        inv_denom_d = 1 / (1 + 4 * k1 * rho2_d + 6 * k2 * rho4_d + 8 * p1 * y_distorted + 8 * p2 * x_distorted)

                        x_undistorted = x_distorted - inv_denom_d * Dx_d
                        y_undistorted = y_distorted - inv_denom_d * Dy_d
                    else:
                        # Recursive distortion model
                        # WISHME: Use Levenberg-Marquardt?
                        n = 6
                        p_distorted = np.dstack((x_distorted, y_distorted))
                        dxy_undistorted = self.distortion(p_distorted)
                        # Approximate value
                        p_undistorted = p_distorted - dxy_undistorted
                        for i in range(n):
                            dxy_undistorted = self.distortion(p_undistorted)
                            p_undistorted = p_distorted - dxy_undistorted

                        x_undistorted = p_undistorted[..., 0]
                        y_undistorted = p_undistorted[..., 1]
            else:
                x_undistorted = x_distorted;
                y_undistorted = y_distorted;

            if self.new_method:
                # WISH:find z of the plane in a more procedural way
                #    However, this is NOT NECESSARY, since the Sphere Model (GUM) will always use an aligned projection plane [Pi] to the XY-plane of the mirror frame [M]
                #     This normalized projection plane [Pi] is orthogonal to the model's Z axis
                proj_plane_z_pos_wrt_M = self.plane_k

                p_u_wrt_M = np.dstack((self.Cp_wrt_M[0] + x_undistorted, self.Cp_wrt_M[1] + y_undistorted, np.zeros_like(x_undistorted) + proj_plane_z_pos_wrt_M))

                line_Cp_to_p_u = camera_models.get_lines_through_single_point3(p_u_wrt_M, self.Cp_wrt_M)
                # Find intersecting point, m, between line_CpPs with normalized projection plane
                # unit_sphere = euclid.Sphere() # Not used anymore
                Ps_intersecting_segment = camera_models.intersect_line3_with_sphere_vectorized(line_Cp_to_p_u)
                # The resulting intersection point(s)...ATTENTION: there can be 2

                # Needs to select the intersection point with the Z-highest (closest to the projection plane)
                # between Ps_intersecting_segment.p1.z or Ps_intersecting_segment.p2.z (based on model orientation)

                # CHECKME: Fix reprojection ambiguity may depend on Z value. However, it seems fine just by taking the first point for Ps
                Ps = Ps_intersecting_segment[..., :3]
            else:  # Using old method (Mei's)
                # Lift normalised points to the sphere (inv_hslash)
                # TODO: implement
                xi = self.precalib_params.xi3

                if xi == 1.0:
                    lambda_val = 2 / (x_undistorted * x_undistorted + y_undistorted * y_undistorted + 1)
                    X = lambda_val * x_undistorted
                    Y = lambda_val * y_undistorted
                    Z = lambda_val - 1
                else:
                    lambda_val = (xi + np.sqrt(1 + (1 - xi * xi) * (x_undistorted * x_undistorted + y_undistorted * y_undistorted))) / (1 + x_undistorted * x_undistorted + y_undistorted * y_undistorted)
                    X = lambda_val * x_undistorted
                    Y = lambda_val * y_undistorted
                    Z = lambda_val - xi

#===============================================================================
#         else:
#             u = m[0]
#             v = m[1]
#             x_distorted = (u - self.precalib_params.u_center) * self.precalib_params.inv_K11
#             y_distorted = (v - self.precalib_params.v_center) * self.precalib_params.inv_K22
#             m_distorted_wrt_Cp = euclid.Point3(x_distorted, y_distorted, 1)
#
#             if debug:
#                 print("DEBUG: Distorted point on projection plane wrt Cp is %s" % m_distorted_wrt_Cp)
#
#             if self.precalib_params.use_distortion:
#                 # Apply inverse distortion model
#                 if True:
#                     # Inverse distortion model
#                     # proposed by Heikkila
#                     x2_d = x_distorted * x_distorted
#                     y2_d = y_distorted * y_distorted
#                     mxy_d = x_distorted * y_distorted
#                     rho2_d = x2_d + y2_d
#                     rho4_d = rho2_d * rho2_d
#                     k1 = self.precalib_params.k1
#                     k2 = self.precalib_params.k2
#                     p1 = self.precalib_params.p1
#                     p2 = self.precalib_params.p2
#
#                     radDist_d = k1 * rho2_d + k2 * rho4_d
#                     Dx_d = x_distorted * radDist_d + p2 * (rho2_d + 2 * x2_d) + 2 * p1 * mxy_d
#                     Dy_d = y_distorted * radDist_d + p1 * (rho2_d + 2 * y2_d) + 2 * p2 * mxy_d
#                     inv_denom_d = 1 / (1 + 4 * k1 * rho2_d + 6 * k2 * rho4_d + 8 * p1 * y_distorted + 8 * p2 * x_distorted)
#
#                     x_undistorted = x_distorted - inv_denom_d * Dx_d
#                     y_undistorted = y_distorted - inv_denom_d * Dy_d
#                 else:
#                     # FIXME2: I don't like it because it seems too unstable (distortion values oscillate through iterations)!
#                     # Recursive distortion model
#                     n = 6;
#                     dx_u, dy_u = self.distortion(x_distorted, y_distorted)
#                     # Approximate value
#                     x_undistorted = x_distorted - dx_u
#                     y_undistorted = y_distorted - dy_u
#                     for i in range(n):
#                         dx_u, dy_u = self.distortion(x_undistorted, y_undistorted)
#                         x_undistorted = x_distorted - dx_u
#                         y_undistorted = y_distorted - dy_u
#                         if debug:
#                             print("DEBUG: dx = %f, dy = %f" % (dx_u, dy_u))
#             else:
#                 x_undistorted = x_distorted;
#                 y_undistorted = y_distorted;
#
#             m_undistorted_wrt_Cp = euclid.Point3(x_undistorted, y_undistorted, 1)
#
#             if debug:
#                 print("DEBUG: Undistorted point on projecton plane wrt Cp = %s" % m_undistorted_wrt_Cp)
#
#             if self.new_method:
#                 # WISH:find z of the plane in a more procedural way
#                 # It's okay for now because the normalized projection plane is orthogonal to the model's Z axis
#                 proj_plane_z_pos_wrt_M = self.normalized_projection_plane.n.z * self.normalized_projection_plane.k
#                 p_u_wrt_M = euclid.Point3(self.Cp_wrt_M.x + x_undistorted, self.Cp_wrt_M.y + y_undistorted, proj_plane_z_pos_wrt_M)
#                 line_Cp_to_p_u = euclid.Line3(self.Cp_wrt_M, p_u_wrt_M)
#                 unit_sphere = euclid.Sphere()
#                 Ps_intersecting_segment = line_Cp_to_p_u.intersect(unit_sphere)  # The resulting intersection point(s)...recall there can be 2
#
#                 # Needs to select the intersection point with the Z-highest (closest to the projection plane)
#                 # between Ps_intersecting_segment.p1.z or Ps_intersecting_segment.p2.z (based on model orientation)
#
#                 # FIXME2: Fix reprojection ambiguity may depend on Z value. However, it seems fine just by taking the p1 for Ps
#                 if isinstance(Ps_intersecting_segment, euclid.Line3):
#                     Ps = Ps_intersecting_segment.p1
#                 else:
#                     print("No intersection of %s with %s. \n Type is: %s" % (line_Cp_to_p_u, unit_sphere, type(Ps_intersecting_segment)))
#                     return False, np.nan, np.nan, np.nan
#                 # If there is only 1 intersection point # With Euclid, it will be the same point if there's only one
#     #                 if Ps_intersecting_segment.p1 == Ps_intersecting_segment.p2:
#     #                     if debug:
#     #                         print("DEBUG: Only 1 intersecting Point:", Ps_intersecting_segment.p1)
#                 # Maybe, it needs negation
#     #                 else:
#     #                     if debug:
#     #                         print("DEBUG: 2 intersecting Points:", Ps_intersecting_segment)
#     #                     if self.z_axis < 0:
#     #                         if Ps_intersecting_segment.p1.z < Ps_intersecting_segment.p2.z:
#     #                             Ps = Ps_intersecting_segment.p1
#     #                         else:
#     #                             Ps = Ps_intersecting_segment.p2
#     #                     else: # Positive z_axis (top mirror)
#     #                         if Ps_intersecting_segment.p1.z < Ps_intersecting_segment.p2.z:
#     #                             Ps = Ps_intersecting_segment.p2
#     #                         else:
#     #                             Ps = Ps_intersecting_segment.p1
#
#                 X = Ps.x
#                 Y = Ps.y
#                 Z = Ps.z
#             else:  # Using old method (Mei's)
#                 # Lift normalised points to the sphere (inv_hslash)
#                 xi = self.precalib_params.xi3
#
#                 if xi == 1.0:
#                     lambda_val = 2 / (x_undistorted * x_undistorted + y_undistorted * y_undistorted + 1)
#                     X = lambda_val * x_undistorted
#                     Y = lambda_val * y_undistorted
#                     Z = lambda_val - 1
#                 else:
#                     lambda_val = (xi + np.sqrt(1 + (1 - xi * xi) * (x_undistorted * x_undistorted + y_undistorted * y_undistorted))) / (1 + x_undistorted * x_undistorted + y_undistorted * y_undistorted)
#                     X = lambda_val * x_undistorted
#                     Y = lambda_val * y_undistorted
#                     Z = lambda_val - xi
#
#             # ONLY working for non-Numpy instances, such as the Euclid approach of a single point
#             if debug:
#                 print("DEBUG: Plane position on Z-axis wrt M is: %f" % proj_plane_z_pos_wrt_M)
#                 print("DEBUG: Undistorted point on projecton plane wrt M is: %s" % p_u_wrt_M)
#                 print("DEBUG: Instersection of %s with %s is %s" % (line_Cp_to_p_u, unit_sphere, Ps_intersecting_segment))
#
#             if visualize:
#                 import matplotlib.pyplot as plt
# #                 from mpl_toolkits.mplot3d import Axes3D
#                 from mplot3d import axes3d
#                 fig = plt.figure()
#                 ax = fig.add_subplot(111, projection='3d')
#                 # Draw GUM skeleton
#                 ax = self.draw_fwd_projection(ax)
#                 # Draw point m on normalized projection plane
#                 ax.scatter(p_u_wrt_M.x, p_u_wrt_M.y, p_u_wrt_M.z, color="red", s=20)
#                 ax.text(p_u_wrt_M.x + 0.1, p_u_wrt_M.y, p_u_wrt_M.z, "m", color="red")
#
#                 # Draw Ps points line segment intersecting with unit sphere
#                 Ps_p1 = Ps_intersecting_segment.p1
#                 Ps_p2 = Ps_intersecting_segment.p2
#                 ax.scatter(Ps_p1.x, Ps_p1.y, Ps_p1.z, color="purple", s=20)
#                 ax.text(Ps_p1.x + 0.1, Ps_p1.y, Ps_p1.z, "Ps1", color="purple")
#                 ax.scatter(Ps_p2.x, Ps_p2.y, Ps_p2.z, color="blue", s=20)
#                 ax.text(Ps_p2.x + 0.1, Ps_p2.y, Ps_p2.z, "Ps2", color="purple")
#                 # Plot the m-Cp-Pss_intersecting_segment line segment
#                 line_Pss_xs = [Ps_p1.x, Ps_p2.x]
#                 line_Pss_ys = [Ps_p1.y, Ps_p2.y]
#                 line_Pss_zs = [Ps_p1.z, Ps_p2.z]
#                 ax.plot3D(line_Pss_xs, line_Pss_ys, line_Pss_zs, color="purple", linestyle='--', linewidth=1.5, alpha=1.0)
#                 line_mPs_xs = [Ps.x, p_u_wrt_M.x]
#                 line_mPs_ys = [Ps.y, p_u_wrt_M.y]
#                 line_mPs_zs = [Ps.z, p_u_wrt_M.z]
#                 ax.plot3D(line_mPs_xs, line_mPs_ys, line_mPs_zs, color="magenta", linestyle='--', linewidth=1.5, alpha=1.0)
#
#                 # Plot the space point P (through chosen Ps) and it's projection to M
#                 P_scale_factor = 2.0
#                 Pw = Ps * P_scale_factor
#                 ax.scatter(Pw.x, Pw.y, Pw.z, color="blue", s=30)
#                 ax.text(Pw.x, Pw.y, Pw.z + 0.05, "Pw", color="blue")
#                 line_PM_xs = [Pw.x, self.Pm.x]
#                 line_PM_ys = [Pw.y, self.Pm.y]
#                 line_PM_zs = [Pw.z, self.Pm.z]
#                 ax.plot3D(line_PM_xs, line_PM_ys, line_PM_zs, color="blue", linestyle='--', linewidth=2.0, alpha=1.0)
#
#                 plt.show()  # Show both figures as subplots (same window)
#
#             Ps = np.array([X, Y, Z])
#===============================================================================

        return Ps

    def get_radial_distortion(self, p, dist_params):
        '''
        @brief Compute the polynomial radial distortion to be applied to an input p (from the normalized plane)

        @param p: array of points on the normalized projection plane
        @param dist_params: List of distortion params (polynomial coefficients)

        @return: the polynomial radial distortion to be applied to point p
        '''
        x_d = p[..., 0]
        y_d = p[..., 1]
        rho_dist_sq = x_d ** 2 + y_d ** 2
        radial_dist_factor = np.ones_like(rho_dist_sq)
        for idx, k in enumerate(dist_params):
            e = idx + 1  # Power term
            radial_dist_factor += k * rho_dist_sq ** e

        return radial_dist_factor[..., np.newaxis]

    def get_radially_distorted_points(self, p, dist_params):
        '''
        @brief Apply the polynomial radial distortion to an input p (from the normalized plane)

        @param p: array of points on the normalized projection plane
        @param dist_params: List of distortion params (polynomial coefficients)

        @return: the distorted point, such that \f$ P_d = P_u * distortion\f$
        '''
        distorted_point = p * self.get_radial_distortion(p, dist_params)
        return distorted_point

#===============================================================================
#     def distortion(self, p_und):
#         '''
#         @brief Apply distortion to input point (from the normalized plane)
#
#         @param p_und: undistorted point on the normalized "undistorted" projection plane
#
#         @retval distortion: distortion value that to be added to the undistorted point \f$P_u\f$ such that the distorted point is obtained by \f$ P_d = P_u + distortion\f$
#         '''
#         x_u = p_und[..., 0]
#         y_u = p_und[..., 1]
#         # Radial distortion parameters
#         k1 = self.precalib_params.k1
#         k2 = self.precalib_params.k2
#         k3 = self.precalib_params.k3
#
#         rho_und_sq = x_u ** 2 + y_u ** 2
#         radial_dist_factor = k1 * rho_und_sq + k2 * rho_und_sq ** 2 + k3 * rho_und_sq ** 3
#
#         if self.new_method:
#             # Unlike original model (by Mei), tangential distortion is no longer needed in the new model (by Xiang)
#             # since the effect has been well represented by Cp coordinates xi_x, xi_y, xi_z (wrt to sphere center or [M] frame).
#             distortion = radial_dist_factor[..., np.newaxis] * p_und
#             return distortion
#         else:
#             mx2_u = x_u * x_u
#             my2_u = y_u * y_u
#             mxy_u = x_u * y_u
#             # p's are tangential distortion parameters (needed for Mei's coaxial model)
#             p1 = self.precalib_params.p1
#             p2 = self.precalib_params.p2
#             dx_u = x_u * radial_dist_factor + 2 * p1 * mxy_u + p2 * (radial_dist_factor + 2 * mx2_u)
#             dy_u = y_u * radial_dist_factor + 2 * p2 * mxy_u + p1 * (radial_dist_factor + 2 * my2_u)
#             return dx_u, dy_u
#===============================================================================

#===============================================================================
#     def distortion_OLD(self, mx_u, my_u):
#         '''
#         @brief Apply distortion to input point (from the normalised plane)
#
#         @param mx_u: undistorted x coordinate of point on the normalised projection plane
#         @param my_u: undistorted y coordinate of point on the normalised projection plane
#
#         @retval dx_u: distortion value that was added to the undistorted point \f$mx_u\f$ such that the distorted point is produced \f$ mx_d = mx_u+dx_u \f$
#         @retval dy_u: distortion value that was added to the undistorted point \f$my_u\f$ such that the distorted point is produced \f$ my_d = my_u+dy_u \f$
#         '''
#
#         mx2_u = mx_u * mx_u
#         my2_u = my_u * my_u
#         mxy_u = mx_u * my_u
#         rho2_u = mx2_u + my2_u
#         k1 = self.precalib_params.k1
#         k2 = self.precalib_params.k2
#         p1 = self.precalib_params.p1
#         p2 = self.precalib_params.p2
#         radial_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u
#
#         if self.new_method:
#             # Unlike original model (by Mei), tangential distortion is no longer needed in the new model (by Xiang)
#             # since the effect has been well represented by xi1and xi2.
#             dx_u = mx_u * radial_dist_u
#             dy_u = my_u * radial_dist_u
#         else:
#             dx_u = mx_u * radial_dist_u + 2 * p1 * mxy_u + p2 * (rho2_u + 2 * mx2_u)
#             dy_u = my_u * radial_dist_u + 2 * p2 * mxy_u + p1 * (rho2_u + 2 * my2_u)
#
#         return dx_u, dy_u
#===============================================================================

class GUMStereo(OmniStereoModel):
    '''
    The vertically-folded omnistereo model using GUM
    '''

    def infer_additional_parameters_from_models(self):
#         # Model parameters
#         self.c1 = self.top_model.c
#         self.k1 = self.top_model.k
#         self.c2 = self.bot_model.c
#         self.k2 = self.bot_model.k
#         self.d = self.bot_model.d
#         # Foci
#         self.F1 = self.top_model.F
#         self.Oc = self.top_model.Fv
#         self.F2 = self.bot_model.F
#         self.F2v = self.bot_model.Fv
#         self.baseline = self.get_baseline()
#         # Radii:
#         self.system_radius = self.top_model.r_max
#         self.reflex_radius = self.top_model.r_min
#         self.camera_hole_radius = self.bot_model.r_min
#         self.system_height, self.height_above, self.height_below = self.get_system_heights()
#         self.resolve_theoretical_bounds()
        self.theoretical_model = None

    def update_optimized_params(self, params, only_intrinsics = False, only_extrinsics = False, suppress_tz_optimization = False, final_update = False):
        '''
        Plays an important role during calibration as the new parameter values have to be updated
        '''
        tz_offset = 2
        if only_intrinsics:
            only_extrinsics = False
        # NOTE: the tz in this case is for the translation in the Transformation of [C] wrt [M]!
        if only_extrinsics:
            top_params = [params[0]]  # just the tz for the top GUM
            bottom_params = [params[1]]  # just the tz for the bottom GUM
        else:
            intrinsic_offset = 11 + tz_offset
            top_params = np.concatenate(([params[0]], params[tz_offset:intrinsic_offset]))
            bottom_params = np.concatenate(([params[1]], params[intrinsic_offset:]))

        if final_update:
            print("FINAL update using optimized parameters:")
        self.top_model.update_optimized_params(params = top_params, only_intrinsics = only_intrinsics, only_extrinsics = only_extrinsics, suppress_tz_optimization = suppress_tz_optimization, final_update = final_update)
        self.bot_model.update_optimized_params(params = bottom_params, only_intrinsics = only_intrinsics, only_extrinsics = only_extrinsics, suppress_tz_optimization = suppress_tz_optimization, final_update = final_update)

        if final_update:
            self.set_params()  # This propagates the globally highest/lowest elevations angles
            T_top_wrt_C = self.top_model.T_model_wrt_C
            T_bottom_wrt_C = self.bot_model.T_model_wrt_C
            self.set_poses(T_top_wrt_C = T_top_wrt_C, T_bottom_wrt_C = T_bottom_wrt_C)
            self.common_vFOV = self.set_common_vFOV(verbose = True)
            self.baseline = self.get_baseline()
            print("Baseline = %.2f %s" % (self.baseline, self.units))

    def get_baseline(self):
        baseline = self.top_model.F[2, 0] - self.bot_model.F[2, 0]
        return baseline

    def init_theoretical_model(self, theoretical_model):
        self.theoretical_model = theoretical_model
        if self.theoretical_model is not None:
            self.theoretical_model.current_omni_img = self.current_omni_img
            self.top_model.init_theoretical_model(theoretical_model.top_model)
            self.bot_model.init_theoretical_model(theoretical_model.bot_model)

            # CHECKME: GUM shouldn't need this!
            #=======================================================================
            # top_pc_params = self.top_model.theoretical_model.precalib_params
            # self.top_model.precalib_params.set_pinhole_cam_params(focal_length=top_pc_params.focal_length, image_size=top_pc_params.image_size, pixel_size=top_pc_params.pixel_size, sensor_size=top_pc_params.sensor_size)
            # bot_pc_params = self.bot_model.theoretical_model.precalib_params
            # self.bot_model.precalib_params.set_pinhole_cam_params(focal_length=bot_pc_params.focal_length, image_size=bot_pc_params.image_size, pixel_size=bot_pc_params.pixel_size, sensor_size=bot_pc_params.sensor_size)
            #=======================================================================

