# -*- coding: utf-8 -*-
# cata_hyper_model.py

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
Definition of the omnistereo model proposed in the related MDPI Sensors article titled:
'Design and Analysis of a Single−Camera Omnistereo Sensor for Quadrotor Micro Aerial Vehicles (MAVs)'.
'''
from __future__ import division
from __future__ import print_function

import numpy as np
from omnistereo.camera_models import OmniCamModel, OmniStereoModel, CamParams
from omnistereo.common_tools import convert_steradian_to_radian, convert_resolution_units

class HyperCataStereo(OmniStereoModel):

    def set_models_params(self, **kwargs):
        self.top_model.set_model_params(**kwargs)
        self.bot_model.set_model_params(**kwargs)
        self.T_Cest_wrt_Rgt = np.identity(4)  # Hand-eye transformation
        self.infer_additional_parameters_from_models()

    def get_baseline(self, **kwargs):
        c1 = kwargs.get("c1", self.c1)
        c2 = kwargs.get("c2", self.c2)
        d = kwargs.get("d", self.d)
        baseline = c1 + c2 - d
        return baseline

    def infer_additional_parameters_from_models(self):
        # Model parameters
        self.c1 = self.top_model.c
        self.k1 = self.top_model.k
        self.c2 = self.bot_model.c
        self.k2 = self.bot_model.k
        self.d = self.bot_model.d
        # Foci
        self.F1 = self.top_model.F
        self.Oc = self.top_model.Fv
        self.F2 = self.bot_model.F
        self.F2v = self.bot_model.Fv
        self.baseline = self.get_baseline()
        # Radii:
        self.system_radius = self.top_model.r_max
        self.reflex_radius = self.top_model.r_min
        self.camera_hole_radius = self.bot_model.r_min
        self.system_height, self.height_above, self.height_below = self.get_system_heights()
        try:
            self.resolve_theoretical_bounds()
        except:
            print("Problem: could not resolve theoretical bounds")
            pass

    def get_baseline_for_modified_FOV(self, alpha_cam, alpha_1, **kwargs):
        '''
        Computes the baseline needed to fit a certain camera field of view angle \f$\alpha_{cam}$\f for a vFOV \f$\alpha_1\f$ on mirror 1

        @param alpha_cam: The camera's FOV (usually the smallest, such as the height-wise FOV on a landscape camera) to be satisfied as solved for \f$c_i\f$
        @param alpha_1: The specific field of view \f$\alpha_1\f$ to be satisfied
        @param kwargs: It could be useful parameters to enforce, for example, \f$k_1\f$: the curvature-related parameter of mirror 1.
            When "get_all" is set to True, all values are returned. Otherwise, only the values starting from the maximum index of the result are returned

        @retval b: the newly computed baseline \f$b\f$ for the given hyperboloid that achieves the desired FOV \f$alpha_{cam}\f$.
        @retval (k1, k2): a tuple of arrays of k_i arguments associated to the baseline computed
        @retval (c1, c2, d): a tuple containing the new values for \f$(c_1, c_2, d)$\f
        '''

        mirror1 = self.top_model
        mirror2 = self.bot_model
        get_all = kwargs.get("get_all", False)
        k1s = kwargs.get("k1", self.k1)
        r_sys, k1 = mirror1.get_rsys_analytically(alpha_1, ki = k1s, get_all = True)
        z1_max = r_sys / np.tan(alpha_cam / 2)
        c1 = k1 * z1_max - np.sqrt(k1 * (k1 - 2)) * np.sqrt(r_sys ** 2 + z1_max ** 2)
#         delta_z = c1 - self.c1
#         d = self.d
#         d_new = d + 2*delta_z
        d_new = self.d
        k2 = self.k2
#         x2 = r_sys
#         y2 = np.zeros_like(x2)  # The y coordinates (all zeros)
#
#         z2 = mirror2.get_z_hyperbola(x2, y2, is_virtual=False)
#         c2 = k2 * (d_new - z2) - np.sqrt(k2 * (k2 - 2) * (d_new ** 2 - 2 * d_new * z2 + r_sys ** 2 + z2 ** 2))
        c2 = self.c2
#         b = self.get_baseline(c1=c1, c2=c2, d=d_new)
        b = self.get_baseline(c1 = c1)
        if get_all:
            return b, (np.zeros_like(b) + k1, np.zeros_like(b) + k2), (c1, c2, d_new)
        else:
            if isinstance(b, np.ndarray):
                if isinstance(k1, np.ndarray):
                    b_max_index = np.argmax(b)  # Get rid of the overshoot for plotting purposes
                else:
                    b_max_index = 0
#                 return b[b_max_index:], ((np.zeros_like(b) + k1)[b_max_index:], (np.zeros_like(b) + k2)[b_max_index:]), (c1[b_max_index:], c2[b_max_index:], d_new[b_max_index:])
                return b[b_max_index:], ((np.zeros_like(b) + k1)[b_max_index:], (np.zeros_like(b) + k2)[b_max_index:])  # , (c1[b_max_index:], c2[b_max_index:], d_new[b_max_index:])
            else:
                return b, (k1, k2)  # , (c1, c2, d_new)

    def get_system_heights(self):
        '''
        Computes the total height of the rig (theoretically)
        '''
        self.height_above = self.top_model.get_z_hyperbola(self.system_radius, 0)
        self.height_below = self.bot_model.get_z_hyperbola(self.system_radius, 0)
        self.system_height = self.height_above - self.height_below  # System height
        return self.system_height, self.height_above, self.height_below

    def print_rig_dimensions(self):
        # Height
        h_sys, _, _ = self.get_system_heights()
        print("System Height = %.2f mm" % (h_sys))

        # Masses:
        density_mirror = 7.19  # Chromium: 7.19 | Brass: 8.4 to 8.73 | Copper: 8.96  [g/cm^3]
        density_tube = 1.17  # Acrylic:  1.17 to 1.20  | Glass: 2.4-2.8 [g/cm^3]
        thickness_mirror = 4  # Big rig: 4.0 mm | Small rig: 2.5 mm
        thickness_tube = 3.5
        r_inner_tube = 38  # mm
        r_outer_tube = 0
        h_tube = 178  # mm
        mass_cam = 25  # g
        mass_mirror_lip = get_hollow_cylinder_mass(density_mirror, 5, self.system_radius - 6.5, 6.5)  # Optional mass
        # Measurents:
        # Big Rig: M1 =  142 g, M2 = 155 g | Small Rig: M1 = 44 g, M2 = 41 g
        mass1 = self.top_model.get_mass(density_mirror, thickness_mirror) + mass_mirror_lip
        mass2 = self.bot_model.get_mass(density_mirror, thickness_mirror) + mass_mirror_lip
        mass_tube = get_hollow_cylinder_mass(density_tube, h_tube, r_inner_tube, thickness_tube)
        mass_sys = mass1 + mass2 + mass_tube + mass_cam
        print("Total mass = %f g:" % (mass_sys))
        print("(M1 = %f g) + (M2 = %f g) + (Mtub = %f g) + (Mcam = %f g)" % (mass1, mass2, mass_tube, mass_cam))

    def get_quadrant_points(self, x, y, z):
        '''
        It takes the 3D coordinates for a point in order to produce the symmetric set of points about each axis (counterclockwise order)
        @return:  The ndarray of 4 points (1 point per axis)
        '''
        return np.array([[[x, y, z, 1], [y, x, z, 1], [x, -y, z, 1], [y, -x, z, 1]]])

    def resolve_theoretical_bounds(self):
        # Top mirror's edge
        x = self.system_radius
        y = 0
        z = self.height_above
        self.top_highest_bound_3D_points = self.get_quadrant_points(x, y, z)
        _, _, self.top_highest_bound_pixels = self.top_model.get_pixel_from_3D_point_wrt_C(self.top_highest_bound_3D_points)
        # Due to reflex mirror
        x = self.reflex_radius
        y = 0
        z = self.d / 2.0
        self.top_lowest_bound_3D_points = self.get_quadrant_points(x, y, z)
        _, _, self.top_lowest_bound_pixels = self.top_model.get_pixel_from_3D_point_wrt_C(self.top_lowest_bound_3D_points)

        # Bottom mirror's bounds:
        self.bottom_highest_bound_3D_points = self.top_highest_bound_3D_points
        _, _, self.bottom_highest_bound_pixels = self.bot_model.get_pixel_from_3D_point_wrt_C(self.bottom_highest_bound_3D_points)

        x = self.system_radius
        y = 0
        z = self.height_below
        self.bottom_lowest_bound_3D_points = self.get_quadrant_points(x, y, z)
        _, _, self.bottom_lowest_bound_pixels = self.bot_model.get_pixel_from_3D_point_wrt_C(self.bottom_lowest_bound_3D_points)

        # Due to camera hole
        x = self.camera_hole_radius
        y = 0
        z = self.bot_model.get_z_hyperbola(self.camera_hole_radius, 0)
        self.cam_hole_3D_points = self.get_quadrant_points(x, y, z)
        _, _, self.cam_hole_pixels = self.bot_model.get_pixel_from_3D_point_wrt_C(self.cam_hole_3D_points)

    def get_min_elevations_theoretical(self):
        '''
        Compute the minimum elevations theoretically with
        \f[
            \theta_{1,min} &= \atan{\frac{d - 2 c_1}{2 r_{ref}}}
        \f]
        and
        \f[
            \theta_{2,min} &= \atan{
            \frac{c_2 - \sqrt {c_2^2{\left( {1
            - \frac{2}{{{k_2}}}} \right) + 2r_{sys}^2\left( {{k_2} - 2} \right)}}}{2 r_{sys}}}
        \f]
        @retval min_elev_1: The theoretical minimum elevation angle for mirror 1
        @retval min_elev_2: The theoretical minimum elevation angle for mirror 2
        '''
        r_sys = self.system_radius
        r_ref = self.reflex_radius

        min_elev_1 = np.arctan((self.d - 2.0 * self.c1) / (2.0 * r_ref))
        min_elev_2 = np.arctan((self.c2 - np.sqrt(self.c2 ** 2 * (1.0 - 2.0 / self.k2) + 2 * r_sys ** 2 * (self.k2 - 2.0))) / (2.0 * r_sys))
        return min_elev_1, min_elev_2

    def get_max_elevations_theoretical(self):
        '''
        Compute the maximum elevations theoretically with
        \f[
            \theta_{1,max} &= \atan{\frac{\sqrt{\frac{(k_1-2) \left(c_1^2+2 k_1 r^2_{sys}
            \right)}{k_1}}-c_1}{2 r_{sys}}}
        \f]
        and
        \f[
            \theta_{2,max} = \atan{\frac{\frac{c_1}{2}+ c_{2} - d\right
            +\sqrt{\frac{c_1^2 \left (k_{1} - 2\right )}{4\cdot k_{1}} + r_{sys}^{2} \left (\dfrac{k_{1}}{2} -1\right )}}
            {r_{sys}}}
        \f]
        @retval max_elev_1: The theoretical maximum elevation angle for mirror 1
        @retval max_elev_2: The theoretical maximum elevation angle for mirror 2
        '''
        r_sys = self.system_radius

        max_elev_1 = np.arctan((np.sqrt((self.k1 - 2.0) * (self.c1 ** 2 + 2 * self.k1 * r_sys ** 2) / self.k1) - self.c1) / (2.0 * r_sys))
        max_elev_2 = np.arctan((self.c1 / 2.0 + (self.c2 - self.d) + np.sqrt((self.c1 ** 2 * (self.k1 - 2.0)) / (4 * self.k1) + r_sys ** 2 * (self.k1 / 2.0 - 1.0))) / r_sys)
        return max_elev_1, max_elev_2

    def compare_elevation_values(self):
        min_elev1, min_elev2 = self.get_min_elevations_theoretical()
        print("MIN Elevations: Mirror 1: %f degrees, Mirror 2: %f degrees" % (np.rad2deg(min_elev1), np.rad2deg(min_elev2)))
        print("VS. Elevation from projections")
        _, lowest_elev_top = self.top_model.get_direction_angles_from_pixel(self.top_lowest_bound_pixels)
        _, lowest_elev_bottom = self.bot_model.get_direction_angles_from_pixel(self.bottom_lowest_bound_pixels)
        print("Mirror 1: %f degrees, Mirror 2: %f degrees" % (np.rad2deg(lowest_elev_top[0, 0]), np.rad2deg(lowest_elev_bottom[0, 0])))

        max_elev1, max_elev2 = self.get_max_elevations_theoretical()
        print("MAX Elevations: Mirror 1: %f degrees, Mirror 2: %f degrees" % (np.rad2deg(max_elev1), np.rad2deg(max_elev2)))
        print("VS. Elevation from projections")
        _, highest_elev_top = self.top_model.get_direction_angles_from_pixel(self.top_highest_bound_pixels)
        _, highest_elev_bottom = self.bot_model.get_direction_angles_from_pixel(self.bottom_highest_bound_pixels)
        _, highest_elev_cam_hole = self.bot_model.get_direction_angles_from_pixel(self.cam_hole_pixels)
        print("Mirror 1: %f degrees, Mirror 2: %f degrees" % (np.rad2deg(highest_elev_top[0, 0]), np.rad2deg(highest_elev_bottom[0, 0])))
        print("NOTE: Mirror 2's highest elevation may be limited by camera hole with : %f degrees" % (np.rad2deg(highest_elev_cam_hole[0, 0])))

    def test_radius_equality_constraint(self):
        from omnistereo.common_tools import unit_test
        print("Radius equality constraint check:")
        mirror1 = self.top_model
        mirror2 = self.bot_model
        r_sys_1 = mirror1.get_r_hyperbola(mirror1.get_z_hyperbola(self.system_radius, 0))
        r_sys_2 = mirror2.get_r_hyperbola(mirror2.get_z_hyperbola(self.system_radius, 0))
        unit_test(r_sys_1, r_sys_2, decimals = 10)

    def lambdify_mid_Pw(self, omnistereo_sym):
        from sympy import lambdify
        args = (omnistereo_sym.direction_vectors_as_symb, omnistereo_sym.c1, omnistereo_sym.c2, omnistereo_sym.d)
        expr = omnistereo_sym.mid_Pw
        mid_Pw_as_function = lambdify(args, expr, "numpy")
        return mid_Pw_as_function

    def lambdify_mid_Pw_expanded(self, omnistereo_sym):
        from sympy import lambdify
        args = (omnistereo_sym.uv_coords, omnistereo_sym.fu, omnistereo_sym.fv, omnistereo_sym.s, omnistereo_sym.uc, omnistereo_sym.vc, omnistereo_sym.k1, omnistereo_sym.k2, omnistereo_sym.c1, omnistereo_sym.c2, omnistereo_sym.d)
        expr = omnistereo_sym.mid_Pw_expanded
        mid_Pw_as_function = lambdify(args, expr, "numpy")
        return mid_Pw_as_function

    def set_pixel_coordinates_covariance(self, stdev_on_pixel_coord):
        '''
        @param stdev_on_pixel_coord: pixel error on feature detection (usually determined from experiments)
        '''
        self.cov_pixel_coords_matrix = stdev_on_pixel_coord ** 2 * np.eye(4)

    def get_covariance_matrix(self, jacobian_matrix, pixel_std_dev = 1.0):
        '''
        @param pixel_std_dev: pixel error on feature detection (usually determined from experiments)
        '''
        self.set_pixel_coordinates_covariance(pixel_std_dev)
        cov_matrix = np.dot(jacobian_matrix, np.dot(self.cov_pixel_coords_matrix, jacobian_matrix.T))
        return cov_matrix.astype('float')

    def draw_radial_bounds_theoretical(self, omni_img = None):
        import cv2

        if omni_img is None:
            omni_img = self.current_omni_img

        img_center_point = self.top_model.precalib_params.center_point
        # circ_top = cv2.fitEllipse(omni_stereo.top_highest_bound_pixels[...,:2].reshape(4,2).astype('float32')) # needs more than 4 points!
        r_high_top = np.linalg.norm(self.top_highest_bound_pixels[0, 0, :2] - img_center_point, axis = -1)
        r_low_top = np.linalg.norm((self.top_lowest_bound_pixels[0, 0, :2] - img_center_point), axis = -1)
        r_high_bot = np.linalg.norm(self.bottom_highest_bound_pixels[0, 0, :2] - img_center_point, axis = -1)
        r_low_bot = np.linalg.norm((self.bottom_lowest_bound_pixels[0, 0, :2] - img_center_point), axis = -1)

        # Make copy of image
        img = omni_img.copy()
        # Show theoretical radial boundaries
        win_name = "Theoretical Radial Bounds"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        # Draw:
        # circle center
        center = (int(img_center_point[0]), int(img_center_point[1]))
        cv2.circle(img, center, 3, (0, 255, 0), -1, 8, 0)
        # circle outline
        cv2.circle(img, center, int(r_high_top), (0, 255, 0), 3, 8, 0)
        cv2.circle(img, center, int(r_low_top), (0, 255, 0), 3, 8, 0)
        cv2.circle(img, center, int(r_high_bot), (0, 255, 0), 3, 8, 0)
        cv2.circle(img, center, int(r_low_bot), (0, 255, 0), 3, 8, 0)

        cv2.imshow(win_name, img)
        pressed_key = cv2.waitKey(10)
        return pressed_key

    def get_FOVcom_from_FOV1(self, vFOV1):
        min_elev_1, min_elev_2 = self.get_min_elevations_theoretical()
        _, max_elev_2 = self.get_max_elevations_theoretical()
        max_elev_1 = vFOV1 - abs(min_elev_1)
        vFOV_common = self.set_common_vFOV(theta1_max = max_elev_1, theta1_min = min_elev_1, theta2_max = max_elev_2, theta2_min = min_elev_2)
        return vFOV_common

class HyperCata(OmniCamModel):

    def __init__(self, number, F, F_virt, c, k, d = None, **kwargs):
        self.c = c
        self.k = k
        self.d = d
        if number == 1:
            self.mirror_name = "top"
            self.mirror_number = number
            self.z_0 = self.c / 2.0  # z_0 is the offset # TODO: add optional adjustment on position of camera.
        elif number == 2:
            self.mirror_name = "bottom"
            self.mirror_number = 2
            self.z_0 = self.d - self.c / 2.0
            self.n_reflex = np.array([0, 0, -1.0, 1]).reshape(4, 1)  # Reflex mirror (plane normal unit vector)

        self.a = self.get_a_hyperbola(self.c, self.k)
        self.b = self.get_b_hyperbola(self.c, self.k)

        self._init_default_values(mirror_number = number, F = F, F_virt = F_virt, **kwargs)

    def set_model_params(self, **kwargs):  # TODO: WIP
        pass  # Nothing to do for now

    def get_reflex_surf_points(self, scale = 1):
        # Planar mirror (reflex on mirror 1)
        u = np.linspace(0, 2 * np.pi, 100)
        v_reflex = np.linspace(0, self.r_min, 50)  # Assuming r_min is the reflex radius
        x_reflex = scale * np.outer(np.cos(u), v_reflex)
        y_reflex = scale * np.outer(np.sin(u), v_reflex)
        z_reflex = np.zeros_like(x_reflex) + scale * self.d / 2.0
        return x_reflex, y_reflex, z_reflex

    def get_mounting_lip_points(self, scale = 1, steps = 100, lip_radius = 10, seam_size = 0):
        '''
        Computes the surface points for the mounting lip of the mirrors

        @param steps: Number of steps in the linear space interpolation
        @param lip_radius: The radial width of the mounting lip (if any)
        @param seam_size:  Trick for joining surfaces so there is no gap
        '''
        x = np.linspace(-self.r_max - lip_radius, self.r_max + lip_radius, steps)
        y = x
        xx_lip, yy_lip = np.meshgrid(x, y)
        r_squared = xx_lip ** 2 + yy_lip ** 2
        lip_mask = np.logical_and(r_squared < (self.r_max + lip_radius) ** 2, r_squared >= (self.r_max - seam_size) ** 2)
        z = self.get_z_hyperbola(self.r_max, 0)
        zz_lip = np.where(lip_mask, z, np.nan)

        x_lip = scale * x
        y_lip = scale * y
        zz_lip = scale * zz_lip
        return x_lip, y_lip, zz_lip

    def get_surf_points(self, scale = 1, steps = 100, dense = False):
        '''
        @param steps: Number of steps in the linear space interpolation
        @param dense: True if desired to get data for all the positions of a square grid (Those where data doesn't exist will be set to nan)
        '''
        if dense:
            x = np.linspace(-self.r_max, self.r_max, steps)
            y = np.linspace(-self.r_max, self.r_max, steps)
            hyper_xx, hyper_yy = np.meshgrid(x, y)
            hyper_r_squared = hyper_xx ** 2 + hyper_yy ** 2
            inner_bound_mask = hyper_r_squared >= self.r_min ** 2
            validation_mask = np.logical_and(inner_bound_mask, hyper_r_squared <= self.r_max ** 2)
            hyper_xx = np.where(validation_mask, hyper_xx, np.nan)
            hyper_yy = np.where(validation_mask, hyper_yy, np.nan)
            hyper_zz = self.get_z_hyperbola(hyper_xx, hyper_yy)
            if self.mirror_number == 1:
                z_reflex_mirror = self.get_z_hyperbola(self.r_min, 0)
                hyper_zz = np.where(inner_bound_mask, hyper_zz, z_reflex_mirror)

            hyper_xx = scale * hyper_xx
            hyper_yy = scale * hyper_yy
            hyper_zz = scale * hyper_zz
            hyper_x = scale * x
            hyper_y = scale * y
            return (hyper_xx, hyper_x), (hyper_yy, hyper_y), hyper_zz
        else:
            u = np.linspace(0, 2 * np.pi, steps)
            v = np.linspace(self.r_min, self.r_max, steps)
            hyper_xx = scale * np.outer(np.cos(u), v)
            hyper_yy = scale * np.outer(np.sin(u), v)
            hyper_zz = scale * self.get_z_hyperbola(hyper_xx, hyper_yy)
            return hyper_xx, hyper_yy, hyper_zz

    def get_t_bp(self, q):
        '''
        Computes the t parameter for the backward projection equation

        @param q: The Euclidean 3D point (as an ndarray of rows x cols x 3) on the normalized projection plane

        @return: The t parameters as an ndarray (rows x cols) corresponding to the point(s) q
        '''
        c, k = self.c, self.k
        return c / (k * q[..., 2] - np.sqrt(k * (k - 2)) * get_vector_magnitude(q))

    # This was just used for testing the back-projection equation
    def get_t_bp_long(self, q):
        c, k = self.c, self.k
        return (c * k + c * np.sqrt(k * (k - 2)) * get_vector_magnitude(q)) / (k * (-k * (q[0] ** 2 + q[1] ** 2) + 2 * (q[0] ** 2 + q[1] ** 2 + q[2] ** 2)))

    def back_project_Q_to_mirror(self, q):
        '''
        Back-projects a 3D point q from the normalized projection plane into its reflection point in the mirror.

        @param q: An ndarray (rows, cols, 3) of the Euclidean vector coordinates on the normalized projection plane to be lifted. Note, z = 1!!!, so coordinates can be thought of given w.r.t. the mirror's virtual focus.
        @return The back-projected reflection point's position vector in homogeneous coordinates (as an rows x cols x 4 ndarray)
        '''
        t = self.get_t_bp(q)
        if self.mirror_number == 1:
            P = t[..., np.newaxis] * q
        elif self.mirror_number == 2:
            P = np.dstack((t, t, self.d - t)) * q  # This is correct. Recall that we must use the q on the virtual plane but wrt C

        if np.ndim(P) > 1:
            P_in_mirror_homo = np.dstack((P, np.ones(P.shape[:-1])))
        else:  # Special case input (single point)
            P_in_mirror_homo = np.hstack((P, np.ones(np.ndim(P))))

        return P_in_mirror_homo

    def get_point_wrt_focus(self, p):
        '''
        @param p: p is an ndarray (rows, cols, 3 or 4) representing a 3D point in space wrt to Oc (origin of common reference frame)
        @return: the position of p wrt the focus
        '''
        if p.shape[-1] == 4:
            v = np.ones_like(p)
            v[..., :3] = p[..., :3] - (self.F.T)[..., :3]
        else:
            v = p - self.F[:-1].T

        return v

    def get_points_wrt_M(self, p):
        return self.get_point_wrt_focus(p)

    def get_point_wrt_origin(self, p):
        '''
        @param p: p is an ndarray (rows, cols, 3 or 4) representing a 3D point in space wrt to the focus
        @return: the position of p wrt the origin of the common frame of reference
        '''
        if p.shape[-1] == 4:
            v = np.ones_like(p)
            v[..., :3] = p[..., :3] + (self.F.T)[..., :3]
        else:
            v = p + self.F[:-1].T

        return v

    def get_gamma_from_xy(self, x, y, k = None):
        use_r_sq = True
        r_square = get_r(x, y, give_r_squared = use_r_sq)
        gamma = self.get_gamma_from_r(r_square, is_r_squared = use_r_sq, k = k)
        return gamma

    def get_gamma_from_r(self, r, is_r_squared = False, k = None):

        if is_r_squared:
            r_square = r
        else:
            r_square = r ** 2

        if k == None:  # Compute with the current model's a and b parameters
            a = self.a
            b = self.b
        else:
            a = self.get_a_hyperbola(self.c, k)
            b = self.get_b_hyperbola(self.c, k)

        gamma = a / b * np.sqrt(b ** 2 + r_square)

        return gamma

    def get_Gamma(self, z, k = None):
        if k == None:
            a = self.a
        else:
            a = self.get_a_hyperbola(self.c, k)

        Gamma = np.sqrt((z - self.z_0) ** 2 / a ** 2 - 1)
        return Gamma

    def get_spatial_resolution_in_2D(self, p, in_pixels = False, in_radians = True, use_spatial_resolution = True):
        '''
        Spatial resolution for the catadioptric sensor using 2D angles

        @param p: The ndarray of 3d point(s) of shape (rows, cols, vect_size) to compute the resolution at
                  Note: p must be given already transformed on the respective frame (e.g. when using mirror 2 it must be in the virtual camera frame)
        @param in_pixels: Set to True in order to provide the spatial resolution normalized in \f$\left[{st}/{px^2}\right]\f$. If False (default), it will be given as \f$\left[{st}/{area}\right]\f$
        @param in_radians: True (default) to provide the spatial resolution in \f$\left[{rad}/{px}\right]\f$. Otherwise it will be given as \f$\left[{deg}/{px}\right]\f$
        @param use_spatial_resolution: If True, it indicates to calculate the spatial resolution (as a ratio) in [length per radian] units instead of [radian per length].

        @return the ndarray of 2D resolutions for the respective profile points
        '''
        eta = self.get_spatial_resolution(p)
        # Convert solid angle to 2D (as crossection of cone)
        # Thinking the infinitesimal surface is a square:
        # eta_2D = np.sqrt(eta) / convert_steradian_to_radian(1)
        # Thinking the infinitesimal surface is a circle:
        eta_2D = 2 * np.sqrt(eta / np.pi) / convert_steradian_to_radian(1)
        if use_spatial_resolution == False:
            eta_2D = 1 / eta_2D
        eta_2D = convert_resolution_units(self.precalib_params.pixel_size[0], in_pixels, in_radians, use_spatial_resolution, eta_2D, in_2D = True)

        return eta_2D

    def get_vertical_range_variation(self, const_rho):
        '''
        @retval delta_z: the ndarray of \f$\Delta z$\f differences in change for the sequence of 1-pixel intervals on the image for points back-projected beyond $\min(\rho_{ns,low}, \rho_{ns,high})$ limits
        @retval z_level: the ndarray of \f$z_w$\f midpoint levels used to compute the results
        '''
        _, elev_all = self.get_all_direction_angles_per_pixel_radially()
        elev_a = elev_all[..., :-1]
        elev_b = elev_all[..., 1:]
        z_of_F = self.F[2]
        z_a = z_of_F + const_rho * np.tan(elev_a)
        z_b = z_of_F + const_rho * np.tan(elev_b)
        delta_z = np.abs(z_a - z_b)
        z_level = (z_a + z_b) / 2.0
        return delta_z, z_level

    def get_horizontal_range_variation(self, const_z):
        '''
        @retval delta_rho: the ndarray of \f$\Delta \rho$\f differences in change for the sequence of 1-pixel intervals on the image for points back-projected within $z_{min,max}$ limits
        @retval rho_level: the ndarray of \f$\rho_w$\f midpoint levels used to compute the results
        '''
        dist_from_const_z_to_F = self.F[2] - const_z
        _, elev_all = self.get_all_direction_angles_per_pixel_radially()
        if dist_from_const_z_to_F > 0:
            elev_all_to_plane = np.where(elev_all < 0, elev_all, np.nan)
        elif dist_from_const_z_to_F < 0:
            elev_all_to_plane = np.where(elev_all > 0, elev_all, np.nan)

        elev_a = elev_all_to_plane[..., :-1]  # elev_a < elev_b for Mirror 1
        elev_b = elev_all_to_plane[..., 1:]
        rho_a = dist_from_const_z_to_F / np.tan(-elev_a)
        rho_b = dist_from_const_z_to_F / np.tan(-elev_b)
        delta_rho = rho_b - rho_a
        rho_level = (rho_a + rho_b) / 2.0
        return delta_rho, rho_level

    def get_angular_range_variation(self, const_mag):
        '''
        @retval delta_phi: the ndarray of \f$\Delta \phi$\f as angle differences wrt to the camera frame for a constant distance to \f$P_w$\f from \f$O_c$\f
        @retval phi_level: the ndarray of \f$\phi_w$\f midpoint levels used to compute the results
        '''
        _, elev_all = self.get_all_direction_angles_per_pixel_radially()
        elev_a = elev_all[..., :-1]
        elev_b = elev_all[..., 1:]
        z_of_F = self.F[2]
        term_sqrt = np.sqrt(const_mag ** 2 - z_of_F ** 2 * np.cos(elev_a) ** 2)
        phi_a = np.arctan((z_of_F * np.cos(elev_a) ** 2 - np.sin(elev_a) * term_sqrt) / (np.cos(elev_a) * (term_sqrt + z_of_F * np.sin(elev_a))))
        phi_b = np.arctan((z_of_F * np.cos(elev_b) ** 2 - np.sin(elev_b) * term_sqrt) / (np.cos(elev_b) * (term_sqrt + z_of_F * np.sin(elev_b))))
        delta_phi = np.abs(phi_a - phi_b)
        phi_level = (phi_a + phi_b) / 2.0
        return delta_phi, phi_level

    def get_spatial_resolution(self, p, in_pixels = False, in_steradians = True, use_spatial_resolution = True):
        '''
        Spatial resolution for the catadioptric sensor using 3D solid angles

        @param p: The ndarray of 3d point(s) of shape (rows, cols, vect_size) to compute the resolution at
                  Note: p must be given already transformed on the respective frame (e.g. when using mirror 2 it must be in the virtual camera frame)
        @param in_pixels: Set to True in order to provide the spatial resolution normalized in \f$\left[{st}/{px^2}\right]\f$. If False (default), it will be given as \f$\left[{st}/{area}\right]\f$
        @param in_steradians: True (default) to provide the spatial resolution in \f$\left[{st}/{area}\right]\f$. Otherwise it will be given as \f$\left[{deg^2}/{area}\right]\f$
               To convert from steradian to square degrees multiply by 3282.810874
        @param use_spatial_resolution: If True, it indicates to calculate the spatial resolution (as a ratio) in [area per st] units instead of [st per area].

        @return the ndarray of resolutions for the respective points
        '''
        f = self.precalib_params.focal_length
        c = self.c
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        r_sq = get_r(x, y, give_r_squared = True)

        if use_spatial_resolution:
            if self.mirror_number == 1:
                eta = (f ** 2 * (r_sq + (c - z) ** 2) * np.sqrt(r_sq + z ** 2)) / (z ** 3)
            else:
                d = self.d
                eta_cam_2 = f ** 2 * ((np.sqrt(r_sq + (d - z) ** 2)) / (d - z)) ** 3
                eta = (r_sq + (c - d + z) ** 2) / (r_sq + (d - z) ** 2) * eta_cam_2
        else:
            eta = (z ** 3) / (f ** 2 * (r_sq + (c - z) ** 2) * np.sqrt(r_sq + z ** 2))

        if in_pixels:  # CHECKME: This is using pixels at z instead of in the camera plane
            eta = self.precalib_params.pixel_area * eta  # [mm^2 / px^2] * [st/mm^2]
        if in_steradians == False:
            # To convert from steradian to square degrees multiply by (180/pi)^2 or 3282.810874
            eta = (180 / np.pi) ** 2 * eta

        return eta  # CHECKME: something is wrong for mirror 2's values

    def get_spatial_resolution_as_BakerNayar(self, p, in_pixels = False, in_steradians = True, use_spatial_resolution = True):
        '''
        Spatial resolution for the catadioptric sensor the origin as the mirror viewpoint (like in Baker & Nayar's 1999 theory paper)

        @param p: The ndarray of 3d point(s) of shape (rows, cols, vect_size) to compute the resolution at
                  Note: p must be given already transformed on the respective frame (e.g. when using mirror 2 it must be in the virtual camera frame)
        @param in_pixels: Set to True in order to provide the spatial resolution normalized in \f$\left[{st}/{px^2}\right]\f$. If False (default), it will be given as \f$\left[{st}/{area}\right]\f$
        @param in_steradians: True (default) to provide the spatial resolution in \f$\left[{st}/{area}\right]\f$. Otherwise it will be given as \f$\left[{deg^2}/{area}\right]\f$
               To convert from steradian to square degrees multiply by 3282.810874
        @param use_spatial_resolution: If True, it indicates to calculate the spatial resolution (as a ratio) in [area per st] units instead of [st per area].

        @return the ndarray of resolutions for the respective points
        '''
        u = self.precalib_params.focal_length
        c = self.c
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        r_sq = get_r(x, y, give_r_squared = True)

        cos_psi_cube = ((c - z) / (np.sqrt((c - z) ** 2 + r_sq))) ** 3
        eta_cam = u ** 2 / cos_psi_cube

        eta = (r_sq + z ** 2) / (r_sq + (c - z) ** 2) * eta_cam
        if use_spatial_resolution == False:
            eta = 1 / eta

        # Alternatively:
        # eta = (self.precalib_params.get_spatial_resolution(p)) * ((r_sq + z ** 2) / (r_sq + (c - z) ** 2))

        if in_pixels:
            eta = self.precalib_params.pixel_area * eta  # [mm^2][px^2]*[st/mm^2]
        if in_steradians == False:
            # To convert from steradian to square degrees multiply by (180/pi)^2 or 3282.810874
            eta = (180 / np.pi) ** 2 * eta

        return eta

    def get_z_hyperbola(self, x, y, is_virtual = False, k = None):
        '''
        @param is_virtual: True if the z-value should be such of the reflected (virtual) point created by the reflex (planar) mirror.
                           It only applies to mirror 2 (at bottom) since its points can be reflected by the reflex mirror
        @param k: The \f$k\f$ parameter. If None is passed, the current model's value is used instead (default)
        '''
        gamma = self.get_gamma_from_xy(x, y, k)
        z = None
        # TODO: add radial constraint from a dictionary of constraints, such as
        # if r_min <= r <= r_max
        if self.mirror_number == 1:
            z = self.z_0 + gamma
        elif self.mirror_number == 2:
            if is_virtual:
                z = self.d - (self.z_0 - gamma)
            else:
                z = self.z_0 - gamma

        return z

    def get_r_hyperbola(self, z, k = None):
        if k == None:
            k = self.k

        Gamma = self.get_Gamma(z, k)
        b = self.get_b_hyperbola(self.c, k)
        r = b * Gamma
        # TODO: add radial constraint from a dictionary of constraints, such as
        # if r_min <= r <= r_max
        return r  # Returns only the positive value of r

    def get_rsys_analytically(self, alpha_i, **kwargs):
        '''
        Computes the system radius necessary to achieve certain vertical field of view angle \f$\alpha_i\f$ on mirror \f$i\f$

        @param alpha_i: The target \f$\alpha_i\f$ angle to be satisfied. If None, the current angles are used.
        @param kwargs: It could be useful parameters to enforce, for example, \f$k_1\f$: the curvature-related parameter of mirror 1.
            When "get_all" is set to True, all values are returned. Otherwise, only the values starting from the maximum index of the result are returned
        @retval r_sys: the system radius that achieves the desired vFOV \f$\alpha_i\f$
        @retval k1: the array of k arguments associated to the baseline computed

        '''

        r_sys = None
        get_all = kwargs.get("get_all", True)

        d = kwargs.get("d", self.d)
        c1 = kwargs.get("c1", self.c)
        ki = kwargs.get("ki", self.k)

        if alpha_i == None:
            alpha_i = self.vFOV

        if self.mirror_number == 1:
            r_ref = kwargs.get("r_ref", self.r_min)
            term1 = np.arctan((2 * c1 - d) / (2 * r_ref))
            term2 = np.cos(alpha_i - term1)
            r_sys = np.abs(((c1 * ki * np.tan(alpha_i - term1) * np.abs(np.cos(alpha_i - term1)) + c1 * np.sqrt(ki * (ki - 2))) * (term2 ** 2)) / (ki * (ki * term2 ** 2 - 2) * np.abs(term2)))

        if self.mirror_number == 2:
            c2 = kwargs.get("c2", self.c)
            k2 = kwargs.get("k2", self.k)
            r_cam = kwargs.get("r_cam", self.r_min)
            # TODO: implement (not needed for now)

        if get_all:
            return r_sys, ki
        else:
            r_max_index = np.argmax(r_sys)
            return r_sys[r_max_index:], ki[r_max_index:]

    def get_vFOV_analytically(self, **kwargs):
        '''
        Computes limiting elevation angles analyically (from equations on paper) and returns the correspoinding vertical field of view

        @param kwargs: The model parameters can be passed at will in order to use different values in the computations instead of the currently set for the model

        @return: the vertical field of view computed from the analytical elevation angle limits
        '''
        print_info = kwargs.get("print_info", False)
        if print_info:
            print("----------------------------------")
            print("ANALYTICAL VALUES:")

        theta_max = None
        theta_min = None

        d = kwargs.get("d", self.d)
        r_sys = kwargs.get("r_sys", self.r_max)
        c1 = kwargs.get("c1", self.c)
        k1 = kwargs.get("k1", self.k)

        if self.mirror_number == 1:
            r_ref = kwargs.get("r_ref", self.r_min)
            theta_max = np.arctan((np.sqrt((k1 - 2) * (c1 ** 2 + 2 * k1 * r_sys ** 2) / k1) - c1) / (2 * r_sys))
            theta_min = np.arctan((d - 2 * c1) / (2 * r_ref))

        if self.mirror_number == 2:
            c2 = kwargs.get("c2", self.c)
            k2 = kwargs.get("k2", self.k)
            r_cam = kwargs.get("r_cam", self.r_min)
            # theta_max_due_to_cam_hole = -np.arctan((np.sqrt(c2 ** 2 * (k2 - 2) / (4 * k2) + r_cam ** 2 * (k2 / 2 - 1)) - c2 / 2) / r_cam)
            # or
            theta_max_due_to_cam_hole = np.arctan((c2 - np.sqrt(c2 ** 2 * (1 - 2 / k2) + 2 * r_cam ** 2 * (k2 - 2))) / (2 * r_cam))
            if print_info:
                print("Highest elevation angle (due to camera hole): %f degrees" % np.rad2deg(theta_max_due_to_cam_hole))
            theta_max = np.arctan((c1 / 2 + c2 - d + np.sqrt(c1 ** 2 * (k1 - 2) / (4 * k1) + r_sys ** 2 * (k1 / 2 - 1))) / r_sys)
            theta_min = np.arctan((c2 - np.sqrt(c2 ** 2 * (1 - 2 / k2) + 2 * r_sys ** 2 * (k2 - 2))) / (2 * r_sys))

        alpha_vFOV = theta_max - theta_min
        if print_info:
            self.print_params(vFOV = alpha_vFOV, max_elevation = theta_max, min_elevation = theta_min)
            print("----------------------------------")

        return alpha_vFOV

    def _volume_integrated(self, z):
        a = self.a
        b = self.b
        z0 = self.z_0
        tau = self.thickness
        Gamma = self.get_Gamma(z)
        log_part = np.abs(a * Gamma + z0 - z)
        V = np.pi * tau * (b * (z - z0) * Gamma + a * b * np.log(log_part) - tau * z)
        return V

    def get_volume(self, thickness):
        '''
        @param thickness: The thickness of the mirror in mm
        '''
        self.thickness = thickness
        self.z_min = self.get_z_hyperbola(self.r_min, 0)
        self.z_max = self.get_z_hyperbola(self.r_max, 0)
        V = np.abs(self._volume_integrated(self.z_max) - self._volume_integrated(self.z_min))
        if self.mirror_number == 1:  # Mirror 1 on top has a reflex mirror part
            V = V + thickness * np.pi * self.r_min ** 2

        return V

    def get_mass(self, density, thickness):
        '''
        @param density: The material's density in [g / cm^3] units
        @param thickness: The thickness of the mirror in mm

        @return: mass in grams.
        '''
        # self.density = density * 10 ** 6  # or 1e6 : Transforms to [g / m^3] units
        self.density = density * 10 ** -3  # or 1e-3 : Transforms to [g / mm^3] units
        self.thickness = thickness
        V = self.get_volume(thickness)
        print("Volume = %f mm^3" % (V))
        mass = V * self.density
        return mass

    def lift_pixel_to_projection_plane(self, m_homo):
        '''
        @param m_homo: An ndarray of 2D pixel point in homogeneous coordinates, so shape is (rows, cols, 3)

        @return: The point back-projected to the image projection plane \f$\pi_{img}$\f (usually normalized) as a 3x1 position vector wrt \f$O_c$\f
        '''
        q = np.dot(m_homo, self.precalib_params.K_inv.T)
        # It could also be written as:
        # q = np.einsum("ij,mnj->mni",self.precalib_params.K_inv, m_homo)

        return q

    def lift_pixel_to_mirror_surface(self, m, visualize = False, debug = False):
        '''
        @brief Lifts a pixel point from the image plane to the surface of the mirror with respecto to the system reference frame
        @param m: A ndarray of k image point coordinates [u, v] per cell (e.g. shape may be rows x cols x 2)
        @param visualize: Indicates if visualization will take place
        @param debug: Indicates to print debugging statements
        @retval Ps: The homogeneous coordinates (as an rows x cols x 4 ndarray) of the point(s) on the sphere (w.r.t. camera frame).
        '''
        # TODO: First, validate that image point m is considered valid if it is located within the imaged radial bounds
        # See equation 3.11 and 3.12

        # Enforce the pixel is in homogeneous coordinates
        if m.shape[-1] == 2:
            if np.ndim(m) > 1:
                m_homo = np.dstack((m, np.ones(m.shape[:-1])))
            else:  # Special case input (single point)
                m_homo = np.hstack((m, np.ones(np.ndim(m))))
        else:
            m_homo = m

        q = self.lift_pixel_to_projection_plane(m_homo)
        p_bp = self.back_project_Q_to_mirror(q)
        return p_bp

    def lift_pixel_to_unit_sphere_wrt_focus(self, m, visualize = False, debug = False):
        '''
        @brief Lifts a pixel point from the image plane to the unit sphere centered at the mirror.
        @param m: A ndarray of k (rows x columns) image point coordinates [u, v] per cell (e.g. shape may be rows x cols x 2)
        @param visualize: Indicates if visualization will take place
        @param debug: Indicates to print debugging statements
        @retval Ps: The Euclidean coordinates (as a rows x cols x 3 ndarray) of the point(s) on the sphere (w.r.t. the mirror focus).
        '''
        # TODO: First, validate that image point m is considered valid if it is located within the imaged radial bounds
        # See equation 3.11 and 3.12
        p_bp = self.lift_pixel_to_mirror_surface(m, visualize, debug)

        v = self.get_point_wrt_focus(p_bp)
        v_mag = get_vector_magnitude(v)
        # Normalize onto unit sphere
        Ps = v[..., :3] / v_mag[..., np.newaxis]

        return Ps

    def get_3D_point_from_angles_wrt_focus(self, azimuth, elevation):
        '''
        Finds a world point using the given projection angles towards the focus of the mirror.
        @note: We arbitrarily scale up the position of the point (e.g. by 1000 times) so it is not inside the mirror

        @return: The numpy ndarray of 3D points (in homogeneous coordinates) w.r.t. origin of coordinates (\f$O_C$\f)
        '''
        P_on_sphere = self.map_angles_to_unit_sphere(elevation, azimuth)
        # The points in the sphere is with respect to F, but we need the points w.r.t the system origin Oc
        # Also, we arbitrarily scale up the position of the point so it is not inside the mirror
        # resize_factor = 1000
        # Pw = (P_on_sphere[..., :3] * resize_factor)
        # Pw = (P_on_sphere[..., :3] * resize_factor)
        # Pw_homo = np.dstack((Pw, P_on_sphere[..., -1]))  # Put back the ones for the homogeneous coordinates
        # return Pw_homo
        return P_on_sphere

    def get_pixel_from_3D_point_wrt_C(self, Pw_homo_wrt_C, visualize = False):
        '''
        @brief Project a three-dimensional numpy array (rows x cols x 4) of 3D homogeneous points (eg. [x, y, z, 1]) as row-vectors to the image plane in (\a u,\a v).
        This function is already vectorized for Numpy performance.

        @param Pw_homo_wrt_C: the multidimensional array of homogeneous coordinates of the points (wrt the origin of the projection frame, e.g. camera pinhole)
        @param visualize: To indicate if a 3D visualization will be shown

        @retval u: the resulting ndarray of u coordinates on the image plane
        @retval v: the resulting ndarray of v coordinates on the image plane
        @retval m_homo: The pixel point(s) as numpy array in homogeneous coordinates
        '''

        Q = self.project_3D_point_to_normalized_plane(Pw_homo_wrt_C)
        # TODO: Add distortion parameters

        # The transf_matrix is of size (4, 4) as the homogeneous transformation encoding the 3x3 rotation matrix and 3x1 translation vector (plus the usual padding),
        # Example, using only 3D matrix and the "inner product", such that
        #    result = np.dot(matrix_of_pts,transf_matrix.T)
        #                 where
        #                    the matrix_of_pts is of size (rows, cols, 4), so that each point on the grid is a row vector of size (4,) indicating the homogeneous position vector.
        # Example of Numpy tensordot method:
        #    result = np.tensordot(transf_matrix, matrix_of_pts, axes=([1],[2])).transpose([1,2,0,3])
        #                 where
        #                    the matrix_of_pts is of size (rows, cols, 4, 1), so that each point on the grid is shaped as a (4x1) homogeneous position vector.
        #                    We specify the axes upon which the products should be summed.
        #                    Finally, we need to swap the resulting axes since we want to preserve the original form of the matrix of points. We achieve this with the transpose command
        #                    where we specify the indices of the axes in the desired order.

        m_homo = np.dot(Q, self.precalib_params.K.T)
        u, v = m_homo[..., 0], m_homo[..., 1]
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
        P_wrt_C = self.get_point_wrt_origin(Pw_wrt_M)  # Directly, project points onto the unit sphere
        return self.get_pixel_from_3D_point_wrt_C(P_wrt_C, visualize = visualize)

    def project_3D_point_to_normalized_plane(self, Pw_homo):
        '''
        @brief Project a ndarray of shape (rows x cols x 4) of 3D homogeneous points (eg. [x, y, z, 1]) as row-vectors to the 3D normalized projection plane

        @param Pw_homo: the multidimensional array of homogeneous coordinates of the points (wrt the origin of the projection frame, e.g. camera pinhole)
        @return: The resulting ndarray of 3D point from the projection onto the normalized camera plane. Note that the size of the resulting array is of size (rows,cols,3)
        '''
        P_on_mirror_wrt_Oc = self.get_primer_point_for_projection(Pw_homo)
        return self.project_mirror_point_to_normalized_plane(P_on_mirror_wrt_Oc)

    def project_mirror_point_to_normalized_plane(self, P_mirror):
        '''
        @brief Project a ndarray of shape (rows x cols x 4) of mirror surface points (eg. [x1, y1, z1, 1]) as row-vectors to the 3D normalized projection plane

        @param P_mirror: the multidimensional array of homogeneous coordinates of the mirror points (wrt the origin of the projection frame, e.g. camera pinhole)
        @return: The resulting ndarray of 3D point from the projection onto the normalized camera plane. Note that the size of the resulting array is of size (rows,cols,3)
        '''
        z_p_mirror = P_mirror[..., 2]
        # Project to normalized camera plane
        Q = P_mirror[..., :3] / z_p_mirror[..., np.newaxis]
        return Q

    def get_projection_matrix(self, Pw_homo, is_a_real_reflection = False, k = None):
        '''
        @brief Computes the necessary projection matrix as part of the chain of transformations needed to find the appropriated points on the mirror surface

        @param Pw_homo: the multidimensional array of homogeneous coordinates of the points (wrt the origin of the projection frame, e.g. camera pinhole)
        @param is_a_real_reflection: A boolean to indicate that the matrix for the real reflection point should be returned.
        @param k: The desired k parameter for the mirror curvature. If None, the current (optimal) k will be automatically assumed.

        @return: The ndarray of shape (cols,rows,3,4) for the respective Pw_home that will achieve its projection to the mirror surface.
        '''
        # lamb: A single value or a ndarray of values obeying the frame shape of Pw_homo, such as rows x cols
        lamb = self.get_lambda(Pw_homo, k)
        ident = np.identity(3)
        F = self.F[:-1].reshape(ident.shape[0], 1)  # The translation from the focus
        if self.mirror_number == 1 or is_a_real_reflection:
            # Using multidimensional matrices, so the components are shaped appropriately
            rotation_part = np.outer(lamb, ident).reshape(lamb.shape + ident.shape)
            translation_part = np.outer((1 - lamb), F).reshape(lamb.shape + F.shape)
        elif self.mirror_number == 2:
            diag_normal = np.diag(self.n_reflex[:-1, -1])
            rotation_part = np.outer(lamb, (ident + 2 * diag_normal)).reshape(lamb.shape + ident.shape)
            f2v = self.Fv[:-1].reshape(ident.shape[0], 1)
            normal_x_F = self.n_reflex[:-1] * F
            lamb_norm_x_F = np.outer((1 - lamb), normal_x_F).reshape(lamb.shape + normal_x_F.shape)
            translation_part = f2v + lamb_norm_x_F

        K_proj = np.concatenate((rotation_part, translation_part), axis = -1)  # A shaped ndarray as (cols,rows,3,4)

        return K_proj

    def get_reflection_point_top(self, xw, yw, zw, c1, lamb, d):
        xr = lamb * xw
        yr = lamb * yw
        zr = zw * lamb + c1 * (1 - lamb)
        dist_sq = (xw - xr) ** 2 + (yw - yr) ** 2 + (zw - zr) ** 2
        return xr, yr, zr, dist_sq

    def get_2D_profile_wrt_itself(self, num_of_points, r_size = None):
        if r_size == None:
            r_max = self.r_max
        else:
            r_max = r_size

        r = np.linspace(-r_max, r_max, num_of_points)  # Also x coordinates
        y = np.zeros_like(r)  # The y coordinates (all zeros)
        gamma = self.get_gamma_from_xy(r, y)
        z = self.z_0 - gamma

#         p = np.dstack((r, y, z))
        return r, z

    def get_reflection_point_bottom(self, xw, yw, zw, c2, lamb, d):
        '''
        Old/simple method (not being used)
        '''
        xr = lamb * xw
        yr = lamb * yw
        zr = zw * lamb + (d - c2) * (1 - lamb)
        dist_sq = (xw - xr) ** 2 + (yw - yr) ** 2 + (zw - zr) ** 2
        return xr, yr, zr, dist_sq

    def get_primer_point_for_projection(self, Pw_homo):
        '''
        @brief It resolves for the "pseudo" reflection point to be used in the direct projection to the image plane.
        @attention: This method doesn't find the real reflection point for mirror 2 due to the use of a reflex mirror (so it's virtual).
                    Use get_reflection_point_on_mirror() for that instead.

        @param Pw_homo: An ndarray of shape (rows,cols,4) of the homogeneous coordinates (w.r.t. to the system origin Oc) for 3D points in space
        @returns P: The projected points (non-homogeneous!) onto the mirror surface with respect to common frame C (??? not sure if for mirror 2, too!)
        '''
        K_mirror_proj = self.get_projection_matrix(Pw_homo, is_a_real_reflection = False)

        #=======================================================================
        # from time import process_time  # , perf_counter
        # start_time = process_time()
        # P_matrix_iter = np.ones(shape=(rows, cols, 3), dtype=Pw_homo.dtype)
        # rows = K_mirror_proj.shape[0]
        # cols = K_mirror_proj.shape[1]
        # for row in range(rows):
        #     for col in range(cols):
        #         K = K_mirror_proj[row, col]  # The 3x4 projection matrix
        #         P = np.dot(K, Pw_homo[row, col])  # Equivalent to: P = np.dot(Pw_homo[row, col], K.T)
        #         P_matrix_iter[row, col] = P
        # end_time = process_time()
        # time_ellapsed_1 = end_time - start_time
        # print("Time elapsed: {time:.8f} seconds".format(time=time_ellapsed_1))
        #=======================================================================

        #=======================================================================
        # start_time = process_time()
        #=======================================================================
        # Now, using a vectorized approach to this inner matrix vector multiplication
        P_matrix = np.einsum("ijtf,ijf->ijt", K_mirror_proj, Pw_homo)
        #=======================================================================
        # end_time = process_time()
        # time_ellapsed_2 = end_time - start_time
        # print("Time elapsed: {time:.8f} seconds".format(time=time_ellapsed_2))
        # print("Time DIFF: {time:.8f} seconds".format(time=(time_ellapsed_1 - time_ellapsed_2)))
        # if np.allclose(P_matrix, P_matrix_iter, equal_nan=True):
        #     print("SAME Answer!")
        # else:
        #     print("DIFFERENT Answer!")
        #=======================================================================

        return P_matrix

    def get_reflection_point_on_mirror(self, Pw_homo, k = None):
        '''
        @param Pw_homo: An ndarray of shape (rows,cols,4) of the homogeneous coordinates (w.r.t. to the system origin Oc) for 3D points in space
        @param k: The desired k parameter for the mirror curvature. If None, the current (optimal) k will be automatically assumed.
        @returns P: The projected points onto the mirror surface with respect to common frame C
        '''
        K_mirror_proj = self.get_projection_matrix(Pw_homo, is_a_real_reflection = True, k = k)

        # WISH: Use a vectorized approach to this inner matrix vector multiplication
        P_homo = np.ones_like(Pw_homo)
        for row in range(K_mirror_proj.shape[0]):
            for col in range(K_mirror_proj.shape[1]):
                K = K_mirror_proj[row, col]  # The 3x4 projection matrix
                P = np.dot(K, Pw_homo[row, col])  # Equivalent to: P = np.dot(Pw_homo[row, col], K.T)
                P_homo[row, col, :-1] = P

        return P_homo

    def get_reflection_point_reflex_vector(self, P2_homo):
#         # NOTE: This point to plain intersection works only for this kind of plane described as n^T x p = -d/2
#         # where d is the distance from C to f2v
        P2_alone = P2_homo[0, 0, :]
        Fv = self.Fv[:, 0]
        d = Fv[2]
        z2 = P2_alone[2]
        tr = float(d / (2.0 * (d - z2)))
        f2v_to_P2 = P2_alone - Fv
        Pr = Fv + tr * f2v_to_P2

    # TODO:: Expand implementation since this is the original method meant to be used with a uni-dimensional numpy array
#         f2v_to_P2 = (P2_homo - self.Fv.T)
#         P_on_plane = np.array([0, 0, self.d / 2, 1]).reshape(4, 1)
#         k_for_plane = np.dot(self.n_reflex.T, P_on_plane)
#         n_ref = self.n_reflex.T[np.newaxis, ...]
#         proj_line_to_plane = np.dot(n_ref, f2v_to_P2.T)  # It should be used to check if plane and line are parallel
#         proj_tail_line_to_plane = np.dot(self.n_reflex.T, self.Fv)
#         tr = (k_for_plane - proj_tail_line_to_plane) / proj_line_to_plane
#         Pr = self.Fv + tr * f2v_to_P2

        return Pr  # point at reflex mirror in homogeneous coordinates

    def get_lambda(self, Pw_homo, k = None):
        '''
        Computes the lambda variable on the forward-projection equation for a 3D homogeneous point (or ndarray of points) Pw_homo

        @param P_home: the multidimensional array of homogeneous coordinates of the points (wrt the origin of the projection frame, e.g. camera pinhole)
        @param k: The desired k parameter for the mirror curvature. If None, the current (optimal) k will be automatically assumed.

        @return: the lambda value in the shape of the input array
        '''
        x = Pw_homo[..., 0]
        y = Pw_homo[..., 1]
        z = Pw_homo[..., 2]

        sign_value = 1.0  # TODO: work out the equation so this conditional doesn't need to be specified.

        if self.mirror_number == 1:
            sign_value = -1.0

        if k == None:
            k = self.k

        return self.c / (np.sqrt(k * (k - 2)) * get_magnitude(x - self.F[0], y - self.F[1], z - self.F[2]) + k * (z - self.F[2]) * sign_value)  # Delta_z = z - z_f2 = z - (d-c2)

    # WISH: re=implement to support multidimentional arrays for p_homo
    def transform_due_to_reflex(self, p_homo):
        '''
        It takes a single homogeneous vector (a 1D numpy array for now) and it transforms the point with respect to the reflex mirror,
        so it can uses the virtual focus of mirror 2 as its origin of coordinates (Useful for testing projection via mirror 2)
        '''
        ident = np.identity(3, self.n_reflex.dtype)
        diag_normal = np.diag(self.n_reflex[:-1])
        T = ident + 2 * diag_normal
        T = np.hstack((T, self.Fv[:-1].reshape((-1, 1))))
    #     print("Transform matrix", T)
    #     p_transformed = np.dot(T, p_homo)
        p_transformed = np.dot(T, p_homo)
        return p_transformed

def get_theoretical_OmniStereo(omni_img, radial_bounds_filename, theoretical_params_filename, model_version, is_synthetic, use_existing_radial_bounds = False):
    import cv2
    import os.path as osp

    if type(omni_img) is list:
        n_images = len(omni_img)
        gray_images = n_images * [None]
        for img_idx in range(n_images):
            if omni_img[img_idx].ndim == 3:
                gray_images[img_idx] = cv2.cvtColor(omni_img[img_idx], cv2.COLOR_BGR2GRAY)
            else:
                gray_images[img_idx] = omni_img[img_idx].copy()

        # Compute the average image for initializing (extracting radial boundaries and center)
        # via blending them
        omni_img_avg = gray_images[0]
        blending_counter = 1
        for img_gray in gray_images[1:]:
            blending_counter = blending_counter + 1
            omni_img_avg = cv2.addWeighted(omni_img_avg, 1.0 - 1.0 / blending_counter, img_gray, 1.0 / blending_counter, gamma = 0)
    else:
        omni_img_avg = omni_img

    # NOTE: The convention of digital images sizes (width, height)
    image_size = np.array([omni_img_avg.shape[1], omni_img_avg.shape[0]])
    # sensor_size = np.array([4, 3])  # (width, height) in [mm]
#     image_size = np.array([752, 480])  # BlueFox-MLC = (752x480) | PointGrey Chameleon = (1280x960)

    # Radial Pixel boundaries
    # Refine manually
    radial_initial_values = []
    from omnistereo.common_cv import find_center_and_radial_bounds
    file_exists = osp.isfile(radial_bounds_filename)
    if not file_exists:
        radial_initial_values = [[(image_size / 2.0).astype("int") - 1, (image_size / 2.0).astype("int") - 1, None, None], [(image_size / 2.0) - 1, (image_size / 2.0) - 1, None, None]]

    # Manual refinement of borders
    [[center_pixel_top_outer, center_pixel_top_inner, radius_top_outer, radius_top_inner], [center_pixel_bottom_outer, center_pixel_bottom_inner, radius_bottom_outer, radius_bottom_inner]] = find_center_and_radial_bounds(omni_img_avg, initial_values = radial_initial_values, radial_bounds_filename = radial_bounds_filename, save_to_file = False, is_stereo = True, use_auto_circle_finder = False, use_existing_radial_bounds = use_existing_radial_bounds)

    center_pixel_top_avg = (center_pixel_top_outer + center_pixel_top_inner) / 2.0  # Get the average between the two points
    center_pixel_bottom_avg = (center_pixel_bottom_outer + center_pixel_bottom_inner) / 2.0  # Get the average between the two points

    # Changing boundaries manually
    # gums.top_model.set_occlussion_boundaries(321, 580)
    # gums.bot_model.set_occlussion_boundaries(75, 320)

    # THEORETICAL VALUES:
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    from omnistereo.camera_models import PinholeCamera
    from omnistereo.common_tools import get_theoretical_params_from_file
    c1, c2, k1, k2, d, r_sys, r_reflex, r_cam = get_theoretical_params_from_file(theoretical_params_filename, file_units = "cm")
    # Points as homogeneous column vectors:
    Oc = np.array([0, 0, 0, 1]).reshape(4, 1)  # also F1'
    F1 = np.array([0, 0, c1, 1]).reshape(4, 1)  # F1
    F2 = np.array([0, 0, d - c2, 1]).reshape(4, 1)  # F2
    F2v = np.array([0, 0, d, 1]).reshape(4, 1)  # F2' (virtual camera, also)
    mirror1 = HyperCata(1, F1, Oc, c1, k1, d)
    mirror2 = HyperCata(2, F2, F2v, c2, k2, d)

    from omnistereo.common_tools import get_length_units_conversion_factor
    units_scale_factor_sensor = get_length_units_conversion_factor(input_units = "mm", output_units = "m")
    if is_synthetic:
        focal_length = 1. * units_scale_factor_sensor  # Camera Focal length: 1 [mm] into [m] (for synthetic images)

        if model_version == "new":
            cam_hor_FOV = 38  # Horizontal FOV of "synthetic" perspective camera
        elif model_version == "old":
            cam_hor_FOV = 45  # Horizontal FOV of "synthetic" perspective camera
                                # With our 4:3 aspect ratio, the vertical FOV of the camera is about 34.5 degrees
        # pixel_size = np.array([6, 6]) * (10 ** -3)  # in [mm]: BlueFox-MLC = 6x6 um
        # Only for synthetic images vvvvvvvvvvvv
        img_cols = image_size[0]  # the width
        synthetic_pixel_size_horizontal = 2 * focal_length * np.tan(np.deg2rad(cam_hor_FOV) / 2.0) / img_cols
        # square pixels: we get [ 0.00064721  0.00064721] mm ~ 6x6 um for the 1280x960 POV-Ray image with 45deg FOV camera
        pixel_size = np.array([synthetic_pixel_size_horizontal, synthetic_pixel_size_horizontal])  # in [mm]: Simulated parameters for camera (in POV-Ray)
    else:  # For real cameras
        # For Logitech HD Pro Webcam C910:  Sensor size: 1/2.5" or  5.270  [mm] x 3.960[mm] -> diagonal = 6.592 [mm]
        #=======================================================================
        # aperture_width = 5.270 * units_scale_factor_sensor  # [mm] into [m]
        # aperture_height = 3.960 * units_scale_factor_sensor  # [mm] into [m]
        # sensor_size = np.array([aperture_width, aperture_height])  # (width, height) in [m]
        # pixel_size = sensor_size / image_size
        #=======================================================================
        # For PointGray Black Fly: Sony IMX249, 1/1.2", 5.86 um
        um_to_m_factor = 1e-6
        pixel_size = 2 * [5.86 * um_to_m_factor]
        z_at_r_sys_top = mirror1.get_z_hyperbola(x = r_sys, y = 0)
        f_u = radius_top_outer * (z_at_r_sys_top / r_sys)  # Camera Focal length in pixels (NOT [mm])
        # Infer focal length and pixel size from image for REAL camera!
        focal_length = f_u * pixel_size[0]

    cam_mirror1 = PinholeCamera(mirror1, image_size_pixels = image_size, focal_length = focal_length, pixel_size = pixel_size, custom_center = center_pixel_top_avg)  # Sets mirror1 as parent for this cam_mirror1
    mirror1.precalib_params = cam_mirror1
    mirror1.set_radial_limits(r_reflex, r_sys)
    mirror1.set_radial_limits_in_pixels_mono(inner_img_radius = radius_top_inner, outer_img_radius = radius_top_outer)

    cam_mirror2 = PinholeCamera(mirror2, image_size_pixels = image_size, focal_length = focal_length, pixel_size = pixel_size, custom_center = center_pixel_bottom_avg)  # Sets mirror2 as parent for this cam_mirror2
    mirror2.precalib_params = cam_mirror2
    mirror2.set_radial_limits(r_cam, r_sys)
    mirror2.set_radial_limits_in_pixels_mono(inner_img_radius = radius_bottom_inner, outer_img_radius = radius_bottom_outer)
    theoretical_omni_stereo = HyperCataStereo(mirror1, mirror2)

    img_index = 0
    if type(omni_img) is list:
        omni_img_setup = omni_img[img_index]
    else:
        omni_img_setup = omni_img
    pano_width = np.pi * np.linalg.norm(theoretical_omni_stereo.bot_model.lowest_img_point - theoretical_omni_stereo.bot_model.precalib_params.center_point)
    theoretical_omni_stereo.set_current_omni_image(omni_img_setup, pano_width_in_pixels = pano_width, generate_panoramas = True, idx = img_index , view = True)

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    return theoretical_omni_stereo

def get_hollow_cylinder_mass(density, height, r_inner, thickness, r_outer = None):
    density = density * 10 ** -3  # or 1e-3 : Transforms to [g / mm^3] units
    if r_outer == None:
        r_outer = r_inner + thickness
        # area = np.pi * (2 * r_inner * thickness + thickness ** 2)
    area = np.pi * (r_outer ** 2 - r_inner ** 2)
    V_tube_cyl = area * height
    m_tube_cyl = density * V_tube_cyl
    return m_tube_cyl

def get_magnitude(x, y, z):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)

def get_r(x, y, give_r_squared = False):
    r_square = x ** 2 + y ** 2  # = r^2
    if give_r_squared:
        return r_square
    else:
        return np.sqrt(r_square)

def get_vector_magnitude(v):
    '''
    Computes the vector magnitude of v

    @param v: A ndarray of vectors in the last dimension, such as v.shape = (rows,cols,>3)
    @return: The corresponding ndarray of magnitudes for v
    '''
    return get_magnitude(v[..., 0], v[..., 1], v[..., 2])
