import numpy as np
from collections import deque
from tools.pixel_coords import CameraCoords


class Ball3DReconstruct:
    """
    Reconstructs 3D position of a ball using 2D detections, camera intrinsics, and known ball size.

    Pinhole model relationships:
        Z = f * R_real / r_pixels
        X = X_norm * Z
        Y = Y_norm * Z
    """

    def __init__(self, coords: CameraCoords, ball_radius_m: float):
        self.coords = coords
        self.ball_radius = ball_radius_m
        self.z_buffer = deque(maxlen=5)  # smoothing memory

    def bbox_to_radius(self, bbox):
        """
        Convert bounding box to an approximate ball radius in pixels.
        Args:
            bbox (list or tuple): [x1, y1, x2, y2]
        Returns:
            float: radius in pixels
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        r_pixels = (width + height) / 4.0
        return r_pixels

    def compute_3d_position(self, bbox):
        """
        Compute (X, Y, Z) position of the ball relative to the camera.
        Args:
            bbox (list): [x1, y1, x2, y2]
        Returns:
            tuple: (X, Y, Z) in meters
        """
        x1, y1, x2, y2 = bbox
        u_center = (x1 + x2) / 2.0
        v_center = (y1 + y2) / 2.0

        # Normalized coordinates
        X_norm, Y_norm = self.coords.get_pixel_coords(u_center, v_center)

        # Ball radius in pixels
        r_pixels = self.bbox_to_radius(bbox)
        if r_pixels <= 0:
            return None, None, None

        # Z from pinhole geometry
        Z = (self.coords.fx * self.ball_radius) / r_pixels

        # Smooth Z over time
        self.z_buffer.append(Z)
        Z = np.median(self.z_buffer)  # exponential moving average

        # Compute X, Y in meters
        X = X_norm * Z
        Y = Y_norm * Z

        return X, Y, Z

    @staticmethod
    def compute_distance(X, Y, Z):
        """
        Compute Euclidean distance between camera and 3D point.
        """
        if None in (X, Y, Z):
            return None
        return float(np.sqrt(X ** 2 + Y ** 2 + Z ** 2))
