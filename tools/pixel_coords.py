import numpy as np


class CameraCoords:
    """
    Precompute normalized camera coordinate grids and provide sub-pixel lookups.
    Normalized coordinates are (u - cx) / fx and (v - cy) / fy.

    Usage:
        coords = Coords(width, height, fx, fy, cx, cy)
        x_norm, y_norm = coords.get_pixel_coords(u, v, interp=True)
    """

    def __init__(self, width: int, height: int, fx: float, fy: float, cx: float = None, cy: float = None):
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx) if cx is not None else (self.width - 1) / 2.0
        self.cy = float(cy) if cy is not None else (self.height - 1) / 2.0

        # Precompute normalized coordinate grids (float32)
        xs = (np.arange(self.width, dtype=np.float32) - self.cx) / self.fx
        ys = (np.arange(self.height, dtype=np.float32) - self.cy) / self.fy
        self.x_grid, self.y_grid = np.meshgrid(xs, ys)  # shape (height, width)

    def _compute_grids(self):
        """
        Computes normalized camera coordinates for each pixel.
        Returns:
            x_grid, y_grid (np.ndarray): grids of shape (height, width)
        """
        xs = (np.arange(self.width) - self.cx) / self.fx
        ys = (np.arange(self.height) - self.cy) / self.fy
        x_grid, y_grid = np.meshgrid(xs, ys)
        return x_grid, y_grid

    def get_pixel_coords(self, u: int, v: int):
        """
        Converts pixel coordinates (u,v) to normalized camera coordinates (X_norm, Y_norm).
        """
        X_norm = (u - self.cx) / self.fx
        Y_norm = (v - self.cy) / self.fy
        return X_norm, Y_norm

    def pixel_to_camera_frame(self, u: float, v: float, Z: float):
        """
        Converts pixel coordinates (u,v) with known depth Z (m)
        into 3D camera coordinates (X,Y,Z).
        """
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        return X, Y, Z

    def estimate_depth_from_bbox(self, bbox_height_px: float, ball_diameter_m: float):
        """
        Estimate the ball's depth (Z) using the pinhole model:
        Z = (fy * real_height) / pixel_height
        """
        if bbox_height_px <= 0:
            return None
        return (self.fy * ball_diameter_m) / bbox_height_px

    def get_pixel_coords(self, u: float, v: float, interp: bool = True):
        """
        Return normalized camera coordinates (X_norm, Y_norm) for pixel (u, v).
        Args:
            u (float): x pixel coordinate (column)
            v (float): y pixel coordinate (row)
            interp (bool): if True, use bilinear interpolation for sub-pixel values
        Returns:
            X_norm (float), Y_norm (float)
        """
        # Clamp coordinates to valid range
        if u < 0: u = 0.0
        if v < 0: v = 0.0
        if u > self.width - 1: u = float(self.width - 1)
        if v > self.height - 1: v = float(self.height - 1)

        if not interp:
            iu = int(round(u))
            iv = int(round(v))
            return float(self.x_grid[iv, iu]), float(self.y_grid[iv, iu])

        # Bilinear interpolation
        x0 = int(np.floor(u))
        x1 = x0 + 1
        y0 = int(np.floor(v))
        y1 = y0 + 1

        if x1 >= self.width:
            x1 = self.width - 1
        if y1 >= self.height:
            y1 = self.height - 1

        # If integer coords (no interpolation required)
        if x0 == x1 and y0 == y1:
            return float(self.x_grid[y0, x0]), float(self.y_grid[y0, x0])

        # weights
        wx = u - x0
        wy = v - y0

        # four corners
        x00 = self.x_grid[y0, x0]
        x10 = self.x_grid[y0, x1]
        x01 = self.x_grid[y1, x0]
        x11 = self.x_grid[y1, x1]

        y00 = self.y_grid[y0, x0]
        y10 = self.y_grid[y0, x1]
        y01 = self.y_grid[y1, x0]
        y11 = self.y_grid[y1, x1]

        x_top = (1 - wx) * x00 + wx * x10
        x_bot = (1 - wx) * x01 + wx * x11
        x_interp = (1 - wy) * x_top + wy * x_bot

        y_top = (1 - wx) * y00 + wx * y10
        y_bot = (1 - wx) * y01 + wx * y11
        y_interp = (1 - wy) * y_top + wy * y_bot

        return float(x_interp), float(y_interp)
