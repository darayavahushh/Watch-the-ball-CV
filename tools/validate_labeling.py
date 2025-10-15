import json
import os
from typing import List, Dict
import numpy as np
import cv2


class CoordinateFixer:
    def __init__(self, video_path: str, json_path: str, expected_points: int = 5):
        """
        Initialize the coordinate fixer.

        Args:
            video_path: Path to the input video file
            json_path: Path to the input JSON file
            expected_points: Expected number of coordinate points per frame (default: 5)
        """
        self.video_path = video_path
        self.json_path = json_path
        self.expected_points = expected_points
        self.total_frames = self.count_video_frames()
        self.data = {}

    def load_json(self) -> Dict:
        """Load JSON data from file."""
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        return self.data

    def count_video_frames(self) -> int:
        """
        Count and return the number of frames in a video file.

        Returns:
            int: Number of frames in the video, or -1 if error occurs
        """
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {self.video_path}")
                return -1

            # OpenCV property
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Release the video capture
            cap.release()

            return total_frames

        except Exception as e:
            print(f"Error processing video: {e}")
            return -1

    def fix_negative_coordinates(self) -> int:
        """
        Fix negative coordinate values by setting them to 0.

        Returns:
            Number of coordinates fixed
        """
        fixed_count = 0
        for frame_key, coordinates in self.data.items():
            for i, point in enumerate(coordinates):
                for j, coord in enumerate(point):
                    if coord < 0:
                        self.data[frame_key][i][j] = 0.0
                        fixed_count += 1
                        print(f"Fixed negative value in {frame_key}, point {i}, coordinate {j}: {coord} -> 0.0")

        return fixed_count

    def interpolate_coordinates(self, frame1_coords: List[List[float]],
                                frame2_coords: List[List[float]],
                                num_steps: int) -> List[List[List[float]]]:
        """
        Interpolate coordinates between two frames using linear interpolation.

        Args:
            frame1_coords: Coordinates from the first frame
            frame2_coords: Coordinates from the second frame
            num_steps: Number of intermediate frames to create

        Returns:
            List of interpolated coordinate sets (excluding start and end frames)
        """
        if len(frame1_coords) != len(frame2_coords):
            raise ValueError(f"Frame coordinate counts don't match: {len(frame1_coords)} vs {len(frame2_coords)}")

        frame1_arr = np.array(frame1_coords)
        frame2_arr = np.array(frame2_coords)

        interpolated = []
        for step in range(1, num_steps + 1):
            alpha = step / (num_steps + 1)
            interp_coords = frame1_arr * (1 - alpha) + frame2_arr * alpha
            interpolated.append(interp_coords.tolist())

        return interpolated

    def get_frame_number(self, frame_key: str) -> int:
        """Extract frame number from frame key (e.g., 'frame_0' -> 0)."""
        return int(frame_key.split('_')[1])

    def validate_frame_coordinates(self) -> List[str]:
        """
        Validate that each frame has the expected number of coordinate points.

        Returns:
            List of warning messages for frames with incorrect point counts
        """
        warnings = []
        for frame_key, coordinates in self.data.items():
            if len(coordinates) != self.expected_points:
                warning = f"WARNING: {frame_key} has {len(coordinates)} points, expected {self.expected_points}"
                warnings.append(warning)
                print(warning)

        return warnings

    def fill_missing_frames(self) -> int:
        """
        Fill in missing frames by interpolating between known frames.

        Returns:
            Number of frames added
        """
        if not self.data:
            return 0

        # Get sorted list of frame numbers
        frame_numbers = sorted([self.get_frame_number(key) for key in self.data.keys()])

        if not frame_numbers:
            return 0

        frames_added = 0
        new_data = {}

        # Process each gap between consecutive known frames
        for i in range(len(frame_numbers) - 1):
            current_frame = frame_numbers[i]
            next_frame = frame_numbers[i + 1]

            current_key = f"frame_{current_frame}"
            next_key = f"frame_{next_frame}"

            # Add current frame
            new_data[current_key] = self.data[current_key]

            # Check if there's a gap
            gap = next_frame - current_frame - 1
            if gap > 0:
                print(f"Filling gap between {current_key} and {next_key} ({gap} frames)")

                # Interpolate missing frames
                interpolated = self.interpolate_coordinates(
                    self.data[current_key],
                    self.data[next_key],
                    gap
                )

                # Add interpolated frames
                for j, coords in enumerate(interpolated):
                    frame_num = current_frame + j + 1
                    new_data[f"frame_{frame_num}"] = coords
                    frames_added += 1

        # Add the last frame
        last_frame = frame_numbers[-1]
        new_data[f"frame_{last_frame}"] = self.data[f"frame_{last_frame}"]

        self.data = new_data
        return frames_added

    def extend_to_total_frames(self) -> int:
        """
        Extend coordinates to cover all frames up to total_frames.
        Copies the last known frame's coordinates for any missing frames.

        Returns:
            Number of frames extended
        """
        if self.total_frames == -1:
            print("Total frames not specified, skipping extension")
            return 0

        if not self.data:
            return 0

        # Get the highest frame number in current data
        frame_numbers = [self.get_frame_number(key) for key in self.data.keys()]
        max_frame = max(frame_numbers)

        if max_frame >= self.total_frames - 1:
            print(f"All frames up to {self.total_frames - 1} already covered")
            return 0

        # Get coordinates from the last known frame
        last_coords = self.data[f"frame_{max_frame}"]

        frames_extended = 0
        for frame_num in range(max_frame + 1, self.total_frames):
            self.data[f"frame_{frame_num}"] = last_coords.copy()
            frames_extended += 1

        print(f"Extended {frames_extended} frames from frame_{max_frame} to frame_{self.total_frames - 1}")
        return frames_extended

    def save_json(self, output_path: str = None):
        """
        Save the fixed JSON data to file.

        Args:
            output_path: Path for output file (default: input_path with '_fixed' suffix)
        """
        if output_path is None:
            base, ext = os.path.splitext(self.json_path)
            output_path = f"{base}_fixed{ext}"

        # Sort frames by number before saving
        sorted_data = {}
        frame_numbers = sorted([self.get_frame_number(key) for key in self.data.keys()])
        for frame_num in frame_numbers:
            sorted_data[f"frame_{frame_num}"] = self.data[f"frame_{frame_num}"]

        with open(output_path, 'w') as f:
            json.dump(sorted_data, f, indent=4)

        print(f"\nFixed JSON saved to: {output_path}")

    def process(self, output_path: str = None):
        """
        Run the complete fixing process.

        Args:
            output_path: Path for output file (optional)
        """
        print("=== Starting Coordinate Fixing Process ===\n")

        # Check video lenght
        if self.total_frames == -1:
            print("Error: Could not determine total frames from video.")
            return

        # Load data
        print("Loading JSON file...")
        self.load_json()
        print(f"Loaded {len(self.data)} frames\n")

        # Fix negative coordinates
        print("Fixing negative coordinates...")
        fixed_count = self.fix_negative_coordinates()
        print(f"Fixed {fixed_count} negative coordinates\n")

        # Validate coordinates
        print("Validating coordinate counts...")
        warnings = self.validate_frame_coordinates()
        if not warnings:
            print("All frames have correct number of points\n")
        else:
            print()

        # Fill missing frames
        print("Filling missing frames...")
        frames_added = self.fill_missing_frames()
        print(f"Added {frames_added} interpolated frames\n")

        # Extend to total frames
        if self.total_frames:
            print(f"Extending to total {self.total_frames} frames...")
            frames_extended = self.extend_to_total_frames()
            print()

        # Save result
        self.save_json(output_path)
        print("\n=== Process Complete ===")
        print(f"Final frame count: {len(self.data)}")


# Example usage
if __name__ == "__main__":
    # Configure these parameters
    INPUT_VIDEO_PATH = "resources/rgb.avi"  # Path to your input video file
    INPUT_JSON_PATH = "resources/outputs/annotated_ground.json"  # Path to your input JSON file
    OUTPUT_JSON_PATH = "resources/outputs/coordinates_fixed.json"  # Path for output
    EXPECTED_POINTS = 5  # Number of coordinate points expected per frame

    # Create fixer instance and process
    fixer = CoordinateFixer(
        video_path=INPUT_VIDEO_PATH,
        json_path=INPUT_JSON_PATH,
        expected_points=EXPECTED_POINTS,
    )

    fixer.process(output_path=OUTPUT_JSON_PATH)