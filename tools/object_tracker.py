import numpy as np

class ObjectTracker:
    """
    Multi-object tracker that assigns IDs to detections based on proximity.
    """

    def __init__(self, max_lost=10, distance_thresh=80):
        self.next_id = 0
        self.tracks = {}  # id -> {center, lost, trajectory}
        self.max_lost = max_lost
        self.distance_thresh = distance_thresh

    def _distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def update(self, detections):
        """
        detections: list of tuples (u, v, bbox)
        Returns list of tuples (id, u, v, bbox)
        """
        updated_tracks = {}

        for (u, v, bbox) in detections:
            matched_id = None
            min_dist = float("inf")

            # Find the closest existing track
            for track_id, data in self.tracks.items():
                dist = self._distance((u, v), data["center"])
                if dist < self.distance_thresh and dist < min_dist:
                    matched_id = track_id
                    min_dist = dist

            if matched_id is not None:
                # Update existing track
                updated_tracks[matched_id] = {
                    "center": (u, v),
                    "lost": 0,
                    "trajectory": self.tracks[matched_id]["trajectory"] + [(u, v)]
                }
            else:
                # Start new track
                updated_tracks[self.next_id] = {
                    "center": (u, v),
                    "lost": 0,
                    "trajectory": [(u, v)]
                }
                self.next_id += 1

        # Increment lost count for unmatched tracks
        for track_id, data in self.tracks.items():
            if track_id not in updated_tracks:
                data["lost"] += 1
                if data["lost"] < self.max_lost:
                    updated_tracks[track_id] = data

        self.tracks = updated_tracks
        # Return current track list
        output = []
        for tid, data in self.tracks.items():
            u, v = data["center"]
            bbox = detections[0][2] if detections else [0, 0, 0, 0]
            output.append((tid, u, v, bbox))
        return output
