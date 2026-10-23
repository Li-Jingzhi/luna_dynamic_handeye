"""Yellow-ball candidate selection for the SH5 preview."""
import cv2
import numpy as np


def select_yellow(stats, centroids, previous=None, max_step_px=30.0):
    candidates = []
    for j in range(1, len(stats)):
        _, _, width, height, area = stats[j]
        fill = area / max(width * height, 1)
        if 200 <= area <= 8000 and .75 <= width / height <= 1.33 and fill >= .55:
            candidates.append((area, np.asarray(centroids[j], float)))
    if not candidates:
        return None
    if previous is None:
        return max(candidates, key=lambda item: item[0])[1]
    point = min(candidates, key=lambda item: np.linalg.norm(item[1] - previous))[1]
    return point if np.linalg.norm(point - previous) <= max_step_px else None


def _yellow_center(image, previous=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (5, 100, 100), (40, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    point = select_yellow(stats, centroids, previous)
    if point is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if len(contour) >= 5]
    if not contours:
        return point
    fits = [cv2.fitEllipse(contour) for contour in contours]
    return np.asarray(min(fits, key=lambda fit: np.linalg.norm(np.asarray(fit[0])-point))[0])


def track_yellow(frame, previous=None):
    """Acquire at full resolution, then fit the ball boundary in a small ROI."""
    if previous is None:
        return _yellow_center(frame)
    x, y = np.rint(previous).astype(int)
    radius = 60
    x0, x1 = max(0, x-radius), min(frame.shape[1], x+radius)
    y0, y1 = max(0, y-radius), min(frame.shape[0], y+radius)
    point = _yellow_center(frame[y0:y1, x0:x1], previous-np.array([x0, y0]))
    return None if point is None else point + np.array([x0, y0])
