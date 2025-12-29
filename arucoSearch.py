import numpy as np
import cv2
import glob
import os

class ArucoSearch:
    def __init__(self, dictionary=cv2.aruco.DICT_4X4_50):
        self.dictionary = dictionary
        # how many 'marker sizes' to move along the marker's X/Y-axis
        self.calibration_data = {1: (np.float32(0.9569842), np.float32(0.95627487)), 2: (np.float32(-0.9557947), np.float32(-0.9388234))}
        self.calibrated = True

    def search(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dictionary = cv2.aruco.getPredefinedDictionary(self.dictionary)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        return corners, ids, rejected

    def get_marker_basis(self, corner):
        c = corner[0]
        center = np.mean(c, axis=0)
        
        # corners: 0=TopLeft, 1=TopRight, 2=BottomRight, 3=BottomLeft
        top_left = c[0]
        top_right = c[1]
        bottom_left = c[3]
        
        # edge vectors
        vec_x_raw = top_right - top_left
        vec_y_raw = bottom_left - top_left
        scale = np.linalg.norm(vec_x_raw)

        u = vec_x_raw / (scale + 1e-6)
        v = vec_y_raw / (np.linalg.norm(vec_y_raw) + 1e-6)
        
        return center, u, v, scale

    def get_center(self, img, debug=False):
        corners, ids, rejected = self.search(img)
        
        midpoint = None
        x_vec_out = None
        y_vec_out = None

        if ids is None:
            print("No markers detected.")
            return None, None, None

        ids_flat = ids.flatten()
        has_id1 = 1 in ids_flat
        has_id2 = 2 in ids_flat

        # both markers found
        if has_id1 and has_id2:
            idx1 = np.where(ids_flat == 1)[0][0]
            idx2 = np.where(ids_flat == 2)[0][0]
            
            p1, u1, v1, s1 = self.get_marker_basis(corners[idx1])
            p2, u2, v2, s2 = self.get_marker_basis(corners[idx2])

            midpoint = (p1 + p2) / 2.0

            vec_to_center_1 = midpoint - p1

            rel_x1 = np.dot(vec_to_center_1, u1) / s1
            rel_y1 = np.dot(vec_to_center_1, v1) / s1
            self.calibration_data[1] = (rel_x1, rel_y1)

            vec_to_center_2 = midpoint - p2

            rel_x2 = np.dot(vec_to_center_2, u2) / s2
            rel_y2 = np.dot(vec_to_center_2, v2) / s2
            self.calibration_data[2] = (rel_x2, rel_y2)

            self.calibrated = True
            print("Calibration successful.")
            print("Calibration data:", self.calibration_data)

            x_vec_out = u1
            y_vec_out = v1
            
            if debug:
                print(f"Calibrated! ID1 offset: {self.calibration_data[1]}, ID2 offset: {self.calibration_data[2]}")

        # one marker found
        elif self.calibrated and (has_id1 or has_id2):
            found_id = 1 if has_id1 else 2
            idx = np.where(ids_flat == found_id)[0][0]
            
            p, u, v, s = self.get_marker_basis(corners[idx])
            rx, ry = self.calibration_data[found_id]
            
            # calcluate midpoint with calibration data
            midpoint = p + (u * rx * s) + (v * ry * s)
            midpoint = np.array(midpoint)
            
            x_vec_out = u
            y_vec_out = v

        # draw for debugging
        if debug and midpoint is not None:
            print("Detected Midpoint at x:", int(midpoint[0]), "y:", int(midpoint[1]))
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            cv2.circle(img, (int(midpoint[0]), int(midpoint[1])), 8, (0, 0, 255), -1)

            if x_vec_out is not None:
                p_end_x = midpoint + x_vec_out * 50
                p_end_y = midpoint + y_vec_out * 50
                cv2.line(img, (int(midpoint[0]), int(midpoint[1])), (int(p_end_x[0]), int(p_end_x[1])), (0, 255, 0), 2)
                cv2.line(img, (int(midpoint[0]), int(midpoint[1])), (int(p_end_y[0]), int(p_end_y[1])), (255, 0, 0), 2)

            cv2.imshow("result", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return midpoint, x_vec_out, y_vec_out

if __name__ == "__main__":
    folder = "./ARUCO"
    for path in glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.jpeg")):
        img = cv2.imread(path)
        if img is None:
            continue
        print(path)
        aruco = ArucoSearch()
        aruco.get_center(img, debug=True)
     