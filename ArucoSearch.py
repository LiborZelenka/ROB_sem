import numpy as np
import cv2

import glob
import os

class ArucoSearch:
    def __init__(self, dictionary=cv2.aruco.DICT_4X4_50) :
        self.dictionary = dictionary

    def search(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dictionary = cv2.aruco.getPredefinedDictionary(self.dictionary)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        return corners, ids, rejected

    def Get_center(self, img, debug=False):
        corners, ids, rejected = self.search(img)
        if debug:
            print("Nalezené ID:", ids)
            print("Rohy (souřadnice pixelů):", corners)
            # volitelně vykreslit
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(img, corners, ids)

        centers = []
        for c in corners:
            pts = c[0]  # tvar (4,2)
            center = np.mean(pts, axis=0)
            centers.append(center)

        # střed mezi dvěma markery
        midpoint = np.mean(centers, axis=0)

        i1 = np.where(ids == 1)[0][0]
        i2 = np.where(ids == 2)[0][0]
        p1 = corners[i1][0].mean(axis=0)  # střed prvního
        p2 = corners[i2][0].mean(axis=0)  # střed druhého
        v = p2 - p1
        angle_rad = np.arctan2(v[1], v[0])

        if debug:
            print("Středy markerů:", centers)
            print("Střed mezi nimi:", midpoint)
            print("Úhel mezi markery:", angle_rad)
            for center in centers:
                cv2.circle(img, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
            cv2.circle(img, (int(midpoint[0]), int(midpoint[1])), 7, (0, 255, 0), -1)
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 2)
            cv2.putText(img, f"{angle_rad:.1f} rad", (int(midpoint[0]), int(midpoint[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("result", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return midpoint, angle_rad


folder = "./ARUCO"
for path in glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.jpeg")):
    img = cv2.imread(path)
    if img is None:
        continue
    print(path)
    aruco = ArucoSearch()
    aruco.Get_center(img, debug=True)
