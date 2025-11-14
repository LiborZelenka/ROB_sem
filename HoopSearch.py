#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2025-09-21
#     Author: Martin Cífka <martin.cifka@cvut.cz>
#
from typing import List
from numpy.typing import ArrayLike
import numpy as np
import cv2  # noqa


def find_hoop_homography(images: ArrayLike, hoop_positions: List[dict]) -> np.ndarray:
    """
    Find homography based on images containing the hoop and the hoop positions loaded from
    the hoop_positions.json file in the following format:

    [{
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093259019899434, -0.17564068853313258, 0.04918733225140541]
    },
    {
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093569397977782, -0.08814069881074972, 0.04918733225140541]
    },
    ...
    ]
    """

    images = np.asarray(images)
    assert images.shape[0] == len(hoop_positions)

    # todo HW03: Detect circle in each image

    circle_coord = []
    plane_coord = []

    for i in range(len(images)):
        img = images[i]
        hoop = hoop_positions[i]

        #color to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #find circles using hough transform
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=250,
            param1=100, param2=200, minRadius=50, maxRadius=1000
        )

        #take circle
        u, v, _ = circles[0][0] #_ is the radius, i dont need it
        circle_coord.append([u, v])
        #get plain points
        x, y, z = hoop["translation_vector"] # i dont need z to find homography
        plane_coord.append([x, y])

    # todo HW03: Find homography using cv2.findHomography. Use the hoop positions and circle centers.

    #convert lists to np.array
    circle_coord = np.array(circle_coord)
    plane_coord = np.array(plane_coord)

    # find homography H
    H, status = cv2.findHomography(circle_coord, plane_coord, method=0)

    return H
