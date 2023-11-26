import cv2
import numpy as np
import yaml

def generate_chessboard_points(board_size, square_size):
    points_3d = []
    for y in range(board_size[1]):
        for x in range(board_size[0]):
            points_3d.append((x * square_size, y * square_size, 0))
    return np.array(points_3d, dtype=np.float32)

img1 = cv2.imread("data/chessboard01.jpg")
img2 = cv2.imread("data/chessboard02.jpg")

board_size = (9, 6)
square_size = 30
found1, corners1 = cv2.findChessboardCorners(img1, board_size)
found2, corners2 = cv2.findChessboardCorners(img2, board_size)

if found1 and found2:
    board_points_3d = generate_chessboard_points(board_size, square_size)

    fs = cv2.FileStorage("data/camera_intrinsics.yml", cv2.FILE_STORAGE_READ)
    intrinsics = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coefs").mat()
    # _, rvec, tvec = cv2.solvePnP(board_points_3d, corners1, intrinsics, dist_coeffs)
    _, rvec, tvec = cv2.solvePnP(board_points_3d, corners2, intrinsics, dist_coeffs)

    axis_points = np.float32([[60, 0, 0], [0, 60, 0], [0, 0, -60]]).reshape(-1, 3)
    image_points, _ = cv2.projectPoints(axis_points, rvec, tvec, intrinsics, dist_coeffs)
    # img_with_axes = img1.copy()
    img_with_axes = img2.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, (color, point) in enumerate(zip(colors, image_points)):
        # img_with_axes = cv2.line(img_with_axes, tuple(corners1[0].ravel()), tuple(point.ravel()), color, 3)
        img_with_axes = cv2.line(img_with_axes, tuple(corners2[0].ravel()), tuple(point.ravel()), color, 3)

    cube_points_3d = np.array([
        (0, 0, 0), (square_size * 3, 0, 0), (square_size * 3, square_size * 3, 0),
        (0, square_size * 3, 0), (0, 0, -square_size * 3), (square_size * 3, 0, -square_size * 3),
        (square_size * 3, square_size * 3, -square_size * 3), (0, square_size * 3, -square_size * 3)
    ], dtype=np.float32)
    cube_image_points, _ = cv2.projectPoints(cube_points_3d, rvec, tvec, intrinsics, dist_coeffs)
    cube_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                  (4, 5), (5, 6), (6, 7), (7, 4),
                  (0, 4), (1, 5), (2, 6), (3, 7)]
    edge_colors = [(0, 255, 0), (0, 255, 0), (0, 255, 0),
                   (0, 255, 0), (0, 0, 255), (0, 0, 255),
                   (0, 0, 255), (0, 0, 255), (255, 0, 0),
                   (255, 0, 0), (255, 0, 0), (255, 0, 0)]
    # img_with_cube = img1.copy()
    img_with_cube = img2.copy()
    for (start, end), color in zip(cube_edges, edge_colors):
        start_point = tuple(cube_image_points[start].ravel())
        end_point = tuple(cube_image_points[end].ravel())
        img_with_cube = cv2.line(img_with_cube, start_point, end_point, color, 3)

    cv2.imshow("Chessboard with Axes", img_with_axes)
    cv2.imshow("Chessboard with Cube", img_with_cube)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Chessboard not found in both images.")