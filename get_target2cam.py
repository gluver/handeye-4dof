import cv2
import numpy as np
import glob

def read_images(image_folder):
    images = []
    for filename in sorted(glob.glob(f"{image_folder}/*.png"), key=lambda x: int(x.split('/')[-1].split('_')[0])):
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
    return images

def find_corners(images, pattern_size, square_size):
    obj_points = []
    img_points = []
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    
    for i,img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            img_points.append(corners)
            obj_points.append(objp)
        else:
            print(f"\033[91mWarning: Failed to find corners for image {i}\033[0m")
            cv2.imshow("img",img)
            cv2.waitKey(500)
    return obj_points, img_points

def solve_pnp(obj_points, img_points, camera_matrix, dist_coeffs):
    rvecs, tvecs = [],[]
    for objp, imgp in zip(obj_points, img_points):
        ret, rvec, tvec = cv2.solvePnP(objp, imgp, camera_matrix, dist_coeffs)
        if ret:
            rvecs.append(rvec)
            tvecs.append(tvec)
        else:
            print("\033[91mWarning: Failed to find ret for one of the images.\033[0m")
    return rvecs, tvecs

def write_transform(file_path, rvecs, tvecs):
    with open(file_path, 'w') as f:
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            rot_matrix, _ = cv2.Rodrigues(rvec)
            transform = np.eye(4)
            transform[:3, :3] = rot_matrix
            transform[:3, 3] = tvec.flatten()
            f.write(f"##################      Calibration Pose {i+1}      ##################\n")
            for row in transform:
                f.write('  ' + ',  '.join(f"{val: .6f}" for val in row) + '\n')

def main():
    image_folder = '/home/credog/Desktop/handeye-4dof/run1'
    output_file = '/home/credog/Desktop/handeye-4dof/run1/target2cam.txt'
    pattern_size = (5, 8)  # Example pattern size, adjust as needed
    square_size = 20 # Example square size in mm, adjust as needed
    camera_matrix = np.array([[909.10498046875, 0, 645.127075195312], 
                              [0, 908.8701171875, 357.441741943359], 
                              [0, 0, 1]])
    dist_coeffs = np.array([0, 0, 0, 0, 0])

    images = read_images(image_folder)
    obj_points, img_points = find_corners(images, pattern_size, square_size)
    rvecs, tvecs = solve_pnp(obj_points, img_points, camera_matrix, dist_coeffs)
    write_transform(output_file, rvecs, tvecs)

if __name__ == '__main__':
    main()
