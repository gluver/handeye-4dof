import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
from handeye_4dof import Calibrator4DOF, robot_pose_selector


np.set_printoptions(suppress=True)

def read_internet_data(file_path):

    # Actual code to read from the provided file and convert each matrix to a numpy array
    matrices = []  # List to store each 4x4 matrix

    # Path to the uploaded file

    with open(file_path, 'r') as file:
        # Read file line by line
        lines = file.readlines()

        # Temporary list to hold the current 4x4 matrix
        current_matrix = []
        
        for line in lines:
            if line.strip().startswith('#') or not line.strip():
                # If we encounter a comment or empty line, and current_matrix has data, save the matrix
                if current_matrix:
                    matrices.append(np.array(current_matrix))
                    current_matrix = []  # Reset for the next matrix
            else:
                # Convert the current line to a list of floats and append to the current matrix
                row = list(map(float, line.split(',')))
                current_matrix.append(row)
        
        # Add the last matrix if the file doesn't end with a comment
        if current_matrix:
            matrices.append(np.array(current_matrix))

    return matrices


def main():
    
    base_to_hand = read_internet_data('run1/base2gripper.txt')
    camera_to_marker = read_internet_data('run1/target2cam.txt')
    # Obtain optimal motions as dual quaternions.
    camera_to_marker = [np.linalg.pinv(t2c) for t2c in camera_to_marker]
    motions = robot_pose_selector(base_to_hand, camera_to_marker)

    # Initialize calibrator with precomputed motions.
    cb = Calibrator4DOF(motions)

    # Our camera and end effector z-axes are antiparallel so we apply a 180deg x-axis rotation.
    dq_x = cb.calibrate(antiparallel_screw_axes=True)

    # Hand to Camera TF obtained from handeye calibration.
    #ca_hand_to_camera = np.linalg.inv(dq_x.as_transform())
    ca_hand_to_camera = dq_x.as_transform()

    ca_rotation = np.rad2deg(R.from_matrix(ca_hand_to_camera[:3, :3]).as_euler('xyz'))
    
    np.set_printoptions(precision=5)
    print("Hand to Camera Transform Comparisons")
    print("Translations: Calibration  {}".format(ca_hand_to_camera[:3, -1]))
    print("Rotations:    Calibration  {}".format(ca_rotation))
    print(ca_hand_to_camera)

if __name__ == '__main__':
    main()
