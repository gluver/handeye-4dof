import numpy as np

#supress numpy print notation
np.set_printoptions(suppress=True)
def read_csv(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    return data

def calculate_transform(data):
    transforms = []
    for row in data:
        x, y, z, u = row
        u_rad = np.deg2rad(u)
        
        # Rotation matrix around the z-axis
        Rz = np.array([
            [np.cos(u_rad), -np.sin(u_rad), 0],
            [np.sin(u_rad), np.cos(u_rad), 0],
            [0, 0, 1]
        ])
        
        # Translation vector
        t = np.array([x, y, z])
        
        # Combine into a single transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = Rz
        transform[:3, 3] = t
        print(transform)
        transforms.append(transform)
    
    return transforms

def write_transform(file_path, transforms):
    with open(file_path, 'w') as f:
        for i, transform in enumerate(transforms):
            f.write(f"##################      Calibration Pose {i+1}      ##################\n")
            for row in transform:
                f.write('  ' + ',  '.join(f"{val: .6f}" for val in row) + '\n')

def main():
    input_file = '/home/credog/Desktop/handeye-4dof/run1/robot1.csv'
    output_file = '/home/credog/Desktop/handeye-4dof/run1/base2gripper.txt'
    
    data = read_csv(input_file)
    transform = calculate_transform(data)
    write_transform(output_file, transform)

if __name__ == '__main__':
    main()
