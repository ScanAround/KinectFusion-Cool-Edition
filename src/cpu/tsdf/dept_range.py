import cv2
import numpy as np
import os

# This function loops over all depth images in a directory and compute the average minimum and 
# maximum depths and the average depth.
def get_depth_range(directory):
    min_depths = []
    max_depths = []
    avg_depths = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            # Load the depth image
            depth_image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_UNCHANGED)

            # Convert to floating point depth in meters
            depth_image = depth_image.astype(np.float32) / 5000.0

            # Mask out zero values (missing data)
            mask = depth_image > 0
            depth_image = depth_image[mask]

            if depth_image.size > 0: # Check that there is data
                # Compute minimum and maximum depth
                min_depth = np.min(depth_image)
                max_depth = np.max(depth_image)
                avg_depth = np.mean(depth_image)
                min_depths.append(min_depth)
                max_depths.append(max_depth)
                avg_depths.append(avg_depth)

    # Compute average minimum, maximum and average depth
    avg_min_depth = np.mean(min_depths)
    avg_max_depth = np.mean(max_depths)
    avg_avg_depth = np.mean(avg_depths)

    return avg_min_depth, avg_max_depth, avg_avg_depth

directory = "/home/anil/Desktop/kinect_fusion_project/rgbd_dataset_freiburg1_xyz/depth" # replace with your directory
avg_min_depth, avg_max_depth, avg_avg_depth = get_depth_range(directory)

print(f'Average minimum depth: {avg_min_depth} meters')
print(f'Average maximum depth: {avg_max_depth} meters')
print(f'Average depth: {avg_avg_depth} meters')