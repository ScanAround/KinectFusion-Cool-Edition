import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define image dimensions
image_width = 640
image_height = 480

# Define sphere parameters
sphere_radius = 100
sphere_center_x = image_width // 2
sphere_center_y = image_height // 2

# Create an empty depth map
depth_map = np.zeros((image_height, image_width))

# Generate the depth map of the sphere
for y in range(image_height):
    for x in range(image_width):
        # Calculate the distance from the current pixel to the sphere center
        distance_to_center = np.sqrt((x - sphere_center_x) ** 2 + (y - sphere_center_y) ** 2)

        # If the distance is within the sphere radius, set the depth to a positive value
        if distance_to_center < sphere_radius:
            depth_map[y, x] = sphere_radius - distance_to_center

# Convert the depth map to a PIL image
depth_map_image = Image.fromarray((depth_map * 255 / np.max(depth_map)).astype(np.uint8))

# Save the depth map as an image
depth_map_image.save('sphere_depth_map.png')
