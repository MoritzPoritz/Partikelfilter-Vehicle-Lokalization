straight_x_line_name = "straight_in_x"
data_suffix = '__data'
image_and_image_data_prefix = 'map_image__'
image_suffix = '__image'
image_data_suffix = '__image_data'
distance_map_suffix = '__distance_map'

# stuff for particle filter
# acc, orientation, distance
sensor_std = [.2, .2, .2]
N = 500
neff_threshold = 50
std = [0, 0]

v_range = [0, 10]
a_range = [0, 10]
# processmodel specifics
L = 2.743
dt = 0.05

# vehicle specifics
vehicle = 'mustang'
max_steering_angle = 70




