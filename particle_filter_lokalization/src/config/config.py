straight_x_line_name = "straight_in_x"
curve_line_name = "curve"
s_curve_name_constant_velocity = "s_curve_constant_velocity"
s_curve_name_variable_velocity = "s_curve_variable_velocity"

data_suffix = '__data'
image_and_image_data_prefix = 'map_image__'
image_suffix = '__image'
image_data_suffix = '__image_data'
distance_map_suffix = '__distance_map'

# data generation specifics
map_border = 100

# stuff for particle filter
# acc, orientation, distance
sensor_std = [.3, .3, .3]
N = 5000
neff_threshold = 20
std = [0, 0]

v_range = [0, 10]
a_range = [0, 10]
# processmodel specifics
L = 2.743
dt = 0.05

# vehicle specifics
vehicle = 'mustang'
max_steering_angle = 70




