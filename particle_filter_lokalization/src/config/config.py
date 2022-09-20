import numpy as np
# vehicle specifics
vehicle = 'mustang'
max_steering_angle = 70
# data generation specifics
straight_x_constant_velocity_line_name = "straight_in_x_constant_velocity"
straight_x_variable_velocity_line_name = "straight_in_x_variable_velocity"
curve_line_name = "curve"
s_curve_name_constant_velocity = "s_curve_constant_velocity"
s_curve_name_variable_velocity = "s_curve_variable_velocity"


all_road_types = [straight_x_constant_velocity_line_name, straight_x_variable_velocity_line_name, curve_line_name, s_curve_name_constant_velocity, s_curve_name_variable_velocity]


data_suffix = '__data'
image_and_image_data_prefix = 'map_image__'
image_suffix = '__image'
image_data_suffix = '__image_data'
distance_map_suffix = '__distance_map'
point_cloud_appendix = '__pc'
point_cloud_measured_appendix = '__pcm'
imu_data_appendix = '__imu'
lidar_data_appendix = '__lidar'
map_data_appendix = '__map_data'

map_border = 100
# stuff for particle filter
# acc, orientation, distance
N = 1000

imu_sensor_std = [.3, .3, .3]

imu_neff_threshold = 20
imu_std = [0, 0]

lidar_sensor_std = .02
lidar_neff_threshold = 10
lidar_std = [0, 0]

v_range = [0, 10]
a_range = [0, 10]
theta_range = [0, 2*np.pi]
delta_range = [-max_steering_angle+20,max_steering_angle-20]

initial_pos_radius = 30

lidar_range = 10

# processmodel specifics
L = 2.743
dt = 0.05



rain_rates = np.arange(0, 25, 5) ##mm/h

# Variables for statistical survey
sample_size = 10


# for my private pc
'''
paths = {
    'data_path' : "C:\\Users\\Modulo\\Documents\\Uni\\Projekt 2\\Partikelfilter-Vehicle-Lokalization\\particle_filter_lokalization\\data\\",
    'filter_results_path': "C:\\Users\\Modulo\\Documents\\Uni\\Projekt 2\\Partikelfilter-Vehicle-Lokalization\\particle_filter_lokalization\\data\\filter_results\\",
    'map_path': "C:\\Users\\Modulo\\Documents\\Uni\\Projekt 2\\Partikelfilter-Vehicle-Lokalization\\particle_filter_lokalization\\data\\map\\",
    'evaluation_results_path': "C:\\Users\\Modulo\\Documents\\Uni\\Projekt 2\\Partikelfilter-Vehicle-Lokalization\\particle_filter_lokalization\\data\\evaluation_results\\"
}
'''
# for my work pc
paths = {
    'data_path' : "C:\\Users\\mso\\Documents\\Uni\\Partikelfilter-Vehicle-Lokalization\\particle_filter_lokalization\\data\\",
    'filter_results_path': "C:\\Users\\mso\\Documents\\Uni\\Partikelfilter-Vehicle-Lokalization\\particle_filter_lokalization\\data\\filter_results\\",
    'map_path': "C:\\Users\\mso\\Documents\\Uni\\Partikelfilter-Vehicle-Lokalization\\particle_filter_lokalization\\data\\map\\",
    'evaluation_results_path': "C:\\Users\\mso\\Documents\\Uni\\Partikelfilter-Vehicle-Lokalization\\particle_filter_lokalization\\data\\evaluation_results\\"
}




