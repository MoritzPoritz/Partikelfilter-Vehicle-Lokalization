straight_x_line_name = "straight_in_x"
curve_line_name = "curve"
s_curve_name_constant_velocity = "s_curve_constant_velocity"
s_curve_name_variable_velocity = "s_curve_variable_velocity"

overall_data_file = "overall"

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

# carla map related
carla_map_name ="carla_map_unscaled"
carla_distance_map_name ="carla_distance_map_unscaled"
world_to_img_scaling_x = 4.06
world_to_img_scaling_y = -4.06
world_to_img_translate_x = 963
world_to_img_translate_y = 373 
# data generation specifics
map_border = 100
N = 1000
# stuff for particle filter
# acc, orientation, distance
imu_sensor_std = [.3, .3, .3]

imu_neff_threshold = 20
imu_std = [0, 0]

lidar_sensor_std = .3
lidar_neff_threshold = 10
lidar_std = [0, 0]

v_range = [0, 10]
a_range = [0, 10]

initial_pos_radius = 30

lidar_range = 50

# processmodel specifics
L = 2.743
dt = 0.05

# vehicle specifics
vehicle = 'mustang'
max_steering_angle = 70

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




