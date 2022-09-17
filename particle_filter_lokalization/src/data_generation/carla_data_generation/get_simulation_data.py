from __future__ import print_function
import glob
import os
import numpy as np
import sys
import cv2 as cv

try:
    sys.path.append(glob.glob('..\\carla\\dist\\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname('..\\src'), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
import config.config as config
import utils.csv_handler as csv_handler
import utils.image_handler as image_handler
import utils.json_handler as json_handler
from scipy import stats

import keyboard


class DataRetierver: 
    def __init__(self,client, args):
        # carla related stuff
        
        self.client = client
        self.args = args
        self.world = self.client.get_world()
        self.car = None
        self.spectator = self.world.get_spectator()
        self.debug = self.world.debug

        # Stuff for creation of the map data
        self.positions = []
        self.x_range = None
        self.y_range = None

        # imu data
        self.acceleration_measurement = []
        self.orientation_measurement = []
        self.acceleration_input_imu = []
        self.steering_input_imu = []
        self.positions_x_ground_truth_imu = []
        self.positions_y_ground_truth_imu = []
        self.velocities_ground_truth_imu = []
        self.timestamps_imu = []

        # lidar data
        self.lidar_measurements = []
        self.acceleration_input_lidar = []
        self.steering_input_lidar = []
        self.positions_x_ground_truth_lidar = []
        self.positions_y_ground_truth_lidar = []
        self.velocities_ground_truth_lidar = []
        self.timestamps_lidar = []        

        # sensors
        self.imu_sensor = None
        self.lidar_sensor = None

        # carla stuff
        self.actor_list = []
        self.spawn_point_index = 10
        self.traffic_manager = None
        self.blueprint_library = self.world.get_blueprint_library()

    def tick(self): 
        self.world.tick() 

    def set_settings(self): 
        print("Set settings")
        settings = self.world.get_settings()
        if not settings.synchronous_mode:
            settings.synchronous_mode = True
        # fixed_delta_seconds need  <= max_substep_delta_time * max_substeps
        settings.fixed_delta_seconds = .05
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        self.world.apply_settings(settings)



    def add_lidar_sensor(self):
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(config.lidar_range))
        lidar_bp.set_attribute('noise_stddev', '0')
        lidar_bp.set_attribute('upper_fov', '0.0')
        lidar_bp.set_attribute('lower_fov', '0.0')
        lidar_bp.set_attribute('channels', '64.0')
        lidar_bp.set_attribute('rotation_frequency', '20.0')
        lidar_bp.set_attribute('points_per_second','10000')
        lidar_bp.set_attribute('sensor_tick', '0.05')
        lidar_init_trans = carla.Transform(carla.Location(z=2))
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=self.car)
        self.actor_list.append(self.lidar_sensor)
        self.actor_list.append(self.lidar_sensor)
        self.lidar_sensor.listen(lambda data: self.lidar_callback(data, self.lidar_sensor.get_transform()))


    def lidar_callback(self,point_cloud, transform):
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        # get the x,y,z coordinates
        points = data[:3]
        distances = np.linalg.norm(points, axis=1)
        if (len(distances) > 0):
            self.lidar_measurements.append(stats.mode(distances)[0][0])
        else: 
            self.lidar_measurements.append(0)

        self.timestamps_lidar.append(point_cloud.timestamp)
        self.positions_x_ground_truth_lidar.append(self.car.get_transform().location.x)
        self.positions_y_ground_truth_lidar.append(self.car.get_transform().location.y)
        self.acceleration_input_lidar.append(self.car.get_acceleration().x)
        self.steering_input_lidar.append(self.car.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel))
        self.velocities_ground_truth_lidar.append(np.linalg.norm(np.array([self.car.get_acceleration().x, self.car.get_acceleration().y])))


    def add_imu_sensor(self): 
        imu_bp = self.blueprint_library.find('sensor.other.imu')
        imu_location = carla.Location(0,0,0)
        imu_rotation = carla.Rotation(0,0,0)
        imu_transform = carla.Transform(imu_location,imu_rotation)
        imu_bp.set_attribute("sensor_tick",str(config.dt))
        self.imu_sensor = self.world.spawn_actor(imu_bp,imu_transform,attach_to=self.car, attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.imu_sensor)
        self.imu_sensor.listen(lambda imu: self.imu_callback(imu))
    
    def imu_callback(self,imu):
        self.acceleration_measurement.append(np.array(imu.accelerometer.x))
        
        # turn orientation from carla (compass with 0° at -y axis to filter with 0° at x axis)
        carla_orientation = imu.compass
        rotated_orientation = (carla_orientation- np.pi/2)%(2*np.pi)
        vec_from_orientation = np.array([np.cos(rotated_orientation), np.sin(rotated_orientation)])
        new_orientation = np.arctan2(vec_from_orientation[1], vec_from_orientation[0])% (np.pi*2)
        self.orientation_measurement.append(new_orientation)


        self.timestamps_imu.append(imu.timestamp)
        self.positions_x_ground_truth_imu.append(self.car.get_transform().location.x)
        self.positions_y_ground_truth_imu.append(self.car.get_transform().location.y)
        self.acceleration_input_imu.append(self.car.get_acceleration().x)
        self.steering_input_imu.append(self.car.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel))
        self.velocities_ground_truth_imu.append(np.linalg.norm(np.array([self.car.get_acceleration().x, self.car.get_acceleration().y])))

        

    def spawn_vehicle(self): 
        blueprint_library = self.world.get_blueprint_library()
        bp_vehicle = blueprint_library.filter('mustang')[0]
        transform = self.world.get_map().get_spawn_points()[self.spawn_point_index]    
        self.car = self.world.spawn_actor(bp_vehicle, transform)
        self.spectator.set_transform(self.car.get_transform())
        self.actor_list.append(self.car)
        print('created %s' % self.car.type_id)
        
        self.traffic_manager = self.client.get_trafficmanager()
        tm_port = self.traffic_manager.get_port()
        self.traffic_manager.set_synchronous_mode(True)
        self.car.set_autopilot(True, tm_port)
        

    def data_retrievement(self): 
        print("Started data retrievement")
        self.spawn_vehicle()
        print("spawned vehicle")
        self.add_imu_sensor()
        #self.add_lidar_sensor()
        while True: 
            self.tick()
            if keyboard.is_pressed('q'): 
                print("user stopped the loop")
                break
        
        print("End sensor retrievment")

        print("Write imu data")
        self.write_imu_data_to_csv()
        self.write_lidar_data_to_csv()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])


        print('done.')

    def write_lidar_data_to_csv(self): 
        self.lidar_measurements = np.array(self.lidar_measurements)
        self.acceleration_input_lidar = np.array(self.acceleration_input_lidar)
        self.steering_input_lidar = np.array(self.steering_input_lidar)
        self.positions_x_ground_truth_lidar = np.array(self.positions_x_ground_truth_lidar)
        self.positions_y_ground_truth_lidar = np.array(self.positions_y_ground_truth_lidar)
        self.velocities_ground_truth_lidar = np.array(self.velocities_ground_truth_lidar)
        self.timestamps_lidar = np.array(self.timestamps_lidar)    
        data = {
            'acceleration_input': self.acceleration_input_lidar,
            'steering_input': self.steering_input_lidar, 
            'measurements': self.lidar_measurements, 
            'positions_x_ground_truth': self.positions_x_ground_truth_lidar,
            'positions_y_ground_truth': self.positions_y_ground_truth_lidar,
            'velocities_ground_truth': self.velocities_ground_truth_lidar, 
            'timestamps': self.timestamps_lidar
        }
        csv_handler.write_structured_data_to_csv(config.paths['data_path']+ config.overall_data_file + config.lidar_data_appendix + config.data_suffix, data)


    def write_imu_data_to_csv(self): 
        self.acceleration_measurement = np.array(self.acceleration_measurement)
        self.orientation_measurement = np.array(self.orientation_measurement)
        self.acceleration_input_imu = np.array(self.acceleration_input_imu)
        self.steering_input_imu = np.array(self.steering_input_imu)
        self.positions_x_ground_truth_imu = np.array(self.positions_x_ground_truth_imu)
        self.positions_y_ground_truth_imu = np.array(self.positions_y_ground_truth_imu)
        self.velocities_ground_truth_imu = np.array(self.velocities_ground_truth_imu)
        self.timestamps_imu = np.array(self.timestamps_imu)

        data = {
            'acceleration_input': self.acceleration_input_imu,
            'steering_input': self.steering_input_imu, 
            'acceleration_measurement': self.acceleration_measurement, 
            'orientation_measurement': self.orientation_measurement, 
            'positions_x_ground_truth': self.positions_x_ground_truth_imu,
            'positions_y_ground_truth': self.positions_y_ground_truth_imu,
            'velocities_ground_truth': self.velocities_ground_truth_imu, 
            'timestamps': self.timestamps_imu
        }
        csv_handler.write_structured_data_to_csv(config.paths['data_path']+ config.overall_data_file + config.imu_data_appendix + config.data_suffix, data)
