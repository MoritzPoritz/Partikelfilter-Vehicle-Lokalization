from __future__ import print_function
import glob
import os
import numpy as np
import sys

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
import keyboard

class MapPointCloudRetriever: 
    def __init__(self,client, args):
        self.sensor_positions = []
        self.sensor_rotations = []
        self.points = []
        self.client = client
        self.args = args

    def lidar_callback(self,point_cloud, sensor_transform): 
    # first reshape the pointcloud array
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        # get the x,y,z coordinates
        points = data
        # invert the y axis
        points[:,:1] = -points[:,:1]
        # append points to point list
        positions_array = np.full(points[:, :-1].shape, np.array([sensor_transform.location.x, sensor_transform.location.y, sensor_transform.location.z]))
        rotations_array = np.full(points[:, :-1].shape, np.array([sensor_transform.rotation.roll, sensor_transform.rotation.pitch, sensor_transform.rotation.yaw]))
        #print(positions_array)

        '''
        if (len(self.points) == 0 and len(self.sensor_positions) == 0 and len(self.sensor_rotations) == 0): 
            for i in range(len(points)): 
                self.points.append(points[i])
                self.sensor_positions.append(positions_array[i])
                self.sensor_rotations.append(rotations_array[i])
        else: 
            #print("Values there")
            np.append(self.points, points)
            np.append(self.sensor_positions, positions_array, axis=0)
            np.append(self.sensor_rotations, rotations_array, axis=0)

            #print(self.sensor_positions, positions_array)
        '''
        for i in range(len(points)): 
            self.points.append(points[i])
            self.sensor_positions.append(positions_array[i])
            self.sensor_rotations.append(rotations_array[i])
    
        



    def map_point_cloud_retrievement(self): 

        actor_list = []
    
        # Get the world and apply world settings
        world = self.client.get_world()
        # set world settings, synchronous mode and delta seconds
        settings = world.get_settings()
        if not settings.synchronous_mode:
            settings.synchronous_mode = True
        # fixed_delta_seconds need  <= max_substep_delta_time * max_substeps
        settings.fixed_delta_seconds = .05
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        print("Settings", settings)
        world.apply_settings(settings)


        # setup vehicle 
        blueprint_library = world.get_blueprint_library()

        bp_vehicle = blueprint_library.filter('mustang')[0]
        transform = np.random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp_vehicle, transform)

        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)
        
        tm = self.client.get_trafficmanager()
        tm_port = tm.get_port()
        tm.set_synchronous_mode(True)
        vehicle.set_autopilot(True, tm_port)
        
        # setup the lidar sensor
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '500.0')
        lidar_bp.set_attribute('noise_stddev', '0')
        lidar_bp.set_attribute('upper_fov', '1.0')
        lidar_bp.set_attribute('lower_fov', '-1.0')
        lidar_bp.set_attribute('channels', '64.0')
        lidar_bp.set_attribute('rotation_frequency', '20.0')
        lidar_bp.set_attribute('points_per_second','10000')
        lidar_bp.set_attribute('sensor_tick', '0.05')

        # apply lidar sensor to vehicle
        lidar_init_trans = carla.Transform(carla.Location(z=2))
        lidar = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=vehicle)
        actor_list.append(lidar)
        # set callback function
        lidar.listen(lambda data: self.lidar_callback(data, lidar.get_transform()))

        while True: 
            if self.args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            if keyboard.is_pressed('q'): 
                print("user stopped the loop")
                break
        
        self.write_results_to_csv()
        print("End sensor retrievment")
        #lidar_bp.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')

    def write_results_to_csv(self): 

        self.points = np.array(self.points)
        self.sensor_positions = np.array(self.sensor_positions, dtype=float)
        self.sensor_rotations = np.array(self.sensor_rotations, dtype=float)

        print(len(self.points), len(self.sensor_positions), len(self.sensor_rotations))
        data = {
            'point_x': self.points[:,0],
            'point_y': self.points[:,1], 
            'point_z': self.points[:,2], 
            'intensity': self.points[:,3],
            'loc_x': self.sensor_positions[:,0],
            'loc_y': self.sensor_positions[:,1],
            'loc_z': self.sensor_positions[:,2], 
            'rot_roll': self.sensor_rotations[:,0],
            'rot_pitch': self.sensor_rotations[:,1], 
            'rot_yaw': self.sensor_rotations[:,2]
        }
        csv_handler.write_to_csv(config.paths['map_path']+'map_pc', data)
