import particle_filter_imu.particle_filter_imu as pf_imu
import config.config as config
import plotting.plot_animated_filter as animated
def main(): 
    #pf = pf_imu.ParticleFilterIMU(5000, config.straight_x_line_name)
    pf = pf_imu.ParticleFilterIMU(500, config.curve_line_name)
    
    pf.run_pf_imu()
    animated.plot_results_animated(pf.particles_at_t, pf.weights_at_t, pf.xs, pf.ground_truth,pf.dm, pf.Ts)


if __name__ == "__main__": 
    main()