#ifndef BEFORE_SIMULATION
#define BEFORE_SIMULATION

#include "pid_controller.hpp"

extern "C"{
    #include "../asv-swarm/include/asv.h"
    #include "../asv-swarm/include/io.h"
}

void set_asv_and_pid_controller(struct Asv (&asv)[16], struct pid_controller (&pc)[16], struct Dimensions (&waypoints)[16], double initial_x, double initial_y, double wave_ht, double wave_heading){

    waypoints[0] = {initial_x-20, initial_y, 0};
    waypoints[1] = {initial_x-40, initial_y, 0};
    waypoints[2] = {initial_x-20, initial_y-20, 0};
    waypoints[3] = {initial_x-40, initial_y-40, 0};
    waypoints[4] = {initial_x, initial_y-20, 0};
    waypoints[5] = {initial_x, initial_y-40, 0};
    waypoints[6] = {initial_x+20, initial_y-20, 0};
    waypoints[7] = {initial_x+40, initial_y-40, 0};
    waypoints[8] = {initial_x+20, initial_y, 0};
    waypoints[9] = {initial_x+40, initial_y, 0};
    waypoints[10] = {initial_x+20, initial_y+20, 0};
    waypoints[11] = {initial_x+40, initial_y+40, 0};
    waypoints[12] = {initial_x, initial_y+20, 0};
    waypoints[13] = {initial_x, initial_y+40, 0};
    waypoints[14] = {initial_x-20, initial_y+20, 0};
    waypoints[15] = {initial_x-40, initial_y+40, 0};

    
    struct Waypoints useless;
    char in_file[50] = "../asv-swarm/example_input.toml";
    for(int j = 0; j < 16; j++){
      set_input(in_file, &asv[j], &useless);
      if(wave_ht != 0.0){
        asv[j].wave_type = irregular_wave;
        wave_init(&asv[j].wave, wave_ht, wave_heading, 3);
      }
      pid_controller_init(&pc[j]);
      set_waypoint(&pc[j], waypoints[j]);
    }

}

float compute_energy_cost(struct Asv* asv){
    
    float temp_energy_cost = 0.0;
    for(int i =0; i < 4; i++)
        temp_energy_cost += ((asv -> propellers[i].thrust) / 2);

    return temp_energy_cost/4.0;
}

void get_random_wave_height_and_heading(double &wave_ht, double &wave_heading){
    //range from 0 - 3 meters
    srand(time(NULL));
    rand();
    wave_ht = 0.01 + 3 * ((double)rand() / RAND_MAX);
    wave_heading = 2 * M_PI * ((double)rand() / RAND_MAX);
}




#endif