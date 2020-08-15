extern "C"{
    #include "asv-swarm/include/asv.h"
    #include "asv-swarm/include/io.h"
}
#include<iostream>
#include <cstdlib>

#include "controller/nn_before_simulation.hpp"

int main(){


    double initial_x = 100.0;
    double initial_y = 100.0;
    struct Asv asv[16];
    struct Dimensions waypoints[16];
    bool simulating[16];
    float last_distance[16];
    float distance_update[16];
    float time_cost[16];

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
    for(int i = 0; i < 16; i++){
        set_input(in_file, &asv[i], &useless);
            asv_init(&asv[i]);
            simulating[i] = true;
            last_distance[i] = compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoints[i].x, waypoints[i].y);
            distance_update[i] = 0.0;
            time_cost[i] = 0.0;
        }

        int max_time_step = 6000;
        int t = 0;
        for(;t<max_time_step;t++){
            for(int i = 0; i < 16; i++)
                if(simulating[i]){

                    for(int p = 0; p < 4; p ++){
                    asv[i].propellers[p].thrust = 2.0;
                    asv[i].propellers[p].orientation = (struct Dimensions){0, 0, compute_angle(asv[i].cog_position.x, asv[i].cog_position.y, waypoints[i].x, waypoints[i].y)};
                }

                    double _time = t * 40;
                    asv_compute_dynamics(&asv[i], _time);
                    time_cost[i] += 1.0;
                    
                    float current_distance = compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoints[i].x, waypoints[i].y);
                    distance_update[i] += (last_distance[i] - current_distance);
                    last_distance[i] = current_distance;

                    if(current_distance <= 1)
                        simulating[i] = false;
                }
        }
        float sum_average_distance_update = 0;
        for(int i =0; i < 16; i++){
            sum_average_distance_update += distance_update[i]/time_cost[i];
        }

        std::cout << sum_average_distance_update*25/16 <<std::endl;




}