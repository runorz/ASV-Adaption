#ifndef LINEAR_CONTROLLER_HPP
#define LINEAR_CONTROLLER_HPP

extern "C"{
#include "../asv-swarm/include/asv.h"
    #include "../asv-swarm/include/io.h"
}

struct linear_controller
{
    struct Dimensions waypoint;
    double w1_1;
    double w1_2;
    double b1;
    double w2_1;
    double w2_2;
    double b2;
};

void set_weights_and_bias(struct linear_controller* lc, double o_w1, double o_w2, double o_b, double f_w1, double f_w2, double f_b){
    lc -> w1_1 = o_w1;
    lc -> w1_2 = o_w2;
    lc -> b1 = o_b;
    lc -> w2_1 = f_w1;
    lc -> w2_2 = f_w2;
    lc -> b2 = f_b;
};

void set_waypoint(struct linear_controller* lc, struct Dimensions w){
    lc -> waypoint.x = w.x;
    lc -> waypoint.y = w.y;
    lc -> waypoint.z = w.z;
};

double activation(double input){
    //sigmoid
    return 1/(1+exp(-input));
};

double compute_angle(double asv_x, double asv_y, double waypoint_x, double waypoint_y){
  double d_x = waypoint_x - asv_x;
  double d_y = waypoint_y - asv_y;
  if(d_x <= 0){
    return acos(d_y/(sqrt(d_x*d_x+d_y*d_y) * sqrt(1)));
  }else{
    return -acos(d_y/(sqrt(d_x*d_x+d_y*d_y) * sqrt(1)));
  }
}

double compute_distance(double asv_x, double asv_y, double waypoint_x, double waypoint_y){
  double x2 = pow(asv_x - waypoint_x, 2);
  double y2 = pow(asv_y - waypoint_y, 2);
  return sqrt(x2+y2);
}

void step(struct linear_controller* lc, struct Asv* asv){

    double error_orient = compute_angle(asv->cog_position.x, asv->cog_position.y, lc->waypoint.x, lc->waypoint.y);
    error_orient = (error_orient+M_PI)/(2*M_PI);
    double error_dist = compute_distance(asv->cog_position.x, asv->cog_position.y, lc->waypoint.x, lc->waypoint.y);
    error_dist = 2 * atan(error_dist) / M_PI;
    
    double orientation = activation(lc->w1_1*error_orient + lc->w1_2*error_dist + lc->b1);
    double force = activation(lc->w2_1*error_orient + lc->w2_2*error_dist + lc->b2);

    orientation = -M_PI + 2*M_PI*orientation - asv->attitude.z;
    force = 2 * force;
    for(int i = 0; i < 4; i ++){
        asv -> propellers[i].thrust = force;
        asv -> propellers[i].orientation = (struct Dimensions){0, 0, orientation};
    }
    
}

void set_asv_and_linear_controller(struct Asv (&asv)[16], struct linear_controller (&lc)[16], struct Dimensions (&waypoints)[16], double initial_x, double initial_y, double wave_ht, double wave_heading){

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
      set_waypoint(&lc[j], waypoints[j]);
    }

}




#endif