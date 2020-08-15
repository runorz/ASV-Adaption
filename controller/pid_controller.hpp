#ifndef PID_CONTROLLER_HPP
#define PID_CONTROLLER_HPP

extern "C"{
#include "../asv-swarm/include/geometry.h"
#include "../asv-swarm/include/asv.h"
}

struct pid_controller
{
    struct Dimensions waypoint;

    double kp_orientation;
    double ki_orientation;
    double kd_orientation;
    double kp_distance;
    double ki_distance;
    double kd_distance;

    double error_orientation;
    double error_int_orientation;
    double error_deri_orientation;
    double error_distance;
    double error_int_distance;
    double error_deri_distance;  
};

void pid_controller_init(struct pid_controller* pc){
    pc -> error_orientation = 0;
    pc -> error_int_orientation = 0;
    pc -> error_deri_orientation = 0;
    pc -> error_distance = 0;
    pc -> error_int_distance = 0;
    pc -> error_deri_distance = 0;
}

void set_orientation_gain(struct pid_controller* pc, double p, double i, double d){
    pc -> kp_orientation = p;
    pc -> ki_orientation = i;
    pc -> kd_orientation = d;
}

void set_distance_gain(struct pid_controller* pc, double p, double i, double d){
    pc -> kp_distance = p;
    pc -> ki_distance = i;
    pc -> kd_distance = d;
}

void set_waypoint(struct pid_controller* pc, struct Dimensions w){
    pc -> waypoint.x = w.x;
    pc -> waypoint.y = w.y;
    pc -> waypoint.z = w.z;
}

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

double activation(double input){
    //sigmoid
    return 1/(1+exp(-input));
}

void step(struct pid_controller* pc, struct Asv* asv){

    double error_orient = compute_angle(asv->cog_position.x, asv->cog_position.y, pc->waypoint.x, pc->waypoint.y);
    pc->error_int_orientation += error_orient;
    pc->error_deri_orientation = error_orient - pc->error_orientation;
    pc->error_orientation = error_orient;

    double error_dist = compute_distance(asv->cog_position.x, asv->cog_position.y, pc->waypoint.x, pc->waypoint.y);
    pc->error_int_distance += error_dist;
    pc->error_deri_distance = error_dist - pc->error_distance;
    pc->error_distance = error_dist;

    double orientation = activation(pc->kp_orientation * pc->error_orientation + pc->ki_orientation * pc->error_int_orientation + pc->kd_orientation * pc->error_deri_orientation);
    double force = activation(pc->kp_distance * pc->error_distance + pc->ki_distance * pc->error_int_distance + pc->kd_distance * pc->error_deri_distance);

    orientation = -M_PI + 2*M_PI*orientation - asv->attitude.z;
    force = 2 * force;

    for(int i = 0; i < 4; i ++){
        asv -> propellers[i].thrust = force;
        asv -> propellers[i].orientation = (struct Dimensions){0, 0, orientation};
    }
}











#endif
