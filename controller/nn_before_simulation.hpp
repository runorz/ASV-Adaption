#ifndef NN_BEFORE_SIMULATION
#define NN_BEFORE_SIMULATION

extern "C"{
    #include "../asv-swarm/include/asv.h"
    #include "../asv-swarm/include/io.h"
}

void set_asv_and_waypoint(struct Asv (&asv)[8], struct Dimensions (&waypoints)[8], double initial_x, double initial_y, double wave_ht, double wave_heading){

    waypoints[0] = {initial_x-20, initial_y, 0};
    waypoints[1] = {initial_x-40, initial_y, 0};
    waypoints[2] = {initial_x-20, initial_y+20, 0};
    waypoints[3] = {initial_x-40, initial_y+40, 0};
    waypoints[4] = {initial_x+20, initial_y, 0};
    waypoints[5] = {initial_x+40, initial_y, 0};
    waypoints[6] = {initial_x+20, initial_y-20, 0};
    waypoints[7] = {initial_x+40, initial_y-40, 0};


    
    struct Waypoints useless;
    char in_file[50] = "../asv-swarm/example_input.toml";
    for(int j = 0; j < 8; j++){
      set_input(in_file, &asv[j], &useless);
      if(wave_ht != 0.0){
        asv[j].wave_type = irregular_wave;
        wave_init(&asv[j].wave, wave_ht, wave_heading, 3);
      }
      asv_init(&asv[j]);
    }

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

float compute_energy_cost(struct Asv* asv){
    
    float temp_energy_cost = 0.0;
    for(int i =0; i < 4; i++)
        temp_energy_cost += ((asv -> propellers[i].thrust) / 2);

    return temp_energy_cost/4.0;
}

struct wave_generator
{
  inline static int i = 0;
  inline static int j = 0;
};

void get_wave_height_direction(struct wave_generator* wg, double &wave_ht, double &wave_heading){
  wave_ht = 2.5*wg->i / 30;
  wave_heading = -M_PI + wg->j*2*M_PI/30;
  wg->i++;
  if(wg->i == 30){
    wg->i = 0;
    wg->j++;
    if(wg->j == 30)
      wg->j = 0;
  }
}

void get_random_wave_height_direction(double &wave_ht, double &wave_heading){
  srand (time(NULL));
  wave_ht = 0.01 + 2.0 * ((double)rand()/RAND_MAX);
  wave_heading = -M_PI + 2 * M_PI * ((double)rand()/RAND_MAX);
}

void init(struct Asv &asv, double &wave_ht, double &wave_heading){

//    get_random_wave_ht(wave_ht, wave_heading);
    struct Waypoints useless;
    char in_file[100] = "../asv-swarm/example_input.toml";
    set_input(in_file, &asv, &useless);
    if(wave_ht != 0.0)
    {
        std::cout << "????????????" << std::endl;
        asv.wave_type = irregular_wave;
        wave_init(&asv.wave, wave_ht, wave_heading, 0);
    }
    asv_init(&asv);

}

void get_random_waypoint(struct Dimensions &waypoint){

    srand(time(NULL));
    rand();
    double angle = 2 * M_PI * ((double)rand()/RAND_MAX);
//    double distance = 20 + 80 * ((double)rand()/RAND_MAX);
    double distance = 100;
    // std::cout << "angle: " << angle << ", distance: " << distance << std::endl;
    waypoint = (struct Dimensions){distance*cos(angle), distance*sin(angle), 0};

}

float get_normalized_attitude(double a){
  float out = remainder(a, (2 * M_PI));
  if (out > M_PI){
    return -(2*M_PI - out);
  }
  else if (out <= -M_PI)
  {
    return 2 * M_PI + out;
  }
  else{
    return out;
  }
  
}


#endif