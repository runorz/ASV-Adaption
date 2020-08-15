//Algorithm include
#define VERSION "version_unknown"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <array>

#include <boost/foreach.hpp>
#include <boost/multi_array.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/include/for_each.hpp>

#include <boost/test/unit_test.hpp>

#include "sferes2/sferes/eval/parallel.hpp"
#include "sferes2/sferes/gen/evo_float.hpp"
#include "sferes2/sferes/phen/parameters.hpp"
#include "sferes2/sferes/modif/dummy.hpp"
#include "sferes2/sferes/run.hpp"
#include "sferes2/sferes/stat/best_fit.hpp"

#include "map_elites/map_elites.hpp"
#include "map_elites/fit_map.hpp"
#include "map_elites/stat_map.hpp"

// only for sampled / evofloat
#include "map_elites/stat_map_binary.hpp"

#ifndef NO_PARALLEL
#include <sferes/eval/parallel.hpp>
#else
#include <sferes/eval/eval.hpp>
#endif

#include "controller/pid_controller.hpp"
#include "controller/before_simulation.hpp"


using namespace sferes;
using namespace sferes::gen::evo_float;

//ASV-Swarm include
extern "C"{
    #include "asv-swarm/include/asv.h"
    #include "asv-swarm/include/io.h"
}
#include<iostream>
#include <cstdlib>

struct Params {
  struct ea {
    SFERES_CONST size_t behav_dim = 2;
    SFERES_CONST double epsilon = 0;//0.05;
    SFERES_ARRAY(size_t, behav_shape, 10, 10);
  };
  struct pop {
    // number of initial random points
    SFERES_CONST size_t init_size = 200;
    // size of a batch
    SFERES_CONST size_t size = 200;
    SFERES_CONST size_t nb_gen = 1001;
    SFERES_CONST size_t dump_period = 100;
  };
  struct parameters {
    SFERES_CONST float min = -5;
    SFERES_CONST float max = 5;
  };
  struct evo_float {
    SFERES_CONST float cross_rate = 0.25f;
    SFERES_CONST float mutation_rate = 0.1f;
    SFERES_CONST float eta_m = 10.0f;
    SFERES_CONST float eta_c = 10.0f;
    SFERES_CONST mutation_t mutation_type = polynomial;
    SFERES_CONST cross_over_t cross_over_type = sbx;
  };
};

Dimensions set_random_waypoint(){
    srand(time(NULL));
    rand();
    double random_angle = (double)rand()/RAND_MAX * (2.0*M_PI);
    double random_dist = 50 + (rand() % 50);
    double random_x = sin(random_angle) * random_dist + 100;
    double random_y = cos(random_angle) * random_dist + 100;
    return (struct Dimensions){random_x, random_y, 0};
}

FIT_MAP(ASV_SWARM){
  public :
  template <typename Indiv>
  void eval(Indiv & ind) {
    //behavior descriptor, 2-dim, energy cost and time spent
    std::vector<float> descriptor;

    struct Asv asv[16];
    struct pid_controller pc[16];
    struct Dimensions waypoints[16];

    double initial_x = 100.0;
    double initial_y = 100.0;

    double wave_ht = 0.0; 
    double wave_heading = 0;

    set_asv_and_pid_controller(asv, pc, waypoints, initial_x, initial_y, wave_ht, wave_heading);

    bool simulating[16];
    float time_cost[16];
    float energy_cost[16];
    for(int i = 0; i < 16; i++){
        asv_init(&asv[i]);
        set_orientation_gain(&pc[i], ind.data(0), ind.data(1), ind.data(2));
        set_distance_gain(&pc[i], ind.data(3), ind.data(4), ind.data(5));
        simulating[i] = true;
        time_cost[i] = 0.0;
        energy_cost[i] = 0.0;
    }
    //sucessful rate
    float f = 0.0;

    int max_time_step = 6000;
    int t = 0;

    for(;t<max_time_step;t++){

        for(int i = 0; i < 16; i++)
            if(simulating[i]){
                step(&pc[i], &asv[i]);
                double _time = t * 40;
                asv_compute_dynamics(&asv[i], _time);
                energy_cost[i] += compute_energy_cost(&asv[i]);
                time_cost[i] += 1.0;
                if(compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoints[i].x, waypoints[i].y) <= 1)
                    simulating[i] = false;
            }

    }

    float sum_time_cost = 0;
    float sum_energy_cost = 0;
    for(int i =0; i < 16; i++){
      if(compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoints[i].x, waypoints[i].y) < 1)
        f += 1.0;
      sum_time_cost += time_cost[i]/max_time_step;
      sum_energy_cost += energy_cost[i]/time_cost[i];
    }
    descriptor.push_back(sum_time_cost/16.0);
    descriptor.push_back(sum_energy_cost/16.0);
    std::cout<<"Fitness: "<<f<<" "<<", time cost: "<<descriptor[0]<<", energy cost: "<<descriptor[1]<<std::endl;
    this->_value = f;
    this->set_desc(descriptor);
  }

  bool dead() {
    return false;
  }
};

int main(int argc, char** argv){
  using namespace sferes;

  typedef ASV_SWARM<Params> fit_t;
  typedef gen::EvoFloat<6, Params> gen_t;
  typedef phen::Parameters<gen_t, fit_t, Params> phen_t;
  typedef eval::Parallel<Params> eval_t;
  typedef boost::fusion::vector<
        stat::Map<phen_t, Params>
		,stat::BestFit<phen_t, Params>
        , stat::MapBinary<phen_t, Params> 
        > stat_t;
  typedef modif::Dummy<> modifier_t;
  typedef ea::MapElites<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;

  ea_t ea;
  // ea.load("ASV-adaption_2020-07-23_13_02_15_11426/gen_600");
  // std::ofstream ofs1("output");
  // ea.show_stat(0, ofs1, 0);
  // phen_t ind = *boost::fusion::at_c<0>(ea.stat())._archive[0];
  // for(int i = 0; i< ind.size(); i++)
  //   std::cout << ind.data(i) << std::endl;


  struct Asv asv;
  struct Waypoints waypoints;
  char in_file[50] = "../asv-swarm/example_input.toml";
  set_input(in_file, &asv, &waypoints);
  int max_time_step = 10000;
  int t = 0;
  struct Dimensions waypoint = {200,200,0};
  double wave_elevation = 0.0;
  double wave_ht = 0.03; 
  double wave_heading = 0.6*2*M_PI;
  double _time = 0.0;
  long rand_seed = 3;
  std::cout << "Waypoint:" << waypoint.x << "," << waypoint.y << std::endl;

  if(wave_ht != 0.0)
  {
    asv.wave_type = irregular_wave;
    wave_init(&asv.wave, wave_ht, wave_heading * PI/180.0, rand_seed);
  }

  asv_init(&asv);




  struct pid_controller pc;
  pid_controller_init(&pc);
  set_waypoint(&pc, waypoint);
  // set_orientation_gain(&pc, ind.data(0), ind.data(1), ind.data(2));
  // set_distance_gain(&pc, ind.data(3), ind.data(4), ind.data(5));
  set_orientation_gain(&pc, 1.50184, -0.000730038, 4.1845);
  set_distance_gain(&pc, 4.51028, 0.0970273, 2.38809);

  clock_t start, end;
  double simulation_time = 0.0;
  start = clock();
  char out_file[10] = "path";


  for(;t < max_time_step; t++){

      if(t >= OUTPUT_BUFFER_SIZE)
      {
        fprintf(stderr, "ERROR: output buffer exceeded.\n");
        // write output to file
        end = clock();
        simulation_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        write_output(out_file, 
                     t, 
                     wave_ht, 
                     wave_heading, 
                     rand_seed, 
                     _time, 
                     simulation_time);
	      exit(1);
      }

      
      step(&pc, &asv);
      _time = t * 40;
      asv_compute_dynamics(&asv, _time);
      // if(asv.wave_type == irregular_wave)
      //   {
      //     wave_elevation = wave_get_elevation(&asv.wave, 
      //                                         &asv.cog_position, _time);
      //   }

        buffer[t].sig_wave_ht = asv.wave.significant_wave_height;
        buffer[t].wave_heading = asv.wave.heading * 180.0/PI;
        buffer[t].random_number_seed = asv.wave.random_number_seed;
        buffer[t].time = _time;
        // buffer[t].wave_elevation = wave_elevation;
        buffer[t].cog_x = asv.cog_position.x;
        buffer[t].cog_y = asv.cog_position.y;
        buffer[t].cog_z = asv.cog_position.z - (asv.spec.cog.z - asv.spec.T);
        buffer[t].heel = asv.attitude.x * 180.0/PI;
        buffer[t].trim = asv.attitude.y * 180.0/PI;
        buffer[t].heading = asv.attitude.z * 180.0/PI;
        buffer[t].surge_velocity = asv.dynamics.V[surge];
        buffer[t].surge_acceleration = asv.dynamics.A[surge];

        if(compute_distance(asv.cog_position.x, asv.cog_position.y, waypoint.x, waypoint.y) < 1)
            break;
  }

  end = clock();
  simulation_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  write_output(out_file, 
               t, 
               wave_ht, 
               wave_heading, 
               rand_seed, 
               _time, 
               simulation_time);
}