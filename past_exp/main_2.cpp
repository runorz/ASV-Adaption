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
    SFERES_CONST size_t behav_dim = 4;
    SFERES_CONST double epsilon = 0;//0.05;
    SFERES_ARRAY(size_t, behav_shape, 10, 10, 10, 10);
  };
  struct pop {
    // number of initial random points
    SFERES_CONST size_t init_size = 50;
    // size of a batch
    SFERES_CONST size_t size = 50;
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

    bool simulating[16];
    float time_cost[16];
    float energy_cost[16];
    float last_distance[16];
    float distance_update[16];
    
    //wave setting
    double wave_ht = 0.0;
    double wave_heading = 0.0;
    get_random_wave_height_and_heading(wave_ht, wave_heading);

    std::cout<<"wave_ht: "<<wave_ht <<", wave_heading: " << wave_heading <<std::endl;

    set_asv_and_waypoint(asv, pc, waypoints, initial_x, initial_y, wave_ht, wave_heading);

    for(int i = 2; i < 16; i+=4){
        asv_init(&asv[i]);
        set_orientation_gain(&pc[i], ind.data(0), ind.data(1), ind.data(2));
        set_distance_gain(&pc[i], ind.data(3), ind.data(4), ind.data(5));
        simulating[i] = true;
        time_cost[i] = 0.0;
        energy_cost[i] = 0.0;
        last_distance[i] = compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoints[i].x, waypoints[i].y);
        distance_update[i] = 0.0;
    }
    //sucessful rate
    float f = 0.0;

    int max_time_step = 3000;
    int t = 0;

    for(;t<max_time_step;t++){

        for(int i = 2; i < 16; i+=4)
            if(simulating[i]){
                step(&pc[i], &asv[i]);
                double _time = t * 40;
                asv_compute_dynamics(&asv[i], _time);
                energy_cost[i] += compute_energy_cost(&asv[i]);
                time_cost[i] += 1.0;
                
                float current_distance = compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoints[i].x, waypoints[i].y);
                distance_update[i] += (last_distance[i] - current_distance);
                last_distance[i] = current_distance;

                if(current_distance <= 1)
                    simulating[i] = false;
            }

    }

    float sum_time_cost = 0;
    float sum_energy_cost = 0;
    float sum_average_distance_update = 0;
    for(int i =2; i < 16; i+=4){
      if(compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoints[i].x, waypoints[i].y) < 1)
        f += 1.0;
      sum_time_cost += time_cost[i]/max_time_step;
      sum_energy_cost += energy_cost[i]/time_cost[i];
      sum_average_distance_update += distance_update[i]/time_cost[i];
    }
    descriptor.push_back(sum_time_cost/4.0);
    descriptor.push_back(sum_energy_cost/4.0);

    if(wave_ht!=0.0){
        descriptor.push_back(wave_ht/3.01);
        descriptor.push_back(wave_heading/(2*PI));
    }else{
        descriptor.push_back(0.0);
        descriptor.push_back(0.0);
    }

    this->_value = sum_average_distance_update / 4;
    std::cout<<"Fitness: "<<this->_value<<" "<<", time cost: "<<descriptor[0]<<", energy cost: "<<descriptor[1] <<", wave_ht: " <<wave_ht<<", wave_heading: " << wave_heading<<std::endl;
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
  run_ea(argc, argv, ea);
  float best = ea.stat<1>().best()->fit().value();
  std::cout<<"best fit (map_elites):" << best << std::endl;
}