#include <algorithm>
#include <cmath>
#include <iostream>
#include <array>

#include <boost/foreach.hpp>
#include <boost/multi_array.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/include/for_each.hpp>

#include <boost/test/unit_test.hpp>

#include "sferes2/sferes/parallel.hpp"
#include "sferes2/sferes/fit/fitness.hpp"
#include "sferes2/sferes/gen/evo_float.hpp"
#include "sferes2/sferes/phen/parameters.hpp"
#include "sferes2/sferes/modif/diversity.hpp"
#include "sferes2/sferes/modif/dummy.hpp"
#include "sferes2/sferes/ea/nsga2.hpp"
#include "sferes2/sferes/stat/pareto_front.hpp"
#include "sferes2/sferes/stat/mean_fit.hpp"
#include "sferes2/sferes/eval/parallel.hpp"
#include "sferes2/sferes/run.hpp"

#include "nn2/gen_dnn.hpp"
#include "nn2/gen_dnn_ff.hpp"
#include "nn2/phen_dnn.hpp"

#include "map_elites/map_elites.hpp"
#include "map_elites/fit_map.hpp"
#include "map_elites/stat_map.hpp"

// only for sampled / evofloat
#include "map_elites/stat_map_binary.hpp"

#include "controller/nn_before_simulation.hpp"

#ifndef NO_PARALLEL
#include <sferes/eval/parallel.hpp>
#else
#include <sferes/eval/eval.hpp>
#endif

using namespace sferes;
using namespace sferes::gen::dnn;
using namespace sferes::gen::evo_float;

//ASV-Swarm include
extern "C"{
    #include "asv-swarm/include/asv.h"
    #include "asv-swarm/include/io.h"
}
#include<iostream>
#include <cstdlib>

struct Params{
    struct ea {
        SFERES_CONST size_t behav_dim = 2;
        SFERES_CONST double epsilon = 0;//0.05;
        SFERES_ARRAY(size_t, behav_shape, 10, 10);
    };
    struct pop {
        // number of initial random points
        SFERES_CONST size_t init_size = 1000;
        // size of a batch
        SFERES_CONST size_t size = 100;
        SFERES_CONST size_t nb_gen = 1000;
        SFERES_CONST size_t dump_period = 100;
    };

    struct parameters {
        // maximum value of parameters (weights and bias)
        SFERES_CONST float min = -5.0f;
        // minimum value
        SFERES_CONST float max = 5.0f;
    };

    struct evo_float {
        SFERES_CONST float mutation_rate = 0.1f;
        SFERES_CONST float cross_rate = 0.1f;
        SFERES_CONST mutation_t mutation_type = polynomial;
        SFERES_CONST cross_over_t cross_over_type = sbx;
        SFERES_CONST float eta_m = 15.0f;
        SFERES_CONST float eta_c = 15.0f;
    };

    struct dnn {
        SFERES_CONST size_t nb_inputs = 2;
        SFERES_CONST size_t nb_outputs = 2;

        SFERES_CONST size_t min_nb_neurons = 0;
        SFERES_CONST size_t max_nb_neurons = 20;
        SFERES_CONST size_t min_nb_conns = 0;
        SFERES_CONST size_t max_nb_conns = 40;

        SFERES_CONST float m_rate_add_conn = 0.15f;
        SFERES_CONST float m_rate_del_conn = 0.15f;
        SFERES_CONST float m_rate_change_conn = 0.15f;
        SFERES_CONST float m_rate_add_neuron = 0.10f;
        SFERES_CONST float m_rate_del_neuron = 0.10f;

        SFERES_CONST int io_param_evolving = true;

        SFERES_CONST init_t init = random_topology;;
  };

};
FIT_MAP(ASV_SWARM){
    public :
    template <typename Indiv>
    void eval(Indiv & ind) {

        ind.nn().init();

        struct Asv asv[4];
        struct Dimensions waypoints[4];

        double initial_x = 100.0;
        double initial_y = 100.0;

        double wave_ht = 0.0;
        double wave_heading = 0.0;

        bool simulating[4];
        float time_cost[4];
        float energy_cost[4];
        float last_distance[4];
        float distance_update[4];

        std::vector<float> descriptor;

        set_asv_and_waypoint(asv, waypoints, initial_x, initial_y, wave_ht, wave_heading);

        for(int i = 0; i < 4; i++){
            simulating[i] = true;
            time_cost[i] = 0.0;
            energy_cost[i] = 0.0;
            last_distance[i] = compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoints[i].x, waypoints[i].y);
            distance_update[i] = 0.0;
        }

        float f = 0.0;

        int max_time_step = 3000;
        int t = 0;

        for(;t<max_time_step;t++){

            for(int i = 0; i < 4; i++)
                if(simulating[i]){

                    float angle = compute_angle(asv[i].cog_position.x, asv[i].cog_position.y, waypoints[i].x, waypoints[i].y);
                    float distance = compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoints[i].x, waypoints[i].y);
                    angle = sin(angle);
                    distance = 2 * atan(distance) / M_PI;
                    static const std::vector<float> inputs = {angle, distance};

                    ind.nn().step(inputs);

                    const std::vector<float> &outf = ind.nn().get_outf();

                    for(int p = 0; p < 4; p++)
                    {
                        asv[i].propellers[p].thrust = 2 * ((1 + outf[0])/2.0); //N
                        asv[i].propellers[p].orientation = (struct Dimensions){0, 0, outf[1] * M_PI};
                    }


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
        for(int i =0; i < 4; i++){
            if(compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoints[i].x, waypoints[i].y) < 1)
                f += 1.0;
            sum_time_cost += time_cost[i]/max_time_step;
            sum_energy_cost += energy_cost[i]/time_cost[i];
            sum_average_distance_update += distance_update[i]/time_cost[i];
        }
        descriptor.push_back(sum_time_cost/4.0);
        descriptor.push_back(sum_energy_cost/4.0);

        this->_value = sum_average_distance_update * 25 / 4;
        std::cout<<"Fitness: "<<this->_value<<" "<<", time cost: "<<descriptor[0]<<", energy cost: "<<descriptor[1] <<std::endl;
        this->set_desc(descriptor);


    }

    bool dead() {
        return false;
    }
};

int main(int argc, char** argv){
  using namespace sferes;

  typedef ASV_SWARM<Params> fit_t;
  typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> weight_t;
  typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> bias_t;
  typedef nn::PfWSum<weight_t> pf_t;
  typedef nn::AfTanh<bias_t> af_t;
  typedef nn::Neuron<pf_t, af_t> neuron_t;
  typedef nn::Connection<weight_t> connection_t;
  typedef gen::Dnn<neuron_t, connection_t, Params> gen_t;
  typedef phen::Dnn<gen_t, fit_t, Params> phen_t;


  // typedef gen::EvoFloat<10, Params> gen_t;
  // typedef phen::Parameters<gen_t, fit_t, Params> phen_t;
  typedef eval::Parallel<Params> eval_t;
  typedef boost::fusion::vector<
        stat::Map<phen_t, Params>
        , stat::MeanFit<phen_t, Params>
        , stat::MapBinary<phen_t, Params> 
        > stat_t;
  typedef modif::Dummy<> modifier_t;
  typedef ea::MapElites<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;

  ea_t ea;
//   run_ea(argc, argv, ea);
  ea.load("map_nn_2020-07-27_18_12_55_22853/gen_900");
  std::ofstream ofs1("output");
  ea.show_stat(0, ofs1, 59);
  phen_t ind = *boost::fusion::at_c<0>(ea.stat())._archive[59];
//   std::ofstream ofs("testttt");
//   ind.gen().write(ofs);
  std::vector<float> input1 = {0.5, -0.1};
  std::vector<float> input2 = {-0.4, 0.8};

  ind.nn().init();
  ind.nn().step(input1);
  struct Asv asv;
  struct Waypoints waypoints;
  char in_file[50] = "../asv-swarm/example_input.toml";
  set_input(in_file, &asv, &waypoints);
  int max_time_step = 10000;
  int t = 0;
  asv_init(&asv);
  struct Dimensions waypoint = {100, 120, 0};
  double wave_elevation = 0.0;
  double wave_ht = 0.0; 
  double wave_heading = 0.0;
  double _time = 0.0;
  std::cout << "Waypoint:" << waypoint.x << "," << waypoint.y << std::endl;


  for(;t < max_time_step; t++){

      float angle = compute_angle(asv.cog_position.x, asv.cog_position.y, waypoint.x, waypoint.y);
      float distance = compute_distance(asv.cog_position.x, asv.cog_position.y, waypoint.x, waypoint.y);
      angle = sin(angle);
      distance = 2 * atan(distance) / M_PI;

      static const std::vector<float> inputs = {angle, distance};
      ind.nn().step(inputs);
      const std::vector<float> &outf = ind.nn().get_outf();
      for(int p = 0; p < 4; p++)
      {
        asv.propellers[p].thrust = 2 * ((1 + outf[0])/2.0); //N
        asv.propellers[p].orientation = (struct Dimensions){0, 0, outf[1] * M_PI};
      }
      _time = t * 40;
      asv_compute_dynamics(&asv, _time);

      buffer[t].sig_wave_ht = asv.wave.significant_wave_height;
        buffer[t].wave_heading = asv.wave.heading * 180.0/PI;
        buffer[t].random_number_seed = asv.wave.random_number_seed;
        buffer[t].time = _time;
        buffer[t].wave_elevation = wave_elevation;
        buffer[t].cog_x = asv.cog_position.x;
        buffer[t].cog_y = asv.cog_position.y;
        buffer[t].cog_z = asv.cog_position.z - (asv.spec.cog.z - asv.spec.T);
        buffer[t].heel = asv.attitude.x * 180.0/PI;
        buffer[t].trim = asv.attitude.y * 180.0/PI;
        buffer[t].heading = asv.attitude.z * 180.0/PI;
        buffer[t].surge_velocity = asv.dynamics.V[surge];
        buffer[t].surge_acceleration = asv.dynamics.A[surge];
  }

  double simulation_time = 0.0;
  clock_t start, end;
  end = clock();
  simulation_time = ((double)(end - start));
  char out_file[10] = "path";
  long rand_seed = 3;
  write_output(out_file, 
               t, 
               wave_ht, 
               wave_heading, 
               rand_seed, 
               _time, 
               simulation_time);






  
//   for (const phen_t* i = ea.archive().data(); i < (ea.archive().data() + ea.archive().num_elements()); ++i) {
//     if (j == 36)
//     ind = *i;
//   }
//   float best = ea.stat<1>().best()->fit().value();
//   std::cout<<"best fit (map_elites):" << best << std::endl;
//   BOOST_CHECK(best > -1e-3);
}