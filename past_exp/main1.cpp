//Algorithm include
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
#include "sferes2/sferes/stat/best_fit.hpp"
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

#ifndef NO_PARALLEL
#include <sferes/eval/parallel.hpp>
#else
#include <sferes/eval/eval.hpp>
#endif

using namespace sferes;
using namespace sferes::gen::dnn;
using namespace sferes::gen::evo_float;

#include "controller/pid_controller.hpp"
#include "controller/before_simulation.hpp"

//ASV-Swarm include
extern "C"{
    #include "asv-swarm/include/asv.h"
    #include "asv-swarm/include/io.h"
}
#include<iostream>
#include <cstdlib>

struct Params {
  struct ea {
    SFERES_CONST size_t behav_dim = 1;
    SFERES_CONST double epsilon = 0;//0.05;
    SFERES_ARRAY(size_t, behav_shape, 10);
  };
  struct pop {
    // number of initial random points
    SFERES_CONST size_t init_size = 2;
    // size of a batch
    SFERES_CONST size_t size = 200;
    SFERES_CONST size_t nb_gen = 1000;
    SFERES_CONST size_t dump_period = 100;
  };


  // minimum and maximum weight/bias
  struct parameters {
      // maximum value of parameters (weights and bias)
      SFERES_CONST float min = -5.0f;
      // minimum value
      SFERES_CONST float max = 5.0f;
  };
  // mutations of the weights and biases
  // these parameters are *per weight*
  // (e.g., 0.1 = 0.1 chance for each each weight)
  struct evo_float {
      SFERES_CONST float mutation_rate = 0.1f;
      SFERES_CONST float cross_rate = 0.1f;
      SFERES_CONST mutation_t mutation_type = polynomial;
      SFERES_CONST cross_over_t cross_over_type = sbx;
      SFERES_CONST float eta_m = 15.0f;
      SFERES_CONST float eta_c = 15.0f;
  };
  
    // neural network
  struct dnn {
      // number of inputs (no need for bias here)
      SFERES_CONST size_t nb_inputs = 2;
      // number of outputs
      SFERES_CONST size_t nb_outputs = 2;
      // minimum number of neurons used for the random generation
      SFERES_CONST size_t min_nb_neurons = 2;
      // maximum number of neurons used for the random generation
      SFERES_CONST size_t max_nb_neurons = 5;
      // minimum number of connections used for the random generation
      SFERES_CONST size_t min_nb_conns = 20;
      // maximum number of connections used for the random generation
      SFERES_CONST size_t max_nb_conns = 25;

      // probability to add a connection between two existing neurons
      SFERES_CONST float m_rate_add_conn = 0.05f;
      // probability to remove one connection
      SFERES_CONST float m_rate_del_conn = 0.04f;
      // probability to change the start or the end of an existing connection
      SFERES_CONST float m_rate_change_conn = 0.1f;
      // probability to add a neuron to on an existing connection
      SFERES_CONST float m_rate_add_neuron = 0.025f;
      // probability to delete a neuron (and its connections)
      SFERES_CONST float m_rate_del_neuron = 0.02;

      // do we evolve the bias of the input/output?
      SFERES_CONST int io_param_evolving = true;

      // we initialize with a feed-forward network
      SFERES_CONST init_t init = ff;
  };
};

// double compute_distance(double x, double y, double initial_x, double initial_y){
//   double x2 = pow(x-initial_x, 2);
//   double y2 = pow(y-initial_y, 2);
//   return sqrt(x2+y2);
// };

// double compute_angle(double x, double y, double initial_x, double initial_y){
//   double dx = initial_x - x;
//   double dy = initial_y - y;
//   if(dx <= 0){
//     return acos(dy/(sqrt(dx*dx+dy*dy) * sqrt(1)));
//   }else{
//     return -acos(dy/(sqrt(dx*dx+dy*dy) * sqrt(1)));
//   }
// };

float normalize_distance(double distance, bool is_disc){
  if(is_disc){
      return exp(-distance);
  }else{ 
      return -1.0 + 2*exp(-distance);
  }
};

float normalize_angle(double angle, bool is_disc){
  if(is_disc)
    return (angle+M_PI)/(2*M_PI);
  else{
    return angle/M_PI;
  }
};

struct Dimensions set_random_waypoint(struct Dimensions center, double distance ){
    srand(time(NULL));
    rand();
    double random_angle = (double)rand()/RAND_MAX * (2.0*M_PI);
    double random_dist =  (double)rand()/RAND_MAX * distance;
    double random_x = sin(random_angle) * random_dist + center.x;
    double random_y = cos(random_angle) * random_dist + center.y;
    return (struct Dimensions){random_x, random_y, 0};
}


// Rastrigin
FIT_MAP(Rastrigin){
  public :
  template <typename Indiv>
  void eval(Indiv & ind) {


    // struct pid_controller pc;
    // pid_controller_init(&pc);

    ind.nn().init();

    //simulate and get result
    struct Asv asv[16];
    struct Waypoints useless;
    char in_file[50] = "../asv-swarm/example_input.toml";
    for(int j = 0; j < 16; j++){
      set_input(in_file, &asv[j], &useless);
      asv_init(&asv[j]);
    }
  
    //random wave
    // srand(time(NULL));
    // rand();
    // wave_ht = (double)rand()/RAND_MAX * 5;
    // wave_heading = (double)rand()/RAND_MAX * 360;


    // printf("waypoints: %f, %f\n", waypoint.x, waypoint.y);


    //parameters
    int max_time_step = 5000;
    int max_distance = 50;
    int t = 0;
    double initial_x = 100.0;
    double initial_y = 100.0;

    struct Dimensions waypoints[16] = {{initial_x-20, initial_y, 0}
                                    ,{initial_x-40, initial_y, 0}
                                    ,{initial_x-20, initial_y-20, 0}
                                    ,{initial_x-40, initial_y-40, 0}
                                    ,{initial_x, initial_y-20, 0}
                                    ,{initial_x, initial_y-40, 0}
                                    ,{initial_x+20, initial_y-20, 0}
                                    ,{initial_x+40, initial_y-40, 0}
                                    ,{initial_x+20, initial_y, 0}
                                    ,{initial_x+40, initial_y, 0}
                                    ,{initial_x+20, initial_y+20, 0}
                                    ,{initial_x+40, initial_y+40, 0}
                                    ,{initial_x, initial_y+20, 0}
                                    ,{initial_x, initial_y+40, 0}
                                    ,{initial_x-20, initial_y+20, 0}
                                    ,{initial_x-40, initial_y+40, 0}
                                    };

    // for(int j = 0; j < 16; j++)
    //   waypoints[j] = set_random_waypoint(conters[j], 5);


    std::vector<float> descriptors;

    float f = 0.0;
    // float energy_cost = 0.0;
    float mean_time_cost = 0.0;
    int time_cost[16];
    bool simulating[16];
    for(int i = 0; i < 16; i++){
      simulating[i] = true;
      time_cost[i] = 0;
    }
    for(;t < max_time_step; t++){
      // if(compute_distance(asv.cog_position.x, asv.cog_position.y, initial_x, initial_y) >= max_distance){
      //   this->_value = (float)(max_time_step - t)/max_time_step;
      //   this->_objs.resize(1);
      //   this->_objs[0] = (float)(max_time_step - t)/max_time_step;
      //   data.push_back(normalize_angle(compute_angle(asv.cog_position.x, asv.cog_position.y, initial_x, initial_y), true));
      //   data.push_back(normalize_distance(compute_distance(asv.cog_position.x, asv.cog_position.y, initial_x, initial_y), max_distance, true));
      //   break;
      // }
      // if(t >= max_time_step){
      //   this->_value = (float)(max_time_step - t)/max_time_step;
      //   this->_objs.resize(1);
      //   this->_objs[0] = (float)(max_time_step - t)/max_time_step;
      //   data.push_back(normalize_angle(compute_angle(asv.cog_position.x, asv.cog_position.y, initial_x, initial_y), true));
      //   data.push_back(normalize_distance(compute_distance(asv.cog_position.x, asv.cog_position.y, initial_x, initial_y), max_distance, true));
      //   break;
      // }

      for(int i = 0; i < 16; i ++){
        if(simulating[i]){
          float angle = normalize_angle(compute_angle(asv[i].cog_position.x, asv[i].cog_position.y, waypoints[i].x, waypoints[i].y), false);
          float distance = normalize_distance(compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoints[i].x, waypoints[i].y), false);
          static const std::vector<float> inputs = {angle, distance};
          for (int j = 0; j < ind.gen().get_depth() + 1; ++j)
            ind.nn().step(inputs);
          const std::vector<float> &outf = ind.nn().get_outf();
          for(int p = 0; p < 4; p++)
          {
            asv[i].propellers[p].thrust = 5.0 * ((1 + outf[0])/2.0); //N
            asv[i].propellers[p].orientation = (struct Dimensions){0, 0, outf[1] * M_PI};
          }
          double _time = t * 40;
          asv_compute_dynamics(&asv[i], _time);
          time_cost[i] += 1;
        }
        if(compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoints[i].x, waypoints[i].y) <= 1)
          simulating[i] = false;
      }
      // f += normalize_distance(compute_distance(asv.cog_position.x, asv.cog_position.y, waypoint.x, waypoint.y), true)/100;
      // float temp_energy_cost = 0.0;
      // for(int p = 0; p < 4; p++)
        // temp_energy_cost += (asv.propellers[p].thrust/5.0);
      // energy_cost += (temp_energy_cost/4.0);

      // if(!is_arrived && compute_distance(asv.cog_position.x, asv.cog_position.y, waypoint.x, waypoint.y) < 5 ){
        // is_arrived = true;
        // descriptors.push_back((float)t/max_time_step);
      // }
    }
    // float f = 10 * ind.size();
    // for (size_t i = 0; i < ind.size(); ++i)
    // f += ind.data(i) * ind.data(i) - 10 * cos(2 * M_PI * ind.data(i));
    // this->_value = -f;

    // std::vector<float> data;
    // data.push_back(ind.gen().data(0));
    // data.push_back(ind.gen().data(1));
    int sum_cost = 0;
    for(int i =0; i < 16; i++){
      if(compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoints[i].x, waypoints[i].y) < 1)
        f += 1.0;
      sum_cost += time_cost[i];
    }
    std::cout << f << std::endl;
    mean_time_cost = (float)sum_cost / 16;
    descriptors.push_back(mean_time_cost/max_time_step);

    this->_value = f;
    this->_objs.resize(1);
    this->_objs[0] = f;

    // if (descriptors.size() == 0)
    // descriptors.push_back(1.0);

    // descriptors.push_back(energy_cost/max_time_step);

    // std::cout<<descriptors[0]<<" "<<descriptors[1]<<std::endl;
    this->set_desc(descriptors);
  }

  bool dead() {
    return false;
  }
};


int main(int argc, char** argv){
  using namespace sferes;

  typedef Rastrigin<Params> fit_t;

  // NEURAL NETWORK CONFIGURATION
  // type of the weights (no need for fitness here)
  typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> weight_t;
  // type of the bias (you can use a different Params class here if needed)
  typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> bias_t;
  // the potential function is the weighted sum of inputs
  typedef nn::PfWSum<weight_t> pf_t;
  // the activation functions is tanh(x)
  typedef nn::AfTanh<bias_t> af_t;
  // our neuron type (standard)
  typedef nn::Neuron<pf_t, af_t> neuron_t;
  // our connection type (standard)
  typedef nn::Connection<weight_t> connection_t;
  // for a feed-forward neural network (no recurrence)
  typedef gen::DnnFF<neuron_t, connection_t, Params> gen_t;
  // typedef sferes::gen::Dnn<neuron_t,  connection_t, Params> gen_t;
  // phenotype (developped neural network) -- we need the fitness here
  typedef phen::Dnn<gen_t, fit_t, Params> phen_t;


  // typedef gen::EvoFloat<10, Params> gen_t;
  // typedef phen::Parameters<gen_t, fit_t, Params> phen_t;
  typedef eval::Parallel<Params> eval_t;
  typedef boost::fusion::vector<
        stat::Map<phen_t, Params>
				, stat::BestFit<phen_t, Params>
        // , stat::MeanFit<phen_t, Params>
        , stat::MapBinary<phen_t, Params> 
        > stat_t;
  typedef modif::Dummy<> modifier_t;
  typedef ea::MapElites<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;

  ea_t ea;
  run_ea(argc, argv, ea);
  float best = ea.stat<1>().best()->fit().value();
  std::cout<<"best fit (map_elites):" << best << std::endl;
  BOOST_CHECK(best > -1e-3);
}


// int main(){
//     struct Asv asv;
//     struct Waypoints waypoints;
//     char in_file[50] = "../asv-swarm/example_input.toml";
//     set_input(in_file, &asv, &waypoints);
//     std::cout << "hello, world" << std::endl;
//     return 1;
// }
