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
#include "sferes2/sferes/stat/best_fit.hpp"
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
#include<cstdlib>

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
        SFERES_CONST size_t size = 200;
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
        SFERES_CONST size_t nb_inputs = 7;
        SFERES_CONST size_t nb_outputs = 4;

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

        SFERES_CONST init_t init = random_topology;
  };

};
FIT_MAP(ASV_SWARM){
    public :
    template <typename Indiv>
    void eval(Indiv & ind) {

        // struct Asv asv[8];
        // struct Dimensions waypoints[8];
        struct Asv asv[2];
        struct Dimensions waypoint[2];
        std::ofstream f;
        std::string record_file = "runing_record.csv";

        f.open(record_file, std::ios::out | std::ios::in | std::ofstream::trunc);
        f << "cog_x,cog_y\n";

        double wave_ht = 0;
        double wave_heading = 0;
        float time_cost[2];
        float energy_cost[2];
        bool is_simulated[2];
        for(int i = 0; i < 2; i++){
            init(asv[i], wave_ht, wave_heading);    
            time_cost[i] = 0.0;
            energy_cost[i] = 0.0;
            is_simulated[i] = true;
        }
        waypoint[0] = (struct Dimensions){0.0, -100.0, 0.0};
        waypoint[1] = (struct Dimensions){100 * sin(M_PI/4), 100*cos(M_PI/4), 0.0};

        std::vector<float> descriptor;
        std::vector<float> dist1;
        std::vector<float> dist2;   
                        
        int max_time_step = 20000;



        for(int i = 0; i < 2; i++){
            ind.nn().init();
            int t = 0;
            for(; t < max_time_step; t++){
                float angle = compute_angle(asv[i].cog_position.x, asv[i].cog_position.y, waypoint[i].x, waypoint[i].y);
                float distance = compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoint[i].x, waypoint[i].y);
                float normalized_distance = 2 * atan(distance/50.0) / M_PI;
                float dynamic_x = get_normalized_attitude(asv[i].attitude.x);
                float dynamic_y = get_normalized_attitude(asv[i].attitude.y);
                float dynamic_z = get_normalized_attitude(asv[i].attitude.z);
                static const std::vector<float> inputs = {angle, normalized_distance, dynamic_x, dynamic_y, dynamic_z, (float)asv[i].dynamics.V[surge], (float)asv[i].dynamics.A[surge]};
                ind.nn().step(inputs);
                const std::vector<float> &outf = ind.nn().get_outf();

                for(int p = 0; p < 4; p++)
                {
                    asv[i].propellers[p].thrust = 2 * ((1 + outf[p])/2.0); //N
                    asv[i].propellers[p].orientation = (struct Dimensions){0, 0, 0};
                }
                double _time = t * 40;
                asv_compute_dynamics(&asv[i], _time);
                energy_cost[i] += compute_energy_cost(&asv[i]);
                time_cost[i] += 1.0;
                if(i == 0)
                    dist1.push_back(compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoint[i].x, waypoint[i].y));
                else{
                    dist2.push_back(compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoint[i].x, waypoint[i].y));
                    f << asv[i].cog_position.x << "," << asv[i].cog_position.y << "\n";
                }
                         
                if(compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoint[i].x, waypoint[i].y) < 1){
                    is_simulated[i] = false;
                }


            }
        }
        float sum_energy_cost = 0;
        float sum_time = 0;
        float sum_fitness = 0;
        float sum_fitness1 = 0;
        float sum_fitness2 = 0;
        for(int i =0 ; i < 2; i ++){
            sum_energy_cost += (energy_cost[i]/time_cost[i]);
            sum_time += time_cost[i]/max_time_step;
        }
        for(int j = 0; j < dist1.size(); j++)
            sum_fitness1 += dist1[j];
        std::cout << "dist2" << std::endl;
        for(int j = 0; j < dist2.size(); j++){
            sum_fitness2 += dist2[j];
            std::cout << dist2[j] << std::endl;
        }

        descriptor.push_back(sum_energy_cost/2);
        descriptor.push_back(sum_time/2);

        this->_value = -(sum_fitness1 + sum_fitness2)/(dist1.size() + dist2.size());

        std::cout << "first fit: " << sum_fitness1/dist1.size() << ", second fit: " << sum_fitness2/dist2.size() << std::endl;

        

        std::cout<<"Fitness: "<<this->_value<<" " <<", energy cost: "<<descriptor[0] << ", time cost: " << descriptor[1] <<std::endl;
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
        , stat::BestFit<phen_t, Params>
        , stat::MapBinary<phen_t, Params> 
        > stat_t;
  typedef modif::Dummy<> modifier_t;
  typedef ea::MapElites<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;

  ea_t ea;

  ea.load("map_nn_2020-08-13_10_04_57_5210/gen_300");
  std::ofstream ofs1("output");
  ea.show_stat(0, ofs1, 29);
  phen_t ind = *boost::fusion::at_c<0>(ea.stat())._archive[29];
  std::cout << "Fit value: " << ind.fit().value() << std::endl;
  ind.nn().init();
  
  struct Asv asv;
  struct Dimensions waypoint = {100 * sin(M_PI/4), 100*cos(M_PI/4), 0.0};
//   get_random_waypoint(waypoint);
  std::cout << "waypoint x :" << waypoint.x << ", waypoint y: " << waypoint.y << std::endl;
  double wave_ht = 0;
  double wave_heading = 0;

  init(asv, wave_ht, wave_heading);
  std::ofstream f;
  std::string record_file = "record.csv";

  f.open(record_file, std::ios::out | std::ios::in | std::ofstream::trunc);
  f << "cog_x,cog_y\n";
  f << asv.cog_position.x << "," << asv.cog_position.y << "\n";

  int max_time_step = 20000;
        int t = 0;

        for(;t<max_time_step;t++){
            
            float angle = compute_angle(asv.cog_position.x, asv.cog_position.y, waypoint.x, waypoint.y);
            float distance = compute_distance(asv.cog_position.x, asv.cog_position.y, waypoint.x, waypoint.y);
            float normalized_distance = 2 * atan(distance/50.0) / M_PI;
            float dynamic_x = get_normalized_attitude(asv.attitude.x);
            float dynamic_y = get_normalized_attitude(asv.attitude.y);
            float dynamic_z = get_normalized_attitude(asv.attitude.z);
            static const std::vector<float> inputs = {angle, normalized_distance, dynamic_x, dynamic_y, dynamic_z, (float)asv.dynamics.V[surge], (float)asv.dynamics.A[surge]};
            ind.nn().step(inputs);

            const std::vector<float> &outf = ind.nn().get_outf();

            for(int p = 0; p < 4; p++)
            {
                asv.propellers[p].thrust = 2 * ((1 + outf[p])/2.0); //N
                asv.propellers[p].orientation = (struct Dimensions){0, 0, 0};
            }


            double _time = t * 40;
            asv_compute_dynamics(&asv, _time);
            
                    
            float current_distance = compute_distance(asv.cog_position.x, asv.cog_position.y, waypoint.x, waypoint.y);

            if(current_distance <= 1){
                
                f << asv.cog_position.x << "," << asv.cog_position.y << "\n";
                break;
            }
            f << asv.cog_position.x << "," << asv.cog_position.y << "\n";

        }


}