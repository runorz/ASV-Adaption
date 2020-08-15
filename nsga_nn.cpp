#include <algorithm>
#include <cmath>
#include <iostream>

#include <sferes/parallel.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/phen/parameters.hpp>
#include <sferes/modif/diversity.hpp>
#include <sferes/ea/nsga2.hpp>
#include <sferes/stat/pareto_front.hpp>
#include <sferes/stat/best_fit.hpp>
#include <sferes/eval/parallel.hpp>
#include <sferes/run.hpp>

#include <modules/nn2/gen_dnn.hpp>
#include <modules/nn2/gen_dnn_ff.hpp>
#include <modules/nn2/phen_dnn.hpp>

using namespace sferes;
using namespace sferes::gen::dnn;
using namespace sferes::gen::evo_float;

extern "C"{
    #include "asv-swarm/include/asv.h"
    #include "asv-swarm/include/io.h"
}

struct Params {
    // NEURAL NETWORK PARAMETERS

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
    // minimum and maximum weight/bias
    struct parameters {
        // maximum value of parameters (weights and bias)
        SFERES_CONST float min = -5.0f;
        // minimum value
        SFERES_CONST float max = 5.0f;
    };
    // neural network
    struct dnn {
        // number of inputs (no need for bias here)
        SFERES_CONST size_t nb_inputs = 2;
        // number of outputs
        SFERES_CONST size_t nb_outputs = 1;
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

    // NSGA-2PARAMETERS
    struct pop {
        // population size
        SFERES_CONST unsigned size = 200;
        // number of generation
        SFERES_CONST unsigned nb_gen = 500;
        // period to write intermediate results
        SFERES_CONST int dump_period = 100;
        // multiplier for initial population size
        SFERES_CONST int initial_aleat = 2;
    };
};

// clang-format off
// FITNESS function
SFERES_FITNESS(FitXOR, sferes::fit::Fitness)
{
public:
    FitXOR() {}
    template <typename Indiv> 
    void eval(Indiv& indiv)
    {
        // we need two objectives: one for the fitness, one for the diversity score
        this->_objs.resize(2);
        _behavior.resize(4);

        float fitness = 0;
        static const std::vector<std::vector<float>> inputs = {{-1, 1,}, {-1, -1}, {1, -1,}, {1, 1}};
        static const std::vector<float> outputs = {-1, 1, -1, 1};
        
        int n_waypoints = 4;

        // struct Asv asv[8];
        // struct Dimensions waypoints[8];
        struct Asv asv[n_waypoints];
        struct Dimensions waypoint[n_waypoints];

        double wave_ht = 0;
        double wave_heading = 0;
        float time_cost[n_waypoints];
        float energy_cost[n_waypoints];
        float dist[n_waypoints];
        for(int i = 0; i < n_waypoints; i++){
            init(asv[i], wave_ht, wave_heading);    
            time_cost[i] = 0.0;
            energy_cost[i] = 0.0;
            dist[i] = 0.0;
        }

        waypoint[0] = (struct Dimensions){0.0, 20.0, 0.0};
        waypoint[1] = (struct Dimensions){20 * sin(M_PI/4), 20*cos(M_PI/4), 0.0};
        waypoint[2] = (struct Dimensions){-20 * sin(M_PI/4), -20*cos(M_PI/4), 0.0};
        waypoint[3] = (struct Dimensions){-20.0, 0.0, 0.0};

        std::vector<float> descriptor;

        int max_time_step = 3000;

        for(int i = 0; i < n_waypoints; i++){
            int t = 0;
            ind.nn().init();
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
                    dist[i] += compute_distance(asv[i].cog_position.x, asv[i].cog_position.y, waypoint[i].x, waypoint[i].y);
                         
                    if(dist[i] < 1){
                        break;
                    }
            }
        }


        float sum_energy_cost = 0;
        float sum_time = 0;
        float sum_fitness = 0;
        float sum_dist = 0;
        for(int i =0 ; i < n_waypoints; i ++){
            sum_energy_cost += (energy_cost[i]/time_cost[i]);
            sum_time += time_cost[i]/max_time_step;
            sum_dist += dist[i]/time_cost[i];
        }

        this->_objs[0] = -sum_dist/n_waypoints;
        this->_value = -sum_dist/n_waypoints;
    }

    // behavioral distance for the behavioral diversity modifier
    template<typename Indiv>
    float dist(const Indiv& ind) const {
        assert(_behavior.size() == 4);
        double d = 0;
        for (size_t i = 0; i < _behavior.size(); ++i)
            d += std::powf(_behavior[i] - ind.fit()._behavior[i], 2.0);
        return d;
    }
private:
    // store the behavior
    std::vector<float> _behavior;
};
// clang-format on

int main(int argc, char** argv)
{   
    // FITNESS FUNCTION
    typedef FitXOR<Params> fit_t;

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
    // for a neural network with possible recurrences
    //typedef sferes::gen::Dnn<neuron_t,  connection_t, Params> gen_t;
    // phenotype (developped neural network) -- we need the fitness here
    typedef phen::Dnn<gen_t, fit_t, Params> phen_t;

    // EVOLUTIONARY ALGORITHM CONFIGURATION
    // parallel evaluator
    typedef eval::Parallel<Params> eval_t;
    // list of statistics
    typedef boost::fusion::vector<stat::BestFit<phen_t, Params>, stat::ParetoFront<phen_t, Params>> stat_t;
    // we use a behavioral diversity modifier
    typedef modif::Diversity<phen_t> modifier_t;
    typedef ea::Nsga2<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;
    ea_t ea;

    // RUN THE ALGORITHM
    run_ea(argc, argv, ea);

    return 0;
}