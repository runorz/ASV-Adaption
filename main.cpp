#define USE_LIBCMAES true

#include <iostream>

// you can also include <limbo/limbo.hpp> but it will slow down the compilation
#include <limbo/bayes_opt/boptimizer.hpp>
#include "controller/linear_controller.hpp"

using namespace limbo;

struct Params {
    struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
    };

// depending on which internal optimizer we use, we need to import different parameters
#ifdef USE_NLOPT
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
    };
#elif defined(USE_LIBCMAES)
    struct opt_cmaes : public defaults::opt_cmaes {
    };
#else
    struct opt_gridsearch : public defaults::opt_gridsearch {
    };
#endif

    // enable / disable the writing of the result files
    struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
        BO_PARAM(int, stats_enabled, true);
    };

    // no noise
    struct kernel : public defaults::kernel {
        BO_PARAM(double, noise, 1e-10);
    };

    struct kernel_maternfivehalves : public defaults::kernel_maternfivehalves {
    };

    // we use 10 random samples to initialize the algorithm
    struct init_randomsampling {
        BO_PARAM(int, samples, 10);
    };

    // we stop after 40 iterations
    struct stop_maxiterations {
        BO_PARAM(int, iterations, 10000);
    };

    // we use the default parameters for acqui_ucb
    struct acqui_ucb : public defaults::acqui_ucb {
    };
};

struct Eval {
    // number of input dimension (x.size())
    BO_PARAM(size_t, dim_in, 6);
    // number of dimensions of the result (res.size())
    BO_PARAM(size_t, dim_out, 1);

    // the function to be optimized
    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        struct Asv asv[16];
        struct linear_controller lc[16];
        struct Dimensions waypoints[16];
        bool simulating[16];
        float last_distance[16];
        float distance_update[16];
        float time_cost[16];


        double wave_ht = 0.0;
        double wave_heading = 0.0;
        set_asv_and_linear_controller(asv, lc, waypoints, 100, 100, wave_ht, wave_heading);

        for(int i = 0; i < 16; i++){
            asv_init(&asv[i]);
            set_weights_and_bias(&lc[i], x(0), x(1), x(2), x(3), x(4), x(5));
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
                    step(&lc[i], &asv[i]);
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

        return tools::make_vector(sum_average_distance_update / 16);
    }
};


int main()
{
    // we use the default acquisition function / model / stat / etc.
    bayes_opt::BOptimizer<Params> boptimizer;
    // run the evaluation
    boptimizer.optimize(Eval());
    // the best sample found
    std::cout << "Best sample: " << boptimizer.best_sample()(0) << " - Best observation: " << boptimizer.best_observation()(0) << std::endl;
    return 0;
}
