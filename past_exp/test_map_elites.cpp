#define VERSION "version_unknown"
#include <iostream>
#include <cmath>
#include <array>

#include <boost/foreach.hpp>
#include <boost/multi_array.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/include/for_each.hpp>

#include <boost/test/unit_test.hpp>

#include <sferes/eval/parallel.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/phen/parameters.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/run.hpp>
#include <sferes/stat/best_fit.hpp>

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

using namespace sferes::gen::evo_float;

struct Params {
  struct ea {
    SFERES_CONST size_t behav_dim = 2;
    SFERES_CONST double epsilon = 0;//0.05;
    SFERES_ARRAY(size_t, behav_shape, 128, 128);
  };
  struct pop {
    // number of initial random points
    SFERES_CONST size_t init_size = 1000;
    // size of a batch
    SFERES_CONST size_t size = 1000;
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

// Rastrigin
FIT_MAP(Rastrigin){
  public :
  template <typename Indiv>
  void eval(Indiv & ind) {
    float f = 10 * ind.size();
    for (size_t i = 0; i < ind.size(); ++i)
    f += ind.data(i) * ind.data(i) - 10 * cos(2 * M_PI * ind.data(i));
    this->_value = -f;

    std::vector<float> data;
    data.push_back(ind.gen().data(0));
    data.push_back(ind.gen().data(1));

    this->set_desc(data);
  }

  bool dead() {
    return false;
  }
};


int main(int argc, char** argv) {
  using namespace sferes;

  typedef Rastrigin<Params> fit_t;
  typedef gen::EvoFloat<10, Params> gen_t;
  typedef phen::Parameters<gen_t, fit_t, Params> phen_t;
  typedef eval::Parallel<Params> eval_t;
  typedef boost::fusion::vector<
        stat::Map<phen_t, Params>
				, stat::BestFit<phen_t, Params>
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
