cmake_minimum_required(VERSION 3.8.2)

project(ASV-adaption VERSION 1.0)


INCLUDE_DIRECTORIES(
  asv-swarm/include
  asv-swarm/dependency/tomlc99
  sferes2
  limbo/src
  )

SET( SOURCE
asv-swarm/dependency/tomlc99/toml.c
asv-swarm/source/io.c
asv-swarm/source/regular_wave.c
asv-swarm/source/wave.c
asv-swarm/source/asv.c
sferes2/sferes/dbg/dbg.cpp
sferes2/sferes/ea/cmaes.cpp
test.cpp
)

SET( SOURCE2
asv-swarm/dependency/tomlc99/toml.c
asv-swarm/source/io.c
asv-swarm/source/regular_wave.c
asv-swarm/source/wave.c
asv-swarm/source/asv.c
sferes2/sferes/dbg/dbg.cpp
sferes2/sferes/ea/cmaes.cpp
nsga_nn.cpp
)

SET(SOURCE3
test_mlpack.cpp)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

SET(Boost_INCLUDE_DIRS /local/software/boost/1.67.0/include)
SET(Boost_LIBRARIES /local/software/boost/1.67.0/lib)
SET(tbb_INCLUDE_DIRS /local/software/intel/2018.4/tbb/include)
SET(TBB_LIBRARIES /local/software/intel/2018.4/tbb/lib/intel64/gcc4.7)
SET(EIGEN3_INCLUDE_DIR /local/software/eigen/3.2.10/include)
SET(EIGEN3_LIBRARIES /local/software/eigen/3.2.10)
SET(NLOPT_INCLUDE_DIR /lyceum/rz2u19/nlopt/include)
SET(NLOPT_LIBRARY /lyceum/rz2u19/nlopt/lib)
SET(LIBCMAES /lyceum/rz2u19/libcmaes)
SET(LIBCMAES_LIBRARY /lyceum/rz2u19/libcmaes/src)

SET(MLPACK_LIB /lyceum/rz2u19/mlpack-3.2.2/build/lib)
SET(MLPACK_INCLUDE /lyceum/rz2u19/mlpack-3.2.2/build/include)

SET(ARMADILLO_INCLUDE /lyceum/rz2u19/armadillo-9.900.2/include)
SET(ARMADILLO_INCLUDES /lyceum/rz2u19/armadillo-9.900.2/include/armadillo_bits)
SET(ARMADILLO_LIB /lyceum/rz2u19/armadillo-9.900.2/)

SET(MKL_LIB /local/software/intel/2018.4/mkl/lib/intel64/)

SET(OPENMPI_LIB /local/software/openmpi/4.0.4/gcc/lib)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

INCLUDE_DIRECTORIES (${EIGEN3_INCLUDE_DIR})

INCLUDE_DIRECTORIES(${tbb_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(${NLOPT_INCLUDE_DIR})

INCLUDE_DIRECTORIES(${LIBCMAES})

INCLUDE_DIRECTORIES(${MLPACK_INCLUDE})

INCLUDE_DIRECTORIES(${ARMADILLO_INCLUDE})
INCLUDE_DIRECTORIES(${ARMADILLO_INCLUDES})


add_executable(ASV-adaption ${SOURCE})
SET_PROPERTY(TARGET ASV-adaption PROPERTY CXX_STANDARD 11)
add_executable(extract_binary_map map_elites/extract_binary_map.cpp)

add_executable(map_nn ${SOURCE2})
SET_PROPERTY(TARGET map_nn PROPERTY CXX_STANDARD 11)

add_executable(mlpack ${SOURCE3})
SET_PROPERTY(TARGET mlpack PROPERTY CXX_STANDARD 11)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(mlpack Threads::Threads)
target_link_libraries(mlpack -fopenmp)
target_link_libraries(mlpack -lpthread)


file(GLOB files "${MLPACK_LIB}/*.so")
foreach(f ${files})
	TARGET_LINK_LIBRARIES(mlpack ${f})
endforeach()

file(GLOB files "${ARMADILLO_LIB}/*.so")
foreach(f ${files})
	TARGET_LINK_LIBRARIES(mlpack ${f})
endforeach()

file(GLOB files "${MKL_LIB}/*.so")
foreach(f ${files})
	TARGET_LINK_LIBRARIES(mlpack ${f})
endforeach()

file(GLOB files "${TBB_LIBRARIES}/*.so")
foreach(f ${files})
	TARGET_LINK_LIBRARIES(mlpack ${f})
endforeach()

file(GLOB files "${OPENMPI_LIB}/*.so")
foreach(f ${files})
	TARGET_LINK_LIBRARIES(mlpack ${f})
endforeach()

file(GLOB files "${Boost_LIBRARIES}/*.a")
foreach(f ${files})
	TARGET_LINK_LIBRARIES(ASV-adaption ${f})
endforeach()

file(GLOB files "${Boost_LIBRARIES}/*.a")
foreach(f ${files})
	TARGET_LINK_LIBRARIES(map_nn ${f})
endforeach()

file(GLOB files "${Boost_LIBRARIES}/*.a")
foreach(f ${files})
	TARGET_LINK_LIBRARIES(extract_binary_map ${f})
endforeach()


file(GLOB files "${TBB_LIBRARIES}/*.so")
foreach(f ${files})
        TARGET_LINK_LIBRARIES(ASV-adaption ${f})
endforeach()

file(GLOB files "${TBB_LIBRARIES}/*.so")
foreach(f ${files})
        TARGET_LINK_LIBRARIES(map_nn ${f})
endforeach()

file(GLOB files "${NLOPT_LIBRARY}/*.a")
foreach(f ${files})
        TARGET_LINK_LIBRARIES(ASV-adaption ${f})
endforeach()

file(GLOB files "${LIBCMAES_LIBRARY}/*.o")
foreach(f ${files})
        TARGET_LINK_LIBRARIES(ASV-adaption ${f})
endforeach()


