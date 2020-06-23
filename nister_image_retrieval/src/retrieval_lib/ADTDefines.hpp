#pragma once
/*satish.lokkoju@gmail.com*/
/*25/07/2015*/
/*Global Defines/Macros*/

#define ENABLE_MULTITHREADING 1
#define ENABLE_OPENMP 1
#define EVALUATE 1

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#include <omp.h>
#endif

#define UKBENCHLOCATION "/home/satish/datasets/ukbench"
#define UKBENCHLOCATION_DATABASE "/home/satish/datasets/db_ukbench.db"

#define USERS_LOCATION "/home/satish/datasets/ukbench_users"

#define USER0_LOCATION "/home/satish/datasets/ukbench_users/user_0"
#define USER1_LOCATION "/home/satish/datasets/ukbench_users/user_1"
#define USER2_LOCATION "/home/satish/datasets/ukbench_users/user_2"
#define USER3_LOCATION "/home/satish/datasets/ukbench_users/user_3"

#define DEBUG 1

#ifdef DEBUG
//#define INFO( msg )  std::cout << __FILE__ << ":" << __LINE__ << ": " << msg << std::endl;
#define INFO( msg )  std::cout << msg << std::endl;
#else
#define INFO( msg )
#endif
			 
enum RetrievalType
{
	INVERTEDINDEX,
	VOCABULARYTREE
};

/// L-norms for normalization
enum LNorm
{
  L1,
  L2
};

/// Weighting type
enum WeightingType
{
  TF_IDF,
  TF,
  IDF,
  BINARY
};

/// Scoring type
enum ScoringType
{
  L1_NORM,
  L2_NORM,
  CHI_SQUARE,
  KL,
  BHATTACHARYYA,
  DOT_PRODUCT,
};
