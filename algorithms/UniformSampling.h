// Instance.h
#ifndef UNIFORMSAMPLING_H
#define UNIFORMSAMPLING_H

#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include "Instance.h"
#include "utils.h"
#include<tuple>

using namespace std;

class UniformSampling {

public:
    
    //the solution (construction)
    string solution=""; // the final LFCSP solution
    string a_prime=""; // the coresponding A' solution (all with '*' mark s)
    double runtime=0.0;
    
    vector<int> M_with_A_matchings; // mathcings between A and M to make an A'
    vector<pair<int, int>> A_with_B_matching; // pairs of matched between A and B for the resulting solution
    Instance* instance; //instance to be solved via this algorithm

    //constructor(s):
    UniformSampling(Instance* in);
    // Destructor
    ~UniformSampling();
    //execution (the main function to run the algorithm):
    void run(int number_solution_generated, string out_path, const set<int>& sample_fix);
    void save(string& path);
    void print();
private: 
   
   std::tuple<int, string, string, vector<pair<int, int>>> sample(const set<int>& sampled_fix);
   
};
 #endif
