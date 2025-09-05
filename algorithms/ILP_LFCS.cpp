#include<limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include<sys/times.h>
#include<unistd.h>
#include <time.h>
#include <ctime>
#include <vector>
#include <set>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "Instance.h"
#include "UniformSampling.h"
#include "utils.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <ilcplex/ilocplex.h>
#include <random>
#include <tuple>
#include <iterator>  

// ./tune-main.sh parameter-files/param instances-lists/instances.txt 5000 ==> Run Irace
 

using namespace std;

ILOSTLBEGIN

/** Declare global vars  **/  
int ilp_algorithm=0; // Basic ILP 
double allowed_time_limit=10.0;
int solution=0;
string File_Name="";
string File_Name_Out="";
double time_limit=10.0;
int t_lim_sub=20.0;
double percentage=0.0; //fully greedy (initialized)
double num_solutions=5; 
int age_max_val=10;//age_x and age_y
int age_max_y=3;
int seed=10;

/** ADAPT-CMSA params: **/
double alpha_lb=0.6;
double alpha_ub=1.0;
int t_limit_ILP=10;
float t_prop=0.5;
float alpha_red=0.2;

        
// Глобални генератор
std::mt19937 gen; 

/** End of global vars declaration **/

void set_global_seed(unsigned int seed) {

    srand(seed);    // Фиксирање за rand()
    gen.seed(seed); // Фиксирање за std::mt19937
}

/** Custom hash function for std::pair<int, int> (required for X vars) **/
struct PairHash {

    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto hash1 = std::hash<T1>{}(p.first);  // Hash for the first element
        auto hash2 = std::hash<T2>{}(p.second); // Hash for the second element
        return hash1 ^ (hash2 << 1); // Combine the hashes using XOR and shift
    }
};
 
// Custom Callback Class for Tracking Best Solution  IncumbentCallbackI 
class BestSolutionCallback : public IloCplex::IncumbentCallbackI {
    private:
        IloNum best_known;  // Best known solution
    
    public:
        // Constructor
        BestSolutionCallback(IloEnv env, int best_known_val) : IloCplex::IncumbentCallbackI(env), best_known(best_known_val) {}//IloInfinity
    
        // Override main() to handle incumbent solutions
        void main() override {
            IloNum objval = getObjValue();  // Get current best solution
    
            if (objval < best_known) {  // If a better solution is found
                std::cout << "New best solution found: " << objval << std::endl;
                best_known = objval;
                abort();  // Stop solving as we found a better solution
            }
        }

    // Clone function required for callbacks
    IloCplex::CallbackI* duplicateCallback() const override {
        return new (getEnv()) BestSolutionCallback(getEnv(), best_known);
    }
};

/** Greedy approach for the LCS problem **/
std::tuple<int, int, float> greedy_value(const string&A, const string&B, int current_pos_in_A, int current_pos_in_B, char letter_to_add)
{

    int next_math_A_letter_to_add=-1;
    int next_math_B_letter_to_add=-1;
    
    float result=std::numeric_limits<double>::max();  
    
    for(int i=current_pos_in_A; i < A.size(); ++i)
    {
        if(letter_to_add == A[i])
        {
            next_math_A_letter_to_add=i;
            break;
        }
    }    
    // the same for B:
    for(int j=current_pos_in_B; j < B.size(); ++j)
    {
        if(letter_to_add == B[j])
        {
            next_math_B_letter_to_add=j;
            break;
        }
    }
    //
    if(next_math_B_letter_to_add>=0 and next_math_A_letter_to_add>=0)
    {
          result = (next_math_A_letter_to_add - current_pos_in_A + 1.0) / (A.size() - current_pos_in_A)  + (next_math_B_letter_to_add - current_pos_in_B+1.0) / (B.size() - current_pos_in_B);      

    }
    
    return {next_math_A_letter_to_add, next_math_B_letter_to_add, result};
}



//tuple<int, int float>: coordinate1: pos_x, coordinate2: pos_y, coordinate_3: greedy_val 
std::tuple<int, int, float> randomization_decisions(std::vector<std::tuple<int, int, float>>& decisions, double percentage)
{
    // Sort the vector by the third element (float)   
    std::sort(decisions.begin(), decisions.end(), [](std::tuple<int, int, float>& a, std::tuple<int, int, float>& b) {
        return std::get<2>(a) < std::get<2>(b); // Compare the third elements
    });
    // randomize (take the first @percentage % best decisions (to be randomized)
    int k = std::max(1, int( (int)decisions.size() * percentage));
    // Create a random device and seed
    
    //std::random_device rd; 
    //std::mt19937 gen(rd()); // Mersenne Twister random number generator
    
    std::uniform_int_distribution<> dist(1, k); // Uniform distribution between 0 and k
    // Generate a random integer
    int random_number = dist(gen);
    return decisions[random_number-1];
} 

std::tuple<string, std::vector<pair<int, int>>> greedy_LCS_approach(const string& A, const string& B,  double percentage_greediness)
{
    
    //store all letters:
    set<char> letters;
    for(char c: A)
        letters.insert(c);
    for(char c: B)
        letters.insert(c);
    
    string solution="";
    int current_pos_A=0;
    int current_pos_B=0;
    
    vector<pair<int, int>> components;
    
    while(current_pos_A < A.size() and current_pos_B < B.size())
    {
        std::vector<std::tuple<int, int, float>> feasible_decisions; // save the extensions 
        for(char c: letters)
        {
            auto [pos_c_A, pos_c_B, value] = greedy_value(A, B, current_pos_A, current_pos_B, c);
            
            if(value < std::numeric_limits<double>::max()) // feasible decision as a greedy value is assigned
               feasible_decisions.push_back({pos_c_A, pos_c_B, value});
        }
        
        if(feasible_decisions.size() == 0)   // no feasible extension
            return { solution, components };
        
        std::tuple<int, int, float> selected_decision = randomization_decisions(feasible_decisions, percentage_greediness);
        // there is at least one feasible solution:
        current_pos_A = std::get<0>(selected_decision) + 1;  //pointer_x+1;
        current_pos_B = std::get<1>(selected_decision) + 1;  //pointer_y+1;
        solution += A[current_pos_A-1];//best_char;
        //add to the components:
        assert(current_pos_A-1 >= 0 and current_pos_B-1 >= 0);
        components.push_back({ current_pos_A-1, current_pos_B-1 });
    }
    return { solution, components };
}

/** CMSA CONSTRUCTION: **/
std::set<pair<int, int>> construct_phase(Instance* instance, int number_solutions, double percentage)
{
       int n_sol=0;
       std::set<std::pair<int, int>> union_all_components;
       
       while(n_sol < number_solutions)
       {
           auto [greedy_sol, components] = greedy_LCS_approach(instance->get_A(), instance->get_B(), percentage);  
           for(auto& pair_comp: components){
               union_all_components.insert(pair_comp); 
           }   
           n_sol++;
       }
       return union_all_components;
}


std::tuple<int, int, double> cplex_solve(IloCplex& cplex)
{
    
    int sol = 0; int status=0; double gap=0.0;
    // Solve the model
    if (cplex.solve()) {// Optimaly??
        
        sol = cplex.getObjValue();
        status=2; gap=0.0; // optimal solution reached
 
    }else{ 
         // Check the status for feasibility
        IloAlgorithm::Status status = cplex.getStatus();    
        if (status == IloCplex::Feasible || status == IloCplex::Optimal) {  // if only a feasible solution has reached
        
            std::cout << "Feasible solution found (not optimal)." << std::endl;
            sol = cplex.getObjValue();
            status=1; // fasible solution foung
            gap = cplex.getMIPRelativeGap();
          
       }else // no any feasible solution is found
            
            std::cerr << "No solution found." << std::endl;
    }
    return {sol, status, gap};
}

// TRUE: if the pair of matchings is in conflict
bool conflict(std::pair<int, int> pair1, std::pair<int, int> pair2) {
    return (pair1.first <= pair2.first && pair1.second >= pair2.second) ||
           (pair1.first >= pair2.first && pair1.second <= pair2.second);
}

string merge_x_and_y_components_into_solution(Instance* instance, vector<int>& lcs_A_B_matched, vector<int>& deleted_positions){     
      //Merge two vectors, sort afterwards 
      string solution_string="";
      lcs_A_B_matched.insert(lcs_A_B_matched.end(),  deleted_positions.begin(), deleted_positions.end());
      std:sort(lcs_A_B_matched.begin(), lcs_A_B_matched.end());
      //extract solution:
      for(auto pos:lcs_A_B_matched)
          solution_string += instance->get_A()[pos];
       
      return solution_string;
}

/** solve a subproblem of the original instance w.r.t. the components C **/
std::tuple<int, float, string, vector<pair<int, int>>> cmsa_solve_phase(Instance* instance,  int time_limit_subinstance,
                                                                        std::set<pair<int, int>>& C, string& out_file)
{
      //useful data structures to read the instances:
      unordered_map<int, vector<int>> C_first_to_second; // C_i
      unordered_map<int, vector<int>> C_second_to_first; // C^j
       
      for(auto& first_second: C)
      {
          if(C_first_to_second.find(first_second.first) == C_first_to_second.end()){
          
              C_first_to_second[first_second.first] = vector<int>();
              C_first_to_second[first_second.first].push_back(first_second.second);
              
          }else
              C_first_to_second[first_second.first].push_back(first_second.second);
          // the same for C_second_to_first:
          if(C_second_to_first.find(first_second.second) == C_second_to_first.end()){
          
              C_second_to_first[first_second.second] = vector<int>();
              C_second_to_first[first_second.second].push_back(first_second.first);         
          }
          else
              C_second_to_first[first_second.second].push_back(first_second.first);         
      }

      // create the (sub-)model:
      try { 
         
         auto start = std::chrono::high_resolution_clock::now();
         // Create CPLEX environment and model
         IloEnv env;
         IloModel model(env); 
         
         int rows = instance->get_A().size();
         int cols = instance->get_B().size();
         string solution_string="";
        // Hash map to store pairs of matching positions and binary variables
        std::unordered_map<std::pair<int, int>, IloBoolVar, PairHash> x; // from C
        
        for(auto& p: C)
        {
            IloBoolVar x_p(env);
            x[p] = x_p;
        }
        IloNumVarArray y(env, rows, 0, 1, ILOINT); // Binary variables y_{i}
        // declare constraints:
        IloRangeArray constraints(env);
        
        for(auto p: C)
        {
            for( auto q: C)
            {
                if(p.first <= q.first) // break symmetries (a, b) and (b, a) are equal
                {    
                    if(p.second != q.second) // p!=q
                        if(conflict(p, q))
                            constraints.add(x[p] + x[q] <= 1);
                }
            }
        }
        // Add constraint: sum(x[i][j]) <= 1 for all j
        for(int j=0; j<cols; ++j)
        {
           IloExpr expr(env);
           for(auto& i: C_second_to_first[j])
               expr += x[{i, j}];
           
           constraints.add(expr <= 1);
           expr.end();
        }
        //Add constraint: y[i] + sum(x[i][j]) <= 1 for all i
        for(int i=0; i<rows; ++i)
        {
            IloExpr expr(env);
            expr += y[i];
            for(auto& j: C_first_to_second[i])
                expr += x[{i, j}];
            
            constraints.add(expr <= 1);
            expr.end();
        }
        //CONSTRAINTS: sum(y[i] where A[i] == sigma) <= M_sigma[sigma] 
        for(auto sigma: instance->sigmas)
        {   
        
            IloExpr expr(env); bool one_match_at_least=false;
            for (int i = 0; i < rows; ++i) {
                if (instance->get_A()[i] == sigma ){
                    expr += y[i];
                    one_match_at_least=true;
                }  
            }
            if(one_match_at_least)
            {   
                 if(instance->M_sigma.find(sigma) != instance->M_sigma.end())
                    constraints.add(expr <= instance->M_sigma[sigma]);
                 else
                    constraints.add(expr == 0);   
            }
            expr.end();
        }
        // Set up the objective function: Maximize sum(y[i]) + sum(x[{i, j}]
        IloExpr obj_expr(env);
        for (int i = 0; i < rows; ++i) {
            obj_expr += y[i];
        }
        for(auto& p: C) 
            obj_expr+=x[ p ];
        // set up to a maximization problem:   
        IloObjective obj = IloMaximize(env, obj_expr);
        
        //Add @constraints and @obj into the model, pass it to CPLEX
        model.add(obj);
        obj_expr.end();
        //Add constraints to the model
        model.add(constraints);
        
        //Create CPLEX solver and configure time limit
        IloCplex cplex(model);
        cplex.setParam(IloCplex::TiLim, time_limit_subinstance);
        cplex.setOut(env.getNullStream());
        
        //Solve the model:
        auto [sol, status, gap] = cplex_solve(cplex);// status => 0: NO SOLUTION, 1: FASIBLE, 2: OPTIMAL
        vector<pair<int, int>> components_of_opt_sol;

        if(status==1 or status==2) //feasible or optimal solution found
        {
          
            // Extract values of y variables
            vector<int> deleted_positions;
            vector<int> lcs_A_B_matched;
            
            for (int i = 0; i < rows; ++i) {
                if(cplex.getValue(y[ i ]) > 0.5) //deleted letter in A
                    deleted_positions.push_back(i);
            }
            for(const auto& pairs: C)
            {
                if(cplex.getValue(x[ pairs ]) > 0.5){
                    lcs_A_B_matched.push_back( pairs.first );
                    components_of_opt_sol.push_back(pairs);   
                }
            }
            //Merge two vectors, sort afterwards 
           /* lcs_A_B_matched.insert(lcs_A_B_matched.end(), deleted_positions.begin(), deleted_positions.end());
            std:sort(lcs_A_B_matched.begin(), lcs_A_B_matched.end());
            //extract solution:
            for(auto pos: lcs_A_B_matched)
                solution_string += instance->get_A()[pos];
        */
           solution_string = merge_x_and_y_components_into_solution(instance, lcs_A_B_matched, deleted_positions );

        }
        // End timing
        auto end = std::chrono::high_resolution_clock::now();
        float time_for_solving = std::chrono::duration<float >(end - start).count();
        
        //save the output into the file:
        //cout << "Sol: " << solution_string << " |LFCS|=" << solution_string.size() << " sol from CPLEX: " << sol << endl;
        cplex.end();//free memory
        env.end();
        
        if(out_file != "" ) // if the out file is provided:
        {
            if(status==1 or status==2)
                save_in_file_ILP(out_file, solution_string, time_for_solving, status, gap,  " ");
            else
                save_in_file_ILP(out_file, "", time_for_solving, 0, 100, " ");
       }
       return {status, gap, solution_string, components_of_opt_sol};
       
       
    }  catch (const IloException& e) {
           std::cerr << "CPLEX Exception: " << e.getMessage() << std::endl;
    }  catch (const std::exception& e) {
           std::cerr << "Standard Exception: " << e.what() << std::endl;
    }
}

/** the ADAPT phase in CMSA **/
void adapt_cmsa(set<pair<int, int>>& C_prime, vector<pair<int, int>> & S_prime_opt, unordered_map<pair<int, int>, int, PairHash>& age, int age_max)
{

     for(pair<int, int> c: C_prime)
         age[c]++;
     
     for(pair<int, int> s_prime_comp: S_prime_opt)
         age[ s_prime_comp ] = 0; 

     for(auto it =C_prime.begin(); it !=C_prime.end(); )
     {  
         if( age[*it] >= age_max ){
             it=C_prime.erase(it);
             age.erase(*it);
         }else
             ++it;
     }
}

/** CMSA design: **/
void CMSA(Instance* inst, int n_a, double p_init, int age_max, int time_limit, int t_limit_subproblem, string& out_file)
{
     
    string S_bsf="";
    set<pair<int, int>> C_prime;
    unordered_map<pair<int, int>, int, PairHash> age;
    double t_current=0.0;
    auto start = std::chrono::high_resolution_clock::now();
    int iteration=0; int iter_best=-1;
    while(t_current < time_limit)
    {
         //generate (random) solutions:
          auto S = construct_phase(inst, n_a, p_init);// construct (promising) x_{i, j} variables (LCS matchings between A and B
          for(pair<int, int> c: S)
          {
              if(C_prime.find(c) == C_prime.end()){  
                  age[c] = 0;
                  C_prime.insert(c);
              }
          }
          //MERGE phase:
          string sx="";
          auto S_prime_opt = cmsa_solve_phase(inst, t_limit_subproblem, C_prime, sx); // no need for storing the output
          
          if(std::get<2>(S_prime_opt).size() > S_bsf.size()){
              S_bsf = std::get<2>(S_prime_opt);
              iter_best = iteration;
          }
          //ADAPT phase:
          adapt_cmsa(C_prime, std::get<3>(S_prime_opt), age, age_max); // update C_prime w.r.t. S_prime_opt and the parameter age_max
              
          // End timing
          auto end = std::chrono::high_resolution_clock::now();
          t_current = std::chrono::duration<float >(end - start).count();
          iteration++;
          cout <<"#iter="<< iteration << "\tsolution size: " << S_bsf.size() << " C_prime: " << C_prime.size() << " time: " << t_current <<  endl;
    }
    //the final stats:
    std::string other_stats="iter_best: " +  std::to_string(iter_best) + "\niteration: " + std::to_string(iteration); // remaining stats
    cout << "LFCS: " << S_bsf << " " << " |LFCS|=" << S_bsf.size() << " time: " << t_current << endl;
    save_in_file_ILP(out_file, S_bsf, t_current, 2, -10000, other_stats );
    
}

/**
    CMSA: version 2 (with generating LFCSP solutions)
**/
// merge_y: y-are chosen  (merge by ILP)
tuple<int, float, string, vector<int>, vector<pair<int, int>> > solve_subinstance_set_x_y(Instance* instance, set<int>& merge_y, 
                                                                                          set<pair<int, int>>& components_x, 
                                                                                          int time_limit_subinstance, int best_solution=0)
{
      
      //TODO: callback integration, pass the best so-far solution, and stop solving once a better primal solution has been found 
      //processing: 
      unordered_map<int, vector<int>> c_first_to_second;
      unordered_map<int, vector<int>> c_second_to_first;
      
      for(pair<int, int> x_c: components_x)
      {
           if(c_first_to_second.find(x_c.first) == c_first_to_second.end())
           {
               c_first_to_second[x_c.first] = vector<int>();
               c_first_to_second[x_c.first].push_back(x_c.second);
           }
           else{
               c_first_to_second[x_c.first].push_back(x_c.second);
           }
           if(c_second_to_first.find(x_c.second) == c_second_to_first.end())
           {
               c_second_to_first[x_c.second] = vector<int>();
               c_second_to_first[x_c.second].push_back(x_c.first);
           }
           else{
              c_second_to_first[x_c.second].push_back(x_c.first);
           }  
      }
      try { 
         
         auto start = std::chrono::high_resolution_clock::now();
         // Create CPLEX environment and model
         IloEnv env;
         IloModel model(env); 
         
         int rows = instance->get_A().size();  //components_x: x
         int cols = instance->get_B().size();
         
         string solution_string="";
         
        // Hash map to store pairs of matching positions and binary variables  
        set<pair<int, int>> pairs;
        std::unordered_map<std::pair<int, int>, IloBoolVar, PairHash> x;
    
        for(auto& p: components_x){
                    // Create binary variable for matching pair
                    IloBoolVar x_ij(env);
                    x[p]=x_ij; 
        }
        std::unordered_map< int, IloBoolVar> y;  
        for(auto i: merge_y)
        {
            IloBoolVar yi(env);
            y[i] = yi;
        }
        // declare constraints
        IloRangeArray constraints(env);
        
        for(auto& p1: components_x)//pairs
        {
            for(auto& p2: components_x) //pairs
            {
                if(p1.first <= p2.first)
                {
                    if(p1.second != p2.second or p1.first != p2.first) //meaning if pairs1_var and pairs2_var are different:
                        if(conflict(p1, p2))
                            constraints.add( x[ p1 ] + x[ p2 ] <= 1);
                }
            }
        }
        // Add constraint: sum(x[i][j]) <= 1 for all j
        for (size_t j = 0; j < instance->get_B().size(); ++j) {
            IloExpr expr(env);
            bool add_i=false;
            for(int i: c_second_to_first[ j ]){
                expr += x[{i, j}];
                add_i=true;
            }
            if(add_i)
               constraints.add(expr <= 1);
            expr.end();
        }
        // Add constraint: y[i] + sum(x[i][j]) <= 1 for all i
        for(int i: merge_y)
        {
            IloExpr expr(env);
            expr += y[i];
            bool add_j=false;
            for(int j: c_first_to_second[ i ]){
                expr += x[{i, j}];
                add_j=true;
            }
             
            constraints.add(expr <= 1);
            expr.end();
        }
        //CONSTRAINTS: sum(y[i] where A[i] == sigma) <= M_sigma[sigma] 
        for(auto sigma: instance->sigmas)
        {
            IloExpr expr(env); bool one_match_at_least=false;
            for (int i: merge_y) 
            {
                if (instance->get_A()[i] == sigma ){ // i \in merge_y
                    
                    expr += y[i];
                    one_match_at_least=true;
                }
            }
            if(one_match_at_least)
            {   
                 if(instance->M_sigma.find(sigma) != instance->M_sigma.end())
                    constraints.add(expr <= instance->M_sigma[sigma]);
                 else
                    constraints.add(expr == 0);   //redundant constr.
            }
            expr.end();
        }
        
        // Set up the objective function: Maximize sum_i(y[i]) + sum_{i, j}(x[{i, j}]
        IloExpr obj_expr(env);
        for(int i: merge_y)
            obj_expr += y[i];
        
        for(auto& p: components_x) 
            obj_expr+=x[ p ];
        // set up to a maximization problem:   
        IloObjective obj = IloMaximize(env, obj_expr);
        
        //Add @constraints and @obj into the model, pass it to CPLEX
        model.add(obj);
        obj_expr.end();
        //Add constraints to the model
        model.add(constraints);
        
        //Create CPLEX solver and configure time limit
        IloCplex cplex(model);
        cplex.setParam(IloCplex::TiLim, time_limit_subinstance);
        cplex.setOut(env.getNullStream());
        if(best_solution != 0)//solution callback implemented and utilized
            cplex.use(new (env) BestSolutionCallback(env, best_solution));  

        //Solve the model:
        auto [sol, status, gap] = cplex_solve(cplex);// status => 0: NO SOLUTION, 1: FASIBLE, 2: OPTIMAL
        //SOLUTION EXTRACTION:
        vector<int> components_of_opt_sol; //y-components
        vector<pair<int, int>> components_x_opt;
        
        if(status==1 or status==2) //feasible or optimal solution found
        {
          
            // Extract values of y variables
            vector<int> deleted_positions;
            vector<int> lcs_A_B_matched;
            
            for (int i: merge_y) {
                if(cplex.getValue(y[ i ]) > 0.5){ //deleted letter in A
                
                    deleted_positions.push_back(i);
                    components_of_opt_sol.push_back(i);  // optimal y vars
                }
            }
            for(const auto& p: components_x)
            {
                if(cplex.getValue(x[ p ]) > 0.5){
                    lcs_A_B_matched.push_back( p.first );
                    components_x_opt.push_back(p);
                }
            }
            //Merge two vectors, sort afterwards      
            solution_string = merge_x_and_y_components_into_solution(instance, lcs_A_B_matched, deleted_positions );

        }
        // End timing
        auto end = std::chrono::high_resolution_clock::now();
        float time_for_solving = std::chrono::duration<float >(end - start).count();
        
        //save the output into the file:
        //cout << "Sol: " << solution_string << " |LFCS|=" << solution_string.size() << " sol from CPLEX: " << sol << endl;
        cplex.end();//free memory
        env.end();

        return {status, gap, solution_string, components_of_opt_sol, components_x_opt}; //  
       
       
    }  catch (const IloException& e) {
           std::cerr << "CPLEX Exception: " << e.getMessage() << std::endl;
    }  catch (const std::exception& e) {
           std::cerr << "Standard Exception: " << e.what() << std::endl;
    }
}

/** merge by DP */
tuple<int, float, string, vector<int>, vector<pair<int, int>> > solve_by_DP_subinstance_set_x_y(Instance* instance, 
                                  set<int>& merge_y, set<pair<int, int>>& components_x, int time_limit_subinstance)
{    
     //prepare A' and B:
     string a_prime_masked="";
     // choose among merge_y w.r.t. coverage constraints of the set M:
     auto coverage = greedy_maximize_matching_with_allowed_y_vars(instance->get_A(), instance->M_sigma, merge_y); 

    /*cout << "Coverage: " << endl;
    for(auto& m: coverage)
        cout << m << " ";
    cout << " " << endl;*/
     
     for(int i=0; i<instance->get_A().size(); ++i)
     {
         if(merge_y.find(i) == merge_y.end()) // i in merge_y: 
             a_prime_masked += "*";
         else
             a_prime_masked += instance->get_A()[i];
     }
     // make * after coverage:
     for(int i=0; i<instance->get_A().size(); ++i)
     {
         if(merge_y.find(i) == merge_y.end()) // i in merge_y: 
             a_prime_masked += "*";
         else
             a_prime_masked += instance->get_A()[i];
     }
     for(auto& mc: coverage)
         a_prime_masked[mc]='*'; // covered by the max... 

     //modify A' and create B w.r.t. feasible components_x:
     set<int> a_side_allowed_positions;
     set<int> b_side_allowed_positions;
     
     for(auto& x_c : components_x)
     {
         a_side_allowed_positions.insert(x_c.first);
         b_side_allowed_positions.insert(x_c.second);
     }

     // modify A':
     for(int i=0; i < a_prime_masked.size(); ++i)
     {
        if(a_prime_masked[i]=='*')
            continue;
        else{

            if(a_side_allowed_positions.find(i) == a_side_allowed_positions.end()){
                 a_prime_masked[i]='*';
            }
        }     
     }
     string b_masked="";
     for(int j=0; j<instance->get_B().size(); ++j)
     {
         if( b_side_allowed_positions.find(j) !=  b_side_allowed_positions.end())
             b_masked += instance->get_B()[j];
         else
             b_masked += "*";
     }
    // calculate LCS(a_prime_masked, b_masked)  
    cout << a_prime_masked << "\n" << b_masked << endl;
    auto [len, sol] = determine_LCS(a_prime_masked, b_masked);
    // optimal comp. of x and y extraction:
    //vector<int> sol_comp_y;
    vector<pair<int, int>> sol_comp_x;
    vector<int> comp_x;
    vector<int> comp_y; int curr_match=0;

    for(int i=0; i < a_prime_masked.size(); ++i)
    {
        if(a_prime_masked[i] == '*')
            continue;
        
        if(a_prime_masked[i] == sol[curr_match])
        {
            curr_match++;
            comp_x.push_back(i);
        }
        if(curr_match == sol.size())
           break;
    }
    // similar for B:
    curr_match=0;
    for(int j=0; j<b_masked.size(); ++j)
    {
        if(b_masked[j]=='*')
           continue;
        
        if(b_masked[j]== sol[curr_match])
        {
            curr_match++;
            comp_y.push_back(j);
        }
        if(curr_match == sol.size())
           break;
    }

    assert(comp_x.size()==sol.size() && comp_y.size()==sol.size());
    for(int i=0; i < sol.size(); ++i)
        sol_comp_x.push_back( {comp_x[i], comp_y[i]} );

    string solution_string = merge_x_and_y_components_into_solution(instance, comp_x,  coverage);//deleted_positions );

    return {2, 1000, solution_string, coverage, sol_comp_x};
}

pair< set<int>, set<pair<int, int>>> generate_subinstance_cmsa2(Instance* inst, int n_a, const set<int>& sample_fix = set<int>() ) // , set<int>& C_x ) 
{
      
       int num=0; set<int> merge_y; set<pair<int, int>> match_A_B;
       while(num<n_a)
       {
           UniformSampling* us = new UniformSampling(inst);
           us->run(1, "", sample_fix);  
           //generated solutions:           
           for(int s: us->M_with_A_matchings){
               merge_y.insert(s);
           }
           for(auto& pair: us->A_with_B_matching){   // need to fix????
               match_A_B.insert(pair);
           }
           delete us;
           num++;
       }   
        
       return {merge_y, match_A_B};
}

void adapt_cmsa2(set<int>& C_prime, vector<int>& S_prime_opt, set<pair<int, int>>& C_x_prime, 
                 vector<pair<int, int>>& S_prime_x_opt, unordered_map<int, int>& age, 
                 unordered_map<pair<int, int>, int, PairHash>& age_x_components, int age_max_x, int age_max_y){

    //update C_prime: y-components:
    for(int c: C_prime)
         age[c]++;
     
    for(int s_prime_comp: S_prime_opt)
         age[ s_prime_comp ] = 0; 
         
    for (auto it = C_prime.begin(); it != C_prime.end(); ) {
        if (age[*it] >= age_max_y) {
            age.erase(*it);  // Erase from age map
            it = C_prime.erase(it); // Erase and update iterator
        } else {
            ++it; // Move to the next element
        }
    }
            
    //update C_x_prime w.r.t. age_x_components (x_ij components)
    for(pair<int, int> x_comp: C_x_prime)
        age_x_components[x_comp]++;
    
    for(pair<int, int> x_comp: S_prime_x_opt)
         age_x_components[ x_comp ] = 0;     

    for(auto it=C_x_prime.begin(); it != C_x_prime.end(); )
    {
         if(age_x_components[*it] >= age_max_x){
             age_x_components.erase(*it);
             it = C_x_prime.erase(it);  
         }    
         else
            ++it;
    }
}  

// CMSA: version 2
void cmsa_2(Instance* inst, int n_a, int age_max_x, int age_max_y, int time_limit, int t_limit_subproblem, string& out_file ){

    string S_bsf="";
    set<int> C_prime; set<pair<int, int>> C_x_prime;
    
    unordered_map<int, int> age; unordered_map< pair<int, int>, int, PairHash> age_x_components;
    double t_current=0.0;
    auto start = std::chrono::high_resolution_clock::now();
    int iteration=0; int iter_best=0;
    auto end = start;
    //execute iterations:
    while(t_current < time_limit)
    {
         //generate (random) solutions:
          auto [ S, S_x ] = generate_subinstance_cmsa2(inst, n_a); // promising y-variables
          for(int c: S)
          {
              if(C_prime.find(c) == C_prime.end()){// c not in C_prime:
                  age[c] = 0;
                  C_prime.insert(c);
              }
          }
          
          // union S_x and C_x_prime:
          for(auto& com_pair: S_x)
          {
              if(C_x_prime.find(com_pair) == C_x_prime.end())
              {
                  age_x_components[com_pair]=0;
                  C_x_prime.insert(com_pair);
              }
          }
          //MERGE phase
          string sx="";
          //auto S_prime_opt = solve_by_DP_subinstance_set_x_y(inst, C_prime, C_x_prime, t_limit_subproblem); //  solve_subinstance_set_x_y(inst, C_prime, C_x_prime, t_limit_subproblem);
          auto S_prime_opt=solve_subinstance_set_x_y(inst, C_prime, C_x_prime, t_limit_subproblem, S_bsf.size());

          if(std::get<2>(S_prime_opt).size() > S_bsf.size()){
              S_bsf = std::get<2>(S_prime_opt);
              iter_best=iteration;
          }
          
          if(std::get<0>(S_prime_opt) > 0) //if empty string delivered by the exact solver, just step to the nest iteration ...
          {

            string a_prime_iter=""; int curr=0;
            for(int i=0; i < inst->get_A().size(); ++i)
            {
                if(i ==  std::get<3>(S_prime_opt)[curr]){
                   a_prime_iter+="*";
                   curr++;
                }    
                else 
                   a_prime_iter+=inst->get_A()[i];
          
            }
            //C_x_prime
            set<int> indices_A_feasible;
            set<int> indices_B_feasible;

            for(auto comp: C_x_prime){
                indices_A_feasible.insert(comp.first);
                indices_B_feasible.insert(comp.second);

            }
            for(int i=0; i < a_prime_iter.size(); ++i)
            {    
                if(a_prime_iter[i] != '*' and indices_A_feasible.find(i) == indices_A_feasible.end())
                  a_prime_iter[i]='*';  
            }
  
            // B masking:
            string b_prime_iter="";
            for(int i=0; i < inst->get_B().size(); ++i)
            {      
                if(indices_B_feasible.find(i) != indices_B_feasible.end())
                    b_prime_iter+=inst->get_B()[i];
                else
                    b_prime_iter+='*';
            }
          }
          
          adapt_cmsa2(C_prime, std::get<3>(S_prime_opt), C_x_prime, std::get<4>(S_prime_opt), age, age_x_components, age_max_x, age_max_y);   
          // End timing
          end = std::chrono::high_resolution_clock::now();
          t_current = std::chrono::duration<float >(end - start).count();
          iteration++;
          cout <<"#iter="<< iteration << "\tsolution size: " << S_bsf.size() << " C_prime: "
                         << C_prime.size() << " C_x_prime: "<< C_x_prime.size() << " time: " << t_current <<  endl;

    }
    // other stats:
    std::string other_stats="iter_best: " +  std::to_string(iter_best) + "\niteration: " + std::to_string(iteration); // remaining stats
    //the final stats:
    cout << "LFCS: " << S_bsf << " " << " |LFCS|=" << S_bsf.size() << " time: " << t_current << endl;
    save_in_file_ILP(out_file, S_bsf, t_current, 2, -10000, other_stats);
    
     
}

/**  ADAPT-CMSA: TODO **/

int getRandomElement(std::set<int>& mySet) {
    if (mySet.empty()) {
        throw std::runtime_error("Set is empty!");
    }
    int index = rand() % mySet.size(); // Generišemo slučajan indeks
    auto it = mySet.begin();
    std::advance(it, index); // Pomeri iterator na slučajan indeks
    return *it;
}

// update C_prime and C_x_y_prime acc. to the value of param. alpha_bsf  (alpha_bsf=1: no changes, alpha_bsf=0: random change)
tuple<set<int>,  set<pair<int, int>> > update_components(Instance* instance, set<int>& C_prime, set<pair<int, int>>& C_x_y_prime, float alpha_bsf )
{  
    
    unordered_map<char, set<int>> C_prime_char_positions;
    for(auto x: C_prime) 
    {
        char c = instance->get_A()[x];  
        if(C_prime_char_positions.find(c) == C_prime_char_positions.end())
        {    
            std::set<int> pos_char; pos_char.insert(x);
            C_prime_char_positions.insert({ c, pos_char});
        }else{
            C_prime_char_positions[c].insert(x);
        }
    }

    unordered_map<char, set<int>> char_remaining_not_in_C_prime;
    for(auto& x: C_prime_char_positions)
    {
        std::set<int> remaining_char;
        for(auto pos_ch: instance->char_to_positions_in_A[x.first]) // positions of all chars 
        {
            if( x.second.find(pos_ch)  == x.second.end())// pos_ch not in x.second
                remaining_char.insert(pos_ch);
        }
        //insert  (x, remaining_char)
        if(remaining_char.size() > 0 )
            char_remaining_not_in_C_prime[ x.first ] = remaining_char;
    }

    //std::random_device rd;  // Nasumični seed
    //std::mt19937 gen(rd()); // Mersenne Twister generator
    
    std::uniform_real_distribution<double> dist(0.0, 1.0); 
    
    set<int> C_prime_mutate;
    //mutate
    for(auto& x: C_prime_char_positions)//walk through C_prime char-by-char positions:
    {
        for(auto pos: x.second)
        {
            double random_num = dist(gen);
            if(random_num > alpha_bsf && char_remaining_not_in_C_prime.find(x.first) != char_remaining_not_in_C_prime.end() )  //change it, choose one from char_remaining_not_in_C_prime
            {
                std::uniform_int_distribution<int> dist(0, char_remaining_not_in_C_prime[x.first].size()-1); //char_remaining_not_in_C_prime[x.first] --> set<int>
                int new_random_index = dist(gen);
                int pos_change = getRandomElement(char_remaining_not_in_C_prime[ x.first ]);
                assert(instance->get_A()[pos_change] == x.first);
                C_prime_mutate.insert(pos_change);
                //swap between @pos and @pos_change:
                char_remaining_not_in_C_prime[ x.first ].insert(pos);
                char_remaining_not_in_C_prime[ x.first ].erase(pos_change);
                //cout << pos<< " " << pos_change << endl;
                assert(instance->get_A()[pos_change] == instance->get_A()[pos]);

            }
            else
                C_prime_mutate.insert(pos);
        }  
    }
    // C_prime ==> C_prime_mutate
    auto [C_best, C_x_y] = generate_subinstance_cmsa2(instance, 1, C_prime_mutate);// when y-variables are fixed; (set<pair<int, int>> type)

    //cout << "Modified solution result (C_prime_mutate): " << (C_best.size() + C_x_y.size()) << endl;
    
    return {C_best, C_x_y};

}

void adaptive_cmsa(Instance* inst, int time_limit, string& out_path, 
                   double alpha_lb, double alpha_ub, int t_limit_ILP, float t_prop, float alpha_red)
{

    //Generate random solution:
    set<int> C_prime; //unordered_map< pair<int, int>, int, PairHash> age_x_components;
    set<pair<int, int>> C_x_y_prime;
    double t_current=0.0;
    auto start = std::chrono::high_resolution_clock::now();
    //stats variables initialized:
    int iteration=0; int iter_best=0; double t_best=0.0;  
    auto end = start;
    //generate (random) solutions:
    auto [ S_best, S_x_best ] = generate_subinstance_cmsa2(inst, 1); // promising y-variables:
    int n_a=1;  double alpha_bsf = alpha_ub;
    //fill in the components 
    for(auto yi:  S_best)
        C_prime.insert(yi);
    for(auto pair_x_y: S_x_best)
        C_x_y_prime.insert(pair_x_y);

    cout << "Initial result: " << (S_best.size() + S_x_best.size()) << endl;
    int best_lfcs_solution=(S_best.size() + S_x_best.size());

    while(t_current < time_limit)    //execute iterations:
    {    
         cout << "#iter: " << iteration << "\n";
         bool incumbent_reached_at_iteration=false;
         //cout << "CONSTRUCT: " << endl;
         for(int i=0; i < n_a; ++i)
         {
             //auto [S, S_x_y] = generate_subinstance_cmsa2(inst, 1);//fully random generation
             auto [S, S_x_y ] = update_components(inst, S_best, S_x_best, alpha_bsf);// pass a seed value 

             if(best_lfcs_solution < S.size() + S_x_y.size())
             {
                 best_lfcs_solution =  S.size() + S_x_y.size();
                 cout <<"New incumbent: " << best_lfcs_solution << endl;
                 S_best = S;
                 S_x_best = S_x_y;
                 incumbent_reached_at_iteration=true;
             }
             // update C_prime, C_x_y_prime:
             for(auto sx: S)
                 C_prime.insert(sx);
 
             for(auto pair_x_y: S_x_y)
                 C_x_y_prime.insert(pair_x_y);
           
        }
        
         //solve the sub_instance C_prime via CPLEX: Line 16
        auto begin_ilp = std::chrono::high_resolution_clock::now();
        // update C_prime, C_x_y_prime according to alpha_bsf:
        auto S_prime_opt=solve_subinstance_set_x_y(inst, C_prime, C_x_y_prime, t_limit_ILP);  
        //cout << "SOLVE PHASE RESULT: " << get<2>(S_prime_opt).size();    
        auto end_ilp = std::chrono::high_resolution_clock::now();
        auto t_solve_ilp = std::chrono::duration<float >(end_ilp - begin_ilp).count();
  
        // LINE 16: compare if tsolve < tprop · tILP ∧ αbsf > αLB then
        if(t_solve_ilp < t_prop * t_limit_ILP && alpha_bsf > alpha_lb) // the instance is easy to solve 
        {
            alpha_bsf -= alpha_red;
        }  
             
        if(std::get<2>(S_prime_opt).size() > S_best.size() + S_x_best.size()) // LINE 19, compare the solutions
        {
             // UPDATE S_best, S_x_best:
             best_lfcs_solution = std::get<2>(S_prime_opt).size();
             //update the structures that align to S_prime_opt (new inclumbent)
             S_best.clear(); S_x_best.clear();
             auto S_best_vec = std::get<3>(S_prime_opt);   
             auto S_x_best_vec = std::get<4>(S_prime_opt);
             //update S_best (matchings between A and the multiset M: the deletions)
             for(auto x: S_best_vec)
                S_best.insert(x);
             //update S_x_best (matchings between A' and B)
             for(auto x_y: S_x_best_vec)
                 S_x_best.insert(x_y); 
             cout << "\nNew inclumbent: " << std::get<2>(S_prime_opt).size() << endl;
             // update n_a, alpha_bsf
             n_a=1;
             alpha_bsf=alpha_ub;
             incumbent_reached_at_iteration=true;

        }else if(std::get<2>(S_prime_opt).size() < S_best.size() + S_x_best.size()){ // LINE 23
             
              if(n_a==1)
                alpha_bsf=std::min(alpha_bsf + alpha_red/10, alpha_ub);
              else{
                n_a=1;
                alpha_bsf=alpha_ub;
              }
          }else{ // equaly good solution has been found - diversify:
              n_a+=1;
          }
 
        // Line 33: Re-initialize the sub-instance C 0 with the solution components of sbsf (update  C_prime, C_x_y_prime)
        C_prime.clear(); C_x_y_prime.clear();
        //fill in (re-initialize from S_best and S_x_best
        for(auto x: S_best)
            C_prime.insert(x);
        for(auto x_y: S_x_best)
            C_x_y_prime.insert(x_y);
        /* THINK ABOUT EFECTIVE RESTART MECHANISM?????
        if(t_solve_ilp > time_limit * 0.1) // long solving process: restart
        {
            n_a=1;
            alpha_bsf=alpha_lb;    
        } 
        */
        // time measure:  
        end = std::chrono::high_resolution_clock::now();
        t_current = std::chrono::duration<float >(end - start).count(); // current time spent
        if(incumbent_reached_at_iteration){
           t_best=t_current; // time stored for when new incumnet is reached 
           iter_best=iteration;
        }
        iteration++;
    }
    string S_best_string = "";
    //construct the LFCS:
    std::vector<int> S_best_vec(S_best.begin(), S_best.end());
    std::vector<int> S_x_best_vec;
    
    for(auto& x_y: S_x_best)
        S_x_best_vec.push_back(x_y.first);
    //sort S_x_best_vec
    std::sort(S_x_best_vec.begin(), S_x_best_vec.end());

    S_best_string = merge_x_and_y_components_into_solution(inst,  S_x_best_vec, S_best_vec); //vector<int>& lcs_A_B_matched, vector<int>& deleted_positions){     
    cout << "Best solution |LFCS|=" << best_lfcs_solution << endl;
    save_in_file_ILP(out_path, S_best_string, t_current, 2, -10000, "time best: " 
                     + std::to_string(t_best) + "\niteration best: " + std::to_string(iter_best)
                     + "\niterations: " + std::to_string(iteration));

    //return {best_lfcs_solution, S_best, S_x_best};
}

/** PURE ILP MODEL **/
int ILP_solve(Instance* instance, int time_limit, string& out_file, bool strengthen=false) {
 
    try {
        // Start timing
        auto start = std::chrono::high_resolution_clock::now();
        // Create CPLEX environment and model
        IloEnv env;
        IloModel model(env);
        // Define 2D decision variables
        int rows = instance->get_A().size();
        int cols = instance->get_B().size();
        string solution_string="";
        // Hash map to store pairs of matching positions and binary variables
        std::unordered_map<std::pair<int, int>, IloBoolVar, PairHash> x;

        // Traverse strings and find matching positions
        for (size_t i = 0; i < instance->get_A().size(); ++i) {
            for (size_t j = 0; j < instance->get_B().size(); ++j) {
                if (instance->get_A()[i] == instance->get_B()[j]) {
                    // Create binary variable for matching pair
                    IloBoolVar x_ij(env);
                    x[{i, j}] = x_ij;
                }
            }
        }

        // Define 1D decision variables
        IloNumVarArray y(env, rows, 0, 1, ILOINT); // Binary variables y_{i}
        IloRangeArray constraints(env);
        
        std::set<char> sigma_present_in_A;
        for(char c: instance->get_A())
             sigma_present_in_A.insert(c);
        
        for(int i=0; i < rows; ++i)
        {
            bool present=false;
            for(auto c: instance->M_sigma)
            {
                if(c.first == instance->get_A()[i])
                {
                   present=true;
                   break;
                }
            }
            if(!present) // @c not in A
                constraints.add(y[i] == 0);
        }
        // Add constraint: y[i] + sum(x[i][j]) <= 1 for all i
        for (int i = 0; i < rows; ++i) {
            IloExpr expr(env);
            expr += y[i];
            for (int j = 0; j < cols; ++j)
                if(instance->get_A()[i] == instance->get_B()[j])
                   expr += x[ {i, j} ];
            
            constraints.add(expr <= 1);
            expr.end();
        }  

        // Add constraint: sum(x[i][j]) <= 1 for all j
        for (int j = 0; j < cols; ++j) {
            IloExpr expr(env);
            for (int i = 0; i < rows; ++i){
                if(instance->get_A()[i] == instance->get_B()[j])
                    expr += x[ {i, j} ];
            }
            constraints.add(expr <= 1);
            expr.end();  
        }
        
        
        //CONSTRAINTS: sum(y[i] where A[i] == sigma) <= M_sigma[sigma]  
        for(auto sigma: instance->sigmas) //instance.M_sigma)
        {
            IloExpr expr(env);
            bool exist_in_A=false;
            for (int i = 0; i < rows; ++i) {
                if (instance->get_A()[i] == sigma ) { //sigma
                    expr += y[i];
                    exist_in_A=true;
                }
            }
            if(exist_in_A)
                constraints.add (expr <= instance->M_sigma[sigma] );   
            
            expr.end();
        }
        // Conflicting vars (2b constraints):
        for(const auto& pairs1_var: x)
        {
            for(auto pairs2_var: x)
            {
                if(pairs1_var.first.first <= pairs2_var.first.first)
                {   
                    if(pairs1_var.first.second != pairs2_var.first.second) //meaning if pairs1_var and pairs2_var are different:
                        if(conflict(pairs1_var.first, pairs2_var.first))
                            constraints.add(pairs1_var.second + pairs2_var.second <= 1);
                }
            }  
        }
        // Set up the objective function: Maximize sum(y[i]) + sum(x[{i, j}]
        IloExpr obj_expr(env);
        for (int i = 0; i < rows; ++i) {
            obj_expr += y[i];
        }
        
        for(const auto& pair_var: x)
            obj_expr+=pair_var.second;
        // set up to a maximization problem:   
        IloObjective obj = IloMaximize(env, obj_expr);
        model.add(obj);
        obj_expr.end();  
        //Add constraints to the model
        model.add(constraints);
        
        //Create CPLEX solver and configure time limit
        IloCplex cplex(model);
        cplex.setParam(IloCplex::TiLim, time_limit);
        
        // Solve the model
        auto [sol, status, gap] = cplex_solve(cplex);
        
        if(status==1 or status==2) //feasible or optima solution found
        {
               // Extract values of y variables
               vector<int> deleted_positions;
               vector<int> lcs_A_B_matched;
            
               for (int i = 0; i < rows; ++i) {
                  if(cplex.getValue(y[i]) > 0.5){ //deleted letter in A
                      deleted_positions.push_back(i); 
                      cout << i << " ";
                  }
               }
               cout << endl;
               for(const auto& pair_vars: x)
                   if(cplex.getValue(x[ pair_vars.first ]) > 0.5)
                       lcs_A_B_matched.push_back( pair_vars.first.first);

            solution_string=merge_x_and_y_components_into_solution(instance, lcs_A_B_matched, deleted_positions);// covert into solution:
        }  
        // End timing
        auto end = std::chrono::high_resolution_clock::now();
        float time_for_solving = std::chrono::duration<float >(end - start).count();
        
        if(status==1 or status==2)
            save_in_file_ILP(out_file, solution_string,  time_for_solving, status, gap, " ");
        else
            save_in_file_ILP(out_file, "", time_for_solving, 0, 100, " ");
        
        // Clean up
        env.end();
    } catch (const IloException& e) {
        std::cerr << "CPLEX Exception: " << e.getMessage() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
    }
  
    return solution;
}

void read_parameters(int argc, char **argv) 
{

    int iarg=1;
    while (iarg < argc) {
        if (strcmp(argv[iarg],"-i")==0) File_Name = argv[++iarg];
        else if (strcmp(argv[iarg],"-a")==0) ilp_algorithm = atoi(argv[++iarg]);//ilp_algorithm
        else if (strcmp(argv[iarg],"-t")==0) time_limit = atof(argv[++iarg]);    
        else if (strcmp(argv[iarg],"-o")==0) File_Name_Out = argv[++iarg];
        else if (strcmp(argv[iarg],"-percentage")==0) percentage = atof(argv[++iarg]);
        else if (strcmp(argv[iarg],"-num_sols")==0) num_solutions = atoi(argv[++iarg]);  
        else if (strcmp(argv[iarg],"-age")==0)   age_max_val = atoi(argv[++iarg]);
        else if (strcmp(argv[iarg],"-age_y")==0) age_max_y = atoi(argv[++iarg]);
        else if (strcmp(argv[iarg],"-seed")==0) seed = atoi(argv[++iarg]);
        ////double aplha_lb, double alpha_ub, int t_limit_ILP, float t_prop, float alpha_red
        else if (strcmp(argv[iarg],"-alpha_lb")==0) alpha_lb = atof(argv[++iarg]);
        else if (strcmp(argv[iarg],"-alpha_ub")==0) alpha_ub = atof(argv[++iarg]);
        else if (strcmp(argv[iarg],"-t_prop")==0) t_prop = atof(argv[++iarg]);
        else if (strcmp(argv[iarg],"-t_limit_ILP")==0) t_limit_ILP = atoi(argv[++iarg]);
        else if (strcmp(argv[iarg],"-alpha_red")==0) alpha_red = atof(argv[++iarg]);        
        iarg++;
    }
}

int main(int argc, char **argv)
{
         read_parameters(argc,argv);
         Instance* instance=new Instance();
         instance->read_instance(File_Name);//fill in with the data
         instance->print_instance_data();
         // set up the value of seed: 
         set_global_seed(seed);

         if(ilp_algorithm==0)// basic  ILP
         {   
             cout << "Run the basic ILP " << endl;
             int sol=ILP_solve(instance, time_limit, File_Name_Out);
         }else //CMSA application:
          if(ilp_algorithm==1){

                 cout << "Start the CMSA procedure: \n";
                 //CMSA(instance, num_solutions, percentage, age_max_val, time_limit, t_lim_sub, File_Name_Out);
                 cmsa_2(instance, num_solutions, age_max_val, age_max_y, time_limit, t_lim_sub, File_Name_Out);
          }else{ //ilp_algorithm=2 ==> adapt CMSA
                 cout << "Start the ADAPT-CMSA \n"; // num sols and  t 
                 adaptive_cmsa(instance, time_limit, File_Name_Out, alpha_lb, alpha_ub, t_limit_ILP, t_prop, alpha_red); 
                 // default: 0.6, 1.0, 10, 0.5, 0.2); //double aplha_lb, double alpha_ub, int t_limit_ILP, float t_prop, float alpha_red
         }
         delete instance;  
         
         return 0;
}
