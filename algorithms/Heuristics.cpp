/* HEURISTICS FOR LFCS PROBLEM */
      
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
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <iterator>
#include <tuple>
#include <algorithm>
#include <chrono>
#include <random>
// custom classes
#include "Instance.h"
#include "UniformSampling.h"
#include "utils.h"


using namespace std;
 
 
/** Declare global vars  **/
double allowed_time_limit=10.0;
int solution=0;
string File_Name="";
string File_Name_Out="";
double time_limit=10.0;
int algorithm=0;
int k_val=0;
int num_solutions=1;
int neighbor_size=3;
/** End of global vars declaration **/
   
//LOCAL SEARCH (Section IV, subsection B)

// Function to generate combinations
void generateCombinations(const std::vector<int>& nums, 
                          std::vector<int>& currentComb, 
                          std::vector<std::vector<int>>& result, 
                          int start, 
                          int m, 
                          Instance* inst) {
    if (currentComb.size() == m) {
        // Combination of size m generated
        result.push_back(currentComb);
        return;
    }
    
    for (int i = start; i < nums.size(); ++i) {
        // Skip duplicates
        if (i > start && nums[i] == nums[i - 1]) {
            continue;
        }
        // Include the current element
        currentComb.push_back(nums[i]);
        
        // Recurse
        generateCombinations(nums, currentComb, result, i + 1, m, inst);
        
        // Backtrack
        currentComb.pop_back();
    }
}

std::vector<std::vector<int>> combinations(std::vector<int> nums, int m, Instance* inst) {

    std::vector<std::vector<int>> result;
    std::vector<int> currentComb;

    // Sort the vector to handle duplicates
    std::sort(nums.begin(), nums.end());
    // Generate combinations
    generateCombinations(nums, currentComb, result, 0, m, inst);
    return result;
}

bool valid_positions(vector<int>& pos, Instance* inst, unordered_map<char, int>& char_upper_bounds)
{

    unordered_map<char, int> vector_occur_pos;
    for(auto px: pos)
    {
            if( vector_occur_pos.find( inst->get_A()[px] ) == vector_occur_pos.end())
                vector_occur_pos[ inst->get_A()[px] ] = 1;
            else
                vector_occur_pos[ inst->get_A()[px] ]++;
    }
    for(auto char_int: vector_occur_pos)
    {
        if( char_upper_bounds.find(char_int.first) == char_upper_bounds.end() or char_upper_bounds[ char_int.first ]==0) // A[[pos] not in M
            return false;
        if(char_upper_bounds[ char_int.first ] < char_int.second) // too much char_int letters taken for this pos (characters deletion)
            return false;
    }    
    return true;
}
//INITIAL STEP (work with the initial instance -- no '*' (matched) symbols)
string extract_solution(Instance* instance, int i, int k, string& sol_string, vector<int>& pos_vector){

    string merged=""; int current_reminder=0; int pos_vector_pointer=0;  vector<int> pos_match_sol_string;
    
    for(int ix=0; ix<instance->get_A().size();++ix)
    {
        if(current_reminder < sol_string.size() && instance->get_A()[ix] == sol_string[current_reminder])
        {
            if(pos_vector_pointer < pos_vector.size() && pos_vector[pos_vector_pointer] == ix)
            {
                pos_vector_pointer++;
                continue;
            }else{
                
                //merged += instance->get_A()[ix];
                pos_match_sol_string.push_back(ix);
                current_reminder++;
            }
        }
    }
    // merge @pos_match_sol_string and @pos_vector
    for(int px: pos_vector)
        pos_match_sol_string.push_back(px);
    //sort:
    std::sort(pos_match_sol_string.begin(), pos_match_sol_string.end());
    
    for(int pos: pos_match_sol_string)
        merged += instance->get_A()[pos];
        
    return merged;
}

string convert_final_solution(Instance* instance, string a_prime)
{
    
   string res_string="";
   auto [sol_length, sol_string] = determine_LCS(a_prime, instance->get_B());// calculate LCS(A', B)
   vector<int> lcs_match;  vector<int>positions_deletion_match; int pointer=0; int index=0;
   
   for(auto ch: a_prime)
   {
       if(ch=='*')
           positions_deletion_match.push_back(index);
       
       if(pointer < sol_string.size() && ch==sol_string[pointer])
       {
           pointer++;
           lcs_match.push_back(index);
       }
       index++;
   }
   // union of two vectors:
   vector<int> all_matches;
   for(auto x: lcs_match)
       all_matches.push_back(x);
    for(auto y: positions_deletion_match)
       all_matches.push_back(y);  
   
   std::sort(all_matches.begin(), all_matches.end());
   
   for(auto pos: all_matches)
       res_string+=instance->get_A()[pos];
   
   cout << "lcs_match " << lcs_match.size() << " positions_deletion_match: " << positions_deletion_match.size() <<endl; 
   return res_string;
}

std::tuple<int, string, vector<int>> perform_ls_step(Instance* instance, int i, int k){// window (i, k)
    
    //Match M for a[i, i+k]
    vector<int> positions;
    int best_solution_in_the_iteration=0; string best_sol_string_in_the_iteration=""; 
    vector<int> best_position_vector; // keep the used (matched) letters)
    unordered_map<char, vector<int>> M_sigma_partial;
    
    // Iterate through all k-consecutive substrings: REMOVE '*' FROM CONSIDERATION 
    int j=i;
    while(j<instance->get_A().size() && positions.size() < k) 
    {
        if( instance->get_A()[j] == '*') // already matched before
        {
            j++;
            continue;
        }
        // if not matched, execute the following commands
        if(M_sigma_partial.find(instance->get_A()[j]) ==  M_sigma_partial.end())
        {
           M_sigma_partial[ instance->get_A()[j] ] = vector<int>();
           M_sigma_partial[ instance->get_A()[j] ].push_back( j );
        }   
        else
           M_sigma_partial[ instance->get_A()[j] ].push_back( j );
       
        positions.push_back(j); // positions to be processed (not '*')
        j++;   
    }
    
    if(positions.size() < k) // not enough positions found (should be k of them) due to frequent presence of '*' symbols
        return {0, "", vector<int>()};
    // CONSTRAINT (upper bound on the number of occurances of each letter from M_sigma)
    unordered_map<char, int> char_upper_bounds;
    
    for(auto char_occur: M_sigma_partial)
    {
          if(M_sigma_partial[char_occur.first].size() <= instance->M_sigma[char_occur.first])
              char_upper_bounds[char_occur.first] = M_sigma_partial[char_occur.first].size();
          else 
              char_upper_bounds[char_occur.first] = instance->M_sigma[char_occur.first];
    }
    // CALCULATE all combinations of @positions, remove those not relevant (not max-covered, by using the function @valid_positions)
    std::vector<std::vector<int>> results_all_partitions;
    for(int dim=1; dim <= k; ++dim){
    
        vector<vector<int>> result = combinations(positions, dim, instance);
        for(auto& pos: result)
        {
            if(valid_positions(pos, instance, char_upper_bounds))
                results_all_partitions.push_back(pos); //results_all_partitions.end(), result.begin(), result.end());
        }
    }
    // Perform LCS on the valid @results_all_partitions
    for(vector<int>& pos_vector: results_all_partitions)
    {
        string a_prime_remainder="";
        for(int ix=0; ix<instance->get_A().size(); ++ix){
            
            if(instance->get_A()[ ix ] == '*') // skip the symbol '*'
                continue;
                
            if (std::find(pos_vector.begin(), pos_vector.end(), ix) == pos_vector.end()) { // A[i] not in A[pos_vec], save into A' (the remainder)
                a_prime_remainder += instance->get_A()[ ix ];    
            }
        }  
        // LCS(a_prime_remainder, B):
        auto [sol_length, sol_string] = determine_LCS(a_prime_remainder, instance->get_B());// calculate LCS(A', B)
        // update the best solution in the current iteration:
        if(pos_vector.size() + sol_length >= best_solution_in_the_iteration) // LCS(A', B) + |deletion|
        {    
           best_solution_in_the_iteration = pos_vector.size() + sol_length;
           best_sol_string_in_the_iteration = extract_solution(instance, i, k, sol_string, pos_vector);
           best_position_vector=pos_vector;
        } 
    }
    
    return   {best_solution_in_the_iteration, best_sol_string_in_the_iteration, best_position_vector};
}

string Local_search(Instance* instance, string& out_file, int k, bool verbose=false)
{
    vector<int>  best_pos_matched;
    string I ="";
    int sol_ls_best=0;
    // Record the start time
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<instance->get_A().size(); ++i)
    {
        auto [sol_len_ls, sol_string_ls, positions_matched] = perform_ls_step(instance, i, k);// window (i, k)
        if(sol_len_ls > sol_ls_best)
        {
            sol_ls_best = sol_len_ls;
            I = sol_string_ls;
            best_pos_matched = positions_matched;
        }
    }

    cout << "#Interation 0:  sol_ls_best=" << sol_ls_best << " " << I << " " << endl;
    string a_prime=""; int curr=0; string symbols_matched="";
    for(int i=0; i < instance->get_A().size(); ++i)
    {
        if(best_pos_matched[curr] == i)
        {
            a_prime += "*";
            symbols_matched += instance->get_A()[i];
            curr++;//move to the next symbol
        }else
            a_prime += instance->get_A()[i];
    }
    int iteration=1;//execute the next iteration
    Instance* I_best = new Instance(a_prime, instance->get_B(), instance->M_sigma, symbols_matched, instance->get_sigma(), 0); // new instance, with updated M and M_sigma vals...

    while(true)
    {
        // preparation for the next iteration:
        int sol_ls_best_iter_best=0; string I_best_iter=""; vector<int> best_pos_matched_iter; symbols_matched="";
        //Exectute the iteration (best-improvement strategy)
        for(int i=0; i<instance->get_A().size(); ++i)
        {    
            auto [sol_len_ls_iter, sol_string_ls_iter, positions_matched_iter] = perform_ls_step(I_best, i, k);// window (i, k), to be updated: '*' symbol to be ignored

            if(sol_len_ls_iter > sol_ls_best_iter_best){ // LCS(A', B) + |deletion|
            
               sol_ls_best_iter_best = sol_len_ls_iter;
               I_best_iter = sol_string_ls_iter;
               best_pos_matched_iter = positions_matched_iter;
            }
        }
        
        for(int px: best_pos_matched_iter)
        {
               symbols_matched += I_best->get_A()[ px ];
               a_prime[ px ] = '*';
        }
        
        if(verbose or true)
        {
            cout  << "symbols_matched: " << symbols_matched << endl;
            for(auto bpmx: best_pos_matched_iter)
                cout << bpmx << " chr: " << I_best->get_A()[bpmx] << endl;
            cout  << endl;
        }
        // Compare to the best-so-far:
        cout << "#iteration " << iteration << " best sol length: "<< sol_ls_best_iter_best  
             << " positions matched " << best_pos_matched_iter.size() << " |M|=" << I_best->get_M().size() 
             <<  " symbols_matched: " << symbols_matched << " a_prime: " << a_prime << endl;
        
        if(symbols_matched.size()>0)  //(improved)
            I_best = new Instance(a_prime, I_best->get_B(), I_best->M_sigma, symbols_matched, I_best->get_sigma(), I_best->number_of_matched) ;
        else // if no symbols anymore to match:
           break;
           
        iteration++;
    }
    
    //convert the best LCS solution from @instance and the last I_best (A')
    string solution_ls_best=convert_final_solution(instance, a_prime);
     // Record the end time
   
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the duration in milliseconds
    std::chrono::duration<double> duration = end - start;
    save_in_file(out_file, a_prime, solution_ls_best, duration.count());
   
    return solution_ls_best; //sol_ls_best;

}
std::tuple<int, string, string, unordered_map<char, int>> approximate_algoritm_1(Instance* instance, string& out_file)
{
    
    auto start = std::chrono::high_resolution_clock::now();
    //STEP 1: LCS(a, b)
    auto [len, sol] = determine_LCS(instance->get_A(), instance->get_B());
    //STEP 2: calculate A_prim:

    string a_prim=""; vector<int> positions_of_solution;
    set<int> R_1_a; int current=0;
    for(int i=0; i<instance->get_A().size(); ++i)
    {
        if(instance->get_A()[i] == sol[current])
        {   
             current+=1;
             R_1_a.insert(i);  
             positions_of_solution.push_back(i);   
        }
        if(current == sol.size())
           break;
    }
    // match wit B string:
    vector<int> position_of_sol_in_B; current=0;
     for(int i=0; i<instance->get_B().size(); ++i)
    {
        if(instance->get_B()[i] == sol[current])
        {   
             current+=1;
             position_of_sol_in_B.push_back(i);   
        }
        if(current == sol.size())
           break;
    }
    //pair of position of the resultsing LCS   
    vector<pair<int, int>> match_lcs_sol;
    for(int i=0; i<sol.size(); ++i)
        match_lcs_sol.push_back({positions_of_solution[i], position_of_sol_in_B[i]});

    //STEP 2: construct a_prime from A:
    for(int i=0; i<instance->get_A().size(); ++i)
    {
        if (R_1_a.find(i) == R_1_a.end())
            a_prim += instance->get_A()[i];
    }
    //STEP 3: Greedy computation of R_{1, i}: set the positions of A' of maximum size that matches M by insertion; Greedy-approx(A', M);
    set<int> R_1_i;
    auto [str_r_1_i, Msigma] = greedy_maximize_matching(a_prim, instance->M_sigma); // a_prim: positions not touched between A and B 
    
    //Convert the proper indices for str_r_1_i w.r.t. a_prime: 
    current=0; 
    for(int i=0; i<instance->get_A().size(); ++i)
    {   
        if(str_r_1_i[current] =='*')
            continue;
        if(R_1_a.find(i) == R_1_a.end() && str_r_1_i[current] == instance->get_A()[i])
        {   
            current++;
            R_1_i.insert(i);
            positions_of_solution.push_back(i);  // add position of match with M (deleted_positions) -- to complete @solution pos. in A
        }  
        if(current == str_r_1_i.size())
            break;
   }
   string a_star_asterix="";
   for(int i=0; i<instance->get_A().size(); ++i)
   {
        if(R_1_i.find(i) == R_1_i.end())
            a_star_asterix += instance->get_A()[i];
        else
            a_star_asterix += "*";
   }
   // Extract the solution
   string extracted_final_slution="";
   // sort positions_of_solution
   std::sort(positions_of_solution.begin(), positions_of_solution.end());
   for(auto px: positions_of_solution)
       extracted_final_slution += instance->get_A()[ px ];
   
   auto end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> duration = end - start;
   
   if(out_file!="")
       save_in_file(out_file, "", extracted_final_slution, duration.count());
   //@ positions_of_solution: position of matches of the solution on string A; 
   // Msigma: 
   // R_1_i: deleted positions stored
   return {R_1_i.size() + R_1_a.size(), extracted_final_slution, a_star_asterix, Msigma};
}
 
/* SEARCH-BASED ALGORITHM **/
pair<string, unordered_map<char, int>> modify(Instance* instance, string& a_prime, unordered_map<char, int>& M_sigma_a_prime, int k)
{
    cout << a_prime << endl;
    unordered_map<char, int> M_sigma_a_prime_copy = M_sigma_a_prime;
    
    vector<int> deleted_positions;
    vector<int> non_deleted_positions;
    
    for(int i=0; i<a_prime.size(); ++i)
        if(a_prime[i] == '*')
            deleted_positions.push_back(i);
        else
            non_deleted_positions.push_back(i);
    
    //pick k of them to be replaced by some others...
    std::random_device rd;
    std::mt19937 gen(rd());

    if(deleted_positions.size() < k)
       k=non_deleted_positions.size();
    
    std::shuffle(deleted_positions.begin(), deleted_positions.end(), gen); // Shuffle elements
    std::vector<int> elements_to_replace(deleted_positions.begin(), deleted_positions.begin() + k); // Pick first k
    string a_prime_modified=a_prime;
    
    for(int pos: elements_to_replace){ // not anymore under '*'
        a_prime_modified[pos] = instance->get_A()[pos];
        M_sigma_a_prime_copy[instance->get_A()[pos]]++;   
    }
    
    /*cout << "non_deleted_letters: ";
    for(int c: non_deleted_positions)
        cout << c << " ";
        
    cout <<"\n" << endl;*/
    cout << "elements_to_replace: ";
    for(int c: elements_to_replace)
        cout << c << " ";
        
    cout <<"\n" << endl;
        
    
    int matched_num=0; std::vector<int> elements_with_whom_to_replace;
    //modify feasibly
    std::shuffle(non_deleted_positions.begin(), non_deleted_positions.end(), gen);
    int index=0;
    
 
    while(matched_num != k)
    {

        if(M_sigma_a_prime_copy[a_prime[non_deleted_positions[ index ] ] ] >= 1) 
        {
            M_sigma_a_prime_copy[a_prime[non_deleted_positions[index] ] ]--;
            elements_with_whom_to_replace.push_back( non_deleted_positions[index] ); // now under '*'
            matched_num++;
            //non_deleted_positions.erase(non_deleted_positions.begin());
        }
        index++;
        if(index>=non_deleted_positions.size())
            break;
    }
    
    
    /*cout << "elements_with_whom_to_replace: ";
    for(int c: elements_with_whom_to_replace)
        cout << c << " ";
        
    cout <<"\n" << endl; */ 
    
     
    for(int pos: elements_with_whom_to_replace)
         a_prime_modified[pos] = '*';
    
    //cout << "new p_prime: " << a_prime_modified << endl;
     
    return {a_prime_modified, M_sigma_a_prime_copy};
 
}

string extract_solution_a_prime_and_LCS(string& A, string& a_prime, string& LCS_remaining)
{
    
    string result="";
    int current_match=0;
    vector<int> pos_lcs_in_a_prime;
    
    for(int i=0; i<a_prime.size(); ++i)
    {
        if(a_prime[i] == '*'){
            result += A[i];
            continue;
        }
            
        if(a_prime[i]==LCS_remaining[current_match])
        {
            result+=a_prime[i];
            current_match++;
        }
        if(current_match >=LCS_remaining.size())
            break;
    }
    return result;
}

void search_algorithm(Instance* instance, double time_allowed=10, string out_file="")
{
    string A=instance->get_A();
    string no_out_needed="";
    //Approx. 1 to initialization
    auto [len_best_lcs, lfcs_best_solution, a_prime_modified, Msigma_modified] = approximate_algoritm_1(instance, no_out_needed);

    string best_a_prime=a_prime_modified;  
    auto M_best = Msigma_modified;     

    // init:
    /*auto [ a_prime_modified, Msigma_modified ] = greedy_maximize_matching(A, instance->M_sigma);
    auto [len_best_lcs, best_lcs_solution] = determine_LCS(a_prime_modified, instance->get_B());
    //extract solution based on a_prime and the obtained LCS on A'
    string lfcs_best_solution = extract_solution_a_prime_and_LCS(A, a_prime_modified, best_lcs_solution);
   
    string best_a_prime=a_prime_modified;  
    auto M_best = Msigma_modified; */

    //tracking runtime
    auto start = std::chrono::high_resolution_clock::now();
    double t_time = 0.0; int iteration=0; int iter_best=0;
    //cout << "Start the basic search strategy (TODO: initialize with the APPROX)" << endl;
    int k_min=2; int k_max=int(instance->get_A().size()/10);
    int k_neighbor=k_min;
    while(t_time < time_allowed)
    {
        iteration++;
        cout << "#iter: " << iteration  << " " << t_time << endl;  
        
        auto [a_prime_modified_it, M_sigma_modified_it] = modify(instance, a_prime_modified, M_best, k_neighbor); //(repair-construct) the best solution
        auto [lcs_len_it, lcs_sol_it] = determine_LCS(a_prime_modified_it, instance->get_B());
        //solution construction: 
        string lfcs_solution_it_constructed = extract_solution_a_prime_and_LCS(A, a_prime_modified_it, lcs_sol_it);
        cout <<"lfcs_solution_it_constructed for #iter " << iteration << " size: " << lfcs_solution_it_constructed.size() << endl;
        // update if a better incumbent found:
        if(lfcs_best_solution.size() < lfcs_solution_it_constructed.size())
        {
             lfcs_best_solution = lfcs_solution_it_constructed;
             best_a_prime = a_prime_modified_it;
             M_best = M_sigma_modified_it; //Msigma_modified_greedy;
             iter_best=iteration;

             k_neighbor=k_min;
            
        }else
             k_neighbor++;
        
        if(k_neighbor>k_max)
           k_neighbor=k_min;

        auto end = std::chrono::high_resolution_clock::now();
        // Calculate the duration in milliseconds
        std::chrono::duration<double> duration = end - start;
        t_time=duration.count();
        
        cout << "---------------------------------------------------------------------------------------" << endl;
    }
    
    cout << "\nbest lfcs: " << lfcs_best_solution << " size: " << lfcs_best_solution.size() << " a_prime: " <<  best_a_prime << " iter best: " << iter_best << endl;
    save_in_file(out_file, best_a_prime, lfcs_best_solution, t_time);
}


void read_parameters(int argc, char **argv) 
{

    int iarg=1;
    while (iarg < argc) {
        if (strcmp(argv[iarg],"-i")==0) File_Name = argv[++iarg];
        else if (strcmp(argv[iarg],"-o")==0) File_Name_Out = argv[++iarg];
        else if (strcmp(argv[iarg],"-t")==0) time_limit = atof(argv[++iarg]);
        else if (strcmp(argv[iarg], "-a")==0) algorithm=atoi(argv[++iarg]);
        else if (strcmp(argv[iarg], "-k")==0) k_val=atoi(argv[++iarg]);//num_solutions
        else if (strcmp(argv[iarg], "-numsol")==0) num_solutions=atoi(argv[++iarg]);//num_solutions
        else if (strcmp(argv[iarg], "-neighbor")==0) neighbor_size=atoi(argv[++iarg]);//neighbor size
        iarg++;
    }

}

int main(int argc, char **argv)
{

         read_parameters(argc,argv);
         cout << " Instance declaration " << endl;
         // declare an instance 
         std::string instance_path=File_Name; //"../instances/RANDOM/64_16_58.txt"; //inst.txt";
         Instance* instance=new Instance();
         // read data
         instance->read_instance(instance_path);
         instance->print_instance_data();
         if(algorithm==0) // uniform sampling
         {
             cout << "=====================  uniform sampling ==================================" << endl;
             UniformSampling* us = new UniformSampling(instance);
             us->run(num_solutions, File_Name_Out, set<int>()); 
             //us->print();
             
         }else
         if(algorithm==1){
             cout << "=====================  Approximate algorithm ==================================" << endl;
             auto [approx_sol, solution, a_prim, Msigma] = approximate_algoritm_1(instance, File_Name_Out);
             cout << "Length: " << approx_sol << endl;
             cout << "LCS solution: " << solution << endl;
         }else
         if(algorithm==2){
         
             cout << "============= Run LS algorithm (S-k):  =========================" << endl;
             string solution = Local_search(instance, File_Name_Out, k_val);
             cout << "Solution size: "<< solution.size() << endl;
             cout << "LCS solution: "<< solution << endl;
         }else
          if(algorithm==3){
          
             cout << "============= Run Basic search algorithm  =========================" << endl;
             search_algorithm(instance, time_limit, File_Name_Out);
          }     
         
         delete instance;
         return 0;
}
