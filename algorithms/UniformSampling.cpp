// UniformSampling.cpp
#include "UniformSampling.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <algorithm>
#include <random>
#include <assert.h>

//define the methods:

UniformSampling::UniformSampling(Instance* in) : solution(""), instance(in){};

std::tuple<int, string, string, vector<pair<int, int>>> UniformSampling::sample(const set<int>& sampled_fix =  std::set<int>() )
{   
    set<int> sampled;

    if(sampled_fix.size() == 0)
    {
        unordered_map<char, vector<int>>  chs_list_indices; // char: list_indices
        set<char> sigma;    
        // Create a random engine
        std::random_device rd;
        std::mt19937 gen(rd());
        // determine sigma from M:
        for(auto ch: this->instance->get_M())
            sigma.insert(ch);
        // initialize chs_list_indices
        for(char c: sigma)
            chs_list_indices[c] = vector<int>();
        
        // fill in chs_list_indices
        for(int i=0; i<this->instance->get_A().size(); ++i)
            if(sigma.find( instance->get_A()[i] ) !=  sigma.end())
                chs_list_indices[ this->instance->get_A()[i] ].push_back(i);
    
        // now insert into <sample> set:
        for(char c: sigma)
        {
            if(chs_list_indices[c].size() <= this->instance->M_sigma[c])
                for(auto cc: chs_list_indices[c])
                    sampled.insert(cc);
             else{ // sample from vector chs_list_indices[c],  instance.M_sigma[c] of indices
               // Shuffle the vector
               std::shuffle(chs_list_indices[c].begin(), chs_list_indices[c].end(), gen);
               // Select the first p elements from the shuffled vector -- random matching
               sampled.insert(chs_list_indices[c].begin(), chs_list_indices[c].begin() + this->instance->M_sigma[c]);
            }
        }
    }else{ // copy to sampled_fix:
        for(auto& x: sampled_fix)
            sampled.insert(x);
    }

    //STEP 3: take the sampled indices (@sampled) -- create A';
    string a_prime_with_asterix=""; string a_prime="";// needed for the export
    for(int i=0; i < this->instance->get_A().size(); ++i)
    {
        if(sampled.find(i) == sampled.end()) // this position is not sampled, concatenate the character at the position to s_prime
        {  
            a_prime +=  this->instance->get_A()[i]; 
            a_prime_with_asterix += this->instance->get_A()[i];
        }else  
            a_prime_with_asterix += "*";
    }
    
    //STEP 4: calculate LCS between A' and B;
    auto [LCS_a_prime_B, a_prime_B_sol] =  determine_LCS(a_prime_with_asterix, this->instance->get_B());
    
    int final_count =  sampled.size() + LCS_a_prime_B;
    int current=0;   
    vector<int> pos_x_vector;
    for(int i=0; i < this->instance->get_A().size(); ++i)
    {
        if(sampled.find(i) == sampled.end() && this->instance->get_A()[i] == a_prime_B_sol[current])
        {
            current++;  
            sampled.insert(i);
            pos_x_vector.push_back(i);

            if(current == LCS_a_prime_B)
               break; 
        }
    }
    // return matchings of a_prime_B_sol:
    int current_matched=0; // count the number of matched letters in the resulting @a_prime_B_sol
      
    vector<pair<int, int>> matching_A_B;
    //vector<int> pos_x_vector; 
    vector<int> pos_y_vector;
    int pointer_y=0;
    
    while(true)
    {
         //find the matching with B
         while(pointer_y < this->instance->get_B().size() && this->instance->get_B()[pointer_y] != a_prime_B_sol[current_matched]) // when the matching happens with current_x
             pointer_y++;
        
         if(pointer_y == this->instance->get_B().size())
             break;
         
         pos_y_vector.push_back(pointer_y); 
         pointer_y++;  current_matched++;


         if(current_matched == this->instance->get_B().size())
             break;
    }       
    //raise an error
    assert(current_matched == a_prime_B_sol.size() && "No full match, something is wrong here!!!");     
        
    for(int i=0; i<current_matched; ++i)
         matching_A_B.push_back({pos_x_vector[i], pos_y_vector[i]});       
   
    // STEP 5: final solution string construction (from the sampled set):
    string solution_final="";
    vector<int> pos_match_sorted; pos_match_sorted.insert(pos_match_sorted.end(), sampled.begin(), sampled.end());
    std::sort(pos_match_sorted.begin(), pos_match_sorted.end());
    
    for(int index: pos_match_sorted)
        solution_final += instance->get_A()[index];
    
    //assert(solution_final.size() == final_count );
    //save solution and pass it further:
    return {final_count, a_prime_with_asterix, solution_final, matching_A_B};
}

void UniformSampling::save(string& out_path)
{
     save_in_file(out_path, this->a_prime, this->solution, this->runtime); 
}

void UniformSampling::run(int number_solution_generated, string out_path, const set<int>& sample_fix= set<int>()) // the main function that generates a solution:
{
 
    int counter=0; int max_sol_length=0; 
    string largest_solution=""; string a_prime_sol_final=""; 
    vector<pair<int, int>> match_A_B_best;
    //start the counter
    auto start = std::chrono::high_resolution_clock::now();
    //run the generation of solutions:
    while(counter < number_solution_generated)
    {
        auto [sol_length, a_prime_sol,  sol_final, positions_A_B] = this->sample(sample_fix);
        if(max_sol_length < sol_length){
            
            max_sol_length = sol_length;
            largest_solution=sol_final;
            a_prime_sol_final=a_prime_sol;
            match_A_B_best = positions_A_B;
            
        }
        counter++;   
    }
    // Record the end time
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the duration in milliseconds
    std::chrono::duration<float> duration = end - start;
    //stats:
    this->a_prime = a_prime_sol_final;
    this->solution = largest_solution;
    this->runtime = duration.count();
    
    //M_with_A_mathchings: -- to fill in (positions of '*' in a_prime):
    for(int i=0; i<this->a_prime.size(); ++i)
        if(this->a_prime[i] == '*')
            this->M_with_A_matchings.push_back(i);
    //A_with_B_matching: fill:             match_A_B_best  
    for(auto& pair: match_A_B_best)
       this->A_with_B_matching.push_back(pair);

       if(out_path != "")
           this->save(out_path);
}


void UniformSampling::print()
{

    cout << "A': " << this->a_prime << endl;
    cout << "|LCS|=" << this->solution << endl;
    cout << "LCS solution: " << this->solution.size() << endl;
    for(auto x_y: this->A_with_B_matching)
        cout << " (" << x_y.first << " " << x_y.second <<  ") " << endl;
    
    cout<<endl;
}
// destructor
UniformSampling::~UniformSampling()
{
    
}


