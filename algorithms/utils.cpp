// utils.cpp
#include "utils.h"
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <random>

using namespace std;

string backtrack(const string& a, const string& b, vector<vector<int>>& LCS)
{

    int i = a.size();
    int j = b.size();
    string sol = "";  

    while (i > 0 && j > 0)
    {
        if (a[i - 1] == b[j - 1] && a[i-1] != '*') // skip the symbol '*'
        {
            sol = a[i - 1] + sol;
            i--;
            j--;  
        }
        else
        {
            if (LCS[i][j] == LCS[i - 1][j])
                i--;
            else
                j--;
        }
    }

    assert(sol.size() == LCS[a.size()][b.size()] && "Not good LCS case!!!");
    return sol;
}

std::tuple<int, string> determine_LCS(const string& a, const string& b)
{
    // Initialization:
    vector<vector<int>> LCS(a.size() + 1, vector<int>(b.size() + 1, 0));

    // Run the DP-recursion:
    for (int i = 0; i < a.size(); ++i)
    {
        for (int j = 0; j < b.size(); ++j)
        {
            if ( a[i] == b[j] && a[i] != '*') //do not consider '*' (deleted letters in A)
                LCS[i + 1][j + 1] = 1 + LCS[i][j];
            else
                LCS[i + 1][j + 1] = std::max(LCS[i + 1][j], LCS[i][j + 1]);
        }
    }

    // Backtracking to extract the solution:
    string a_prime_B_solution = backtrack(a, b, LCS);
    //reversed:
    //std::reverse(a_prime_B_solution.begin(), a_prime_B_solution.end());  // Reverse the string
    return {LCS[(int)a.size()][(int)b.size()], a_prime_B_solution};
}


void save_in_file(string& path_to_file, const string& a_prime, const string& LCS_solution, float time) {
    // Open the file in write mode
    ofstream file(path_to_file);
    
    // Check if the file is successfully opened
    if (!file) {
        cerr << "Error: Could not open file at " << path_to_file << endl;
        return;
    }
    
    // Write the values to the file
    file << "a_prime: " << a_prime << "\n";
    file << "LCS_solution: " << LCS_solution << "\n";
    file<<  "Size: " << LCS_solution.size() << "\n";
    file << "Time: " << time;
    
    // Close the file
    file.close();
    
    // Confirmation message
    cout << "Data successfully saved to " << path_to_file << endl;
}

/** Helper function for designing the ILP **/
void save_in_file_ILP(std::string& path_to_file, const std::string& LCS_solution, float time, int status, float gap, const std::string& other_stats) {
    // Open the file in write mode
    std::ofstream file(path_to_file);
    
    // Check if the file is successfully opened
    if (!file) {
        std::cerr << "Error: Could not open file at " << path_to_file << std::endl;
        return;
    }
    // Write the values to the file
    file << "LCS_solution: " << LCS_solution << "\n";
    file << "Size: " << LCS_solution.size() << "\n";
    file << "Time: " << time << "\n";
    file << "Status: " <<  status <<"\n";
    file << "Gap: " <<  gap <<"\n";
    if(other_stats != " ")
        file << other_stats << "\n";
    // Close the file
    file.close();
    std::cout << "Data successfully saved to " << path_to_file << std::endl;
}
// APPROX alg. from Castelli et al.~[9], Chapter 4 (do we implementing Approximate-algorithm-2?)  
pair<string, unordered_map<char, int>> greedy_maximize_matching(const string& a_prime, unordered_map<char, int>& M_sigma)
{
    //Copy M_sigma;
    unordered_map<char, int> M_sigma_copy;
    
    for(auto key_value: M_sigma)
        M_sigma_copy[key_value.first] = key_value.second;
    
    string aux="";
    for(int i=0; i<a_prime.size(); ++i)
    {    
        if(M_sigma_copy.find( a_prime[i] ) !=  M_sigma_copy.end())
        {
           aux += a_prime[i]; 
           M_sigma_copy[ a_prime[i] ]--;
        }else
           aux += '*';
           
        if(M_sigma_copy[ a_prime[i] ] == 0) // remove the key a_prime[i];
           M_sigma_copy.erase(a_prime[i]);
    }
    return {aux, M_sigma_copy};
}

vector<int> greedy_maximize_matching_with_allowed_y_vars(const string& a_prime, unordered_map<char, int>& M_sigma, set<int>& merge_y)
{
      
     unordered_map<char, vector<int>> chars_allowed_list_pos;
     for(int i=0; i<a_prime.size(); ++i)
     {
          if(merge_y.find(i) == merge_y.end()) // i not in merge_y 
             continue;

          if(chars_allowed_list_pos.find(a_prime[i]) != chars_allowed_list_pos.end())
                chars_allowed_list_pos[a_prime[i]].push_back(i); 
          else{
               
               chars_allowed_list_pos[a_prime[i]] = vector<int>();
               chars_allowed_list_pos[a_prime[i]].push_back(i);
          }
     }
     vector<int> max_match_positions_y;
     // position of char occurences w.r.t. y vars (allowed positions in A):
     // Initialize a random number generator
     random_device rd;  // Random seed
     mt19937 rng(rd()); // Mersenne Twister PRNG

     for(auto& [ch, vec_pos]: chars_allowed_list_pos){

         //std::shuffle(vec_pos.begin(), vec_pos.end(), rng);
         int allowed_ch=M_sigma[ch];
         if(vec_pos.size() < allowed_ch)
            allowed_ch = vec_pos.size();
         // allowed_ch
         for(auto it=vec_pos.begin(); it != vec_pos.begin() + allowed_ch; ++it)
             max_match_positions_y.push_back(*it);
     }
     //max_match_positions_y: reconstruction of string to be returned;
     return max_match_positions_y;

}
