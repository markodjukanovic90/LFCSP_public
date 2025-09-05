#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <tuple>  
#include <unordered_map>
#include <set>

std::string backtrack(const std::string& a, const std::string& b, std::vector<std::vector<int>>& LCS);
std::tuple<int, std::string> determine_LCS(const std::string& a, const std::string& b);
void save_in_file(std::string& path_to_file, const std::string& a_prime, const std::string& LCS_solution, float time);
void save_in_file_ILP(std::string& path_to_file, const std::string& LCS_solution, float time, int status, float gap, const std::string& others );
std::pair<std::string, std::unordered_map<char, int>> greedy_maximize_matching(const std::string& a_prime, std::unordered_map<char, int>& M_sigma);
std::vector<int>  greedy_maximize_matching_with_allowed_y_vars(const std::string& a_prime, 
                                                          std::unordered_map<char, int>& M_sigma, std::set<int>& merge_y);

#endif // UTILS_H

