// Instance.h
#ifndef INSTANCE_H
#define INSTANCE_H

#include <string>
#include <vector>
#include <set>
#include <unordered_map>


class Instance {

private:
    int sigma;
    int size_m;
    std::string A;
    std::string B;
    std::string M; 
    std::vector<std::pair<int, int>> matchings;

public:
    std::unordered_map<char, int> M_sigma;
    int number_of_matched; // number of already matched symbols from M to A (i.e. the number of asterisk symbols in A)
    std::unordered_map<int, char> string_matched; // character matched (ordered increasingly acc. to positions in @A)
    std::set<char> sigmas; // alphabet of the instance
    std::unordered_map<char, std::vector<int>> char_to_positions_in_A; // char: positions = for each char c, its positions in A are assigned

    
public:
    // Constructor
    Instance();
    // Another constructor   
    Instance(std::string Aprim, std::string Bprim, std::unordered_map<char, int>& M_init, std::string positions_matched_from_M, int sig, int matched_symb_from_M);

    // Destructor
    ~Instance();

    // Function to read the instance from a file
    void read_instance(const std::string& path);

    // Getters for debugging or external access
    int get_sigma() const;
    int get_m_size() const;
    std::string get_A() const;
    std::string get_B() const;
    std::string get_M() const;
    std::vector<std::pair<int, int>> get_matchings() const;
  
    void count_letters_in_M_and_sigma();
    // print data:
    void print_instance_data() const;
    void all_matchings();
    void char_positions_in_A();
    bool empty_M() const;
};
 #endif
