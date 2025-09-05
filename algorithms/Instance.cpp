// Instance.cpp
#include "Instance.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <limits>

// Constructor
Instance::Instance() : sigma(0), size_m(0), A(""), B(""), M("") , number_of_matched(0){};

// Another constructor -- build an instance from the old one -- A_init=A, B_init=B; M_init = M affected by the used (i.e. deleted) positions @positions_matched_char
// Solution is A', that maximizes the sum between |A|-|A'| and LCS(A', B) ... 

Instance::Instance(std::string A_init, std::string B_init, std::unordered_map<char, int>& M_sigma_init, std::string string_matched, int sigma, int matched_symb_from_M)
 {
     //Copy to A, and B attributes:
     this->A=A_init;
     this->B=B_init;
     this->M="";
     
     //initialize (copy) M_sigma:
     for(auto char_int: M_sigma_init)
         this->M_sigma[char_int.first] = char_int.second;    
         
     //Decrease the used symbols from M_sigma according to already used symbols with positions from @positions_matched_char:
     for(int i=0; i < string_matched.size(); ++i)
     {   
         this->M_sigma[ string_matched[i] ] =  this->M_sigma[ string_matched[i] ] - 1;
         if(this->M_sigma[ string_matched[i] ] <= 0)
            this->M_sigma.erase( string_matched[i] ); // remove char @c from M (cannot be used anymore)
     }
     //Update M:
     for(auto char_int: this->M_sigma)
     {
         for(int rep=0; rep < char_int.second; ++rep)
             this->M += char_int.first;
     }
     this->number_of_matched=matched_symb_from_M;
     this->number_of_matched+=string_matched.size();
     this->sigma=sigma;
 }

// Destructor
Instance::~Instance() {
    // No dynamic memory to release, but cleanup can be added here if necessary
    this->M_sigma.clear();
}

// Function to read the instance from a file
void Instance::read_instance(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + path);
    }

    // Read the first line
    file >> sigma >> size_m;
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Ignore the rest of the line

    // Read the second line for A
    std::getline(file, A);

    // Read the third line for B
    std::getline(file, B);

    // Read the line for M
    std::getline(file, M);

    file.close();

    // M_sigma fill in
    this->count_letters_in_M_and_sigma();
    //all_matchings();
    char_positions_in_A();

}

// Getters
int Instance::get_sigma() const {
    return sigma;
}

int Instance::get_m_size() const {
    return size_m;
}

std::string Instance::get_A() const {
    return A;
}

std::string Instance::get_B() const {
    return B;
}

std::string Instance::get_M() const {
    return M;
}

std::vector<std::pair<int, int>> Instance::get_matchings() const
{

    return matchings;
}


// Helper function to count occurrences of letters in M
void Instance::count_letters_in_M_and_sigma() {

    for (char ch : M) {
        M_sigma[ch]=0;
    }
    for (char ch : M) {
        M_sigma[ch]++;
    }
    //letters:
    for(char c: this->A)
        this->sigmas.insert(c);
    for(char c: this->B)
        this->sigmas.insert(c);
     for(char c: this->M)
        this->sigmas.insert(c);   
}

bool Instance::empty_M() const{

    return this->M.size() == 0;
}

void Instance::all_matchings()
{
    for(int i=0; i < this->A.size(); ++i)
        for(int j=0; j < this->B.size(); ++j)
        {
            if( this->A[i] == this->B[j])
                this->matchings.push_back({i, j});
        }
}

void Instance::char_positions_in_A()
{

    for(int i=0; i < this->A.size(); ++i)
    {
        if(char_to_positions_in_A.find(this->A[i]) == char_to_positions_in_A.end())
        {
            char_to_positions_in_A[ this->A[i] ] = std::vector<int>();
            char_to_positions_in_A[ this->A[i] ].push_back(i);
        }
        else{
            char_to_positions_in_A[ this->A[i] ].push_back(i);
        }
    }
}

// print data:
void Instance::print_instance_data() const{

    std::cout << "====================== Data od this instance: ======================" << std::endl;
    std::cout << "Sigma size: " << sigma << std::endl;
    std::cout << "A: " << A << std::endl;
    std::cout << "B: " << B << std::endl;
    std::cout << "M: "; // { " << M << "}" << std::endl;
    for (size_t i = 0; i < M.size(); ++i) {
        std::cout << M[i];
        if (i < M.size() - 1) {
            std::cout << ",";
        }
    }
    std::cout << std::endl;
    
    for(auto char_int: this->M_sigma)
        std::cout << char_int.first << " " << char_int.second << std::endl;
    std::cout << std::endl;
    std::cout << "\n=================" << M.size() << " " << M_sigma.size() << std::endl;
    std::cout << "===========================================================" << std::endl;

}
