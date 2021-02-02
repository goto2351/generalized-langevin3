#include "./simulator.hpp"

int main (int argc, char* argv[]) {
    std::string input_toml_path = argv[1];
    generalized_langevin::Simulator simulator(input_toml_path);
    simulator.run();
    return 0;
}