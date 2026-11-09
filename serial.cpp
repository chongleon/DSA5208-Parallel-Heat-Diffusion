
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

// 2D Heat Equation - Serial Implementation
// Numerical method: Explicit finite difference (5-point stencil)
// Boundary conditions: Fixed Dirichlet, T = 0
// Precision: double

constexpr double LAMBDA = 0.2;
constexpr double HOT_TEMPERATURE = 100.0;

// Convert 2D coordinates to 1D row-major index
inline std::size_t index(int i, int j, int N) {
    return static_cast<std::size_t>(i) * N + j;
}

// Initialize the temperature grid
void initialize_grid(std::vector<double>& grid, int N) {
    const int L = N / 5;
    const int start = (N - L) / 2;

    for (int i = start; i < start + L; ++i) {
        for (int j = start; j < start + L; ++j) {
            grid[index(i, j, N)] = HOT_TEMPERATURE;
        }
    }
}

// Perform one iteration of the 2D heat equation
void update_grid(
    const std::vector<double>& old_grid,
    std::vector<double>& new_grid,
    int N
) {
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            const std::size_t idx = index(i, j, N);

            new_grid[idx] = old_grid[idx] + LAMBDA * (
                old_grid[idx - N] +
                old_grid[idx + N] +
                old_grid[idx - 1] +
                old_grid[idx + 1] -
                4.0 * old_grid[idx]
            );
        }
    }
}

// Write the final N x N temperature grid to CSV
void write_csv(const std::vector<double>& grid, int N) {
    std::ofstream file("serial_result.csv");

    if (!file.is_open()) {
        throw std::runtime_error(
            "Failed to open serial_result.csv"
        );
    }

    file << std::setprecision(17);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (j > 0) {
                file << ",";
            }

            const double value = grid[index(i, j, N)];

            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "NaN or Infinity detected in result"
                );
            }

            file << value;
        }
        file << "\n";
    }

    if (!file) {
        throw std::runtime_error("Failed to write CSV");
    }
}

int main(int argc, char* argv[]) {
    try {
        // 1. Parse command-line arguments
        if (argc != 3 && argc != 4) {
            std::cerr
                << "Usage: ./serial N Steps [--validate]\n";
            return 1;
        }

        const std::string n_arg = argv[1];
        const std::string steps_arg = argv[2];

        std::size_t pos_n = 0;
        std::size_t pos_steps = 0;

        const long long n_value = std::stoll(n_arg, &pos_n);
        const long long steps_value =
            std::stoll(steps_arg, &pos_steps);

        if (pos_n != n_arg.size() ||
            pos_steps != steps_arg.size() ||
            n_value < 2 ||
            n_value > std::numeric_limits<int>::max() ||
            steps_value < 0 ||
            steps_value > std::numeric_limits<int>::max()) {
            throw std::invalid_argument(
                "Invalid N or Steps"
            );
        }

        const int N = static_cast<int>(n_value);
        const int Steps = static_cast<int>(steps_value);

        bool validate = false;

        if (argc == 4) {
            if (std::string(argv[3]) != "--validate") {
                throw std::invalid_argument(
                    "Unknown argument"
                );
            }
            validate = true;
        }

        // Prevent integer overflow in grid allocation
        const std::size_t n = static_cast<std::size_t>(N);

        if (n > std::vector<double>().max_size() / n) {
            throw std::length_error("Grid size too large");
        }

        // 2. Allocate two contiguous temperature grids
        std::vector<double> old_grid(n * n, 0.0);
        std::vector<double> new_grid(n * n, 0.0);

        // 3. Initialize the central hot region
        initialize_grid(old_grid, N);

        // 4. Measure iterative solver runtime only
        const auto start_time =
            std::chrono::steady_clock::now();

        for (int step = 0; step < Steps; ++step) {
            update_grid(old_grid, new_grid, N);
            old_grid.swap(new_grid);
        }

        const auto end_time =
            std::chrono::steady_clock::now();

        const double runtime =
            std::chrono::duration<double>(
                end_time - start_time
            ).count();

        // 5. Output standardized benchmark information
        std::cout << "N=" << N << "\n";
        std::cout << "Steps=" << Steps << "\n";
        std::cout << std::setprecision(10);
        std::cout << "Runtime=" << runtime << "\n";

        // 6. Validation output (outside timing scope)
        if (validate) {
            write_csv(old_grid, N);
            std::cout << "Validation=serial_result.csv\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
