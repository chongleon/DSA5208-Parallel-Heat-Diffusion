#include <mpi.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <climits>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3 && argc != 4) {
        if (rank == 0) std::cerr << "Usage: ./mpi N Steps [--validate]\n";
        MPI_Finalize();
        return 1;
    }
    bool validate = false;
    if (argc == 4) {
        if (std::string(argv[3]) != "--validate") {
            if (rank == 0) std::cerr << "Unknown option.\n";
            MPI_Finalize();
            return 1;
        }
        validate = true;
    }

    int N = 0, Steps = 0;
    try {
        std::string n_arg(argv[1]), steps_arg(argv[2]);
        size_t pos_n = 0, pos_steps = 0;
        N = std::stoi(n_arg, &pos_n);
        Steps = std::stoi(steps_arg, &pos_steps);
        if (pos_n != n_arg.size() || pos_steps != steps_arg.size())
            throw std::invalid_argument("Invalid integer");
    } catch (const std::exception&) {
        if (rank == 0) std::cerr << "N and Steps must be integers.\n";
        MPI_Finalize();
        return 1;
    }

    if (N < 5 || Steps < 0 || size > N || 1LL * N * N > INT_MAX) {
        if (rank == 0) std::cerr << "Invalid grid or rank configuration.\n";
        MPI_Finalize();
        return 1;
    }

    // 1D row decomposition, including uneven partitions.
    int local_rows = N / size + (rank < N % size ? 1 : 0);
    int start_row = rank * (N / size) + std::min(rank, N % size);

    // Two ghost rows are included in each local grid.
    std::vector<double> old_grid(static_cast<size_t>(local_rows + 2) * N, 0.0);
    std::vector<double> new_grid(static_cast<size_t>(local_rows + 2) * N, 0.0);

    int hot_size = N / 5;
    int hot_start = (N - hot_size) / 2;
    int hot_end = hot_start + hot_size;
    for (int i = 1; i <= local_rows; ++i) {
        int global_i = start_row + i - 1;
        for (int j = 0; j < N; ++j) {
            if (global_i >= hot_start && global_i < hot_end &&
                j >= hot_start && j < hot_end)
                old_grid[static_cast<size_t>(i) * N + j] = 100.0;
        }
    }

    int up = rank == 0 ? MPI_PROC_NULL : rank - 1;
    int down = rank == size - 1 ? MPI_PROC_NULL : rank + 1;
    const double lambda = 0.2;
    double local_comm = 0.0, local_compute = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int step = 0; step < Steps; ++step) {
        double t0 = MPI_Wtime();
        MPI_Sendrecv(
            old_grid.data() + N, N, MPI_DOUBLE, up, 0,
            old_grid.data() + static_cast<size_t>(local_rows + 1) * N,
            N, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(
            old_grid.data() + static_cast<size_t>(local_rows) * N,
            N, MPI_DOUBLE, down, 1,
            old_grid.data(), N, MPI_DOUBLE, up, 1,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_comm += MPI_Wtime() - t0;

        t0 = MPI_Wtime();
        for (int i = 1; i <= local_rows; ++i) {
            int global_i = start_row + i - 1;
            if (global_i == 0 || global_i == N - 1) continue;
            for (int j = 1; j < N - 1; ++j) {
                size_t idx = static_cast<size_t>(i) * N + j;
                double center = old_grid[idx];
                double top = old_grid[idx - N];
                double bottom = old_grid[idx + N];
                double left = old_grid[idx - 1];
                double right = old_grid[idx + 1];
                new_grid[idx] = center + lambda *
                    (top + bottom + left + right - 4.0 * center);
            }
        }
        old_grid.swap(new_grid);
        local_compute += MPI_Wtime() - t0;
    }

    double local_runtime = MPI_Wtime() - start_time;
    double runtime = 0.0, communication = 0.0, computation = 0.0;
    MPI_Reduce(&local_runtime, &runtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comm, &communication, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_compute, &computation, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (validate) {
        std::vector<int> recvcounts, displs;
        std::vector<double> full_grid;
        if (rank == 0) {
            recvcounts.resize(size);
            displs.resize(size);
            full_grid.resize(static_cast<size_t>(N) * N);
            for (int r = 0; r < size; ++r) {
                int rows = N / size + (r < N % size ? 1 : 0);
                int start = r * (N / size) + std::min(r, N % size);
                recvcounts[r] = rows * N;
                displs[r] = start * N;
            }
        }
        MPI_Gatherv(old_grid.data() + N, local_rows * N, MPI_DOUBLE,
                    rank == 0 ? full_grid.data() : nullptr,
                    rank == 0 ? recvcounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::ofstream file("mpi_result.csv");
            if (!file) {
                std::cerr << "Failed to open mpi_result.csv\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            file << std::setprecision(17);
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    if (j) file << ',';
                    file << full_grid[static_cast<size_t>(i) * N + j];
                }
                file << '\n';
            }
            file.close();
            if (!file) {
                std::cerr << "Failed to write CSV.\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    if (rank == 0) {
        std::cout << "N=" << N << "\nSteps=" << Steps
                  << "\nRanks=" << size << '\n';
        std::cout << std::fixed << std::setprecision(10)
                  << "Runtime=" << runtime << '\n'
                  << "CommunicationTime=" << communication << '\n'
                  << "ComputationTime=" << computation << '\n';
    }
    MPI_Finalize();
    return 0;
}
