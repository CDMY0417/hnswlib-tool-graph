#include <iostream>
#include <vector>
#include <random>
#include <fstream>

#include "hnswlib/hnswlib.h"

// Filter: only allow neighbors whose input label matches source node's output label
struct IOFilter : public hnswlib::BaseFilterFunctor {
    const std::vector<int>& input_labels;
    const std::vector<int>& output_labels;

    int current_output = -1;

    IOFilter(const std::vector<int>& in, const std::vector<int>& out)
        : input_labels(in), output_labels(out) {}

    void set_current_source(hnswlib::labeltype src_label) {
        current_output = output_labels[src_label];
    }

    bool operator()(hnswlib::labeltype candidate_label) override {
        if (current_output < 0) return true;
        return input_labels[candidate_label] == current_output;
    }
};

int main() {
    // --- HNSW parameters ---
    const int dim             = 32;    // high-dimensional space (keep this!)
    const int max_elements    = 1000;  // number of points
    const int M               = 16;
    const int ef_construction = 200;
    const int K               = 8;     // neighbors per node for directed graph

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, max_elements, M, ef_construction);

    // --- random data and labels ---
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_vec(0.0f, 1.0f);
    std::uniform_int_distribution<int> dist_label(1, 5);

    std::vector<float> data(max_elements * dim);
    std::vector<int> input_label(max_elements);
    std::vector<int> output_label(max_elements);

    for (int i = 0; i < max_elements; ++i) {
        for (int d = 0; d < dim; ++d) {
            data[i * dim + d] = dist_vec(rng);
        }
        input_label[i]  = dist_label(rng);
        output_label[i] = dist_label(rng);
        index.addPoint(data.data() + i * dim, i);  // label = index
    }

    // --- build directed adjacency with IO constraint ---
    IOFilter filter(input_label, output_label);
    std::vector<std::vector<int>> adjacency(max_elements);

    for (int i = 0; i < max_elements; ++i) {
        filter.set_current_source(i);

        auto result = index.searchKnnCloserFirst(
            data.data() + i * dim,
            K,
            &filter
        );

        auto& nbrs = adjacency[i];
        nbrs.reserve(result.size());
        for (const auto& p : result) {
            int j = static_cast<int>(p.second);
            if (j == i) continue;  // no self-loop
            nbrs.push_back(j);
        }
    }

    // --- export nodes: id, input, output, d0..d(dim-1) ---
    {
        std::ofstream nf("nodes.csv");
        nf << "id,input,output";
        for (int d = 0; d < dim; ++d) {
            nf << ",d" << d;
        }
        nf << "\n";

        for (int i = 0; i < max_elements; ++i) {
            nf << i << "," << input_label[i] << "," << output_label[i];
            for (int d = 0; d < dim; ++d) {
                nf << "," << data[i * dim + d];
            }
            nf << "\n";
        }
    }

    // --- export edges: src,dst ---
    {
        std::ofstream ef("edges.csv");
        ef << "src,dst\n";
        for (int i = 0; i < max_elements; ++i) {
            for (int j : adjacency[i]) {
                ef << i << "," << j << "\n";
            }
        }
    }

    std::cout << "Wrote nodes.csv and edges.csv\n";
    return 0;
}
