#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>

#include "json.hpp"
#include "hnswlib/hnswlib.h"

using json = nlohmann::json;

struct Tool {
    std::string id;
    std::string desc;
    std::vector<std::string> input_types;
    std::vector<std::string> output_types;
    std::vector<float> embedding;
};

// --------- helpers ----------

std::vector<Tool> load_tools_from_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open JSON file: " + path);
    }

    json j;
    in >> j;

    std::vector<Tool> tools;
    for (const auto& node : j["nodes"]) {
        Tool t;
        t.id           = node.at("id").get<std::string>();
        t.desc         = node.at("desc").get<std::string>();
        t.input_types  = node.at("input-type").get<std::vector<std::string>>();
        t.output_types = node.at("output-type").get<std::vector<std::string>>();
        tools.push_back(std::move(t));
    }
    return tools;
}

std::unordered_map<std::string, std::vector<float>>
load_embeddings_by_id(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open embeddings CSV: " + path);
    }

    std::string header;
    std::getline(in, header);  // first line: column names

    std::unordered_map<std::string, std::vector<float>> emb_by_id;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;

        // first cell = id
        std::getline(ss, cell, ',');
        std::string id = cell;

        std::vector<float> emb;
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                emb.push_back(std::stof(cell));
            }
        }

        if (!emb.empty()) {
            emb_by_id.emplace(std::move(id), std::move(emb));
        }
    }

    return emb_by_id;
}

// Filter functor: allow candidate tool B only if
// output_types(A) == input_types(B) as sequences.
struct ToolTypeFilter : public hnswlib::BaseFilterFunctor {
    const std::vector<std::vector<std::string>>& input_seqs;
    const std::vector<std::vector<std::string>>& output_seqs;

    int current_src = -1;

    ToolTypeFilter(
        const std::vector<std::vector<std::string>>& in_seqs,
        const std::vector<std::vector<std::string>>& out_seqs
    ) : input_seqs(in_seqs), output_seqs(out_seqs) {}

    void set_current_source(hnswlib::labeltype src_label) {
        current_src = static_cast<int>(src_label);
    }

    bool operator()(hnswlib::labeltype candidate_label) override {
        if (current_src < 0) return true;
        int cand = static_cast<int>(candidate_label);

        const auto& out_seq = output_seqs[current_src];
        const auto& in_seq  = input_seqs[cand];

        return out_seq == in_seq;  // exact sequence match
    }
};

int main() {
    try {
        // -------- 1. Load tools and embeddings --------
        std::vector<Tool> tools = load_tools_from_json("multimedia_apis.json");
        std::cout << "Loaded " << tools.size() << " tools from JSON\n";

        auto emb_by_id = load_embeddings_by_id("tool_embeddings.csv");
        std::cout << "Loaded " << emb_by_id.size() << " embeddings\n";

        if (tools.empty()) {
            std::cerr << "No tools loaded.\n";
            return 1;
        }

        // Attach embeddings to tools
        int dim = -1;
        for (auto& t : tools) {
            auto it = emb_by_id.find(t.id);
            if (it == emb_by_id.end()) {
                throw std::runtime_error("No embedding found for tool id: " + t.id);
            }
            t.embedding = it->second;
            if (dim < 0) {
                dim = static_cast<int>(t.embedding.size());
            } else if (static_cast<int>(t.embedding.size()) != dim) {
                throw std::runtime_error("Inconsistent embedding dimension for tool id: " + t.id);
            }
        }

        std::cout << "Embedding dimension: " << dim << "\n";

        int num_tools = static_cast<int>(tools.size());

        // -------- 2. Build HNSW index --------
        int M = 16;
        int ef_construction = 200;
        int ef_search = 100;

        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float> index(&space, num_tools, M, ef_construction);

        for (int i = 0; i < num_tools; ++i) {
            index.addPoint(tools[i].embedding.data(), i);  // label = tool index
        }
        index.setEf(ef_search);

        // Prepare sequences for the filter
        std::vector<std::vector<std::string>> input_seqs(num_tools);
        std::vector<std::vector<std::string>> output_seqs(num_tools);
        for (int i = 0; i < num_tools; ++i) {
            input_seqs[i]  = tools[i].input_types;
            output_seqs[i] = tools[i].output_types;
        }

        ToolTypeFilter filter(input_seqs, output_seqs);

        // -------- 3. Build directed adjacency with type constraint --------
        int K = 10;  // max neighbors per tool
        std::vector<std::vector<int>> adjacency(num_tools);

        for (int i = 0; i < num_tools; ++i) {
            filter.set_current_source(i);

            const float* query_vec = tools[i].embedding.data();
            auto result = index.searchKnnCloserFirst(query_vec, K, &filter);

            auto& nbrs = adjacency[i];
            nbrs.reserve(result.size());
            for (const auto& p : result) {
                int j = static_cast<int>(p.second);
                if (j == i) continue;  // avoid self-loop
                nbrs.push_back(j);
            }
        }

        // -------- 4. Export graph edges to CSV --------
        std::ofstream out("tool_graph_edges.csv");
        out << "src_id,dst_id\n";
        for (int i = 0; i < num_tools; ++i) {
            for (int j : adjacency[i]) {
                out << tools[i].id << "," << tools[j].id << "\n";
            }
        }
        std::cout << "Wrote tool_graph_edges.csv\n";

        // Optional: print a few neighbors
        for (int i = 0; i < num_tools; ++i) {
            std::cout << "Tool: " << tools[i].id << "  (";
            for (size_t k = 0; k < tools[i].output_types.size(); ++k) {
                if (k) std::cout << ", ";
                std::cout << tools[i].output_types[k];
            }
            std::cout << ") -> ";
            for (int j : adjacency[i]) {
                std::cout << tools[j].id << "  ";
            }
            std::cout << "\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
