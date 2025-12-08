#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>

#include "json.hpp"
#include "hnswlib/hnswlib.h"

using json = nlohmann::json;

struct Tool {
    std::string library;               // which JSON this came from
    std::string id;                    // tool id
    std::string desc;
    std::vector<std::string> input_types;
    std::vector<std::string> output_types;
    std::vector<float> embedding;
};

void load_tools_from_json_file(
    const std::string& path,
    const std::string& library_name,
    std::vector<Tool>& out
) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open JSON file: " + path);
    }
    json j;
    in >> j;

    for (const auto& node : j["nodes"]) {
        Tool t;
        t.library      = library_name;
        t.id           = node.at("id").get<std::string>();
        t.desc         = node.at("desc").get<std::string>();
        t.input_types  = node.at("input-type").get<std::vector<std::string>>();
        t.output_types = node.at("output-type").get<std::vector<std::string>>();
        out.push_back(std::move(t));
    }
}

std::string make_tool_key(const std::string& lib, const std::string& id) {
    return lib + "::" + id;
}

std::unordered_map<std::string, std::vector<float>>
load_embeddings_all(const std::string& csv_path) {
    std::ifstream in(csv_path);
    if (!in) {
        throw std::runtime_error("Cannot open embeddings CSV: " + csv_path);
    }

    std::string header;
    std::getline(in, header);

    std::unordered_map<std::string, std::vector<float>> emb_by_key;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;

        // first column: library
        std::getline(ss, cell, ',');
        std::string library = cell;

        // second column: id
        std::getline(ss, cell, ',');
        std::string id = cell;

        std::vector<float> emb;
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                emb.push_back(std::stof(cell));
            }
        }

        if (!emb.empty()) {
            emb_by_key.emplace(make_tool_key(library, id), std::move(emb));
        }
    }
    return emb_by_key;
}

struct ToolTypeFilter : public hnswlib::BaseFilterFunctor {
    const std::vector<std::vector<std::string>>& input_seqs;
    const std::vector<std::vector<std::string>>& output_seqs;
    int current_src = -1;

    ToolTypeFilter(
        const std::vector<std::string>&,
        const std::vector<std::string>&
    ) : input_seqs(*(const std::vector<std::vector<std::string>>*)nullptr),
        output_seqs(*(const std::vector<std::vector<std::string>>*)nullptr) {}

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
        return out_seq == in_seq;
    }
};

int main() {
    try {
        std::vector<Tool> tools;
        load_tools_from_json_file("multimedia_apis.json",  "multimedia", tools);
        load_tools_from_json_file("daily_apis.json",       "daily",      tools);
        load_tools_from_json_file("huggingface_apis.json", "huggingface", tools);

        std::cout << "Total tools: " << tools.size() << "\n";

        auto emb_by_key = load_embeddings_all("tool_embeddings_all.csv");

        int dim = -1;
        for (auto& t : tools) {
            std::string key = make_tool_key(t.library, t.id);
            auto it = emb_by_key.find(key);
            if (it == emb_by_key.end()) {
                throw std::runtime_error("No embedding for tool: " + key);
            }
            t.embedding = it->second;
            if (dim < 0) {
                dim = static_cast<int>(t.embedding.size());
            } else if (static_cast<int>(t.embedding.size()) != dim) {
                throw std::runtime_error("Inconsistent embedding dim for: " + key);
            }
        }
        std::cout << "Embedding dimension: " << dim << "\n";

        int num_tools = static_cast<int>(tools.size());

        std::vector<float> norms(num_tools);
        for (int i = 0; i < num_tools; ++i) {
            const auto& v = tools[i].embedding;
            float sum_sq = 0.0f;
            for (float x : v) sum_sq += x * x;
            norms[i] = std::sqrt(sum_sq);
        }

        auto cosine_similarity = [&](int a, int b) {
            const float* va = tools[a].embedding.data();
            const float* vb = tools[b].embedding.data();
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) {
                dot += va[d] * vb[d];
            }
            float denom = norms[a] * norms[b];
            if (denom == 0.0f) return 0.0f;
            return dot / denom;
        };

        int M = 16;
        int ef_construction = 200;
        int ef_search = 100;

        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float> index(&space, num_tools, M, ef_construction);

        for (int i = 0; i < num_tools; ++i) {
            index.addPoint(tools[i].embedding.data(), i);
        }
        index.setEf(ef_search);

        std::vector<std::vector<std::string>> input_seqs(num_tools);
        std::vector<std::vector<std::string>> output_seqs(num_tools);
        for (int i = 0; i < num_tools; ++i) {
            input_seqs[i]  = tools[i].input_types;
            output_seqs[i] = tools[i].output_types;
        }

        ToolTypeFilter filter(input_seqs, output_seqs);

        int K = 32;                  // search width
        float sim_threshold = 0.3f;   // tune threshold

        std::vector<std::vector<int>> adjacency(num_tools);

        for (int i = 0; i < num_tools; ++i) {
            filter.set_current_source(i);
            const float* q = tools[i].embedding.data();
            auto result = index.searchKnnCloserFirst(q, K, &filter);

            auto& nbrs = adjacency[i];
            for (const auto& p : result) {
                int j = static_cast<int>(p.second);
                if (j == i) continue;

                float sim = cosine_similarity(i, j);
                if (sim >= sim_threshold) {
                    nbrs.push_back(j);
                }
            }
        }

        std::ofstream nodes("layered_nodes.csv");
        nodes << "node_id,node_type,library,tool_id\n";
        nodes << "ENTRY,entry,,\n";
        std::vector<std::string> libraries = { "multimedia", "daily", "huggingface" };
        for (const auto& lib : libraries) {
            nodes << "LIB:" << lib << ",library," << lib << ",\n";
        }

        for (int i = 0; i < num_tools; ++i) {
            std::string node_id = "TOOL:" + tools[i].library + "::" + tools[i].id;
            nodes << node_id << ",tool," << tools[i].library << "," << tools[i].id << "\n";
        }
        nodes.close();

        std::ofstream edges_same("layered_edges_same_lib.csv");
        edges_same << "src,dst,edge_type\n";

        std::ofstream edges_all("layered_edges_all_lib.csv");
        edges_all << "src,dst,edge_type\n";

        for (const auto& lib : libraries) {
            std::string lib_node = "LIB:" + lib;
            edges_same << "ENTRY," << lib_node << ",entry_to_lib\n";
            edges_all  << "ENTRY," << lib_node << ",entry_to_lib\n";
        }

        for (int i = 0; i < num_tools; ++i) {
            std::string tool_node = "TOOL:" + tools[i].library + "::" + tools[i].id;
            std::string lib_node  = "LIB:" + tools[i].library;
            edges_same << lib_node << "," << tool_node << ",lib_to_tool\n";
            edges_all  << lib_node << "," << tool_node << ",lib_to_tool\n";
        }

        for (int i = 0; i < num_tools; ++i) {
            std::string src_node = "TOOL:" + tools[i].library + "::" + tools[i].id;
            for (int j : adjacency[i]) {
                std::string dst_node = "TOOL:" + tools[j].library + "::" + tools[j].id;

                edges_all << src_node << "," << dst_node << ",tool_to_tool\n";

                if (tools[i].library == tools[j].library) {
                    edges_same << src_node << "," << dst_node << ",tool_to_tool\n";
                }
            }
        }

        edges_same.close();
        edges_all.close();

        std::cout << "Wrote layered_nodes.csv\n";
        std::cout << "Wrote layered_edges_same_lib.csv (same-lib edges)\n";
        std::cout << "Wrote layered_edges_all_lib.csv  (all-lib edges)\n";

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
