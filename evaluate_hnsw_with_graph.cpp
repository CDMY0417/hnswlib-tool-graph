#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <cmath>

#include "json.hpp"
#include "hnswlib/hnswlib.h"

using json = nlohmann::json;

struct Tool {
    std::string library;
    std::string id;
    std::vector<std::string> input_types;
    std::vector<std::string> output_types;
    std::vector<float> embedding;
};

struct QueryExample {
    int example_index;
    std::vector<float> embedding;         // query embedding
    std::vector<std::string> gold_keys;   // "lib::tool_id"
};

std::string make_key(const std::string& lib, const std::string& id) {
    return lib + "::" + id;
}

// ---------- Load tools from JSON ----------

void load_tools_from_json_file(
    const std::string& path,
    const std::string& library_name,
    std::vector<Tool>& out
) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open JSON: " + path);
    json j;
    in >> j;
    for (const auto& node : j["nodes"]) {
        Tool t;
        t.library      = library_name;
        t.id           = node.at("id").get<std::string>();
        t.input_types  = node.at("input-type").get<std::vector<std::string>>();
        t.output_types = node.at("output-type").get<std::vector<std::string>>();
        out.push_back(std::move(t));
    }
}

// ---------- Load tool embeddings (tool_embeddings_all.csv) ----------

std::unordered_map<std::string, std::vector<float>>
load_tool_embs(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open " + path);

    std::string header;
    std::getline(in, header);  // skip header

    std::unordered_map<std::string, std::vector<float>> m;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;

        // library
        std::getline(ss, cell, ',');
        std::string lib = cell;
        // id
        std::getline(ss, cell, ',');
        std::string id = cell;

        std::vector<float> emb;
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) emb.push_back(std::stof(cell));
        }
        if (!emb.empty()) {
            m[make_key(lib, id)] = std::move(emb);
        }
    }
    return m;
}

// ---------- Load eval queries (eval_queries.csv) ----------

std::vector<QueryExample> load_queries(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open " + path);

    std::string header;
    std::getline(in, header);

    std::vector<std::string> cols;
    {
        std::stringstream ss(header);
        std::string c;
        while (std::getline(ss, c, ',')) cols.push_back(c);
    }

    int idx_example = -1;
    int idx_gold_seq = -1;
    int idx_emb_start = -1;

    for (int i = 0; i < (int)cols.size(); ++i) {
        if (cols[i] == "example_index") idx_example = i;
        else if (cols[i] == "gold_seq") idx_gold_seq = i;
        else if (cols[i].size() >= 2 && cols[i][0] == 'd' && std::isdigit(cols[i][1]) && idx_emb_start < 0) {
            idx_emb_start = i;
        }
    }

    if (idx_example < 0 || idx_gold_seq < 0 || idx_emb_start < 0) {
        throw std::runtime_error("Missing columns in eval_queries.csv");
    }

    std::vector<QueryExample> qs;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(ss, cell, ',')) cells.push_back(cell);

        if ((int)cells.size() <= idx_emb_start) continue;

        QueryExample q;
        q.example_index = std::stoi(cells[idx_example]);

        // parse gold_seq: "lib::id||lib::id||..."
        {
            std::string seq = cells[idx_gold_seq];
            size_t pos = 0;
            while (true) {
                size_t next = seq.find("||", pos);
                if (next == std::string::npos) {
                    std::string key = seq.substr(pos);
                    if (!key.empty()) q.gold_keys.push_back(key);
                    break;
                } else {
                    std::string key = seq.substr(pos, next - pos);
                    if (!key.empty()) q.gold_keys.push_back(key);
                    pos = next + 2;
                }
            }
        }

        for (int i = idx_emb_start; i < (int)cells.size(); ++i) {
            if (!cells[i].empty()) q.embedding.push_back(std::stof(cells[i]));
        }

        qs.push_back(std::move(q));
    }

    return qs;
}

// ---------- IO filter for graph construction ----------

struct ToolTypeFilter : public hnswlib::BaseFilterFunctor {
    const std::vector<std::vector<std::string>>& input_seqs;
    const std::vector<std::vector<std::string>>& output_seqs;
    int current_src = -1;

    ToolTypeFilter(
        const std::vector<std::vector<std::string>>& in_seqs,
        const std::vector<std::vector<std::string>>& out_seqs
    ) : input_seqs(in_seqs), output_seqs(out_seqs) {}

    void set_current_source(hnswlib::labeltype src_label) {
        current_src = (int)src_label;
    }

    bool operator()(hnswlib::labeltype candidate_label) override {
        if (current_src < 0) return true;
        int cand = (int)candidate_label;
        const auto& out_seq = output_seqs[current_src];
        const auto& in_seq  = input_seqs[cand];
        return out_seq == in_seq;
    }
};

// ---------- Planning over adjacency graph ----------

float cosine_to_query(
    const std::vector<float>& query,
    const std::vector<float>& emb
) {
    float dot = 0.0f;
    int dim = (int)query.size();
    for (int i = 0; i < dim; ++i) {
        dot += query[i] * emb[i];
    }
    return dot;
}

std::vector<int> plan_sequence_with_graph(
    const std::vector<float>& query_emb,
    hnswlib::HierarchicalNSW<float>& index,
    const std::vector<Tool>& tools,
    const std::vector<std::vector<int>>& adjacency,
    int seq_len,
    int k0 = 32,
    bool avoid_repeats = true
) {
    std::vector<int> seq;
    if (seq_len <= 0) return seq;

    std::unordered_set<int> used;
    used.reserve(seq_len);

    // ---- Step 0: pick starting tool via HNSW on query ----
    {
        auto result = index.searchKnnCloserFirst(query_emb.data(), k0);
        int chosen = -1;
        for (auto& p : result) {
            int idx = (int)p.second;
            if (avoid_repeats && used.count(idx)) continue;
            chosen = idx;
            break;
        }
        if (chosen < 0) return seq;
        seq.push_back(chosen);
        used.insert(chosen);
    }

    // ---- Subsequent steps: follow adjacency ----
    for (int step = 1; step < seq_len; ++step) {
        int prev = seq.back();
        const auto& nbrs = adjacency[prev];
        if (nbrs.empty()) break;

        int best = -1;
        float best_score = -1e9f;
        for (int j : nbrs) {
            if (avoid_repeats && used.count(j)) continue;
            float score = cosine_to_query(query_emb, tools[j].embedding);
            if (score > best_score) {
                best_score = score;
                best = j;
            }
        }
        if (best < 0) break;
        seq.push_back(best);
        used.insert(best);
    }

    return seq;
}

// ---------- Sequence similarity metrics ----------

size_t common_prefix_length(const std::vector<int>& a, const std::vector<int>& b) {
    size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

double jaccard_similarity(const std::vector<int>& a, const std::vector<int>& b) {
    std::unordered_set<int> sa(a.begin(), a.end());
    std::unordered_set<int> sb(b.begin(), b.end());
    if (sa.empty() && sb.empty()) return 1.0;

    size_t inter = 0;
    for (int x : sa) {
        if (sb.count(x)) ++inter;
    }
    size_t uni = sa.size() + sb.size() - inter;
    if (uni == 0) return 1.0;
    return (double)inter / (double)uni;
}

int lcs_length(const std::vector<int>& a, const std::vector<int>& b) {
    size_t n = a.size(), m = b.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            if (a[i] == b[j]) {
                dp[i + 1][j + 1] = dp[i][j] + 1;
            } else {
                dp[i + 1][j + 1] = std::max(dp[i][j + 1], dp[i + 1][j]);
            }
        }
    }
    return dp[n][m];
}

int main() {
    try {
        // ---------- 1. Load tools ----------
        std::vector<Tool> tools;
        load_tools_from_json_file("multimedia_apis.json",  "multimedia", tools);
        load_tools_from_json_file("daily_apis.json",       "daily",      tools);
        load_tools_from_json_file("huggingface_apis.json", "huggingface", tools);

        int num_tools = (int)tools.size();
        if (num_tools == 0) throw std::runtime_error("No tools loaded");
        std::cout << "Loaded " << num_tools << " tools\n";

        // ---------- 2. Attach embeddings ----------
        auto emb_map = load_tool_embs("tool_embeddings_all.csv");
        int dim = -1;
        for (auto& t : tools) {
            auto it = emb_map.find(make_key(t.library, t.id));
            if (it == emb_map.end()) {
                throw std::runtime_error("No embedding for " + make_key(t.library, t.id));
            }
            t.embedding = it->second;
            if (dim < 0) dim = (int)t.embedding.size();
            else if ((int)t.embedding.size() != dim) {
                throw std::runtime_error("Inconsistent embedding dim");
            }
        }
        std::cout << "Embedding dim = " << dim << "\n";

        // ---------- 3. Build HNSW index ----------
        int M = 16;
        int ef_construction = 200;
        int ef_search = 100;

        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float> index(&space, num_tools, M, ef_construction);
        for (int i = 0; i < num_tools; ++i) {
            index.addPoint(tools[i].embedding.data(), i);
        }
        index.setEf(ef_search);

        // ---------- 4. Build adjacency graph with IO + similarity threshold ----------
        std::vector<std::vector<std::string>> input_seqs(num_tools), output_seqs(num_tools);
        for (int i = 0; i < num_tools; ++i) {
            input_seqs[i]  = tools[i].input_types;
            output_seqs[i] = tools[i].output_types;
        }

        ToolTypeFilter filter(input_seqs, output_seqs);

        int K_graph = 32;
        float sim_threshold = 0.7f;

        std::vector<std::vector<int>> adjacency(num_tools);

        for (int i = 0; i < num_tools; ++i) {
            filter.set_current_source(i);
            auto result = index.searchKnnCloserFirst(tools[i].embedding.data(), K_graph, &filter);

            auto& nbrs = adjacency[i];
            for (auto& p : result) {
                float dist_sq = p.first;
                int j = (int)p.second;
                if (j == i) continue;

                float cos_sim = 1.0f - 0.5f * dist_sq;
                if (cos_sim >= sim_threshold) {
                    nbrs.push_back(j);
                }
            }
        }

        std::cout << "Built adjacency graph with IO + similarity constraints\n";

        std::unordered_map<std::string, int> key_to_idx;
        for (int i = 0; i < num_tools; ++i) {
            key_to_idx[make_key(tools[i].library, tools[i].id)] = i;
        }

        // ---------- 5. Load eval queries ----------
        auto queries = load_queries("eval_queries.csv");
        std::cout << "Loaded " << queries.size() << " eval queries\n";

        // ---------- 6. Evaluate ----------
        std::ofstream out("hnsw_eval_results.csv");
        out << "example_index,"
            << "gold_seq,pred_seq,"
            << "gold_len,pred_len,"
            << "exact_match,prefix_len,prefix_ratio,jaccard,lcs_len\n";

        int total_eval = 0;
        int exact_matches = 0;
        int prefix_matches = 0;
        double sum_jaccard = 0.0;
        double sum_lcs = 0.0;
        double sum_prefix_ratio = 0.0;

        for (const auto& q : queries) {
            std::vector<int> gold_idx;
            bool ok = true;
            for (const auto& key : q.gold_keys) {
                auto it = key_to_idx.find(key);
                if (it == key_to_idx.end()) {
                    ok = false;
                    break;
                }
                gold_idx.push_back(it->second);
            }
            if (!ok || gold_idx.empty()) continue;

            int seq_len = (int)gold_idx.size();

            auto pred_idx = plan_sequence_with_graph(
                q.embedding, index, tools, adjacency, seq_len, 32, true
            );

            // stringify sequences as "lib::id -> lib::id -> ..."
            auto seq_to_str = [&](const std::vector<int>& seq) {
                std::string s;
                for (size_t i = 0; i < seq.size(); ++i) {
                    if (i) s += " -> ";
                    s += make_key(tools[seq[i]].library, tools[seq[i]].id);
                }
                return s;
            };

            std::string gold_str = seq_to_str(gold_idx);
            std::string pred_str = seq_to_str(pred_idx);

            bool exact = (gold_idx == pred_idx);
            size_t gold_len = gold_idx.size();
            size_t pred_len = pred_idx.size();

            size_t pref_len = common_prefix_length(gold_idx, pred_idx);
            double pref_ratio = gold_len > 0 ? (double)pref_len / (double)gold_len : 0.0;

            double jac = jaccard_similarity(gold_idx, pred_idx);
            int lcs_len = lcs_length(gold_idx, pred_idx);

            total_eval++;
            if (exact) exact_matches++;
            if (pref_len == gold_len && gold_len > 0) prefix_matches++;
            sum_jaccard += jac;
            sum_lcs += (double)lcs_len;
            sum_prefix_ratio += pref_ratio;

            out << q.example_index << ","
                << "\"" << gold_str << "\"" << ","
                << "\"" << pred_str << "\"" << ","
                << gold_len << ","
                << pred_len << ","
                << (exact ? 1 : 0) << ","
                << pref_len << ","
                << pref_ratio << ","
                << jac << ","
                << lcs_len << "\n";
        }

        std::cout << "Evaluated " << total_eval << " examples\n";
        if (total_eval > 0) {
            double exact_acc = (double)exact_matches / (double)total_eval;
            double prefix_acc = (double)prefix_matches / (double)total_eval;
            double mean_jaccard = sum_jaccard / (double)total_eval;
            double mean_lcs = sum_lcs / (double)total_eval;
            double mean_prefix_ratio = sum_prefix_ratio / (double)total_eval;

            std::cout << "Exact match accuracy     = " << exact_acc << "\n";
            std::cout << "Prefix match accuracy    = " << prefix_acc << "\n";
            std::cout << "Mean Jaccard similarity  = " << mean_jaccard << "\n";
            std::cout << "Mean LCS length          = " << mean_lcs << "\n";
            std::cout << "Mean prefix ratio        = " << mean_prefix_ratio << "\n";
        }
        std::cout << "Wrote hnsw_eval_results.csv\n";

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
