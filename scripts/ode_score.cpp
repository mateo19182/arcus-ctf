// Batched NLL/logit scorer for the ode GGUF on the llama.cpp Vulkan backend.
// Feeds raw token IDs (vocab type 'none'), reads logits, scores teacher-forced.
//
// stdin : one candidate per line, comma-separated token ids (e.g. "72,101,108")
// stdout: per line  "<avg_logp> <logit_sum> <n_pred>"   (n_pred = len-1)
//
// args: <model.gguf> [n_seq_per_batch=64] [tok_budget=16384] [ngl=999]
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

static std::vector<int> parse_ids(const std::string& line) {
    std::vector<int> v; const char* p = line.c_str();
    while (*p) { char* e; long x = strtol(p, &e, 10); if (e==p) break; v.push_back((int)x); p=e; while(*p==','||*p==' ')p++; }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [n_seq] [tok_budget] [ngl]\n", argv[0]); return 1; }
    const char* model_path = argv[1];
    int n_seq_max = argc > 2 ? atoi(argv[2]) : 64;
    int tok_budget = argc > 3 ? atoi(argv[3]) : 16384;
    int ngl = argc > 4 ? atoi(argv[4]) : 999;

    llama_log_set([](ggml_log_level lvl, const char* text, void*){ if (lvl>=GGML_LOG_LEVEL_WARN) fputs(text, stderr); }, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "FAILED to load model\n"); return 2; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    fprintf(stderr, "loaded: n_vocab=%d ngl=%d\n", n_vocab, ngl);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = tok_budget; cp.n_batch = tok_budget; cp.n_ubatch = tok_budget;
    cp.n_seq_max = n_seq_max;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "FAILED ctx\n"); return 3; }
    llama_memory_t mem = llama_get_memory(ctx);

    // read all candidates
    std::vector<std::vector<int>> seqs;
    { std::string line; char buf[1<<16];
      while (fgets(buf, sizeof(buf), stdin)) { line=buf; auto v=parse_ids(line); if(v.size()>=2) seqs.push_back(std::move(v)); else seqs.push_back({}); } }

    const bool pertok = getenv("ODE_PERTOK") && atoi(getenv("ODE_PERTOK")) != 0;

    llama_batch batch = llama_batch_init(tok_budget, 0, n_seq_max);
    std::vector<double> out_avg(seqs.size()), out_sum(seqs.size());
    std::vector<int> out_n(seqs.size());
    std::vector<std::vector<float>> out_lp(seqs.size());   // per-token logp (pertok mode)

    auto t0 = std::chrono::high_resolution_clock::now();
    size_t i = 0;
    while (i < seqs.size()) {
        // assemble a group respecting n_seq_max and tok_budget
        batch.n_tokens = 0;
        struct Rec { size_t si; int start; int len; };
        std::vector<Rec> recs;
        int seq_local = 0;
        while (i < seqs.size() && seq_local < n_seq_max) {
            int len = (int)seqs[i].size();
            if (len < 2) { out_avg[i]=0; out_sum[i]=0; out_n[i]=0; i++; continue; }
            if (batch.n_tokens + len > tok_budget) break;
            int start = batch.n_tokens;
            for (int j = 0; j < len; j++) {
                int idx = batch.n_tokens++;
                batch.token[idx]    = seqs[i][j];
                batch.pos[idx]      = j;
                batch.n_seq_id[idx] = 1;
                batch.seq_id[idx][0]= seq_local;
                batch.logits[idx]   = (j < len-1) ? 1 : 0; // logit at j predicts token j+1
            }
            recs.push_back({i, start, len});
            seq_local++; i++;
        }
        if (batch.n_tokens == 0) continue;
        llama_memory_clear(mem, true);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 4; }
        for (auto& r : recs) {
            double sum_lp = 0, sum_raw = 0;
            if (pertok) out_lp[r.si].reserve(r.len-1);
            for (int j = 0; j < r.len-1; j++) {
                float* lg = llama_get_logits_ith(ctx, r.start + j);
                int tgt = seqs[r.si][j+1];
                // log_softmax at tgt
                float mx = lg[0]; for (int k=1;k<n_vocab;k++) if (lg[k]>mx) mx=lg[k];
                double se = 0; for (int k=0;k<n_vocab;k++) se += exp((double)lg[k]-mx);
                double lse = mx + log(se);
                double lp = (double)lg[tgt] - lse;
                sum_lp  += lp;
                sum_raw += (double)lg[tgt];
                if (pertok) out_lp[r.si].push_back((float)lp);
            }
            out_avg[r.si] = sum_lp / (r.len-1);
            out_sum[r.si] = sum_raw;
            out_n[r.si]   = r.len-1;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1-t0).count();

    for (size_t k = 0; k < seqs.size(); k++) {
        if (pertok) {
            // "<avg> <sum> <n> | lp0,lp1,..."  (per-token logp, predicting tokens 1..n)
            printf("%.6f %.6f %d |", out_avg[k], out_sum[k], out_n[k]);
            for (size_t j = 0; j < out_lp[k].size(); j++)
                printf("%s%.4f", j ? "," : " ", out_lp[k][j]);
            printf("\n");
        } else {
            printf("%.6f %.6f %d\n", out_avg[k], out_sum[k], out_n[k]);
        }
    }
    fprintf(stderr, "scored %zu candidates in %.3fs -> %.0f cand/s\n", seqs.size(), dt, seqs.size()/dt);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
