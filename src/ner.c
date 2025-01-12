#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#define MAX_SEQ_LEN 512
#define MIN_SCORE -100.0f

typedef struct {
    int state_size;
    int vocab_size;
    float* transitions;  // [state_size x state_size] matrix
    float* emissions;    // [vocab_size x state_size] matrix
} NERModel;

// Forward declarations
static void viterbi_decode(const NERModel* model, const int* tokens, int seq_len, int* tags);
static float logsumexp(const float* arr, int n);
static inline float max_array(const float* arr, int n, int* argmax);

void* init_model(int state_size, int vocab_size, float* transitions, float* emissions) {
    printf("C: init_model called with state_size=%d, vocab_size=%d\n", state_size, vocab_size);
    printf("C: transitions=%p, emissions=%p\n", (void*)transitions, (void*)emissions);
    
    NERModel* model = (NERModel*)malloc(sizeof(NERModel));
    if (!model) return NULL;
    
    model->state_size = state_size;
    model->vocab_size = vocab_size;
    
    size_t trans_size = state_size * state_size * sizeof(float);
    printf("C: Allocating transitions of size %zu bytes (aligned to %zu)\n", 
           trans_size, (trans_size + 15) & ~15);
    model->transitions = (float*)aligned_alloc(16, (trans_size + 15) & ~15);
    if (!model->transitions) {
        free(model);
        return NULL;
    }
    memcpy(model->transitions, transitions, trans_size);
    
    size_t emis_size = vocab_size * state_size * sizeof(float);
    printf("C: Allocating emissions of size %zu bytes (aligned to %zu)\n",
           emis_size, (emis_size + 15) & ~15);
    model->emissions = (float*)aligned_alloc(16, (emis_size + 15) & ~15);
    if (!model->emissions) {
        free(model->transitions);
        free(model);
        return NULL;
    }
    memcpy(model->emissions, emissions, emis_size);
    
    printf("C: Model initialized successfully\n");
    return model;
}

void free_model(void* model_ptr) {
    if (!model_ptr) return;
    NERModel* model = (NERModel*)model_ptr;
    free(model->transitions);
    free(model->emissions);
    free(model);
}

void predict_tags(void* model_ptr, const int* tokens, int seq_len, int* tags) {
    if (!model_ptr || !tokens || !tags || seq_len <= 0 || seq_len > MAX_SEQ_LEN) return;
    NERModel* model = (NERModel*)model_ptr;
    viterbi_decode(model, tokens, seq_len, tags);
}

static void viterbi_decode(const NERModel* model, const int* tokens, int seq_len, int* tags) {
    const int S = model->state_size;
    float dp[MAX_SEQ_LEN][9];  // Viterbi DP table [seq_len x state_size]
    int prev[MAX_SEQ_LEN][9];  // Backpointers [seq_len x state_size]
    
    // Initialize first position
    const float* emit_probs = model->emissions + tokens[0] * S;
    for (int s = 0; s < S; s++) {
        if (s == 0) {  // O tag
            dp[0][s] = emit_probs[s];  // Use emission probability for O tag
        } else if (s % 2 == 1) {  // B- tags
            dp[0][s] = emit_probs[s];  // Use emission probability for B- tags
        } else {  // I- tags
            dp[0][s] = MIN_SCORE;  // Cannot start with I- tag
        }
        prev[0][s] = -1;
    }
    
    // Forward pass
    for (int t = 1; t < seq_len; t++) {
        const float* emit_probs = model->emissions + tokens[t] * S;
        int is_subword = (tokens[t] >= 1000 && tokens[t] <= 2000);  // Check for subword token
        
        for (int curr_s = 0; curr_s < S; curr_s++) {
            float max_score = MIN_SCORE;
            int best_prev = 0;
            
            // For each possible previous state
            for (int prev_s = 0; prev_s < S; prev_s++) {
                float trans_score = model->transitions[prev_s * S + curr_s];
                float score = dp[t-1][prev_s] + trans_score;
                
                // Apply constraints based on tag type
                if (curr_s % 2 == 0 && curr_s > 0) {  // I- tag
                    // Can only transition to I-X from B-X or I-X of same type
                    if (prev_s != curr_s && prev_s != curr_s - 1) {
                        score = MIN_SCORE;
                    }
                }
                
                if (score > max_score) {
                    max_score = score;
                    best_prev = prev_s;
                }
            }
            
            // Add emission score
            if (is_subword) {
                // For subwords, strongly prefer continuing the previous tag
                if (t > 0) {
                    int prev_tag = tags[t-1];
                    if (curr_s == prev_tag) {
                        dp[t][curr_s] = max_score;  // Keep the tag from previous token
                    } else {
                        dp[t][curr_s] = MIN_SCORE;  // Discourage tag changes for subwords
                    }
                } else {
                    dp[t][curr_s] = max_score + emit_probs[curr_s];
                }
            } else {
                dp[t][curr_s] = max_score + emit_probs[curr_s];
            }
            
            prev[t][curr_s] = best_prev;
        }
        
        // Store the best tag for this position (needed for subword handling)
        float best_score = dp[t][0];
        tags[t] = 0;
        for (int s = 1; s < S; s++) {
            if (dp[t][s] > best_score) {
                best_score = dp[t][s];
                tags[t] = s;
            }
        }
    }
    
    // Backward pass to recover best path
    int curr_tag = 0;
    float max_score = dp[seq_len-1][0];
    
    // Find best final tag
    for (int s = 1; s < S; s++) {
        if (dp[seq_len-1][s] > max_score) {
            max_score = dp[seq_len-1][s];
            curr_tag = s;
        }
    }
    
    // Trace back
    tags[seq_len-1] = curr_tag;
    for (int t = seq_len-2; t >= 0; t--) {
        curr_tag = prev[t+1][curr_tag];
        tags[t] = curr_tag;
    }
}

static float logsumexp(const float* arr, int n) {
    if (n == 0) return MIN_SCORE;
    if (n == 1) return arr[0];
    
    float max_val = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += expf(arr[i] - max_val);
    }
    
    return max_val + logf(sum);
}

static inline float max_array(const float* arr, int n, int* argmax) {
    float max_val = arr[0];
    int max_idx = 0;
    
    for (int i = 1; i < n; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    
    if (argmax) *argmax = max_idx;
    return max_val;
}