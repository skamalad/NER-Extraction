#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#define MAX_SEQ_LEN 512
#define MIN_SCORE -100.0f

typedef struct {
    int state_size;
    int vocab_size;
    float* transitions;  // [state_size x state_size] matrix
    float* emissions;    // [vocab_size x state_size] matrix
} NERModel;

// Function declarations
static void viterbi_decode(const NERModel* model, const int* tokens, int seq_len, int* tags, float* scores);

EXPORT void* init_model(int state_size, int vocab_size, float* transitions, float* emissions) {
    fprintf(stderr, "C [init_model]: Entry point. state_size=%d, vocab_size=%d\n", state_size, vocab_size);
    fprintf(stderr, "C [init_model]: Pointers - transitions=%p, emissions=%p\n", 
            (void*)transitions, (void*)emissions);
    
    // Validate dimensions
    if (state_size <= 0 || state_size > 100) {
        fprintf(stderr, "C [init_model]: Invalid state_size: %d\n", state_size);
        return NULL;
    }
    if (vocab_size <= 0 || vocab_size > 100000) {
        fprintf(stderr, "C [init_model]: Invalid vocab_size: %d\n", vocab_size);
        return NULL;
    }
    
    // Validate pointers
    if (!transitions) {
        fprintf(stderr, "C [init_model]: transitions pointer is NULL\n");
        return NULL;
    }
    if (!emissions) {
        fprintf(stderr, "C [init_model]: emissions pointer is NULL\n");
        return NULL;
    }
    
    // Check for NaN or Inf in transitions
    for (int i = 0; i < state_size * state_size; i++) {
        if (isnan(transitions[i]) || isinf(transitions[i])) {
            fprintf(stderr, "C [init_model]: Invalid value in transitions[%d]: %f\n", 
                    i, transitions[i]);
            return NULL;
        }
    }
    
    // Check for NaN or Inf in emissions
    for (int i = 0; i < vocab_size * state_size; i++) {
        if (isnan(emissions[i]) || isinf(emissions[i])) {
            fprintf(stderr, "C [init_model]: Invalid value in emissions[%d]: %f\n", 
                    i, emissions[i]);
            return NULL;
        }
    }
    
    // Allocate model struct
    fprintf(stderr, "C [init_model]: Allocating model struct\n");
    NERModel* model = (NERModel*)malloc(sizeof(NERModel));
    if (!model) {
        fprintf(stderr, "C [init_model]: Failed to allocate model struct\n");
        return NULL;
    }
    
    // Initialize dimensions
    model->state_size = state_size;
    model->vocab_size = vocab_size;
    
    // Allocate and copy transitions
    size_t trans_size = state_size * state_size * sizeof(float);
    fprintf(stderr, "C [init_model]: Allocating transitions of size %zu bytes\n", trans_size);
    model->transitions = (float*)malloc(trans_size);
    if (!model->transitions) {
        fprintf(stderr, "C [init_model]: Failed to allocate transitions\n");
        free(model);
        return NULL;
    }
    fprintf(stderr, "C [init_model]: Copying transitions data\n");
    memcpy(model->transitions, transitions, trans_size);
    
    // Allocate and copy emissions
    size_t emis_size = vocab_size * state_size * sizeof(float);
    fprintf(stderr, "C [init_model]: Allocating emissions of size %zu bytes\n", emis_size);
    model->emissions = (float*)malloc(emis_size);
    if (!model->emissions) {
        fprintf(stderr, "C [init_model]: Failed to allocate emissions\n");
        free(model->transitions);
        free(model);
        return NULL;
    }
    fprintf(stderr, "C [init_model]: Copying emissions data\n");
    memcpy(model->emissions, emissions, emis_size);
    
    fprintf(stderr, "C [init_model]: Model initialized successfully\n");
    return model;
}

EXPORT void free_model(void* model_ptr) {
    fprintf(stderr, "C [free_model]: Entry point with model_ptr=%p\n", model_ptr);
    if (!model_ptr) {
        fprintf(stderr, "C [free_model]: NULL pointer received\n");
        return;
    }
    
    NERModel* model = (NERModel*)model_ptr;
    fprintf(stderr, "C [free_model]: Freeing transitions at %p\n", (void*)model->transitions);
    free(model->transitions);
    
    fprintf(stderr, "C [free_model]: Freeing emissions at %p\n", (void*)model->emissions);
    free(model->emissions);
    
    fprintf(stderr, "C [free_model]: Freeing model struct\n");
    free(model);
    fprintf(stderr, "C [free_model]: Complete\n");
}

EXPORT void decode_sequence(void* model_ptr, const int* tokens, int seq_len, int* tags, float* scores) {
    fprintf(stderr, "C [decode_sequence]: Entry point\n");
    fprintf(stderr, "C [decode_sequence]: model_ptr=%p, tokens=%p, tags=%p, scores=%p, seq_len=%d\n",
            model_ptr, (void*)tokens, (void*)tags, (void*)scores, seq_len);
    
    if (!model_ptr || !tokens || !tags || !scores) {
        fprintf(stderr, "C [decode_sequence]: NULL pointer received\n");
        return;
    }
    
    if (seq_len <= 0 || seq_len > MAX_SEQ_LEN) {
        fprintf(stderr, "C [decode_sequence]: Invalid sequence length: %d\n", seq_len);
        return;
    }
    
    NERModel* model = (NERModel*)model_ptr;
    viterbi_decode(model, tokens, seq_len, tags, scores);
    fprintf(stderr, "C [decode_sequence]: Complete\n");
}

static void viterbi_decode(const NERModel* model, const int* tokens, int seq_len, int* tags, float* scores) {
    fprintf(stderr, "C [viterbi_decode]: Entry point\n");
    
    const int state_size = model->state_size;
    float* dp = (float*)malloc(seq_len * state_size * sizeof(float));
    int* backpointers = (int*)malloc(seq_len * state_size * sizeof(int));
    
    if (!dp || !backpointers) {
        fprintf(stderr, "C [viterbi_decode]: Memory allocation failed\n");
        free(dp);
        free(backpointers);
        return;
    }
    
    // Initialize first position
    int token_id = tokens[0];
    if (token_id < 0 || token_id >= model->vocab_size) {
        fprintf(stderr, "C [viterbi_decode]: Invalid token_id: %d\n", token_id);
        free(dp);
        free(backpointers);
        return;
    }
    
    for (int j = 0; j < state_size; j++) {
        dp[j] = model->emissions[token_id * state_size + j];
        backpointers[j] = -1;
    }
    
    // Forward pass
    for (int t = 1; t < seq_len; t++) {
        token_id = tokens[t];
        if (token_id < 0 || token_id >= model->vocab_size) {
            fprintf(stderr, "C [viterbi_decode]: Invalid token_id: %d\n", token_id);
            free(dp);
            free(backpointers);
            return;
        }
        
        float* prev_scores = dp + (t-1) * state_size;
        float* curr_scores = dp + t * state_size;
        int* curr_back = backpointers + t * state_size;
        
        for (int j = 0; j < state_size; j++) {
            float best_score = MIN_SCORE;
            int best_prev = -1;
            
            for (int i = 0; i < state_size; i++) {
                float score = prev_scores[i] + 
                            model->transitions[i * state_size + j] +
                            model->emissions[token_id * state_size + j];
                if (score > best_score) {
                    best_score = score;
                    best_prev = i;
                }
            }
            
            curr_scores[j] = best_score;
            curr_back[j] = best_prev;
        }
    }
    
    // Find best final state
    float best_final_score = MIN_SCORE;
    int best_final_state = 0;
    float* final_scores = dp + (seq_len-1) * state_size;
    
    for (int j = 0; j < state_size; j++) {
        if (final_scores[j] > best_final_score) {
            best_final_score = final_scores[j];
            best_final_state = j;
        }
    }
    
    // Backtrack
    int curr_state = best_final_state;
    tags[seq_len-1] = curr_state;
    scores[seq_len-1] = best_final_score;
    
    for (int t = seq_len-2; t >= 0; t--) {
        curr_state = backpointers[(t+1) * state_size + curr_state];
        tags[t] = curr_state;
        scores[t] = dp[t * state_size + curr_state];
    }
    
    free(dp);
    free(backpointers);
    fprintf(stderr, "C [viterbi_decode]: Complete\n");
}