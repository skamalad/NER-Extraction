#ifndef NER_H
#define NER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float* transitions;
    float* emissions;
    int state_size;
    int vocab_size;
} NERModel;

NERModel* init_model(const float* transitions, const float* emissions, int state_size, int vocab_size);

void decode_sequence(const NERModel* model, const int* tokens, int seq_length, int* output_tags, float* output_scores);

void free_model(NERModel* model);

#ifdef __cplusplus
}
#endif

#endif