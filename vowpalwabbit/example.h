/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD
license as described in the file LICENSE.
 */
#ifndef EX_H
#define EX_H

#include <stdint.h>
#include "v_array.h"

struct feature {
  float x; // feature value
  uint32_t weight_index; // feature index (hashed)
  bool operator==(feature j){return weight_index == j.weight_index;}
};

struct audit_data {
  char* space; // feature namespace?
  char* feature; // feature name
  size_t weight_index; // feature hash
  float x; // feature value
  bool alloced; // true, initially
};

typedef float simple_prediction;

struct example // core example datatype.
{
  void* ld; // label data
  simple_prediction final_prediction;

  v_array<char> tag;//An identifier for the example.
  size_t example_counter;

  v_array<size_t> indices; // non-empty namespace indexes
  v_array<feature> atomics[256]; // raw parsed data
  
  v_array<audit_data> audit_features[256];
  
  size_t num_features;//precomputed, cause it's fast&easy.
  size_t pass;
  float partial_prediction;//shared data for prediction.
  v_array<float> topic_predictions;
  float loss;
  float eta_round;
  float eta_global;
  float global_weight;
  float example_t;//sum of importance weights so far.
  float sum_feat_sq[256];//helper for total_sum_feat_sq. (sum of squared feature values)
  float total_sum_feat_sq;//precomputed, cause it's kind of fast & easy.
  float revert_weight;

  bool sorted;//Are the features sorted or not?
  bool in_use; //in use or not (for the parser)
  bool done; //set to false by setup_example() and true by gd.train
};

example *alloc_example(size_t);
void dealloc_example(void(*delete_label)(void*), example&);
void copy_example_data(example*&, example*, size_t);
void update_example_indicies(bool audit, example* ec, size_t amount);

#endif
