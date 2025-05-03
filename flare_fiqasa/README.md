---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: query
    dtype: string
  - name: answer
    dtype: string
  - name: text
    dtype: string
  - name: choices
    sequence: string
  - name: gold
    dtype: int64
  splits:
  - name: train
    num_bytes: 248828
    num_examples: 750
  - name: valid
    num_bytes: 61667
    num_examples: 188
  - name: test
    num_bytes: 77672
    num_examples: 235
  download_size: 0
  dataset_size: 388167
extra_gated_fields:
  First name: text
  Last name: text
  Affiliation: text
  Job title: text
  Email: text
  Country: country
  I want to use this model for:
    type: select
    options:
    - Research
    - Education
    - label: Other
      value: other
  I agree to use this model for non-commercial use ONLY: checkbox
license: mit
task_categories:
- text-classification
language:
- en
tags:
- stock
- finance
- market
- sentiment analysis
- text classification
size_categories:
- 1K<n<10K
---
# Dataset Card for "flare-fiqasa"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)