---
name: memory-profile
description: Use when analyzing TPU pretraining HBM peak occupancy from xplane.pb — locates the peak moment within a step, lists every live buffer at that moment with shape/tf_op/parent-jit attribution, and rolls up by lifetime-class / shape / tf_op / dtype. Reads schema documented by profile-anatomy.
argument-hint: "<profile_dir> [--step N | --step-policy peak|last|first] [--top K]"
---

# Memory Profile

Stub — full content written in Task 12.
