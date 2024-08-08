<div align="center">

# Towards Clean-Label Backdoor Attacks in the Physical World
[![arXiv](https://img.shields.io/badge/arXiv-2407.19203-b31b1b.svg)](https://arxiv.org/abs/2407.19203)

</div>

## Introduction

Deep Neural Networks (DNNs) are vulnerable to backdoor poisoning attacks, with most research focusing on digital triggers, special patterns digitally added to test-time inputs to induce targeted misclassification. In contrast, physical triggers, which are natural objects within a physical scene, have emerged as a desirable alternative since they enable real-time backdoor activations without digital manipulation. However, current physical attacks require that poisoned inputs have incorrect labels, making them easily detectable upon human inspection. In this paper, we collect a facial dataset of 21,238 images with 7 common accessories as triggers and use it to study the threat of clean-label backdoor attacks in the physical world. Our study reveals two findings. First, the success of physical attacks depends on the poisoning algorithm, physical trigger, and the pair of source-target classes. Second, although clean-label poisoned samples preserve ground-truth labels, their perceptual quality could be seriously degraded due to conspicuous artifacts in the images. Such samples are also vulnerable to statistical filtering methods because they deviate from the distribution of clean samples in the feature space. To address these issues, we propose replacing the standard ℓ∞ regularization with a novel pixel regularization and feature regularization that could enhance the imperceptibility of poisoned samples without compromising attack performance. Our study highlights accidental backdoor activations as a key limitation of clean-label physical backdoor attacks. This happens when unintended objects or classes accidentally cause the model to misclassify as the target class.

### Paper: [Towards Clean-Label Backdoor Attacks in the Physical World](https://arxiv.org/abs/2407.19203)
