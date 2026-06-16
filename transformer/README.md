# Transformer Experiment

## 1. Overview
This project is implementation of Transformer encoder using Pytorch.

## 2. Motivation
I implemented transformer to understand transformer in code-level.

## 3. How to Run
python train.py   /*  basic transformer running */
python train_no_pe.py   /*  no PE transformer running  */
python train_one_head.py   /*  1-head transformer running  */
python train_one_layer.py   /*  1-layer transformer running  */
python train_four_layer.py   /*  4-layers transformer running  */


## 4. File Structure
transformer/
├── mini_transformer.py   # Transformer components implemented from scratch
├── train.py              # Training script for text classification
├── train_no_pe.py         # Training script without positional encoding
├── train_one_layer.py         # Training script with 1 layer
├── train_one_head.py         # Training script with 1 head
├── train_four_head.py         # Training script with 4 heads
└── README.md

## 4. Experiment
I observed when using four layers, there was best acc on AG news classification.
Without PE, small degradation was observed. Maybe PE was not important to AG News classification. In news article, key word is important than orders between words.
One head was not good. Learning multi-intents may be very important to NLP task.

But I think differences between experiments were not large.
However, I felt I can't ignore hyperparameter tuning.

## 5. What I Learned
First, I learned about model architecture.
I learned that Q, K, and V are affine-transformed before scaled dot-product attention.
But I implemented only 'transformer/mini_transformer.py'.
So I think I need to implement all training code in near future.

## 6. Important point
I think Q, and K make weight, and it is critical to transformer.
With PE, weight makes model learn sequential signal.
