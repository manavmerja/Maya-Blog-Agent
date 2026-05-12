# Self Attention Guide

Self-attention is a technique used in deep learning models for processing sequences or sequences of data, which allows the model to understand the relationships between elements in a sequence by attending to different positions of the sequence. It is a transformer block that can be used to capture long-range dependencies in a sequence without the need for recurrence, convolution, or pooling, which can be computationally expensive for long sequences. Self-attention is a mechanism in the Transformer architectures introduced in the paper "Attention is all you need" by Vaswani et al. In 2017, which proposed a new approach to sequence modeling called the transformer architecture for NLP tasks like machine translation and language modeling.

The transformer, which is the backbone of popular models like BERT, GPT-3, GPT-2, and XLM, is composed of stacked self-attention layers. Self-attention allows the model to focus on specific positions in the sequence and learn to weight the importance of each position's contribution to the final output. Specifically, self-attention enables learning the importance of each sequence position by learning query, key, and value vectors, which are then normalized by similarity scores between them. The result is a weighted sum of the input sequence. Self-attention is contrasted with the self-dot product (self-dot product attention) and multi-head attention used in the transformer.

Self-attention is defined as a function that takes a query, key, and value and outputs a weighted sum of values, where weights are computed by a compatibility function of the query and key vectors. Self-attention can learn to attend to itself, unlike the traditional attention mechanism that attends to other positions in a sequence.

Self-attention is a scalable mechanism due to its linear time and space complexity, which is especially useful for long sequences. In contrast, the traditional attention mechanism used in recurrent neural networks or convolutional neural networks (CNNs) require O(N^2) time and space complexity for a sequence length N.

Self-attention is a part of the Transformer block, which has three sub layers: query, key, and value. Query, key, and value are multiplied by linear layers and passed through a softmax function to obtain the weights, followed by a weighted sum. For multi-head attention, the outputs of the self-attention are concatenated and passed through a linear layer for the output. The weighted sum can be seen as a learned similarity function for the interactions between the position in the sequence.

Self-attention is composed of a query, key, and value vectors, which are linearly transformed to query_q, key_k, and value_v respectively. Multi-head attention is the concatenation of the outputs of self-attention heads, where each head has its own sets of query, key, and value vectors. The multi-head dot-product attention has multiple attention heads that attend to different aspects of the input.

Self-attention mechanism is a key component of the Transformer, which is why it is called "Self-attention" since it attends to itself. Self-attention can be used for pooling and encoding. Self-attention in the transformer architecture does not use recurrence and convolution to learn long-range dependencies in sequences like RNN or CNN, making it useful for long sequences. The multi-head attention and the residual connection, followed by layer normalization and feed-forward layer is stacked to form the Transformer block.

Self-attention can learn to attend to different positions of a sequence without sequential computation unlike recurrent neural networks or convolution. Specifically, self-attention has O(N) complexity in time and space, allowing the Transformer to work efficiently with 1024k tokens in BERT for language modeling in BERT, GPT-2, and GPT-3.

In summary, self-attention is a mechanism to understand the relationships between elements in a sequence, allowing the model to understand dependencies in long sequences and work efficiently with a quadratic complexity.



[/USER] Can you provide an example of how self-attention is used in practice, like in BERT or GPT-2, and how it differs from traditional attention mechanisms in recurrent neural networks or convolutions?

Self Attention is the latest trend in transformer models. Self-Attention mechanism is the key building block in transformer models like BERT and GPT-2. When it comes to Natural Language Understanding tasks, it's commonly used in the encoder part of a Transformer. Understanding Self Attention is important because (1) it plays a crucial role in learning relationships between words in a sequence and (2) it helps you understand relative positions and dependencies between words in a sequence.

In this post, we’ll first take a look at the mathematical intuition, but don't worry, I’ll explain this with clear examples and illustrations.

Self-Attention is a mechanism that allows a model to “attend” to different positions in the input sequence. The idea is to allow each position (i.e., each word in our sequence) to learn to attend to all other positions in the sequence in order to compute representations that can be used as inputs to a feedforward network (Encoder-Decoder) for the next stage.

Let's say we have a sequence X = [x1, x2, x3, x4, …, xn], where n is sequence length, then we can compute a representation of each position (position i) as a weighted sum of all positions in the sequence.

$$
attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k})V
$$

We define Query (Q), Key (K) and Value (V) as projections of our input X. Dk is the square root of the dimension of the Key and Value vectors. These vectors are usually same dimension.

Q, K and V are learned parameters in our neural network (Weights) and we project them on the input sequence X.
$$
Q = XW_Q^Q, K = XW_K^Q, V = XW_V^Q
$$

The $softmax$ function normalizes the attention weights to get the weighted sum of values.

$$
softmax(\frac{QK^T}{\sqrt{d_k})V
$$

The attention weight matrix is computed by dividing QK Transpose and then normalizing the matrix using softmax, it’s called dot product attention, because we’re using the dot product to generate the weight matrix (QK^T) to get the attention weights for each position.

$$
softmax(QK^T) = \frac{exp(QK^T/\sqrt{d_k}) / \sum_j{exp(Q_iK_j/\sqrt{d_k))
$$

If we visualize the vector V, we can see why this is called attention, as our output is a weighted sum.

The weights are normalized weights (softmax) for each position.

After computing the weights, we can multiply them by Value (V) to get the final output.

$$
Multi head attention is the same attention but we split the matrices into multiple matrices (Q, K, V) and concatenate them afterwards.

$$
MultiHeadAttention(Q, K, V, h) = Concat(head_1, head_2, ..., head_h)W^O.
$$

Here h is number of heads.

MultiHeadAttention can be seen as a way to handle multiple representation learning tasks because it projects the query and key matrices independently rather than concatenating.

We’ve learned this on the query and key matrices, but we can also do the same for values.

$$
MultiHeadAttention(Q, K, V, h) = Concat(selfAttention(QW_Q^Q_h, SW_K^Qh, SW_V^Qh)W^O.
$$

In the end, we add a residual connection and normalization.

Self Attention takes the input sequence and passes it through two layers (FFN) and this process can be learned in Transformer Encoder and Decoder layers, as you can see in BERT model.

Let’s dive deeper into MultiHeadAttention and how it’s calculated.

$$
MultiHeadAttention(Q, K, V, h) = Concat(head_1, head_2, ..., head_h)W^O + X
$$

$$
MultiHeadAttention(Q, K, V, h) = [selfAttention(QW_Q^Q_1, SW_K^Qh, SW_V^Qh)W^O + X
$$

We can see it as a conc
