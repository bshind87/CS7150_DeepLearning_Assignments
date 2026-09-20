### What I learned

Across the two notebooks I covered the basic machinery of training neural networks. 
I used PyTorch's autograd to compute derivatives, including higher-order ones, and saw 
how this supports gradient-based optimization. The McCulloch–Pitts neuron exercise showed 
how simple threshold units implement logical functions, and why non-linearly separable problems 
such as XOR require more than one layer. I also studied softmax together with KL divergence, 
cross-entropy, and squared error, and learned why cross-entropy is preferred for classification 
because it gives better-behaved gradients. The second notebook made the mechanics of backpropagation 
explicit: gradients are obtained by applying the chain rule backwards through the computational graph, 
reusing intermediate results so that the cost of the backward pass is roughly that of the forward pass 
rather than growing with the number of parameters. This efficiency comes at a memory cost, since 
activations from the forward pass must be stored for reuse. Seen this way, backpropagation is a 
particular case of reverse-mode automatic differentiation, which is what autograd implements.

### Challenges and unclear points

Several parts worked in code but remain only partly clear to me. Deriving the softmax Jacobian 
by hand and reducing it to its compact form in terms of the output probabilities was intricate. 
I observed the flattening of squared-error gradients visually, but I do not yet fully follow the 
mathematics of why they saturate so much more severely than cross-entropy far from the optimum. 
Choosing weights and biases for the McCulloch–Pitts neuron also relied on trial and error rather 
than a systematic method. In the backpropagation notebook, I could implement the backward pass 
correctly while still being unsure how to reason carefully about its time and memory complexity, 
and about when forward-mode differentiation would be preferable to reverse mode.
