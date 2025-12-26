# MakeMore-Part-2-V1
# My Notes on Andrej Karpathy's Makemore Series

These are my personal notes and insights from implementing the models in Andrej Karpathy's excellent "makemore" tutorial series (part of his *Neural Networks: Zero to Hero* playlist).

---

**block_size** should be chosen based on how much context actually helps, not based on the shortest sequence. The padding trick (using '.') is why **block_size** can (and should) be larger than the shortest (or even average) name length.

If **block_size = 2**, it means: give every character its own little 2D personality vector. Let the neural network move these 27 points around in 2D space during training so that characters that behave similarly end up close together.

Rule of thumb → set block size to the longest context you can afford (in compute and memory) — more importantly, one that still gives good performance.

### Dataset Splits
When building the dataset (80% training, 10% validation, 10% test):

- **Validation (dev) split** → used frequently to tune hyper-parameters (hidden layer size, embedding size, learning rate, batch size, regularization). You train on training data, evaluate loss on validation to compare configurations and detect overfitting (large gap between train and val loss).
- **Test split** → evaluated only sparingly at the end to report final model performance. This gives an unbiased estimate of generalization to unseen data. Overusing it risks indirectly overfitting to the test set.

We also evaluate on the training data itself (without training on it during evaluation) to monitor progress and detect overfitting.

### What the Network Learns Automatically
The network has discovered that:

- Vowels tend to group together.
- Consonants group together.
- Rare letters 'q', 'x', 'j' are far away.
- The end token '.' is isolated.

All of this emerges just from predicting the next character — no one told it what a vowel is. It learned it because vowels behave similarly in names.

### Embeddings
`C = torch.randn((vocab_size, n_embed))`

- "C" is a trainable matrix that acts as a learned dictionary.
- It represents the coordinates of each character in an `n_embed`-dimensional space.
- Row 0 → learned embedding vector for 'a'  
  Row 1 → for 'b'  
  ...  
  Row 26 → for '.'
- During training, gradients flow back through the lookups and update the rows of "C".

**Couldn't the weights just learn the "coordinates" (embeddings) themselves?**

Yes, in theory "W1" could learn everything, including what each character "means". But it would be less efficient:
- Without "C" → no sharing of knowledge → wastes capacity, learns slower, generalizes worse.
- With embeddings, there's only one vector for 'a', shared across all positions and all examples. Every time 'a' appears (no matter where), the model sees the exact same embedding vector.

**How do the coordinates actually get updated?**

Through backpropagation, just like regular weights.

### Hidden Layer (`n_hidden`)
- Controls how many different patterns, combinations, and ideas the network can think about at the same time.
- The hidden neurons are literally 100 independent little detectives, each allowed to look at the full input (e.g., 70 numbers) and say: "I've noticed something."
- Each neuron gets its own column in W1 and learns to fire strongly when its pattern appears.

**Why not 1 neuron? Why not 1000?**

- 1 → can only learn one pattern.
- 10 → can learn 10 patterns → repetitive names.
- 50 → pretty good names.
- 100 → excellent, diverse, human-like names (sweet spot).
- 300 → slightly better, but barely noticeable on this tiny dataset.
- 1000 → overfits — starts memorizing training names.

Each neuron receives `block_size * n_embed` inputs (e.g., 70 weights).

Every time we see a context like "...e m" and the next character should be 'm', the gradients push on the involved embedding vectors (the '.' paddings, 'e', 'm'). The push says: "move these points a tiny bit so that next time, when flattened, the hidden neurons can more easily predict 'm'."

### Weight Initialization
**Why initialize the weights carefully?**

Instead of wasting the first few thousand epochs squashing down exploding weights, we scale them appropriately before training.

**What is `*(5/3) / (n_embed * block_size)** about?**

It's Kaiming initialization — the standard way to initialize neural nets (as of 2025).
- For tanh: std = gain / √(fan_in), where fan_in = `n_embed * block_size`.
- If we use plain `torch.randn()` (std=1), activations explode or vanish through layers.
- The fix: scale so variance stays ~1 at every layer.

**Why keep variance ~1?**

Otherwise gradients vanish or explode — the vanishing/exploding gradient problem, the #1 reason deep nets didn't work before 2010.

**Where does the "5/3" gain come from?**

It accounts for how much tanh squashes variance. Tanh on std=1 input produces std≈0.63 output, so we boost the input by the inverse factor to keep output variance ≈1.

### Mini-Batches
**What happens when using mini-batches instead of the full 200k examples each step?**

You see the data many times in different combinations. E.g., after 100k steps with batch_size=32 → 3.2 million examples processed → each example seen ~14 times on average, but always with different neighbors.

**Why do this?**

- Noise is your friend — randomness acts as a natural regularizer, prevents memorizing particular order.
- Converges faster in practice because each (noisy) gradient wiggles around the true direction.

### Batch Normalization
**Why use BatchNorm?**

If pre-activations have wildly different scales across neurons, the loss surface becomes stretched/skewed → gradients huge in some directions, tiny in others → slow, unstable optimization.

Normalizing to mean=0, std=1 makes the loss surface more isotropic (rounder, smoother) → gradient descent takes more direct steps. It also allows much higher learning rates.

BatchNorm is so powerful that initialization almost stops mattering. Even with perfect init but no BatchNorm, very deep nets can still get dead tanh neurons.

**What does BatchNorm do?**

- Compute mean and std of each neuron across the batch.
- Subtract mean, divide by std → normalize to ~N(0,1).
- Scale and shift with learnable bngain and bnbias (controls neuron "trigger happiness").

**Line breakdown**

- `hpreact.mean(0, keepdim=True)` → mean along batch dim for each neuron.
- `keepdim=True` → crucial for broadcasting.
- Same for std.
- epsilon → prevents division by zero.
- `(hpreact - mean) / (std + 1e-5)` → normalizes each neuron to mean≈0, std≈1 across the batch.

**Why specifically mean=0 and std=1?**

- Mean=0 centers symmetrically — good for odd functions like tanh.
- Std=1 gives consistent scale; bngain/bnbias can learn any desired scale/shift.

**How does it force std exactly 1?**

Dividing any set by its own std makes the new std exactly 1.

**Why remove bias `b1` after adding BatchNorm?**

Any constant shift from b1 is immediately subtracted by the `-bnmean` term. Shifting is now fully controlled by the learned bnbias.

**Why running statistics (`bnmean_running`, `bnstd_running`) during sampling?**

During training, weights adapted to seeing globally normalized activations. At inference (batch=1):
- Using running stats → same global normalization the model is used to → model feels at home → accurate predictions → great names.
- Normalizing over single example → distribution changes every step → model gets confused → slightly worse names.

**Why `@torch.no_grad()` for split evaluation?**

No gradients needed on val/test — skips building the computation graph → faster forward pass.

### Sampling / Generation
**Why `C[torch.tensor([context])]` instead of `C[torch.tensor(context)]`?**

The model expects a batch dimension (even if B=1). Adding the extra dim ensures correct shape without changing C.

The batch dimension comes entirely from the index tensor shape.

**Generated name lengths** are limited by the context length (`block_size`).

**Does the embedding table `C (27, n_embd)` change during the forward pass?**

No — it stays fixed shape throughout training and generation.

**Softmax**

- Turns logits into probabilities.
- Amplifies differences (exponential).
- Only relative differences matter.
- Used in cross-entropy (combined with log for stability) and in sampling via `torch.multinomial`.

**Why epsilon when sampling?**

Numerical underflow in float32 — very confident predictions can round tiny probabilities to exactly 0.

**Why normalize over dim=0 during training but dim=-1 during generation?**

With batch=1, dim=0 gives std≈0 → division by near-zero → NaNs/infs.

---

These notes capture the key intuitions from building the MLP and introducing BatchNorm in the makemore series. Great way to deeply understand modern neural net practices!
