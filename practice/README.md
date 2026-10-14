# Daily coding practice for ML research interviews

20 days of practice, one notebook per day. Each day has three problems:

- **A. Python fluency:** the language features interviewers expect you to know cold (classes, special methods, generators, closures, mutability).
- **B. Algorithms:** classic interview problems (hash maps, two pointers, BFS/DFS, heaps, binary search, dynamic programming, backtracking).
- **C. NumPy / PyTorch / ML from scratch:** the "implement X without the library" questions ML research interviews are known for (softmax, cross-entropy, k-means, backprop, Adam, LayerNorm, attention, VAE and MDN losses).

Each problem has a function stub to fill in and a test cell that tells you when you've got it right. There are no solutions in this folder on purpose. If you're stuck after 20 minutes, ask Claude for a hint.

## How to use it

1. **One notebook a day, about 60–90 minutes.** Consistency matters more than length. 5 days a week, with the weekend for redos.
2. **Plan before typing.** Write the approach as comments first: inputs, outputs, edge cases, complexity. In an interview this is half the grade.
3. **20-minute rule.** Struggle for 20 minutes, then ask for a hint, not the answer.
4. **Say it out loud.** Every problem has prompts like "what's the complexity?" or "why does this work?". Practise answering them out loud. Explaining your code is a separate skill from writing it.
5. **Redo what was hard.** Anything that needed a hint or took over 20 minutes goes on the redo list below. Solve it again from a blank cell 3 days later.
6. **No autocomplete for the first attempt** if you can turn it off. Interviews are usually in a plain editor.

## Schedule

| | Day | Topics | A (Python) | B (Algorithms) | C (NumPy / PyTorch / ML) |
|---|---|---|---|---|---|
| **Week 1** | | **Python foundations** | | | |
| [ ] | [1](day01.ipynb) | Dicts, hash maps, dtypes | Word counts | Two sum | Frames ↔ model input |
| [ ] | [2](day02.ipynb) | Comprehensions, stacks, broadcasting | Comprehensions and slicing | Valid parentheses | Pairwise distances |
| [ ] | [3](day03.ipynb) | Mutability, sorting, stability | Mutable defaults and copies | Merge intervals | Stable softmax / logsumexp |
| [ ] | [4](day04.ipynb) | Classes, Kadane, cross-entropy | Stack class | Max subarray | One-hot and cross-entropy |
| [ ] | [5](day05.ipynb) | Special methods, binary search, splits | Vector2D operators | Binary search / lower bound | Episode split |
| **Week 2** | | **Data structures and NumPy** | | | |
| [ ] | [6](day06.ipynb) | Generators, heaps, batching | Generators | Top-k frequent | Mini DataLoader |
| [ ] | [7](day07.ipynb) | defaultdict, sliding window, conv shapes | Group anagrams | Longest unique substring | Conv shape arithmetic |
| [ ] | [8](day08.ipynb) | Closures, decorators, grids, cumsum | Counter and decorator | Number of islands | Moving averages |
| [ ] | [9](day09.ipynb) | Sort keys, heaps, k-means | Sorting with keys | Kth largest | K-means++ |
| [ ] | [10](day10.ipynb) | Exceptions, LRU, linear regression | Safe loading | LRU cache | Linear regression ×2 |
| **Week 3** | | **PyTorch and dynamic programming** | | | |
| [ ] | [11](day11.ipynb) | Recursion, DP, tensor shapes | Flatten and permutations | Stairs and coins | Tensor shape drills |
| [ ] | [12](day12.ipynb) | *args/**kwargs, matrices, grad check | Configs | Rotate and spiral | Gradient checking |
| [ ] | [13](day13.ipynb) | Dataclasses, topo sort, nn.Module | TrainConfig | Course schedule | MLP module |
| [ ] | [14](day14.ipynb) | Iterators, prefix products, training | Countdown iterator | Product except self | Training loop |
| [ ] | [15](day15.ipynb) | Debugging, edit distance, sequences | Find the bugs | Edit distance | Sequence Dataset for M |
| **Week 4** | | **ML from scratch** | | | |
| [ ] | [16](day16.ipynb) | Tokenizing, BFS, logistic regression | Tokenizer and vocab | Grid shortest path | Logistic regression |
| [ ] | [17](day17.ipynb) | Vectorizing, two heaps, backprop | Vectorize this | Running median | Backprop by hand |
| [ ] | [18](day18.ipynb) | Inheritance, backtracking, Adam | Optimizer classes | Subsets and parentheses | Adam and LayerNorm |
| [ ] | [19](day19.ipynb) | Gotchas, rotated search, attention | Predict the output | Rotated array search | Attention |
| [ ] | [20](day20.ipynb) | Mock interview, world-model math | Merge k sorted (timed) | Word break | Reparam, KL, MDN loss |

Several C problems are pieces of your world-model project: frame conversion (day 1), episode splits (5), conv shapes for the VAE (7), the sequence Dataset for the MDN-RNN (15), and the VAE and MDN losses (20). You can reuse them.

## Redo list

Add anything that needed a hint or took more than 20 minutes. Redo it from scratch 3 days later.

| Problem | Why it was hard | Redo date | Done |
|---|---|---|---|
| | | | |

## After day 20

- **More algorithms:** work through a standard list (NeetCode 150 or "Blind 75"), about 2 a day, grouped by pattern.
- **More ML from scratch:** BatchNorm with running statistics, a GRU/LSTM cell, multi-head attention as a module, a tiny GPT trained on characters, PCA via SVD, a decision tree.
- **Mock interviews:** ask Claude to run a timed mock interview with a problem you haven't seen, and to grade your explanation as well as your code.
