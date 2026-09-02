# Natural coding-agent benchmark

| Task | Cond | Pass | Wall s | Requests | Orig input | Optimized | Derived saved | relevance_split | search+bridge | search-shaped | Grep mentions | shell rg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | off | PASS | 229.8 | 15 | 484983 | 469627 | 15356 | 0 | 0 | 0 | 4 | 0 |
| T1 | on | PASS | 311.1 | 17 | 722876 | 711771 | 11105 | 0 | 0 | 0 | 11 | 0 |
| T2 | off | PASS | 33.0 | 7 | 97609 | 96106 | 1503 | 0 | 0 | 0 | 1 | 0 |
| T2 | on | PASS | 47.8 | 10 | 150176 | 147397 | 2779 | 0 | 0 | 0 | 1 | 0 |
| T3 | off | PASS | 64.0 | 8 | 185435 | 177941 | 7494 | 0 | 0 | 0 | 3 | 0 |
| T3 | on | PASS | 78.1 | 7 | 151562 | 149314 | 2248 | 0 | 0 | 0 | 1 | 0 |
| T4 | off | PASS | 73.6 | 12 | 212116 | 208498 | 3618 | 0 | 0 | 0 | 5 | 0 |
| T4 | on | PASS | 63.9 | 11 | 155604 | 152534 | 3070 | 0 | 0 | 0 | 3 | 0 |
| T5 | off | PASS | 78.7 | 13 | 385639 | 381503 | 4136 | 0 | 0 | 0 | 7 | 0 |
| T5 | on | PASS | 107.4 | 18 | 769396 | 763292 | 6104 | 0 | 0 | 0 | 9 | 0 |
| T6 | off | PASS | 89.3 | 12 | 294822 | 291628 | 3194 | 0 | 0 | 0 | 5 | 0 |
| T6 | on | PASS | 88.6 | 15 | 429905 | 423992 | 5913 | 0 | 0 | 0 | 6 | 0 |

## Pairwise activation

- T1: OFF pass=True, ON pass=True; ON relevance_split=0, OFF relevance_split=0.
- T2: OFF pass=True, ON pass=True; ON relevance_split=0, OFF relevance_split=0.
- T3: OFF pass=True, ON pass=True; ON relevance_split=0, OFF relevance_split=0.
- T4: OFF pass=True, ON pass=True; ON relevance_split=0, OFF relevance_split=0.
- T5: OFF pass=True, ON pass=True; ON relevance_split=0, OFF relevance_split=0.
- T6: OFF pass=True, ON pass=True; ON relevance_split=0, OFF relevance_split=0.
