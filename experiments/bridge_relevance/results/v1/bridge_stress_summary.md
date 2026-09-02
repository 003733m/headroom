# Bridge-eligible real-bug stress benchmark

| Task | Cond | Pass | Wall s | Requests | Orig input | Optimized | Derived saved | relevance_split | search+bridge | search-shaped | Grep mentions | shell rg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | off | PASS | 85.1 | 15 | 554875 | 494019 | 60856 | 0 | 0 | 1 | 1 | 2 |
| T1 | on | FAIL | 150.7 | 20 | 1201773 | 1158767 | 43006 | 1 | 1 | 2 | 2 | 3 |
| T5 | off | PASS | 92.4 | 16 | 594672 | 490142 | 104530 | 0 | 0 | 2 | 1 | 2 |
| T5 | on | PASS | 169.5 | 21 | 865647 | 701277 | 164370 | 1 | 1 | 3 | 1 | 6 |
| T6 | off | PASS | 129.2 | 21 | 1056119 | 951486 | 104633 | 0 | 0 | 1 | 1 | 2 |
| T6 | on | PASS | 93.7 | 17 | 576656 | 505362 | 71294 | 0 | 0 | 1 | 1 | 2 |

## Pairwise activation

- T1: OFF pass=True, ON pass=False; ON relevance_split=1, OFF relevance_split=0.
- T5: OFF pass=True, ON pass=True; ON relevance_split=1, OFF relevance_split=0.
- T6: OFF pass=True, ON pass=True; ON relevance_split=0, OFF relevance_split=0.
