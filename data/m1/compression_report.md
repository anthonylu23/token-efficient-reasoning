# M1 tokenizer compression report

Items: 12600

## Overall R3 gate

| Tokenizer | Median reduction | Threshold | Result |
|---|---:|---:|---|
| deepseek-r1-distill-qwen-1.5b | 32.0% | 30.0% | pass |
| qwen3-0.6b | 32.0% | 30.0% | pass |

## By family, depth, and representation

| Tokenizer | Family | Depth | Rep. | Tokens p10 / median / p90 | Reduction p10 / median / p90 |
|---|---|---:|---|---:|---:|
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 2 | R0 | 32.0 / 32.0 / 32.0 | 0.0% / 0.0% / 0.0% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 2 | R1 | 28.0 / 28.0 / 28.0 | 12.5% / 12.5% / 12.5% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 2 | R2 | 17.0 / 17.0 / 18.0 | 43.8% / 46.9% / 46.9% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 2 | R3 | 22.0 / 22.0 / 22.0 | 31.2% / 31.2% / 31.2% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 4 | R0 | 64.0 / 64.0 / 64.0 | 0.0% / 0.0% / 0.0% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 4 | R1 | 56.0 / 56.0 / 56.0 | 12.5% / 12.5% / 12.5% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 4 | R2 | 35.0 / 36.0 / 37.0 | 42.2% / 43.8% / 45.3% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 4 | R3 | 44.0 / 44.0 / 44.0 | 31.2% / 31.2% / 31.2% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 8 | R0 | 128.0 / 128.0 / 128.0 | 0.0% / 0.0% / 0.0% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 8 | R1 | 112.0 / 112.0 / 112.0 | 12.5% / 12.5% / 12.5% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 8 | R2 | 71.0 / 72.0 / 74.0 | 42.2% / 43.8% / 44.5% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 8 | R3 | 88.0 / 88.0 / 88.0 | 31.2% / 31.2% / 31.2% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 12 | R0 | 195.0 / 195.0 / 195.0 | 0.0% / 0.0% / 0.0% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 12 | R1 | 171.0 / 171.0 / 171.0 | 12.3% / 12.3% / 12.3% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 12 | R2 | 111.0 / 112.0 / 114.0 | 41.5% / 42.6% / 43.1% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 12 | R3 | 135.0 / 135.0 / 135.0 | 30.8% / 30.8% / 30.8% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 16 | R0 | 263.0 / 263.0 / 263.0 | 0.0% / 0.0% / 0.0% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 16 | R1 | 231.0 / 231.0 / 231.0 | 12.2% / 12.2% / 12.2% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 16 | R2 | 151.0 / 153.0 / 155.0 | 41.1% / 41.8% / 42.6% |
| deepseek-r1-distill-qwen-1.5b | directed_graph_traversal | 16 | R3 | 183.0 / 183.0 / 183.0 | 30.4% / 30.4% / 30.4% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 2 | R0 | 39.0 / 41.0 / 44.0 | 0.0% / 0.0% / 0.0% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 2 | R1 | 26.0 / 29.0 / 31.0 | 26.2% / 29.5% / 33.3% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 2 | R2 | 15.0 / 18.0 / 20.0 | 52.5% / 56.1% / 60.5% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 2 | R3 | 21.0 / 24.0 / 26.0 | 38.1% / 41.9% / 45.2% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 4 | R0 | 78.0 / 83.0 / 88.0 | 0.0% / 0.0% / 0.0% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 4 | R1 | 53.0 / 58.0 / 62.0 | 27.4% / 29.8% / 32.9% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 4 | R2 | 32.0 / 37.0 / 41.0 | 52.3% / 54.9% / 59.3% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 4 | R3 | 44.0 / 49.0 / 53.0 | 38.1% / 40.7% / 44.2% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 8 | R0 | 157.0 / 165.0 / 178.0 | 0.0% / 0.0% / 0.0% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 8 | R1 | 107.0 / 116.0 / 127.0 | 27.5% / 29.7% / 32.1% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 8 | R2 | 66.0 / 75.0 / 86.0 | 51.1% / 54.4% / 57.9% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 8 | R3 | 90.0 / 99.0 / 110.0 | 37.3% / 39.9% / 42.7% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 12 | R0 | 240.0 / 253.5 / 275.0 | 0.0% / 0.0% / 0.0% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 12 | R1 | 166.0 / 180.0 / 200.0 | 26.8% / 29.0% / 31.3% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 12 | R2 | 105.0 / 119.0 / 139.0 | 49.1% / 53.2% / 56.4% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 12 | R3 | 141.0 / 155.0 / 175.0 | 36.0% / 38.9% / 41.6% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 16 | R0 | 325.0 / 342.0 / 369.0 | 0.0% / 0.0% / 0.0% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 16 | R1 | 225.0 / 244.0 / 270.1 | 26.6% / 28.7% / 30.8% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 16 | R2 | 144.0 / 163.0 / 189.1 | 48.5% / 52.5% / 55.6% |
| deepseek-r1-distill-qwen-1.5b | multi_operation_arithmetic | 16 | R3 | 192.0 / 211.0 / 237.1 | 35.5% / 38.5% / 40.9% |
| qwen3-0.6b | directed_graph_traversal | 2 | R0 | 32.0 / 32.0 / 32.0 | 0.0% / 0.0% / 0.0% |
| qwen3-0.6b | directed_graph_traversal | 2 | R1 | 28.0 / 28.0 / 28.0 | 12.5% / 12.5% / 12.5% |
| qwen3-0.6b | directed_graph_traversal | 2 | R2 | 17.0 / 17.0 / 18.0 | 43.8% / 46.9% / 46.9% |
| qwen3-0.6b | directed_graph_traversal | 2 | R3 | 22.0 / 22.0 / 22.0 | 31.2% / 31.2% / 31.2% |
| qwen3-0.6b | directed_graph_traversal | 4 | R0 | 64.0 / 64.0 / 64.0 | 0.0% / 0.0% / 0.0% |
| qwen3-0.6b | directed_graph_traversal | 4 | R1 | 56.0 / 56.0 / 56.0 | 12.5% / 12.5% / 12.5% |
| qwen3-0.6b | directed_graph_traversal | 4 | R2 | 35.0 / 36.0 / 37.0 | 42.2% / 43.8% / 45.3% |
| qwen3-0.6b | directed_graph_traversal | 4 | R3 | 44.0 / 44.0 / 44.0 | 31.2% / 31.2% / 31.2% |
| qwen3-0.6b | directed_graph_traversal | 8 | R0 | 128.0 / 128.0 / 128.0 | 0.0% / 0.0% / 0.0% |
| qwen3-0.6b | directed_graph_traversal | 8 | R1 | 112.0 / 112.0 / 112.0 | 12.5% / 12.5% / 12.5% |
| qwen3-0.6b | directed_graph_traversal | 8 | R2 | 71.0 / 72.0 / 74.0 | 42.2% / 43.8% / 44.5% |
| qwen3-0.6b | directed_graph_traversal | 8 | R3 | 88.0 / 88.0 / 88.0 | 31.2% / 31.2% / 31.2% |
| qwen3-0.6b | directed_graph_traversal | 12 | R0 | 195.0 / 195.0 / 195.0 | 0.0% / 0.0% / 0.0% |
| qwen3-0.6b | directed_graph_traversal | 12 | R1 | 171.0 / 171.0 / 171.0 | 12.3% / 12.3% / 12.3% |
| qwen3-0.6b | directed_graph_traversal | 12 | R2 | 111.0 / 112.0 / 114.0 | 41.5% / 42.6% / 43.1% |
| qwen3-0.6b | directed_graph_traversal | 12 | R3 | 135.0 / 135.0 / 135.0 | 30.8% / 30.8% / 30.8% |
| qwen3-0.6b | directed_graph_traversal | 16 | R0 | 263.0 / 263.0 / 263.0 | 0.0% / 0.0% / 0.0% |
| qwen3-0.6b | directed_graph_traversal | 16 | R1 | 231.0 / 231.0 / 231.0 | 12.2% / 12.2% / 12.2% |
| qwen3-0.6b | directed_graph_traversal | 16 | R2 | 151.0 / 153.0 / 155.0 | 41.1% / 41.8% / 42.6% |
| qwen3-0.6b | directed_graph_traversal | 16 | R3 | 183.0 / 183.0 / 183.0 | 30.4% / 30.4% / 30.4% |
| qwen3-0.6b | multi_operation_arithmetic | 2 | R0 | 39.0 / 41.0 / 44.0 | 0.0% / 0.0% / 0.0% |
| qwen3-0.6b | multi_operation_arithmetic | 2 | R1 | 26.0 / 29.0 / 31.0 | 26.2% / 29.5% / 33.3% |
| qwen3-0.6b | multi_operation_arithmetic | 2 | R2 | 15.0 / 18.0 / 20.0 | 52.5% / 56.1% / 60.5% |
| qwen3-0.6b | multi_operation_arithmetic | 2 | R3 | 21.0 / 24.0 / 26.0 | 38.1% / 41.9% / 45.2% |
| qwen3-0.6b | multi_operation_arithmetic | 4 | R0 | 78.0 / 83.0 / 88.0 | 0.0% / 0.0% / 0.0% |
| qwen3-0.6b | multi_operation_arithmetic | 4 | R1 | 53.0 / 58.0 / 62.0 | 27.4% / 29.8% / 32.9% |
| qwen3-0.6b | multi_operation_arithmetic | 4 | R2 | 32.0 / 37.0 / 41.0 | 52.3% / 54.9% / 59.3% |
| qwen3-0.6b | multi_operation_arithmetic | 4 | R3 | 44.0 / 49.0 / 53.0 | 38.1% / 40.7% / 44.2% |
| qwen3-0.6b | multi_operation_arithmetic | 8 | R0 | 157.0 / 165.0 / 178.0 | 0.0% / 0.0% / 0.0% |
| qwen3-0.6b | multi_operation_arithmetic | 8 | R1 | 107.0 / 116.0 / 127.0 | 27.5% / 29.7% / 32.1% |
| qwen3-0.6b | multi_operation_arithmetic | 8 | R2 | 66.0 / 75.0 / 86.0 | 51.1% / 54.4% / 57.9% |
| qwen3-0.6b | multi_operation_arithmetic | 8 | R3 | 90.0 / 99.0 / 110.0 | 37.3% / 39.9% / 42.7% |
| qwen3-0.6b | multi_operation_arithmetic | 12 | R0 | 240.0 / 253.5 / 275.0 | 0.0% / 0.0% / 0.0% |
| qwen3-0.6b | multi_operation_arithmetic | 12 | R1 | 166.0 / 180.0 / 200.0 | 26.8% / 29.0% / 31.3% |
| qwen3-0.6b | multi_operation_arithmetic | 12 | R2 | 105.0 / 119.0 / 139.0 | 49.1% / 53.2% / 56.4% |
| qwen3-0.6b | multi_operation_arithmetic | 12 | R3 | 141.0 / 155.0 / 175.0 | 36.0% / 38.9% / 41.6% |
| qwen3-0.6b | multi_operation_arithmetic | 16 | R0 | 325.0 / 342.0 / 369.0 | 0.0% / 0.0% / 0.0% |
| qwen3-0.6b | multi_operation_arithmetic | 16 | R1 | 225.0 / 244.0 / 270.1 | 26.6% / 28.7% / 30.8% |
| qwen3-0.6b | multi_operation_arithmetic | 16 | R2 | 144.0 / 163.0 / 189.1 | 48.5% / 52.5% / 55.6% |
| qwen3-0.6b | multi_operation_arithmetic | 16 | R3 | 192.0 / 211.0 / 237.1 | 35.5% / 38.5% / 40.9% |
