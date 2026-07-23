# M1 split-overlap audit

Overall result: **pass**

## Coverage

| Split | Items | Families | Depths |
|---|---:|---|---|
| train | 8000 | directed_graph_traversal: 4000, multi_operation_arithmetic: 4000 | 12: 2000, 2: 2000, 4: 2000, 8: 2000 |
| validation | 800 | directed_graph_traversal: 400, multi_operation_arithmetic: 400 | 12: 200, 2: 200, 4: 200, 8: 200 |
| test | 2000 | directed_graph_traversal: 1000, multi_operation_arithmetic: 1000 | 12: 500, 2: 500, 4: 500, 8: 500 |
| test_compositional | 800 | directed_graph_traversal: 400, multi_operation_arithmetic: 400 | 12: 200, 2: 200, 4: 200, 8: 200 |
| test_depth | 1000 | directed_graph_traversal: 500, multi_operation_arithmetic: 500 | 16: 1000 |

## Cross-split collisions

| Audit | Collisions |
|---|---:|
| seeds | 0 |
| instances | 0 |
| prompts | 0 |
| value_signatures | 0 |
| train_compositional_templates | 0 |
| train_depth_test_depths | 0 |
