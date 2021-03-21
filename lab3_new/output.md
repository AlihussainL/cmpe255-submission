| Experiement | Accuracy | Confusion Matrix | Comment |
|-------------|----------|------------------|---------|
| Baseline    | 0.6770833333333334 | [[114  16] [ 46  16]] |  |
| Solution 1   | 0.7662337662337663  | [[133  13] [ 41  44]] |  Iteration 1 uses glucose the most correlated feature and the split of train and test is set to .3 |
| Solution 2   | 0.7835497835497836  | [[132  14] [ 36  49]] |  Iteration 2 picks 3 fetaures in the order of correlation and the split of train and test is set to .3 |
| Solution 3   | 0.7878787878787878  | [[132  14] [ 35  50]] |  Iteration 3 picks 3 fetures and alos replaces 0 vlaues in data with mean |
