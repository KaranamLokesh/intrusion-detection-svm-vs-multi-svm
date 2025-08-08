<ipython-input-16-875b37cb36e2>:53: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  train_set["label"] = train_set["label"].replace(label_map)
<ipython-input-16-875b37cb36e2>:54: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  test_set["label"] = test_set["label"].replace(label_map)

--- OneVsOne SVM Results ---
Accuracy: 0.7490241305890702
Precision: 0.7853497560868729
Recall: 0.7490241305890702
F1 Score: 0.7009792849093258
Confusion Matrix:
 [[9500   55  150    5    1]
 [1569 5753  138    0    0]
 [ 623  186 1612    0    0]
 [2806    0   64   15    0]
 [  61    0    0    0    6]]
              precision    recall  f1-score   support

           0       0.65      0.98      0.78      9711
           1       0.96      0.77      0.86      7460
           2       0.82      0.67      0.74      2421
           3       0.75      0.01      0.01      2885
           4       0.86      0.09      0.16        67

    accuracy                           0.75     22544
   macro avg       0.81      0.50      0.51     22544
weighted avg       0.79      0.75      0.70     22544


--- Pairwise Meta SVM Results ---
Accuracy: 0.7513307310149042
Precision: 0.8007942533279758
Recall: 0.7513307310149042
F1 Score: 0.7035452136967604
Confusion Matrix:
 [[9502   55  150    3    1]
 [1472 5874  114    0    0]
 [ 701  190 1530    0    0]
 [2799    0   61   25    0]
 [  59    0    0    1    7]]
              precision    recall  f1-score   support

           0       0.65      0.98      0.78      9711
           1       0.96      0.79      0.87      7460
           2       0.82      0.63      0.72      2421
           3       0.86      0.01      0.02      2885
           4       0.88      0.10      0.19        67

    accuracy                           0.75     22544
   macro avg       0.84      0.50      0.51     22544
weighted avg       0.80      0.75      0.70     22544


--- Pairwise Meta SVM (Balanced) Results ---
Accuracy: 0.754125266146203
Precision: 0.8185395616695319
Recall: 0.754125266146203
F1 Score: 0.7263970104198394
Confusion Matrix:
 [[9461   57  149   11   33]
 [1646 5744   70    0    0]
 [ 777  182 1460    0    2]
 [2126    0   84  322  353]
 [  48    0    0    5   14]]
              precision    recall  f1-score   support

           0       0.67      0.97      0.80      9711
           1       0.96      0.77      0.85      7460
           2       0.83      0.60      0.70      2421
           3       0.95      0.11      0.20      2885
           4       0.03      0.21      0.06        67

    accuracy                           0.75     22544
   macro avg       0.69      0.53      0.52     22544
weighted avg       0.82      0.75      0.73     22544


--- Enhanced Pairwise Meta SVM Results ---
Accuracy: 0.7684084457061746
Precision: 0.7870776451721305
Recall: 0.7684084457061746
F1 Score: 0.7297042434188944
Confusion Matrix:
 [[9474   52  166   12    7]
 [ 931 6190  246   60   33]
 [ 789  174 1458    0    0]
 [2581    2   70  194   38]
 [  54    0    2    4    7]]
              precision    recall  f1-score   support

           0       0.69      0.98      0.80      9711
           1       0.96      0.83      0.89      7460
           2       0.75      0.60      0.67      2421
           3       0.72      0.07      0.12      2885
           4       0.08      0.10      0.09        67

    accuracy                           0.77     22544
   macro avg       0.64      0.52      0.52     22544
weighted avg       0.79      0.77      0.73     22544
