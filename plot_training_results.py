import matplotlib.pyplot as plt
import re

# Input text data
data = '''
Epoch 1/100
Train Loss: 0.9736 | Val Loss: 0.7943
Val mIoU: 0.3835 | Val Accuracy: 0.9520
Val Precision: 0.4538 | Val Recall: 0.5263
Val Cls Acc: 0.3333

Class-wise Metrics:
Healthy:
  IoU: 0.0000 | Accuracy: 0.9973
  Precision: 0.0000 | Recall: 0.0000
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.5490 | Accuracy: 0.9626
  Precision: 0.6845 | Recall: 0.7349
  Cls Acc: 0.0000 | Samples: 81
GRED:
  IoU: 0.6015 | Accuracy: 0.8960
  Precision: 0.6768 | Recall: 0.8439
  Cls Acc: 0.0000 | Samples: 63
New best model saved with mIoU: 0.3835

Epoch 2/100
Train Loss: 0.6755 | Val Loss: 0.6574
Val mIoU: 0.4035 | Val Accuracy: 0.9558
Val Precision: 0.4935 | Val Recall: 0.5192
Val Cls Acc: 0.3333

Class-wise Metrics:
Healthy:
  IoU: 0.0000 | Accuracy: 0.9904
  Precision: 0.0000 | Recall: 0.0000
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.5696 | Accuracy: 0.9681
  Precision: 0.7750 | Recall: 0.6825
  Cls Acc: 0.0000 | Samples: 81
GRED:
  IoU: 0.6409 | Accuracy: 0.9088
  Precision: 0.7054 | Recall: 0.8752
  Cls Acc: 0.0000 | Samples: 63
New best model saved with mIoU: 0.4035

Epoch 3/100
Train Loss: 0.5591 | Val Loss: 0.5504
Val mIoU: 0.4354 | Val Accuracy: 0.9613
Val Precision: 0.5290 | Val Recall: 0.5398
Val Cls Acc: 0.3757

Class-wise Metrics:
Healthy:
  IoU: 0.0000 | Accuracy: 0.9958
  Precision: 0.0000 | Recall: 0.0000
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.6493 | Accuracy: 0.9763
  Precision: 0.8841 | Recall: 0.7097
  Cls Acc: 0.0000 | Samples: 81
GRED:
  IoU: 0.6570 | Accuracy: 0.9117
  Precision: 0.7028 | Recall: 0.9098
  Cls Acc: 0.1270 | Samples: 63
New best model saved with mIoU: 0.4354

Epoch 4/100
Train Loss: 0.4614 | Val Loss: 0.5800
Val mIoU: 0.4605 | Val Accuracy: 0.9589
Val Precision: 0.7108 | Val Recall: 0.5701
Val Cls Acc: 0.6667

Class-wise Metrics:
Healthy:
  IoU: 0.1731 | Accuracy: 0.9973
  Precision: 0.5234 | Recall: 0.2054
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.5591 | Accuracy: 0.9713
  Precision: 0.9190 | Recall: 0.5880
  Cls Acc: 0.0000 | Samples: 81
GRED:
  IoU: 0.6494 | Accuracy: 0.9080
  Precision: 0.6900 | Recall: 0.9169
  Cls Acc: 1.0000 | Samples: 63
New best model saved with mIoU: 0.4605

Epoch 5/100
Train Loss: 0.4477 | Val Loss: 0.5009
Val mIoU: 0.4853 | Val Accuracy: 0.9631
Val Precision: 0.6012 | Val Recall: 0.6218
Val Cls Acc: 0.8807

Class-wise Metrics:
Healthy:
  IoU: 0.1118 | Accuracy: 0.9964
  Precision: 0.2601 | Recall: 0.1640
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.6715 | Accuracy: 0.9765
  Precision: 0.8318 | Recall: 0.7770
  Cls Acc: 0.6420 | Samples: 81
GRED:
  IoU: 0.6726 | Accuracy: 0.9163
  Precision: 0.7118 | Recall: 0.9244
  Cls Acc: 1.0000 | Samples: 63
New best model saved with mIoU: 0.4853

Epoch 6/100
Train Loss: 0.3872 | Val Loss: 0.4654
Val mIoU: 0.4501 | Val Accuracy: 0.9679
Val Precision: 0.5569 | Val Recall: 0.5278
Val Cls Acc: 0.9053

Class-wise Metrics:
Healthy:
  IoU: 0.0000 | Accuracy: 0.9970
  Precision: 0.0000 | Recall: 0.0000
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.6491 | Accuracy: 0.9765
  Precision: 0.8967 | Recall: 0.7015
  Cls Acc: 0.7160 | Samples: 81
GRED:
  IoU: 0.7014 | Accuracy: 0.9302
  Precision: 0.7740 | Recall: 0.8819
  Cls Acc: 1.0000 | Samples: 63

Epoch 7/100
Train Loss: 0.3668 | Val Loss: 0.4349
Val mIoU: 0.5309 | Val Accuracy: 0.9695
Val Precision: 0.7294 | Val Recall: 0.6271
Val Cls Acc: 0.9001

Class-wise Metrics:
Healthy:
  IoU: 0.1857 | Accuracy: 0.9974
  Precision: 0.5426 | Recall: 0.2201
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.7077 | Accuracy: 0.9793
  Precision: 0.8492 | Recall: 0.8094
  Cls Acc: 0.7160 | Samples: 81
GRED:
  IoU: 0.6994 | Accuracy: 0.9319
  Precision: 0.7963 | Recall: 0.8517
  Cls Acc: 0.9841 | Samples: 63
New best model saved with mIoU: 0.5309

Epoch 8/100
Train Loss: 0.3143 | Val Loss: 0.4595
Val mIoU: 0.5293 | Val Accuracy: 0.9691
Val Precision: 0.6932 | Val Recall: 0.6449
Val Cls Acc: 0.9053

Class-wise Metrics:
Healthy:
  IoU: 0.2270 | Accuracy: 0.9967
  Precision: 0.3798 | Recall: 0.3607
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.6482 | Accuracy: 0.9768
  Precision: 0.9126 | Recall: 0.6911
  Cls Acc: 0.7160 | Samples: 81
GRED:
  IoU: 0.7128 | Accuracy: 0.9339
  Precision: 0.7872 | Recall: 0.8830
  Cls Acc: 1.0000 | Samples: 63

Epoch 9/100
Train Loss: 0.3167 | Val Loss: 0.4076
Val mIoU: 0.5298 | Val Accuracy: 0.9711
Val Precision: 0.6573 | Val Recall: 0.6361
Val Cls Acc: 0.9012

Class-wise Metrics:
Healthy:
  IoU: 0.1487 | Accuracy: 0.9963
  Precision: 0.2860 | Recall: 0.2364
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.7238 | Accuracy: 0.9812
  Precision: 0.8871 | Recall: 0.7973
  Cls Acc: 0.7037 | Samples: 81
GRED:
  IoU: 0.7168 | Accuracy: 0.9358
  Precision: 0.7989 | Recall: 0.8746
  Cls Acc: 1.0000 | Samples: 63

Epoch 10/100
Train Loss: 0.3026 | Val Loss: 0.4263
Val mIoU: 0.5191 | Val Accuracy: 0.9708
Val Precision: 0.6897 | Val Recall: 0.6015
Val Cls Acc: 0.9053

Class-wise Metrics:
Healthy:
  IoU: 0.1319 | Accuracy: 0.9969
  Precision: 0.3667 | Recall: 0.1708
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.7195 | Accuracy: 0.9810
  Precision: 0.8927 | Recall: 0.7876
  Cls Acc: 0.7160 | Samples: 81
GRED:
  IoU: 0.7058 | Accuracy: 0.9344
  Precision: 0.8099 | Recall: 0.8460
  Cls Acc: 1.0000 | Samples: 63

Epoch 11/100
Train Loss: 0.3416 | Val Loss: 0.4335
Val mIoU: 0.5119 | Val Accuracy: 0.9705
Val Precision: 0.8202 | Val Recall: 0.5699
Val Cls Acc: 0.9053

Class-wise Metrics:
Healthy:
  IoU: 0.1622 | Accuracy: 0.9976
  Precision: 0.7115 | Recall: 0.1736
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.6668 | Accuracy: 0.9782
  Precision: 0.9233 | Recall: 0.7059
  Cls Acc: 0.7160 | Samples: 81
GRED:
  IoU: 0.7066 | Accuracy: 0.9359
  Precision: 0.8259 | Recall: 0.8303
  Cls Acc: 1.0000 | Samples: 63

Epoch 12/100
Train Loss: 0.2827 | Val Loss: 0.3868
Val mIoU: 0.5244 | Val Accuracy: 0.9729
Val Precision: 0.7076 | Val Recall: 0.5976
Val Cls Acc: 0.9053

Class-wise Metrics:
Healthy:
  IoU: 0.1087 | Accuracy: 0.9971
  Precision: 0.3880 | Recall: 0.1312
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.7369 | Accuracy: 0.9827
  Precision: 0.9259 | Recall: 0.7831
  Cls Acc: 0.7160 | Samples: 81
GRED:
  IoU: 0.7276 | Accuracy: 0.9389
  Precision: 0.8090 | Recall: 0.8786
  Cls Acc: 1.0000 | Samples: 63

Epoch 13/100
Train Loss: 0.2554 | Val Loss: 0.3599
Val mIoU: 0.5557 | Val Accuracy: 0.9755
Val Precision: 0.7634 | Val Recall: 0.6353
Val Cls Acc: 0.9053

Class-wise Metrics:
Healthy:
  IoU: 0.1312 | Accuracy: 0.9974
  Precision: 0.5758 | Recall: 0.1452
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.7939 | Accuracy: 0.9858
  Precision: 0.8836 | Recall: 0.8867
  Cls Acc: 0.7160 | Samples: 81
GRED:
  IoU: 0.7419 | Accuracy: 0.9435
  Precision: 0.8307 | Recall: 0.8741
  Cls Acc: 1.0000 | Samples: 63
New best model saved with mIoU: 0.5557

Epoch 14/100
Train Loss: 0.2319 | Val Loss: 0.3494
Val mIoU: 0.5579 | Val Accuracy: 0.9747
Val Precision: 0.6397 | Val Recall: 0.6770
Val Cls Acc: 0.9053

Class-wise Metrics:
Healthy:
  IoU: 0.1305 | Accuracy: 0.9958
  Precision: 0.2303 | Recall: 0.2314
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.7996 | Accuracy: 0.9861
  Precision: 0.8795 | Recall: 0.8979
  Cls Acc: 0.7160 | Samples: 81
GRED:
  IoU: 0.7437 | Accuracy: 0.9422
  Precision: 0.8094 | Recall: 0.9016
  Cls Acc: 1.0000 | Samples: 63
New best model saved with mIoU: 0.5579

Epoch 15/100
Train Loss: 0.2216 | Val Loss: 0.3740
Val mIoU: 0.5554 | Val Accuracy: 0.9733
Val Precision: 0.6373 | Val Recall: 0.6834
Val Cls Acc: 0.9053

Class-wise Metrics:
Healthy:
  IoU: 0.1330 | Accuracy: 0.9963
  Precision: 0.2728 | Recall: 0.2061
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8010 | Accuracy: 0.9858
  Precision: 0.8550 | Recall: 0.9269
  Cls Acc: 0.7160 | Samples: 81
GRED:
  IoU: 0.7323 | Accuracy: 0.9377
  Precision: 0.7843 | Recall: 0.9171
  Cls Acc: 1.0000 | Samples: 63

Epoch 16/100
Train Loss: 0.1996 | Val Loss: 0.3350
Val mIoU: 0.5656 | Val Accuracy: 0.9754
Val Precision: 0.6893 | Val Recall: 0.6573
Val Cls Acc: 0.9053

Class-wise Metrics:
Healthy:
  IoU: 0.1410 | Accuracy: 0.9967
  Precision: 0.3339 | Recall: 0.1962
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8094 | Accuracy: 0.9875
  Precision: 0.9332 | Recall: 0.8591
  Cls Acc: 0.7160 | Samples: 81
GRED:
  IoU: 0.7463 | Accuracy: 0.9421
  Precision: 0.8008 | Recall: 0.9165
  Cls Acc: 1.0000 | Samples: 63
New best model saved with mIoU: 0.5656

Epoch 17/100
Train Loss: 0.2130 | Val Loss: 0.3390
Val mIoU: 0.5648 | Val Accuracy: 0.9764
Val Precision: 0.7035 | Val Recall: 0.6520
Val Cls Acc: 0.9053

Class-wise Metrics:
Healthy:
  IoU: 0.1350 | Accuracy: 0.9970
  Precision: 0.3865 | Recall: 0.1719
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8050 | Accuracy: 0.9868
  Precision: 0.9037 | Recall: 0.8806
  Cls Acc: 0.7160 | Samples: 81
GRED:
  IoU: 0.7542 | Accuracy: 0.9453
  Precision: 0.8202 | Recall: 0.9037
  Cls Acc: 1.0000 | Samples: 63

Epoch 18/100
Train Loss: 0.2231 | Val Loss: 0.3379
Val mIoU: 0.5530 | Val Accuracy: 0.9767
Val Precision: 0.7723 | Val Recall: 0.6153
Val Cls Acc: 0.9053

Class-wise Metrics:
Healthy:
  IoU: 0.1188 | Accuracy: 0.9973
  Precision: 0.5381 | Recall: 0.1322
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.7824 | Accuracy: 0.9859
  Precision: 0.9488 | Recall: 0.8168
  Cls Acc: 0.7160 | Samples: 81
GRED:
  IoU: 0.7577 | Accuracy: 0.9467
  Precision: 0.8301 | Recall: 0.8969
  Cls Acc: 1.0000 | Samples: 63

Epoch 19/100
Train Loss: 0.2167 | Val Loss: 0.3274
Val mIoU: 0.5694 | Val Accuracy: 0.9764
Val Precision: 0.6893 | Val Recall: 0.6590
Val Cls Acc: 0.9136

Class-wise Metrics:
Healthy:
  IoU: 0.1432 | Accuracy: 0.9967
  Precision: 0.3259 | Recall: 0.2036
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8118 | Accuracy: 0.9875
  Precision: 0.9198 | Recall: 0.8737
  Cls Acc: 0.7407 | Samples: 81
GRED:
  IoU: 0.7532 | Accuracy: 0.9452
  Precision: 0.8221 | Recall: 0.8998
  Cls Acc: 1.0000 | Samples: 63
New best model saved with mIoU: 0.5694

Epoch 20/100
Train Loss: 0.1992 | Val Loss: 0.3307
Val mIoU: 0.5676 | Val Accuracy: 0.9771
Val Precision: 0.7959 | Val Recall: 0.6410
Val Cls Acc: 0.9218

Class-wise Metrics:
Healthy:
  IoU: 0.1428 | Accuracy: 0.9975
  Precision: 0.6452 | Recall: 0.1550
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.7990 | Accuracy: 0.9866
  Precision: 0.9166 | Recall: 0.8616
  Cls Acc: 0.7654 | Samples: 81
GRED:
  IoU: 0.7611 | Accuracy: 0.9471
  Precision: 0.8260 | Recall: 0.9065
  Cls Acc: 1.0000 | Samples: 63

Epoch 21/100
Train Loss: 0.1923 | Val Loss: 0.3221
Val mIoU: 0.5772 | Val Accuracy: 0.9776
Val Precision: 0.8481 | Val Recall: 0.6445
Val Cls Acc: 0.9218

Class-wise Metrics:
Healthy:
  IoU: 0.1629 | Accuracy: 0.9976
  Precision: 0.7875 | Recall: 0.1704
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8060 | Accuracy: 0.9871
  Precision: 0.9217 | Recall: 0.8652
  Cls Acc: 0.7654 | Samples: 81
GRED:
  IoU: 0.7628 | Accuracy: 0.9481
  Precision: 0.8352 | Recall: 0.8980
  Cls Acc: 1.0000 | Samples: 63
New best model saved with mIoU: 0.5772

Epoch 22/100
Train Loss: 0.1955 | Val Loss: 0.3490
Val mIoU: 0.5498 | Val Accuracy: 0.9743
Val Precision: 0.6431 | Val Recall: 0.6628
Val Cls Acc: 0.9547

Class-wise Metrics:
Healthy:
  IoU: 0.1087 | Accuracy: 0.9966
  Precision: 0.2696 | Recall: 0.1540
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.7952 | Accuracy: 0.9857
  Precision: 0.8750 | Recall: 0.8970
  Cls Acc: 0.8642 | Samples: 81
GRED:
  IoU: 0.7455 | Accuracy: 0.9405
  Precision: 0.7847 | Recall: 0.9372
  Cls Acc: 1.0000 | Samples: 63

Epoch 23/100
Train Loss: 0.1948 | Val Loss: 0.3385
Val mIoU: 0.5653 | Val Accuracy: 0.9764
Val Precision: 0.7190 | Val Recall: 0.6614
Val Cls Acc: 0.9506

Class-wise Metrics:
Healthy:
  IoU: 0.1377 | Accuracy: 0.9972
  Precision: 0.4688 | Recall: 0.1632
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.7972 | Accuracy: 0.9858
  Precision: 0.8724 | Recall: 0.9024
  Cls Acc: 0.8519 | Samples: 81
GRED:
  IoU: 0.7609 | Accuracy: 0.9463
  Precision: 0.8159 | Recall: 0.9186
  Cls Acc: 1.0000 | Samples: 63

Epoch 24/100
Train Loss: 0.1831 | Val Loss: 0.3311
Val mIoU: 0.5767 | Val Accuracy: 0.9769
Val Precision: 0.7457 | Val Recall: 0.6672
Val Cls Acc: 0.9671

Class-wise Metrics:
Healthy:
  IoU: 0.1647 | Accuracy: 0.9973
  Precision: 0.5293 | Recall: 0.1930
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8040 | Accuracy: 0.9865
  Precision: 0.8854 | Recall: 0.8974
  Cls Acc: 0.9012 | Samples: 81
GRED:
  IoU: 0.7615 | Accuracy: 0.9469
  Precision: 0.8224 | Recall: 0.9114
  Cls Acc: 1.0000 | Samples: 63

Epoch 25/100
Train Loss: 0.1996 | Val Loss: 0.3143
Val mIoU: 0.5857 | Val Accuracy: 0.9774
Val Precision: 0.7772 | Val Recall: 0.6710
Val Cls Acc: 0.9465

Class-wise Metrics:
Healthy:
  IoU: 0.1824 | Accuracy: 0.9975
  Precision: 0.6110 | Recall: 0.2064
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8111 | Accuracy: 0.9871
  Precision: 0.8944 | Recall: 0.8970
  Cls Acc: 0.8395 | Samples: 81
GRED:
  IoU: 0.7636 | Accuracy: 0.9476
  Precision: 0.8262 | Recall: 0.9097
  Cls Acc: 1.0000 | Samples: 63
New best model saved with mIoU: 0.5857

Epoch 26/100
Train Loss: 0.1769 | Val Loss: 0.3142
Val mIoU: 0.5670 | Val Accuracy: 0.9779
Val Precision: 0.7414 | Val Recall: 0.6377
Val Cls Acc: 0.9547

Class-wise Metrics:
Healthy:
  IoU: 0.1124 | Accuracy: 0.9972
  Precision: 0.4617 | Recall: 0.1294
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8223 | Accuracy: 0.9883
  Precision: 0.9347 | Recall: 0.8724
  Cls Acc: 0.8642 | Samples: 81
GRED:
  IoU: 0.7661 | Accuracy: 0.9483
  Precision: 0.8279 | Recall: 0.9113
  Cls Acc: 1.0000 | Samples: 63

Epoch 27/100
Train Loss: 0.1919 | Val Loss: 0.3525
Val mIoU: 0.5423 | Val Accuracy: 0.9759
Val Precision: 0.9046 | Val Recall: 0.6197
Val Cls Acc: 0.9465

Class-wise Metrics:
Healthy:
  IoU: 0.0892 | Accuracy: 0.9975
  Precision: 1.0000 | Recall: 0.0892
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.7826 | Accuracy: 0.9853
  Precision: 0.8997 | Recall: 0.8574
  Cls Acc: 0.8395 | Samples: 81
GRED:
  IoU: 0.7550 | Accuracy: 0.9450
  Precision: 0.8140 | Recall: 0.9124
  Cls Acc: 1.0000 | Samples: 63

Epoch 28/100
Train Loss: 0.2087 | Val Loss: 0.3500
Val mIoU: 0.5588 | Val Accuracy: 0.9768
Val Precision: 0.6826 | Val Recall: 0.6385
Val Cls Acc: 0.9259

Class-wise Metrics:
Healthy:
  IoU: 0.1325 | Accuracy: 0.9961
  Precision: 0.2498 | Recall: 0.2202
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.7855 | Accuracy: 0.9861
  Precision: 0.9440 | Recall: 0.8238
  Cls Acc: 0.7778 | Samples: 81
GRED:
  IoU: 0.7584 | Accuracy: 0.9484
  Precision: 0.8539 | Recall: 0.8715
  Cls Acc: 1.0000 | Samples: 63

Epoch 29/100
Train Loss: 0.1997 | Val Loss: 0.3282
Val mIoU: 0.5606 | Val Accuracy: 0.9772
Val Precision: 0.6804 | Val Recall: 0.6447
Val Cls Acc: 0.9671

Class-wise Metrics:
Healthy:
  IoU: 0.1338 | Accuracy: 0.9962
  Precision: 0.2646 | Recall: 0.2131
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.7807 | Accuracy: 0.9856
  Precision: 0.9316 | Recall: 0.8282
  Cls Acc: 0.9012 | Samples: 81
GRED:
  IoU: 0.7674 | Accuracy: 0.9497
  Precision: 0.8452 | Recall: 0.8929
  Cls Acc: 1.0000 | Samples: 63

Epoch 30/100
Train Loss: 0.1787 | Val Loss: 0.3199
Val mIoU: 0.5541 | Val Accuracy: 0.9780
Val Precision: 0.6789 | Val Recall: 0.6248
Val Cls Acc: 0.9671

Class-wise Metrics:
Healthy:
  IoU: 0.0763 | Accuracy: 0.9968
  Precision: 0.2719 | Recall: 0.0958
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8161 | Accuracy: 0.9879
  Precision: 0.9321 | Recall: 0.8676
  Cls Acc: 0.9012 | Samples: 81
GRED:
  IoU: 0.7699 | Accuracy: 0.9494
  Precision: 0.8325 | Recall: 0.9111
  Cls Acc: 1.0000 | Samples: 63

Epoch 31/100
Train Loss: 0.1721 | Val Loss: 0.3108
Val mIoU: 0.5967 | Val Accuracy: 0.9787
Val Precision: 0.7311 | Val Recall: 0.6770
Val Cls Acc: 0.9753

Class-wise Metrics:
Healthy:
  IoU: 0.1760 | Accuracy: 0.9970
  Precision: 0.4159 | Recall: 0.2337
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8463 | Accuracy: 0.9900
  Precision: 0.9412 | Recall: 0.8936
  Cls Acc: 0.9259 | Samples: 81
GRED:
  IoU: 0.7678 | Accuracy: 0.9492
  Precision: 0.8362 | Recall: 0.9038
  Cls Acc: 1.0000 | Samples: 63
New best model saved with mIoU: 0.5967

Epoch 32/100
Train Loss: 0.1685 | Val Loss: 0.3097
Val mIoU: 0.6439 | Val Accuracy: 0.9795
Val Precision: 0.7887 | Val Recall: 0.7321
Val Cls Acc: 0.9753

Class-wise Metrics:
Healthy:
  IoU: 0.3042 | Accuracy: 0.9976
  Precision: 0.5842 | Recall: 0.3883
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8555 | Accuracy: 0.9906
  Precision: 0.9428 | Recall: 0.9023
  Cls Acc: 0.9259 | Samples: 81
GRED:
  IoU: 0.7719 | Accuracy: 0.9502
  Precision: 0.8392 | Recall: 0.9058
  Cls Acc: 1.0000 | Samples: 63
New best model saved with mIoU: 0.6439

Epoch 33/100
Train Loss: 0.1719 | Val Loss: 0.3074
Val mIoU: 0.6280 | Val Accuracy: 0.9794
Val Precision: 0.7481 | Val Recall: 0.7271
Val Cls Acc: 0.9712

Class-wise Metrics:
Healthy:
  IoU: 0.2636 | Accuracy: 0.9971
  Precision: 0.4528 | Recall: 0.3867
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8431 | Accuracy: 0.9899
  Precision: 0.9543 | Recall: 0.8786
  Cls Acc: 0.9136 | Samples: 81
GRED:
  IoU: 0.7774 | Accuracy: 0.9512
  Precision: 0.8371 | Recall: 0.9160
  Cls Acc: 1.0000 | Samples: 63

Epoch 34/100
Train Loss: 0.1702 | Val Loss: 0.3039
Val mIoU: 0.6557 | Val Accuracy: 0.9795
Val Precision: 0.7559 | Val Recall: 0.7773
Val Cls Acc: 0.9753

Class-wise Metrics:
Healthy:
  IoU: 0.3344 | Accuracy: 0.9973
  Precision: 0.5065 | Recall: 0.4959
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8563 | Accuracy: 0.9905
  Precision: 0.9263 | Recall: 0.9188
  Cls Acc: 0.9259 | Samples: 81
GRED:
  IoU: 0.7763 | Accuracy: 0.9509
  Precision: 0.8348 | Recall: 0.9172
  Cls Acc: 1.0000 | Samples: 63
New best model saved with mIoU: 0.6557

Epoch 35/100
Train Loss: 0.1687 | Val Loss: 0.3131
Val mIoU: 0.6419 | Val Accuracy: 0.9789
Val Precision: 0.7674 | Val Recall: 0.7469
Val Cls Acc: 0.9753

Class-wise Metrics:
Healthy:
  IoU: 0.3206 | Accuracy: 0.9973
  Precision: 0.5097 | Recall: 0.4635
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8308 | Accuracy: 0.9891
  Precision: 0.9584 | Recall: 0.8619
  Cls Acc: 0.9259 | Samples: 81
GRED:
  IoU: 0.7742 | Accuracy: 0.9504
  Precision: 0.8339 | Recall: 0.9155
  Cls Acc: 1.0000 | Samples: 63

Epoch 36/100
Train Loss: 0.1592 | Val Loss: 0.3088
Val mIoU: 0.6617 | Val Accuracy: 0.9800
Val Precision: 0.8442 | Val Recall: 0.7355
Val Cls Acc: 0.9547

Class-wise Metrics:
Healthy:
  IoU: 0.3634 | Accuracy: 0.9980
  Precision: 0.7340 | Recall: 0.4185
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8427 | Accuracy: 0.9899
  Precision: 0.9544 | Recall: 0.8781
  Cls Acc: 0.8642 | Samples: 81
GRED:
  IoU: 0.7790 | Accuracy: 0.9520
  Precision: 0.8441 | Recall: 0.9099
  Cls Acc: 1.0000 | Samples: 63
New best model saved with mIoU: 0.6617

Epoch 37/100
Train Loss: 0.1602 | Val Loss: 0.3056
Val mIoU: 0.6524 | Val Accuracy: 0.9798
Val Precision: 0.9225 | Val Recall: 0.7146
Val Cls Acc: 0.9712

Class-wise Metrics:
Healthy:
  IoU: 0.3413 | Accuracy: 0.9982
  Precision: 0.9897 | Recall: 0.3425
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8352 | Accuracy: 0.9892
  Precision: 0.9376 | Recall: 0.8843
  Cls Acc: 0.9136 | Samples: 81
GRED:
  IoU: 0.7807 | Accuracy: 0.9521
  Precision: 0.8401 | Recall: 0.9170
  Cls Acc: 1.0000 | Samples: 63

Epoch 38/100
Train Loss: 0.1592 | Val Loss: 0.3114
Val mIoU: 0.6804 | Val Accuracy: 0.9795
Val Precision: 0.8502 | Val Recall: 0.7649
Val Cls Acc: 0.9753

Class-wise Metrics:
Healthy:
  IoU: 0.4332 | Accuracy: 0.9982
  Precision: 0.7848 | Recall: 0.4916
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8294 | Accuracy: 0.9887
  Precision: 0.9278 | Recall: 0.8866
  Cls Acc: 0.9259 | Samples: 81
GRED:
  IoU: 0.7786 | Accuracy: 0.9515
  Precision: 0.8381 | Recall: 0.9164
  Cls Acc: 1.0000 | Samples: 63
New best model saved with mIoU: 0.6804

Epoch 39/100
Train Loss: 0.1543 | Val Loss: 0.2995
Val mIoU: 0.6747 | Val Accuracy: 0.9796
Val Precision: 0.8250 | Val Recall: 0.7704
Val Cls Acc: 0.9712

Class-wise Metrics:
Healthy:
  IoU: 0.4093 | Accuracy: 0.9981
  Precision: 0.7255 | Recall: 0.4844
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8345 | Accuracy: 0.9889
  Precision: 0.9120 | Recall: 0.9077
  Cls Acc: 0.9136 | Samples: 81
GRED:
  IoU: 0.7802 | Accuracy: 0.9519
  Precision: 0.8376 | Recall: 0.9193
  Cls Acc: 1.0000 | Samples: 63

Epoch 40/100
Train Loss: 0.1557 | Val Loss: 0.3044
Val mIoU: 0.6818 | Val Accuracy: 0.9798
Val Precision: 0.8665 | Val Recall: 0.7603
Val Cls Acc: 0.9753

Class-wise Metrics:
Healthy:
  IoU: 0.4348 | Accuracy: 0.9983
  Precision: 0.8299 | Recall: 0.4774
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8283 | Accuracy: 0.9887
  Precision: 0.9315 | Recall: 0.8821
  Cls Acc: 0.9259 | Samples: 81
GRED:
  IoU: 0.7823 | Accuracy: 0.9523
  Precision: 0.8381 | Recall: 0.9215
  Cls Acc: 1.0000 | Samples: 63
New best model saved with mIoU: 0.6818

Epoch 41/100
Train Loss: 0.1508 | Val Loss: 0.2989
Val mIoU: 0.6564 | Val Accuracy: 0.9803
Val Precision: 0.7911 | Val Recall: 0.7514
Val Cls Acc: 0.9753

Class-wise Metrics:
Healthy:
  IoU: 0.3372 | Accuracy: 0.9976
  Precision: 0.5924 | Recall: 0.4392
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8483 | Accuracy: 0.9900
  Precision: 0.9333 | Recall: 0.9031
  Cls Acc: 0.9259 | Samples: 81
GRED:
  IoU: 0.7836 | Accuracy: 0.9532
  Precision: 0.8477 | Recall: 0.9121
  Cls Acc: 1.0000 | Samples: 63

Epoch 42/100
Train Loss: 0.1493 | Val Loss: 0.3012
Val mIoU: 0.6836 | Val Accuracy: 0.9803
Val Precision: 0.8359 | Val Recall: 0.7681
Val Cls Acc: 0.9671

Class-wise Metrics:
Healthy:
  IoU: 0.4300 | Accuracy: 0.9981
  Precision: 0.6934 | Recall: 0.5309
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8382 | Accuracy: 0.9897
  Precision: 0.9648 | Recall: 0.8647
  Cls Acc: 0.9012 | Samples: 81
GRED:
  IoU: 0.7827 | Accuracy: 0.9531
  Precision: 0.8494 | Recall: 0.9088
  Cls Acc: 1.0000 | Samples: 63
New best model saved with mIoU: 0.6836

Epoch 43/100
Train Loss: 0.1550 | Val Loss: 0.3072
Val mIoU: 0.6526 | Val Accuracy: 0.9801
Val Precision: 0.8036 | Val Recall: 0.7394
Val Cls Acc: 0.9753

Class-wise Metrics:
Healthy:
  IoU: 0.3252 | Accuracy: 0.9977
  Precision: 0.6288 | Recall: 0.4024
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8511 | Accuracy: 0.9903
  Precision: 0.9397 | Recall: 0.9003
  Cls Acc: 0.9259 | Samples: 81
GRED:
  IoU: 0.7816 | Accuracy: 0.9524
  Precision: 0.8424 | Recall: 0.9154
  Cls Acc: 1.0000 | Samples: 63

Epoch 44/100
Train Loss: 0.1497 | Val Loss: 0.3194
Val mIoU: 0.6511 | Val Accuracy: 0.9793
Val Precision: 0.8236 | Val Recall: 0.7379
Val Cls Acc: 0.9753

Class-wise Metrics:
Healthy:
  IoU: 0.3433 | Accuracy: 0.9979
  Precision: 0.7165 | Recall: 0.3973
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8329 | Accuracy: 0.9888
  Precision: 0.9178 | Recall: 0.9000
  Cls Acc: 0.9259 | Samples: 81
GRED:
  IoU: 0.7772 | Accuracy: 0.9511
  Precision: 0.8364 | Recall: 0.9165
  Cls Acc: 1.0000 | Samples: 63

Epoch 45/100
Train Loss: 0.1507 | Val Loss: 0.3116
Val mIoU: 0.6567 | Val Accuracy: 0.9804
Val Precision: 0.8623 | Val Recall: 0.7228
Val Cls Acc: 0.9671

Class-wise Metrics:
Healthy:
  IoU: 0.3477 | Accuracy: 0.9980
  Precision: 0.7847 | Recall: 0.3844
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8392 | Accuracy: 0.9896
  Precision: 0.9480 | Recall: 0.8797
  Cls Acc: 0.9012 | Samples: 81
GRED:
  IoU: 0.7832 | Accuracy: 0.9535
  Precision: 0.8540 | Recall: 0.9043
  Cls Acc: 1.0000 | Samples: 63

Epoch 46/100
Train Loss: 0.1447 | Val Loss: 0.2995
Val mIoU: 0.6610 | Val Accuracy: 0.9804
Val Precision: 0.8418 | Val Recall: 0.7342
Val Cls Acc: 0.9753

Class-wise Metrics:
Healthy:
  IoU: 0.3484 | Accuracy: 0.9980
  Precision: 0.7270 | Recall: 0.4009
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8528 | Accuracy: 0.9905
  Precision: 0.9515 | Recall: 0.8915
  Cls Acc: 0.9259 | Samples: 81
GRED:
  IoU: 0.7818 | Accuracy: 0.9528
  Precision: 0.8470 | Recall: 0.9104
  Cls Acc: 1.0000 | Samples: 63

Epoch 47/100
Train Loss: 0.1463 | Val Loss: 0.3060
Val mIoU: 0.6624 | Val Accuracy: 0.9803
Val Precision: 0.8371 | Val Recall: 0.7363
Val Cls Acc: 0.9753

Class-wise Metrics:
Healthy:
  IoU: 0.3666 | Accuracy: 0.9980
  Precision: 0.7044 | Recall: 0.4333
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8405 | Accuracy: 0.9896
  Precision: 0.9466 | Recall: 0.8822
  Cls Acc: 0.9259 | Samples: 81
GRED:
  IoU: 0.7800 | Accuracy: 0.9532
  Precision: 0.8601 | Recall: 0.8934
  Cls Acc: 1.0000 | Samples: 63

Epoch 48/100
Train Loss: 0.1480 | Val Loss: 0.2981
Val mIoU: 0.6550 | Val Accuracy: 0.9809
Val Precision: 0.8364 | Val Recall: 0.7279
Val Cls Acc: 0.9753

Class-wise Metrics:
Healthy:
  IoU: 0.3116 | Accuracy: 0.9979
  Precision: 0.7150 | Recall: 0.3558
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8672 | Accuracy: 0.9914
  Precision: 0.9487 | Recall: 0.9099
  Cls Acc: 0.9259 | Samples: 81
GRED:
  IoU: 0.7861 | Accuracy: 0.9536
  Precision: 0.8455 | Recall: 0.9180
  Cls Acc: 1.0000 | Samples: 63

Epoch 49/100
Train Loss: 0.1472 | Val Loss: 0.2998
Val mIoU: 0.6682 | Val Accuracy: 0.9801
Val Precision: 0.7975 | Val Recall: 0.7680
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3609 | Accuracy: 0.9978
  Precision: 0.6230 | Recall: 0.4618
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8629 | Accuracy: 0.9910
  Precision: 0.9374 | Recall: 0.9156
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7807 | Accuracy: 0.9516
  Precision: 0.8321 | Recall: 0.9267
  Cls Acc: 1.0000 | Samples: 63

Epoch 50/100
Train Loss: 0.1428 | Val Loss: 0.3012
Val mIoU: 0.6641 | Val Accuracy: 0.9802
Val Precision: 0.8379 | Val Recall: 0.7427
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3617 | Accuracy: 0.9980
  Precision: 0.7278 | Recall: 0.4183
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8491 | Accuracy: 0.9902
  Precision: 0.9429 | Recall: 0.8951
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7815 | Accuracy: 0.9525
  Precision: 0.8432 | Recall: 0.9145
  Cls Acc: 1.0000 | Samples: 63

Epoch 51/100
Train Loss: 0.1420 | Val Loss: 0.2959
Val mIoU: 0.6682 | Val Accuracy: 0.9803
Val Precision: 0.8115 | Val Recall: 0.7606
Val Cls Acc: 0.9877

Class-wise Metrics:
Healthy:
  IoU: 0.3671 | Accuracy: 0.9979
  Precision: 0.6593 | Recall: 0.4530
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8538 | Accuracy: 0.9904
  Precision: 0.9358 | Recall: 0.9069
  Cls Acc: 0.9630 | Samples: 81
GRED:
  IoU: 0.7838 | Accuracy: 0.9527
  Precision: 0.8395 | Recall: 0.9219
  Cls Acc: 1.0000 | Samples: 63

Epoch 52/100
Train Loss: 0.1436 | Val Loss: 0.3007
Val mIoU: 0.6603 | Val Accuracy: 0.9802
Val Precision: 0.8172 | Val Recall: 0.7456
Val Cls Acc: 0.9835

Class-wise Metrics:
Healthy:
  IoU: 0.3483 | Accuracy: 0.9979
  Precision: 0.6717 | Recall: 0.4198
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8515 | Accuracy: 0.9902
  Precision: 0.9357 | Recall: 0.9044
  Cls Acc: 0.9506 | Samples: 81
GRED:
  IoU: 0.7810 | Accuracy: 0.9524
  Precision: 0.8441 | Recall: 0.9127
  Cls Acc: 1.0000 | Samples: 63

Epoch 53/100
Train Loss: 0.1351 | Val Loss: 0.3036
Val mIoU: 0.6602 | Val Accuracy: 0.9804
Val Precision: 0.8027 | Val Recall: 0.7509
Val Cls Acc: 0.9712

Class-wise Metrics:
Healthy:
  IoU: 0.3453 | Accuracy: 0.9978
  Precision: 0.6260 | Recall: 0.4350
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8521 | Accuracy: 0.9903
  Precision: 0.9331 | Recall: 0.9076
  Cls Acc: 0.9136 | Samples: 81
GRED:
  IoU: 0.7833 | Accuracy: 0.9532
  Precision: 0.8491 | Recall: 0.9100
  Cls Acc: 1.0000 | Samples: 63

Epoch 54/100
Train Loss: 0.1470 | Val Loss: 0.3038
Val mIoU: 0.6651 | Val Accuracy: 0.9804
Val Precision: 0.8044 | Val Recall: 0.7596
Val Cls Acc: 0.9835

Class-wise Metrics:
Healthy:
  IoU: 0.3608 | Accuracy: 0.9978
  Precision: 0.6390 | Recall: 0.4531
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8494 | Accuracy: 0.9900
  Precision: 0.9302 | Recall: 0.9072
  Cls Acc: 0.9506 | Samples: 81
GRED:
  IoU: 0.7852 | Accuracy: 0.9533
  Precision: 0.8440 | Recall: 0.9186
  Cls Acc: 1.0000 | Samples: 63

Epoch 55/100
Train Loss: 0.1396 | Val Loss: 0.2990
Val mIoU: 0.6686 | Val Accuracy: 0.9806
Val Precision: 0.8177 | Val Recall: 0.7579
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3690 | Accuracy: 0.9979
  Precision: 0.6758 | Recall: 0.4484
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8498 | Accuracy: 0.9901
  Precision: 0.9314 | Recall: 0.9066
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7869 | Accuracy: 0.9537
  Precision: 0.8458 | Recall: 0.9187
  Cls Acc: 1.0000 | Samples: 63

Epoch 56/100
Train Loss: 0.1444 | Val Loss: 0.2985
Val mIoU: 0.6677 | Val Accuracy: 0.9806
Val Precision: 0.8273 | Val Recall: 0.7528
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3661 | Accuracy: 0.9980
  Precision: 0.7052 | Recall: 0.4323
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8505 | Accuracy: 0.9901
  Precision: 0.9313 | Recall: 0.9074
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7865 | Accuracy: 0.9536
  Precision: 0.8454 | Recall: 0.9187
  Cls Acc: 1.0000 | Samples: 63

Epoch 57/100
Train Loss: 0.1367 | Val Loss: 0.3002
Val mIoU: 0.6640 | Val Accuracy: 0.9805
Val Precision: 0.8165 | Val Recall: 0.7523
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3580 | Accuracy: 0.9979
  Precision: 0.6745 | Recall: 0.4328
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8472 | Accuracy: 0.9899
  Precision: 0.9297 | Recall: 0.9052
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7867 | Accuracy: 0.9537
  Precision: 0.8454 | Recall: 0.9188
  Cls Acc: 1.0000 | Samples: 63

Epoch 58/100
Train Loss: 0.1401 | Val Loss: 0.2996
Val mIoU: 0.6689 | Val Accuracy: 0.9807
Val Precision: 0.8262 | Val Recall: 0.7532
Val Cls Acc: 0.9753

Class-wise Metrics:
Healthy:
  IoU: 0.3698 | Accuracy: 0.9979
  Precision: 0.6949 | Recall: 0.4415
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8493 | Accuracy: 0.9901
  Precision: 0.9313 | Recall: 0.9061
  Cls Acc: 0.9259 | Samples: 81
GRED:
  IoU: 0.7875 | Accuracy: 0.9542
  Precision: 0.8523 | Recall: 0.9119
  Cls Acc: 1.0000 | Samples: 63

Epoch 59/100
Train Loss: 0.1351 | Val Loss: 0.3023
Val mIoU: 0.6700 | Val Accuracy: 0.9806
Val Precision: 0.8179 | Val Recall: 0.7607
Val Cls Acc: 0.9753

Class-wise Metrics:
Healthy:
  IoU: 0.3738 | Accuracy: 0.9979
  Precision: 0.6812 | Recall: 0.4530
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8491 | Accuracy: 0.9900
  Precision: 0.9252 | Recall: 0.9116
  Cls Acc: 0.9259 | Samples: 81
GRED:
  IoU: 0.7871 | Accuracy: 0.9539
  Precision: 0.8472 | Recall: 0.9173
  Cls Acc: 1.0000 | Samples: 63

Epoch 60/100
Train Loss: 0.1363 | Val Loss: 0.3041
Val mIoU: 0.6688 | Val Accuracy: 0.9806
Val Precision: 0.8198 | Val Recall: 0.7569
Val Cls Acc: 0.9753

Class-wise Metrics:
Healthy:
  IoU: 0.3727 | Accuracy: 0.9979
  Precision: 0.6794 | Recall: 0.4522
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8468 | Accuracy: 0.9899
  Precision: 0.9315 | Recall: 0.9031
  Cls Acc: 0.9259 | Samples: 81
GRED:
  IoU: 0.7868 | Accuracy: 0.9539
  Precision: 0.8484 | Recall: 0.9156
  Cls Acc: 1.0000 | Samples: 63

Epoch 61/100
Train Loss: 0.1388 | Val Loss: 0.3033
Val mIoU: 0.6680 | Val Accuracy: 0.9806
Val Precision: 0.8229 | Val Recall: 0.7552
Val Cls Acc: 0.9753

Class-wise Metrics:
Healthy:
  IoU: 0.3709 | Accuracy: 0.9979
  Precision: 0.6920 | Recall: 0.4442
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8461 | Accuracy: 0.9898
  Precision: 0.9286 | Recall: 0.9050
  Cls Acc: 0.9259 | Samples: 81
GRED:
  IoU: 0.7871 | Accuracy: 0.9539
  Precision: 0.8481 | Recall: 0.9164
  Cls Acc: 1.0000 | Samples: 63

Epoch 62/100
Train Loss: 0.1372 | Val Loss: 0.3005
Val mIoU: 0.6695 | Val Accuracy: 0.9807
Val Precision: 0.8275 | Val Recall: 0.7552
Val Cls Acc: 0.9753

Class-wise Metrics:
Healthy:
  IoU: 0.3736 | Accuracy: 0.9980
  Precision: 0.7055 | Recall: 0.4426
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8470 | Accuracy: 0.9899
  Precision: 0.9283 | Recall: 0.9064
  Cls Acc: 0.9259 | Samples: 81
GRED:
  IoU: 0.7879 | Accuracy: 0.9541
  Precision: 0.8487 | Recall: 0.9166
  Cls Acc: 1.0000 | Samples: 63

Epoch 63/100
Train Loss: 0.1320 | Val Loss: 0.3042
Val mIoU: 0.6695 | Val Accuracy: 0.9807
Val Precision: 0.8263 | Val Recall: 0.7553
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3751 | Accuracy: 0.9980
  Precision: 0.6997 | Recall: 0.4471
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8452 | Accuracy: 0.9898
  Precision: 0.9280 | Recall: 0.9045
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7883 | Accuracy: 0.9544
  Precision: 0.8512 | Recall: 0.9143
  Cls Acc: 1.0000 | Samples: 63

Epoch 64/100
Train Loss: 0.1335 | Val Loss: 0.3031
Val mIoU: 0.6695 | Val Accuracy: 0.9806
Val Precision: 0.8198 | Val Recall: 0.7586
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3740 | Accuracy: 0.9979
  Precision: 0.6835 | Recall: 0.4524
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8461 | Accuracy: 0.9898
  Precision: 0.9270 | Recall: 0.9064
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7883 | Accuracy: 0.9542
  Precision: 0.8488 | Recall: 0.9171
  Cls Acc: 1.0000 | Samples: 63

Epoch 65/100
Train Loss: 0.1338 | Val Loss: 0.3017
Val mIoU: 0.6698 | Val Accuracy: 0.9806
Val Precision: 0.8152 | Val Recall: 0.7621
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3751 | Accuracy: 0.9979
  Precision: 0.6735 | Recall: 0.4585
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8463 | Accuracy: 0.9898
  Precision: 0.9266 | Recall: 0.9071
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7881 | Accuracy: 0.9540
  Precision: 0.8454 | Recall: 0.9208
  Cls Acc: 1.0000 | Samples: 63

Epoch 66/100
Train Loss: 0.1383 | Val Loss: 0.3094
Val mIoU: 0.6709 | Val Accuracy: 0.9805
Val Precision: 0.8140 | Val Recall: 0.7646
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3791 | Accuracy: 0.9979
  Precision: 0.6716 | Recall: 0.4654
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8455 | Accuracy: 0.9897
  Precision: 0.9248 | Recall: 0.9079
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7880 | Accuracy: 0.9540
  Precision: 0.8455 | Recall: 0.9206
  Cls Acc: 1.0000 | Samples: 63

Epoch 67/100
Train Loss: 0.1388 | Val Loss: 0.3016
Val mIoU: 0.6707 | Val Accuracy: 0.9806
Val Precision: 0.8197 | Val Recall: 0.7610
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3770 | Accuracy: 0.9979
  Precision: 0.6859 | Recall: 0.4556
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8471 | Accuracy: 0.9899
  Precision: 0.9279 | Recall: 0.9068
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7880 | Accuracy: 0.9539
  Precision: 0.8454 | Recall: 0.9206
  Cls Acc: 1.0000 | Samples: 63

Epoch 68/100
Train Loss: 0.1376 | Val Loss: 0.3029
Val mIoU: 0.6706 | Val Accuracy: 0.9807
Val Precision: 0.8217 | Val Recall: 0.7590
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3762 | Accuracy: 0.9979
  Precision: 0.6874 | Recall: 0.4538
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8475 | Accuracy: 0.9899
  Precision: 0.9292 | Recall: 0.9060
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7883 | Accuracy: 0.9542
  Precision: 0.8485 | Recall: 0.9173
  Cls Acc: 1.0000 | Samples: 63

Epoch 69/100
Train Loss: 0.1419 | Val Loss: 0.2994
Val mIoU: 0.6698 | Val Accuracy: 0.9807
Val Precision: 0.8284 | Val Recall: 0.7545
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3732 | Accuracy: 0.9980
  Precision: 0.7044 | Recall: 0.4425
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8477 | Accuracy: 0.9900
  Precision: 0.9319 | Recall: 0.9036
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7886 | Accuracy: 0.9543
  Precision: 0.8491 | Recall: 0.9172
  Cls Acc: 1.0000 | Samples: 63

Epoch 70/100
Train Loss: 0.1376 | Val Loss: 0.3058
Val mIoU: 0.6708 | Val Accuracy: 0.9807
Val Precision: 0.8187 | Val Recall: 0.7613
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3775 | Accuracy: 0.9979
  Precision: 0.6810 | Recall: 0.4586
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8463 | Accuracy: 0.9898
  Precision: 0.9273 | Recall: 0.9064
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7886 | Accuracy: 0.9542
  Precision: 0.8477 | Recall: 0.9188
  Cls Acc: 1.0000 | Samples: 63

Epoch 71/100
Train Loss: 0.1379 | Val Loss: 0.2989
Val mIoU: 0.6711 | Val Accuracy: 0.9807
Val Precision: 0.8211 | Val Recall: 0.7597
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3775 | Accuracy: 0.9979
  Precision: 0.6844 | Recall: 0.4570
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8478 | Accuracy: 0.9899
  Precision: 0.9300 | Recall: 0.9055
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7881 | Accuracy: 0.9542
  Precision: 0.8490 | Recall: 0.9166
  Cls Acc: 1.0000 | Samples: 63

Epoch 72/100
Train Loss: 0.1378 | Val Loss: 0.2980
Val mIoU: 0.6702 | Val Accuracy: 0.9807
Val Precision: 0.8241 | Val Recall: 0.7571
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3749 | Accuracy: 0.9980
  Precision: 0.6930 | Recall: 0.4496
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8473 | Accuracy: 0.9899
  Precision: 0.9304 | Recall: 0.9047
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7884 | Accuracy: 0.9542
  Precision: 0.8489 | Recall: 0.9171
  Cls Acc: 1.0000 | Samples: 63

Epoch 73/100
Train Loss: 0.1486 | Val Loss: 0.3019
Val mIoU: 0.6698 | Val Accuracy: 0.9807
Val Precision: 0.8156 | Val Recall: 0.7601
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3740 | Accuracy: 0.9979
  Precision: 0.6660 | Recall: 0.4604
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8467 | Accuracy: 0.9899
  Precision: 0.9312 | Recall: 0.9033
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7887 | Accuracy: 0.9543
  Precision: 0.8497 | Recall: 0.9166
  Cls Acc: 1.0000 | Samples: 63

Epoch 74/100
Train Loss: 0.1369 | Val Loss: 0.3040
Val mIoU: 0.6682 | Val Accuracy: 0.9806
Val Precision: 0.8160 | Val Recall: 0.7581
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3706 | Accuracy: 0.9979
  Precision: 0.6699 | Recall: 0.4533
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8459 | Accuracy: 0.9898
  Precision: 0.9296 | Recall: 0.9037
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7881 | Accuracy: 0.9542
  Precision: 0.8486 | Recall: 0.9171
  Cls Acc: 1.0000 | Samples: 63

Epoch 75/100
Train Loss: 0.1384 | Val Loss: 0.3007
Val mIoU: 0.6677 | Val Accuracy: 0.9807
Val Precision: 0.8174 | Val Recall: 0.7556
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3667 | Accuracy: 0.9979
  Precision: 0.6694 | Recall: 0.4478
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8479 | Accuracy: 0.9900
  Precision: 0.9336 | Recall: 0.9024
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7884 | Accuracy: 0.9543
  Precision: 0.8492 | Recall: 0.9167
  Cls Acc: 1.0000 | Samples: 63

Epoch 76/100
Train Loss: 0.1414 | Val Loss: 0.3029
Val mIoU: 0.6684 | Val Accuracy: 0.9805
Val Precision: 0.8028 | Val Recall: 0.7665
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3692 | Accuracy: 0.9978
  Precision: 0.6394 | Recall: 0.4663
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8482 | Accuracy: 0.9899
  Precision: 0.9262 | Recall: 0.9097
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7877 | Accuracy: 0.9537
  Precision: 0.8427 | Recall: 0.9234
  Cls Acc: 1.0000 | Samples: 63

Epoch 77/100
Train Loss: 0.1335 | Val Loss: 0.2983
Val mIoU: 0.6674 | Val Accuracy: 0.9807
Val Precision: 0.8123 | Val Recall: 0.7583
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3638 | Accuracy: 0.9979
  Precision: 0.6590 | Recall: 0.4482
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8497 | Accuracy: 0.9901
  Precision: 0.9307 | Recall: 0.9070
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7888 | Accuracy: 0.9542
  Precision: 0.8471 | Recall: 0.9198
  Cls Acc: 1.0000 | Samples: 63

Epoch 78/100
Train Loss: 0.1331 | Val Loss: 0.2965
Val mIoU: 0.6679 | Val Accuracy: 0.9808
Val Precision: 0.8153 | Val Recall: 0.7558
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3638 | Accuracy: 0.9979
  Precision: 0.6599 | Recall: 0.4477
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8510 | Accuracy: 0.9902
  Precision: 0.9348 | Recall: 0.9048
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7889 | Accuracy: 0.9545
  Precision: 0.8513 | Recall: 0.9150
  Cls Acc: 1.0000 | Samples: 63

Epoch 79/100
Train Loss: 0.1373 | Val Loss: 0.2986
Val mIoU: 0.6667 | Val Accuracy: 0.9807
Val Precision: 0.8121 | Val Recall: 0.7574
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3632 | Accuracy: 0.9979
  Precision: 0.6571 | Recall: 0.4481
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8481 | Accuracy: 0.9900
  Precision: 0.9320 | Recall: 0.9041
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7889 | Accuracy: 0.9542
  Precision: 0.8471 | Recall: 0.9199
  Cls Acc: 1.0000 | Samples: 63

Epoch 80/100
Train Loss: 0.1339 | Val Loss: 0.3005
Val mIoU: 0.6657 | Val Accuracy: 0.9807
Val Precision: 0.8127 | Val Recall: 0.7552
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3601 | Accuracy: 0.9979
  Precision: 0.6578 | Recall: 0.4431
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8484 | Accuracy: 0.9900
  Precision: 0.9314 | Recall: 0.9050
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7887 | Accuracy: 0.9543
  Precision: 0.8489 | Recall: 0.9175
  Cls Acc: 1.0000 | Samples: 63

Epoch 81/100
Train Loss: 0.1422 | Val Loss: 0.2997
Val mIoU: 0.6672 | Val Accuracy: 0.9807
Val Precision: 0.8104 | Val Recall: 0.7590
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3648 | Accuracy: 0.9979
  Precision: 0.6534 | Recall: 0.4523
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8478 | Accuracy: 0.9899
  Precision: 0.9300 | Recall: 0.9056
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7890 | Accuracy: 0.9543
  Precision: 0.8478 | Recall: 0.9192
  Cls Acc: 1.0000 | Samples: 63

Epoch 82/100
Train Loss: 0.1386 | Val Loss: 0.2998
Val mIoU: 0.6677 | Val Accuracy: 0.9808
Val Precision: 0.8101 | Val Recall: 0.7597
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3662 | Accuracy: 0.9979
  Precision: 0.6513 | Recall: 0.4555
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8474 | Accuracy: 0.9899
  Precision: 0.9288 | Recall: 0.9062
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7895 | Accuracy: 0.9545
  Precision: 0.8500 | Recall: 0.9173
  Cls Acc: 1.0000 | Samples: 63

Epoch 83/100
Train Loss: 0.1393 | Val Loss: 0.2968
Val mIoU: 0.6676 | Val Accuracy: 0.9806
Val Precision: 0.8032 | Val Recall: 0.7645
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3670 | Accuracy: 0.9978
  Precision: 0.6378 | Recall: 0.4636
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8471 | Accuracy: 0.9899
  Precision: 0.9262 | Recall: 0.9085
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7886 | Accuracy: 0.9541
  Precision: 0.8456 | Recall: 0.9213
  Cls Acc: 1.0000 | Samples: 63

Epoch 84/100
Train Loss: 0.1406 | Val Loss: 0.2997
Val mIoU: 0.6687 | Val Accuracy: 0.9808
Val Precision: 0.8087 | Val Recall: 0.7617
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3692 | Accuracy: 0.9978
  Precision: 0.6464 | Recall: 0.4627
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8476 | Accuracy: 0.9899
  Precision: 0.9291 | Recall: 0.9062
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7893 | Accuracy: 0.9545
  Precision: 0.8507 | Recall: 0.9163
  Cls Acc: 1.0000 | Samples: 63

Epoch 85/100
Train Loss: 0.1376 | Val Loss: 0.3012
Val mIoU: 0.6691 | Val Accuracy: 0.9808
Val Precision: 0.8115 | Val Recall: 0.7607
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3693 | Accuracy: 0.9979
  Precision: 0.6539 | Recall: 0.4590
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8488 | Accuracy: 0.9900
  Precision: 0.9305 | Recall: 0.9063
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7893 | Accuracy: 0.9545
  Precision: 0.8502 | Recall: 0.9169
  Cls Acc: 1.0000 | Samples: 63

Epoch 86/100
Train Loss: 0.1337 | Val Loss: 0.3011
Val mIoU: 0.6691 | Val Accuracy: 0.9808
Val Precision: 0.8094 | Val Recall: 0.7620
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3698 | Accuracy: 0.9978
  Precision: 0.6483 | Recall: 0.4626
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8477 | Accuracy: 0.9900
  Precision: 0.9315 | Recall: 0.9041
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7898 | Accuracy: 0.9545
  Precision: 0.8486 | Recall: 0.9194
  Cls Acc: 1.0000 | Samples: 63

Epoch 87/100
Train Loss: 0.1302 | Val Loss: 0.3006
Val mIoU: 0.6694 | Val Accuracy: 0.9807
Val Precision: 0.8081 | Val Recall: 0.7639
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3703 | Accuracy: 0.9979
  Precision: 0.6488 | Recall: 0.4632
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8484 | Accuracy: 0.9900
  Precision: 0.9286 | Recall: 0.9075
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7894 | Accuracy: 0.9543
  Precision: 0.8469 | Recall: 0.9209
  Cls Acc: 1.0000 | Samples: 63

Epoch 88/100
Train Loss: 0.1333 | Val Loss: 0.2977
Val mIoU: 0.6686 | Val Accuracy: 0.9809
Val Precision: 0.8143 | Val Recall: 0.7578
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3660 | Accuracy: 0.9979
  Precision: 0.6590 | Recall: 0.4516
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8498 | Accuracy: 0.9901
  Precision: 0.9327 | Recall: 0.9054
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7898 | Accuracy: 0.9547
  Precision: 0.8511 | Recall: 0.9165
  Cls Acc: 1.0000 | Samples: 63

Epoch 89/100
Train Loss: 0.1378 | Val Loss: 0.3043
Val mIoU: 0.6705 | Val Accuracy: 0.9808
Val Precision: 0.8128 | Val Recall: 0.7623
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3729 | Accuracy: 0.9979
  Precision: 0.6592 | Recall: 0.4620
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8494 | Accuracy: 0.9900
  Precision: 0.9298 | Recall: 0.9076
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7890 | Accuracy: 0.9544
  Precision: 0.8493 | Recall: 0.9174
  Cls Acc: 1.0000 | Samples: 63

Epoch 90/100
Train Loss: 0.1391 | Val Loss: 0.2988
Val mIoU: 0.6698 | Val Accuracy: 0.9808
Val Precision: 0.8251 | Val Recall: 0.7557
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3696 | Accuracy: 0.9979
  Precision: 0.6952 | Recall: 0.4410
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8505 | Accuracy: 0.9901
  Precision: 0.9312 | Recall: 0.9075
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7895 | Accuracy: 0.9544
  Precision: 0.8488 | Recall: 0.9186
  Cls Acc: 1.0000 | Samples: 63

Epoch 91/100
Train Loss: 0.1339 | Val Loss: 0.3029
Val mIoU: 0.6704 | Val Accuracy: 0.9808
Val Precision: 0.8200 | Val Recall: 0.7590
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3726 | Accuracy: 0.9979
  Precision: 0.6815 | Recall: 0.4512
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8495 | Accuracy: 0.9900
  Precision: 0.9288 | Recall: 0.9087
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7890 | Accuracy: 0.9544
  Precision: 0.8496 | Recall: 0.9170
  Cls Acc: 1.0000 | Samples: 63

Epoch 92/100
Train Loss: 0.1382 | Val Loss: 0.3043
Val mIoU: 0.6700 | Val Accuracy: 0.9808
Val Precision: 0.8152 | Val Recall: 0.7607
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3713 | Accuracy: 0.9979
  Precision: 0.6680 | Recall: 0.4553
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8493 | Accuracy: 0.9900
  Precision: 0.9282 | Recall: 0.9091
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7893 | Accuracy: 0.9544
  Precision: 0.8494 | Recall: 0.9177
  Cls Acc: 1.0000 | Samples: 63

Epoch 93/100
Train Loss: 0.1349 | Val Loss: 0.2989
Val mIoU: 0.6698 | Val Accuracy: 0.9808
Val Precision: 0.8186 | Val Recall: 0.7594
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3701 | Accuracy: 0.9979
  Precision: 0.6804 | Recall: 0.4480
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8501 | Accuracy: 0.9901
  Precision: 0.9291 | Recall: 0.9091
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7892 | Accuracy: 0.9543
  Precision: 0.8464 | Recall: 0.9212
  Cls Acc: 1.0000 | Samples: 63

Epoch 94/100
Train Loss: 0.1342 | Val Loss: 0.2988
Val mIoU: 0.6690 | Val Accuracy: 0.9809
Val Precision: 0.8253 | Val Recall: 0.7535
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3672 | Accuracy: 0.9979
  Precision: 0.6920 | Recall: 0.4390
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8502 | Accuracy: 0.9901
  Precision: 0.9322 | Recall: 0.9063
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7895 | Accuracy: 0.9546
  Precision: 0.8518 | Recall: 0.9152
  Cls Acc: 1.0000 | Samples: 63

Epoch 95/100
Train Loss: 0.1398 | Val Loss: 0.3025
Val mIoU: 0.6693 | Val Accuracy: 0.9808
Val Precision: 0.8157 | Val Recall: 0.7591
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3693 | Accuracy: 0.9979
  Precision: 0.6677 | Recall: 0.4524
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8495 | Accuracy: 0.9901
  Precision: 0.9301 | Recall: 0.9075
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7890 | Accuracy: 0.9544
  Precision: 0.8495 | Recall: 0.9173
  Cls Acc: 1.0000 | Samples: 63

Epoch 96/100
Train Loss: 0.1375 | Val Loss: 0.2961
Val mIoU: 0.6700 | Val Accuracy: 0.9808
Val Precision: 0.8226 | Val Recall: 0.7566
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3695 | Accuracy: 0.9979
  Precision: 0.6858 | Recall: 0.4448
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8513 | Accuracy: 0.9902
  Precision: 0.9329 | Recall: 0.9069
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7892 | Accuracy: 0.9544
  Precision: 0.8490 | Recall: 0.9180
  Cls Acc: 1.0000 | Samples: 63

Epoch 97/100
Train Loss: 0.1309 | Val Loss: 0.3019
Val mIoU: 0.6706 | Val Accuracy: 0.9808
Val Precision: 0.8084 | Val Recall: 0.7651
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3709 | Accuracy: 0.9979
  Precision: 0.6491 | Recall: 0.4639
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8516 | Accuracy: 0.9902
  Precision: 0.9296 | Recall: 0.9103
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7892 | Accuracy: 0.9543
  Precision: 0.8465 | Recall: 0.9211
  Cls Acc: 1.0000 | Samples: 63

Epoch 98/100
Train Loss: 0.1362 | Val Loss: 0.3017
Val mIoU: 0.6695 | Val Accuracy: 0.9808
Val Precision: 0.8136 | Val Recall: 0.7608
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3683 | Accuracy: 0.9979
  Precision: 0.6638 | Recall: 0.4527
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8513 | Accuracy: 0.9902
  Precision: 0.9286 | Recall: 0.9109
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7890 | Accuracy: 0.9543
  Precision: 0.8482 | Recall: 0.9187
  Cls Acc: 1.0000 | Samples: 63

Epoch 99/100
Train Loss: 0.1354 | Val Loss: 0.3073
Val mIoU: 0.6706 | Val Accuracy: 0.9808
Val Precision: 0.8121 | Val Recall: 0.7633
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3709 | Accuracy: 0.9979
  Precision: 0.6602 | Recall: 0.4584
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8518 | Accuracy: 0.9902
  Precision: 0.9299 | Recall: 0.9102
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7890 | Accuracy: 0.9542
  Precision: 0.8461 | Recall: 0.9212
  Cls Acc: 1.0000 | Samples: 63

Epoch 100/100
Train Loss: 0.1391 | Val Loss: 0.2983
Val mIoU: 0.6697 | Val Accuracy: 0.9809
Val Precision: 0.8273 | Val Recall: 0.7532
Val Cls Acc: 0.9794

Class-wise Metrics:
Healthy:
  IoU: 0.3683 | Accuracy: 0.9979
  Precision: 0.6950 | Recall: 0.4393
  Cls Acc: 1.0000 | Samples: 6
Polyp:
  IoU: 0.8516 | Accuracy: 0.9903
  Precision: 0.9357 | Recall: 0.9045
  Cls Acc: 0.9383 | Samples: 81
GRED:
  IoU: 0.7893 | Accuracy: 0.9545
  Precision: 0.8510 | Recall: 0.9158
  Cls Acc: 1.0000 | Samples: 63'''
# Extract all training losses and add 0.2682 to each
train_losses = []
for line in data.strip().split('\n'):
    match = re.search(r'Train Loss: (\d+\.\d+)', line)
    if match:
        original_loss = float(match.group(1))
        adjusted_loss = original_loss
        train_losses.append(adjusted_loss)

# Create epochs list (1-50)
epochs = list(range(1, len(train_losses) + 1))

# Plot configuration
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_losses, 'r--o', linewidth=1.5, markersize=6, markerfacecolor='blue')
plt.title('Training Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.8)
plt.xticks(range(0, 100, 10))

plt.tight_layout()
plt.show()