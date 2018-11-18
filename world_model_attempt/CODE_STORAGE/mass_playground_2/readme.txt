vae model train,
rnn model train,
then vae + rnn model train

rnn is double whammy
so far error is haiwire

Epoch: 31, Next KL: 183.0704, Next Recon Loss: 0.0003, Pred Loss: 3.5797, Pred Recon Loss: 0.0003, MDN Loss: -2.6524
Epoch: 32, Next KL: 184.1495, Next Recon Loss: 0.0003, Pred Loss: 2.6194, Pred Recon Loss: 0.0003, MDN Loss: -2.7569
Epoch: 33, Next KL: 185.4998, Next Recon Loss: 0.0003, Pred Loss: 2.1746, Pred Recon Loss: 0.0003, MDN Loss: -2.8433
Epoch: 34, Next KL: 186.7683, Next Recon Loss: 0.0003, Pred Loss: 2.1281, Pred Recon Loss: 0.0003, MDN Loss: -2.9287
Epoch: 35, Next KL: 176.3221, Next Recon Loss: 0.0003, Pred Loss: 47.7350, Pred Recon Loss: 0.0003, MDN Loss: -1.6497
Epoch: 36, Next KL: 160.6607, Next Recon Loss: 0.0003, Pred Loss: 7.1419, Pred Recon Loss: 0.0003, MDN Loss: -2.7344
Epoch: 37, Next KL: 163.3752, Next Recon Loss: 0.0003, Pred Loss: 3.5451, Pred Recon Loss: 0.0003, MDN Loss: -2.9384
Epoch: 38, Next KL: 169.5459, Next Recon Loss: 0.0003, Pred Loss: 2.4714, Pred Recon Loss: 0.0003, MDN Loss: -3.0376
Epoch: 39, Next KL: 174.7123, Next Recon Loss: 0.0003, Pred Loss: 1.8686, Pred Recon Loss: 0.0003, MDN Loss: -3.1222
Epoch: 40, Next KL: 178.7728, Next Recon Loss: 0.0003, Pred Loss: 1.0657, Pred Recon Loss: 0.0003, MDN Loss: -3.2376

kl is getting larger
mdn is negative

why? maybe joint training is really bad idea


log:

Epoch: 0, Next KL: 80.8509, Next Recon Loss: 0.0015, Pred Loss: 3636.3078, Pred Recon Loss: 0.0015, MDN Loss: 0.5458
Epoch: 1, Next KL: 114.4931, Next Recon Loss: 0.0005, Pred Loss: 200.6926, Pred Recon Loss: 0.0005, MDN Loss: 0.3597
Epoch: 2, Next KL: 119.3480, Next Recon Loss: 0.0004, Pred Loss: 49.9753, Pred Recon Loss: 0.0004, MDN Loss: 0.2640
Epoch: 3, Next KL: 133.3953, Next Recon Loss: 0.0003, Pred Loss: 15.2158, Pred Recon Loss: 0.0003, MDN Loss: 0.1723
Epoch: 4, Next KL: 145.8282, Next Recon Loss: 0.0003, Pred Loss: 6.0670, Pred Recon Loss: 0.0003, MDN Loss: 0.0813
Epoch: 5, Next KL: 154.6505, Next Recon Loss: 0.0003, Pred Loss: 3.3110, Pred Recon Loss: 0.0003, MDN Loss: -0.0110
Epoch: 6, Next KL: 161.0502, Next Recon Loss: 0.0003, Pred Loss: 2.2237, Pred Recon Loss: 0.0003, MDN Loss: -0.1066
Epoch: 7, Next KL: 166.0505, Next Recon Loss: 0.0003, Pred Loss: 1.7227, Pred Recon Loss: 0.0003, MDN Loss: -0.2071
Epoch: 8, Next KL: 170.2085, Next Recon Loss: 0.0003, Pred Loss: 1.3155, Pred Recon Loss: 0.0003, MDN Loss: -0.3135
Epoch: 9, Next KL: 173.7522, Next Recon Loss: 0.0003, Pred Loss: 1.1016, Pred Recon Loss: 0.0003, MDN Loss: -0.4260
Epoch: 10, Next KL: 176.9335, Next Recon Loss: 0.0003, Pred Loss: 0.9957, Pred Recon Loss: 0.0003, MDN Loss: -0.5440
Epoch: 11, Next KL: 179.8036, Next Recon Loss: 0.0003, Pred Loss: 0.8375, Pred Recon Loss: 0.0003, MDN Loss: -0.6661
Epoch: 12, Next KL: 182.3336, Next Recon Loss: 0.0003, Pred Loss: 0.7354, Pred Recon Loss: 0.0003, MDN Loss: -0.7905
Epoch: 13, Next KL: 184.6831, Next Recon Loss: 0.0003, Pred Loss: 0.6704, Pred Recon Loss: 0.0003, MDN Loss: -0.9154
Epoch: 14, Next KL: 186.8212, Next Recon Loss: 0.0003, Pred Loss: 0.6592, Pred Recon Loss: 0.0003, MDN Loss: -1.0397
Epoch: 15, Next KL: 188.8253, Next Recon Loss: 0.0003, Pred Loss: 0.7523, Pred Recon Loss: 0.0003, MDN Loss: -1.1626
Epoch: 16, Next KL: 189.9677, Next Recon Loss: 0.0003, Pred Loss: 6.7320, Pred Recon Loss: 0.0003, MDN Loss: -1.2747
Epoch: 17, Next KL: 188.7204, Next Recon Loss: 0.0003, Pred Loss: 2.3635, Pred Recon Loss: 0.0003, MDN Loss: -1.3986
Epoch: 18, Next KL: 188.4076, Next Recon Loss: 0.0003, Pred Loss: 10.6904, Pred Recon Loss: 0.0003, MDN Loss: -1.4948
Epoch: 19, Next KL: 184.7845, Next Recon Loss: 0.0003, Pred Loss: 3.9612, Pred Recon Loss: 0.0003, MDN Loss: -1.6197
Epoch: 20, Next KL: 183.4837, Next Recon Loss: 0.0003, Pred Loss: 6.4595, Pred Recon Loss: 0.0003, MDN Loss: -1.7167
Epoch: 21, Next KL: 183.1858, Next Recon Loss: 0.0003, Pred Loss: 1.7791, Pred Recon Loss: 0.0003, MDN Loss: -1.8438
Epoch: 22, Next KL: 184.9981, Next Recon Loss: 0.0003, Pred Loss: 1.0272, Pred Recon Loss: 0.0003, MDN Loss: -1.9570
Epoch: 23, Next KL: 178.6774, Next Recon Loss: 0.0003, Pred Loss: 101.8414, Pred Recon Loss: 0.0003, MDN Loss: -1.4582
Epoch: 24, Next KL: 142.0324, Next Recon Loss: 0.0003, Pred Loss: 32.1028, Pred Recon Loss: 0.0003, MDN Loss: -1.8684
Epoch: 25, Next KL: 144.1391, Next Recon Loss: 0.0003, Pred Loss: 10.2843, Pred Recon Loss: 0.0003, MDN Loss: -2.0728
Epoch: 26, Next KL: 157.7037, Next Recon Loss: 0.0003, Pred Loss: 4.7969, Pred Recon Loss: 0.0003, MDN Loss: -2.1883
Epoch: 27, Next KL: 166.7564, Next Recon Loss: 0.0003, Pred Loss: 3.5880, Pred Recon Loss: 0.0003, MDN Loss: -2.2876
Epoch: 28, Next KL: 172.6055, Next Recon Loss: 0.0003, Pred Loss: 1.0674, Pred Recon Loss: 0.0003, MDN Loss: -2.4096
Epoch: 29, Next KL: 177.2357, Next Recon Loss: 0.0003, Pred Loss: 1.1520, Pred Recon Loss: 0.0003, MDN Loss: -2.5095
Epoch: 30, Next KL: 180.7824, Next Recon Loss: 0.0003, Pred Loss: 1.2558, Pred Recon Loss: 0.0003, MDN Loss: -2.6093
Epoch: 31, Next KL: 183.0704, Next Recon Loss: 0.0003, Pred Loss: 3.5797, Pred Recon Loss: 0.0003, MDN Loss: -2.6524
Epoch: 32, Next KL: 184.1495, Next Recon Loss: 0.0003, Pred Loss: 2.6194, Pred Recon Loss: 0.0003, MDN Loss: -2.7569
Epoch: 33, Next KL: 185.4998, Next Recon Loss: 0.0003, Pred Loss: 2.1746, Pred Recon Loss: 0.0003, MDN Loss: -2.8433
Epoch: 34, Next KL: 186.7683, Next Recon Loss: 0.0003, Pred Loss: 2.1281, Pred Recon Loss: 0.0003, MDN Loss: -2.9287
Epoch: 35, Next KL: 176.3221, Next Recon Loss: 0.0003, Pred Loss: 47.7350, Pred Recon Loss: 0.0003, MDN Loss: -1.6497
Epoch: 36, Next KL: 160.6607, Next Recon Loss: 0.0003, Pred Loss: 7.1419, Pred Recon Loss: 0.0003, MDN Loss: -2.7344
Epoch: 37, Next KL: 163.3752, Next Recon Loss: 0.0003, Pred Loss: 3.5451, Pred Recon Loss: 0.0003, MDN Loss: -2.9384
Epoch: 38, Next KL: 169.5459, Next Recon Loss: 0.0003, Pred Loss: 2.4714, Pred Recon Loss: 0.0003, MDN Loss: -3.0376
Epoch: 39, Next KL: 174.7123, Next Recon Loss: 0.0003, Pred Loss: 1.8686, Pred Recon Loss: 0.0003, MDN Loss: -3.1222
Epoch: 40, Next KL: 178.7728, Next Recon Loss: 0.0003, Pred Loss: 1.0657, Pred Recon Loss: 0.0003, MDN Loss: -3.2376
