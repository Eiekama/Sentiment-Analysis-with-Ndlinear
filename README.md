# Sentiment-Analysis-with-Ndlinear
Will update this readme soon with proper documentation, as well as updating the git with the final miscellaneous housekeeping stuff.

# Results
Trained model for 10 epochs with batch size 128, 4 heads in attention, evaluate on 10000 reviews separated out as test set

Baseline:<br>
Accuracy: 0.7778<br>
TP: 3949.0<br>
FP: 1208.0<br>
FN: 1014.0<br>
TN: 3829.0<br>
Average runtime for block: 2.97067 ms<br>
Average memory usage for block: 487.47 MB<br>
Total runtime according to tqdm: 25min 59s, 155.97s per epoch

NdLinear variant:<br>
Accuracy: 0.8528<br>
TP: 4055<br>
FP: 564<br>
FN: 908<br>
TN: 4473<br>
Average runtime for block: 4.2829 ms<br>
Average memory usage for block: 491.46 MB<br>
Total runtime according to tqdm: 25min 59s, 155.97s per epoch

 ![loss comparison](./loss_plot.png)