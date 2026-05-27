- We should decrease the threshold for the phashes. I am seeing very similar images still.
- We could change the normalization from (0.5 0.5 0.5) to use our own stats, which is technically optimal. This is for the not pretrained model.
- Check dataloader. We may have to put workers=0 when running on the cloud.
- There is some leaking with the splits, maybe separate the train/val/test using different datasets (we have 3).
- Change the confusion matrix to be relative instead of absolute (since we have class imbalance).

- Inference time on personal computer (CPU)
- See if we can use real time inference (with the camera of the laptop, predict in real time).