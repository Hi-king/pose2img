# pose2img

# With reference images (ver. 2018)

## Training

```
## openpose
./build/examples/openpose/openpose.bin -no_display -disable_blending --image_dir /mnt/dataset/Market-1501/Market-1501-v15.09.15/gt_bbox --write_images /mnt/dataset/Market-1501/Market-1501-v15.09.15/gt_bbox_openpose
```

## References

* Pose Guided Person Image Generation
    * https://arxiv.org/abs/1705.09368

# Without Condition (ver. 2017)

![](https://media.githubusercontent.com/media/Hi-king/pose2img/master/sample/images/movie_result.gif)

```
python predict.py \
  sample/images/sample_pose.png \
  --outfile predicted.png \
  --encoder sample/models/encoder_model_13402000.npz \
  --decoder sample/models/decoder_model_13402000.npz
```

## References

* Image-to-Image Translation with Conditional Adversarial Networks
    * https://github.com/phillipi/pix2pix
