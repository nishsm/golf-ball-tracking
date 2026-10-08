# Golf Ball Detection and Tracking

Detecting and tracking a golf ball through a swing in ordinary phone video. The ball is tiny, blurs badly once it's hit, and is often covered by the club head or the golfer's hands at address. An open-source project I built for analysing golf swings from phone video.

> **Status:** write-up first. Training and tracking code is being moved over from my training machine.
> <!-- TODO(Nish): add train script, SAM 2 tracker, configs, and a demo GIF from the laptop -->

## The interesting part: a 0.935 mAP model that didn't work

| | Validation | Real swing videos |
|---|---|---|
| YOLO11s, heavy augmentation, 300 epochs (~35 h) | P 0.999 · R 0.998 · mAP@50 0.995 · **mAP@50-95 0.935** | ball found in **under 10%** of frames |

The validation numbers looked almost perfect, but the model barely worked on real video. Two causes:

1. **Validation was augmented the same way as training**, so it was scoring a synthetic distribution. Real footage has natural motion blur, compression artifacts and a ball only a few pixels wide.
2. **Default inference settings** (conf 0.25, NMS IoU 0.45) were throwing away the weak, small detections that real frames produce.

What changed:

- **Clean retrain:** no augmentation, 416 px, 100 epochs with early stopping.
- **Memory-safe training on an 8 GB RTX 4060 laptop:** batch 2, workers 0, no caching, mixed precision (AMP), and a checkpoint every 10 epochs after repeated crashes.
- **Small-object inference:** conf 0.01, NMS IoU 0.6, and inference at 640 px on a model trained at 416 px.

## Hybrid tracker: YOLO for the stationary ball, SAM 2 once it moves

A single detector couldn't handle both phases of the shot, so the problem is split in two:

```
address / backswing (ball still, often occluded)      impact onward (ball moving, blurred)
        │                                                     │
occlusion-aware YOLO11 ──── detections seed ────►  SAM 2 video mask propagation
        ▲                                                     │
        └──── re-detect every 30 frames, or when SAM 2 confidence drops (fixes drift, recovers lost tracks)
```

- **Occlusion-aware detector.** I extended the dataset from 804 to 2,400+ images with synthetic occlusions: club head, hand silhouettes and shadows covering 10%, 25% or 50% of the ball. Occluded balls get separate classes and class weighting.
- **SAM 2 for motion.** YOLO detections start SAM 2 mask propagation, and bounding boxes are read off the masks. Periodic YOLO re-detection pulls the track back when it drifts.

## Lessons

- Validation metrics are not deployment performance. Evaluate on real target video.
- Only augment for variation that actually shows up in deployment.
- Default COCO thresholds are wrong for tiny objects.
- When one model can't do both jobs, split the problem.
- The conservative config I was forced into by GPU limits (small batch, no augmentation) generalized better.

Thanks to Oliver for the idea of splitting the problem into stationary and moving ball.

## Tech

Python, PyTorch, Ultralytics YOLO11, SAM 2, OpenCV.

See also: [golf-club-tracking](https://github.com/nishsm/golf-club-tracking), the club shaft and head tracker from the same project.
