#!/usr/bin/env bash
# Downloads MediaPipe pose models (full is committed; heavy is optional, more accurate, slower).
cd "$(dirname "$0")"
for m in lite full heavy; do
  [ -f pose_landmarker_$m.task ] || curl -fL -o pose_landmarker_$m.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_$m/float16/latest/pose_landmarker_$m.task
done
