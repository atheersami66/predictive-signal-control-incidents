# Experiment A — Accident detection on TADS (YOLO + ByteTrack)

## 1. Setup
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
A GPU (CUDA) is strongly recommended: ~260k frames. On CPU, use `--model yolo11n.pt --imgsz 640`.

## 2. Data layout
Download TADS (link + extraction code in the TADS GitHub README), then unzip:
```
rgb_frames/<video_name>/*.jpg     # 20 long videos, one folder each
label_ta_upload.xlsx
train_test_split.pkl
```

## 3. Run
```bash
# (a) see the real column names of the label file and the split structure
python tads_detect.py inspect

# (b) detect + track every video (slow, cached in tracks/, safe to resume)
python tads_detect.py track --model yolo11s.pt

# (c) tune on train split, report on test split -> tads_results.json
python tads_detect.py evaluate --fps 25 --col-video <col> --col-start <col> --col-end <col> --col-id <col>
```
Set `--col-*` to the column names printed in step (a). Check the real frame rate of TADS (`--fps`).
If the folder names in `rgb_frames/` differ from the video names in the Excel file, rename the folders to match.

## 4. Send back
Send **tads_results.json**. It fills Table 2 of the paper (precision, recall, F1, false alarms per hour,
time-to-detect, FPS), and its `ttd_all_s` list replaces the latency values used in the SUMO experiments.

## Method (as described in the paper)
- COCO-pretrained YOLO detects car / motorcycle / bus / truck; ByteTrack assigns IDs.
- A track is *stopped* if its centre moves less than `v_stop` box-heights per second over `t_stop` seconds.
- A frame is flagged if at least `min_stopped` tracks are stopped; an alarm is confirmed after `confirm` consecutive flagged frames.
- Parameters are grid-searched on the training split only.
- An accident segment counts as detected if an alarm falls in [start − 2 s, end]; alarms outside any segment (+10 s margin) are false alarms.
