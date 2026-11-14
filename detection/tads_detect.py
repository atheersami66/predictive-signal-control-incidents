#!/usr/bin/env python3
"""
Experiment A: incident (accident) detection on TADS with YOLO + ByteTrack.

Step 0  inspect   : print label-file columns and split-file structure (run once, then set --col-* args)
Step 1  track     : run YOLO + ByteTrack on every long video, cache tracks to tracks/<video>.npz
Step 2  evaluate  : tune the stopped-vehicle rule on the TRAIN split, report on the TEST split,
                    write tads_results.json (numbers that go into the paper's Table 2)

Example
  python tads_detect.py inspect  --labels label_ta_upload.xlsx --split train_test_split.pkl
  python tads_detect.py track    --frames rgb_frames --model yolo11s.pt
  python tads_detect.py evaluate --labels label_ta_upload.xlsx --split train_test_split.pkl --fps 25
"""
import argparse, glob, json, os, pickle, time, itertools
import numpy as np

VEHICLE_CLASSES = [2, 3, 5, 7]          # COCO: car, motorcycle, bus, truck
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# ------------------------------------------------------------------ step 0
def cmd_inspect(a):
    import pandas as pd
    df = pd.read_excel(a.labels)
    print("Label file columns:", list(df.columns))
    print(df.head(10).to_string())
    with open(a.split, "rb") as f:
        sp = pickle.load(f)
    print("\nSplit object type:", type(sp))
    if isinstance(sp, dict):
        for k, v in sp.items():
            print(" key", repr(k), type(v), (v[:5] if hasattr(v, "__getitem__") else v))
    elif isinstance(sp, (list, tuple)):
        for i, v in enumerate(sp[:4]):
            print(" item", i, type(v), (v[:5] if hasattr(v, "__getitem__") else v))
    print("\nSet --col-video/--col-start/--col-end (and --col-id if the split uses segment ids).")


# ------------------------------------------------------------------ step 1
def list_videos(frames_root):
    vids = sorted(d for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d)))
    return vids


def cmd_track(a):
    from ultralytics import YOLO
    os.makedirs(a.cache, exist_ok=True)
    timing = {}
    for vid in list_videos(a.frames):
        out = os.path.join(a.cache, f"{vid}.npz")
        if os.path.exists(out):
            print("skip", vid); continue
        model = YOLO(a.model)          # fresh model = fresh tracker state per video
        frames = sorted(f for f in os.listdir(os.path.join(a.frames, vid)) if f.lower().endswith(IMG_EXT))
        rows, t0 = [], time.time()
        for fi, fn in enumerate(frames):
            r = model.track(os.path.join(a.frames, vid, fn), persist=True, tracker="bytetrack.yaml",
                            classes=VEHICLE_CLASSES, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
            if r.boxes is None or r.boxes.id is None:
                continue
            ids = r.boxes.id.cpu().numpy().astype(int)
            xywh = r.boxes.xywh.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(ids)):
                rows.append((fi, ids[i], *xywh[i], cls[i]))
        dt = time.time() - t0
        timing[vid] = len(frames) / max(dt, 1e-6)
        arr = np.array(rows, dtype=np.float32).reshape(-1, 7)
        np.savez_compressed(out, tracks=arr, n_frames=len(frames))
        print(f"{vid}: {len(frames)} frames, {len(np.unique(arr[:,1])) if len(arr) else 0} tracks, {timing[vid]:.1f} FPS")
    json.dump(timing, open(os.path.join(a.cache, "fps.json"), "w"), indent=1)


# ------------------------------------------------------------------ step 2
def alarms_for_video(tracks, n_frames, fps, v_stop, t_stop, confirm, min_stopped):
    """Return list of alarm onset frames.
    A track is 'stopped' at frame f if its centre moved less than v_stop * (box height) per second
    over the last t_stop seconds (speed normalised by box size -> camera-height independent).
    A frame is flagged if >= min_stopped tracks are stopped; an alarm is confirmed after
    `confirm` consecutive flagged frames."""
    win = max(int(round(t_stop * fps)), 1)
    stopped_count = np.zeros(n_frames, dtype=np.int32)
    if len(tracks):
        for tid in np.unique(tracks[:, 1]):
            tr = tracks[tracks[:, 1] == tid]
            tr = tr[np.argsort(tr[:, 0])]
            f = tr[:, 0].astype(int); cx, cy, h = tr[:, 2], tr[:, 3], tr[:, 5]
            if len(f) < 2:
                continue
            j = 0
            for i in range(len(f)):
                while f[i] - f[j] > win:
                    j += 1
                if f[i] - f[j] < win * 0.8:      # not enough history yet
                    continue
                disp = np.hypot(cx[i] - cx[j], cy[i] - cy[j]) / max(h[i], 1.0)
                speed = disp / ((f[i] - f[j]) / fps)      # box-heights per second
                if speed < v_stop:
                    stopped_count[f[i]] += 1
    flag = stopped_count >= min_stopped
    onsets, run = [], 0
    for fi in range(n_frames):
        run = run + 1 if flag[fi] else 0
        if run == confirm:
            onsets.append(fi)
    return onsets


def score(segments, onsets, n_frames, fps, pre_tol_s=2.0, post_margin_s=10.0):
    """segments: list of (start, end) accident frames for one video."""
    pre, post = int(pre_tol_s * fps), int(post_margin_s * fps)
    hit, ttd = [], []
    used = set()
    for (s, e) in segments:
        cand = [o for o in onsets if s - pre <= o <= e]
        if cand:
            hit.append(1); ttd.append((cand[0] - s) / fps); used.update(o for o in onsets if s - pre <= o <= e + post)
        else:
            hit.append(0)
    fa = [o for o in onsets if o not in used and not any(s - pre <= o <= e + post for s, e in segments)]
    return hit, ttd, len(fa), n_frames / fps / 3600.0


def load_segments(a):
    import pandas as pd
    df = pd.read_excel(a.labels)
    with open(a.split, "rb") as f:
        sp = pickle.load(f)
    if isinstance(sp, dict):
        tr_key = next(k for k in sp if "train" in str(k).lower())
        te_key = next(k for k in sp if "test" in str(k).lower())
        train_ids, test_ids = set(map(str, sp[tr_key])), set(map(str, sp[te_key]))
    else:
        train_ids, test_ids = set(map(str, sp[0])), set(map(str, sp[1]))
    seg = {"train": {}, "test": {}}
    for _, r in df.iterrows():
        sid = str(r[a.col_id]) if a.col_id else str(_)
        part = "train" if sid in train_ids else "test" if sid in test_ids else None
        if part is None:
            continue
        seg[part].setdefault(str(r[a.col_video]), []).append((int(r[a.col_start]), int(r[a.col_end])))
    return seg


def evaluate(seg_part, cache, fps, p):
    hits, ttds, fa, hours = [], [], 0, 0.0
    for vid, segs in seg_part.items():
        z = np.load(os.path.join(cache, f"{vid}.npz"))
        on = alarms_for_video(z["tracks"], int(z["n_frames"]), fps, **p)
        h, t, f, hr = score(segs, on, int(z["n_frames"]), fps)
        hits += h; ttds += t; fa += f; hours += hr
    tp = sum(hits); fn = len(hits) - tp
    prec = tp / max(tp + fa, 1); rec = tp / max(len(hits), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(precision=prec, recall=rec, f1=f1, false_alarms_per_hour=fa / max(hours, 1e-9),
                ttd_mean_s=float(np.mean(ttds)) if ttds else None,
                ttd_median_s=float(np.median(ttds)) if ttds else None,
                ttd_p90_s=float(np.percentile(ttds, 90)) if ttds else None,
                ttd_all_s=ttds, n_segments=len(hits), miss_rate=fn / max(len(hits), 1))


def cmd_evaluate(a):
    seg = load_segments(a)
    grid = dict(v_stop=[0.05, 0.1, 0.2], t_stop=[2.0, 4.0, 6.0], confirm=[5, 15, 30], min_stopped=[1, 2])
    best, best_p = -1, None
    for vals in itertools.product(*grid.values()):
        p = dict(zip(grid.keys(), vals))
        r = evaluate(seg["train"], a.cache, a.fps, p)
        if r["f1"] > best:
            best, best_p = r["f1"], p
        print(p, f"F1={r['f1']:.3f}")
    test = evaluate(seg["test"], a.cache, a.fps, best_p)
    fps_file = os.path.join(a.cache, "fps.json")
    test["inference_fps"] = float(np.mean(list(json.load(open(fps_file)).values()))) if os.path.exists(fps_file) else None
    out = dict(best_params=best_p, train_f1=best, test=test, model=a.model)
    json.dump(out, open("tads_results.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in test.items() if k != "ttd_all_s"}, indent=1))
    print("Saved tads_results.json  ->  send this file back to update the paper.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inspect", "track", "evaluate"])
    ap.add_argument("--frames", default="rgb_frames")
    ap.add_argument("--labels", default="label_ta_upload.xlsx")
    ap.add_argument("--split", default="train_test_split.pkl")
    ap.add_argument("--cache", default="tracks")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--fps", type=float, default=25.0, help="frame rate of TADS videos (check the paper/README)")
    ap.add_argument("--col-video", dest="col_video", default="video")
    ap.add_argument("--col-start", dest="col_start", default="start")
    ap.add_argument("--col-end", dest="col_end", default="end")
    ap.add_argument("--col-id", dest="col_id", default=None)
    a = ap.parse_args()
    {"inspect": cmd_inspect, "track": cmd_track, "evaluate": cmd_evaluate}[a.cmd](a)
