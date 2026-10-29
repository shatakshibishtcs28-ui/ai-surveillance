"""
AI Surveillance & Threat Detection - Context-Aware Risk Scoring
Run: python surveillance.py            (webcam)
     python surveillance.py video.mp4  (recorded video)
Press 'q' to quit.
"""
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG (yahan badlo) ----------------
# ---- Camera config: config.json se aata hai (har camera ki jagah ek baar set) ----
import json


def _load_config():
    cfg = {"camera_id": "CAM-1", "location": "shop", "source": 0}
    if os.path.exists("config.json"):
        with open("config.json") as f:
            cfg.update(json.load(f))
    return cfg


CFG = _load_config()
CAMERA_ID = CFG["camera_id"]
LOCATION = CFG["location"]  # canteen / shop / corridor / lab / server_room

# Sirf demo/test ke liye (asli use me nahi lagte):
#   python surveillance.py video.mp4 --hour 2 --location server_room
_argv = sys.argv[1:]
HOUR_OVERRIDE = None
if "--hour" in _argv:
    i = _argv.index("--hour")
    HOUR_OVERRIDE = int(_argv[i + 1])
    del _argv[i:i + 2]
if "--location" in _argv:
    i = _argv.index("--location")
    LOCATION = _argv[i + 1]
    del _argv[i:i + 2]
SOURCE = _argv[0] if _argv else CFG["source"]
LOCATION_SENSITIVITY = {"canteen": 0.1, "shop": 0.2, "corridor": 0.4, "lab": 0.6, "server_room": 0.9}

# Restricted zone polygon (apni video ke hisaab se points badlo)
ZONE = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.int32)
if os.path.exists("zone.json"):  # pick_zone.py se bani file
    import json
    ZONE = np.array(json.load(open("zone.json")), dtype=np.int32)

LOITER_SECONDS = 10      # itni der ek jagah ruke to loitering
LOITER_RADIUS = 80       # pixels
CROWD_LIMIT = 5          # itne se zyada log = crowd
ABANDON_SECONDS = 8      # bag akela itni der
ABANDON_DIST = 150       # pixels, bag ke paas koi person nahi
COOLDOWN = 10            # same event dobara alert na ho (seconds)

FALL_ENABLED = True      # fall detection band karni ho to False
FALL_HOLD_SECONDS = 1.0  # itni der leta rahe to fall
FALL_RECENT_UPRIGHT = 4  # leta hone se pehle itne second ke andar khada tha

# Risk weights (sum = 1.0)
WEIGHTS = {"location": 0.25, "time": 0.15, "crowd": 0.10,
           "environment": 0.05, "severity": 0.40, "confidence": 0.05}

SEVERITY = {"restricted_zone": 0.8, "loitering": 0.25,
            "crowd": 0.5, "abandoned_object": 0.9, "fall": 0.85}

PERSON = 0
BAGS = {24, 26, 28}  # backpack, handbag, suitcase (COCO)

OUT_DIR = "alerts"
CSV_PATH = "alerts.csv"
# -------------------------------------------------------


def time_factor(hour):
    """Raat = zyada risk, din = kam."""
    if hour >= 22 or hour < 6:
        return 1.0
    if hour >= 18:
        return 0.5
    return 0.1


def crowd_factor(event, persons):
    """Report ke hisaab se: akela/sunsan jagah = zyada risk,
    bheed wali jagah = kam risk. Sirf 'crowd' event me bheed zyada = zyada risk."""
    if event == "crowd":
        return min(persons / (2 * CROWD_LIMIT), 1.0)
    return 1.0 - min(persons / 10, 1.0)


def environment_factor(brightness):
    """Sirf kam roshni (andhera) risk badhati hai. Normal roshni = 0."""
    return min(max((90 - brightness) / 90, 0.0), 1.0)


def risk_score(event, location, persons, brightness, confidence, hour):
    f = {
        "location": LOCATION_SENSITIVITY.get(location, 0.5),
        "time": time_factor(hour),
        "crowd": crowd_factor(event, persons),
        "environment": environment_factor(brightness),
        "severity": SEVERITY[event],
        "confidence": confidence,
    }
    score = 100 * sum(WEIGHTS[k] * f[k] for k in WEIGHTS)
    return round(score, 1)


def risk_level(score):
    """Report wale levels: 0-25 Low, 26-50 Medium, 51-75 High, 76-100 Critical"""
    if score <= 25:
        return "Low"
    if score <= 50:
        return "Medium"
    if score <= 75:
        return "High"
    return "Critical"


COLORS = {"Low": (0, 200, 0), "Medium": (0, 220, 255),
          "High": (0, 140, 255), "Critical": (0, 0, 255)}


def pose_state(box, kxy, kcf):
    """Pose keypoints + box se: 'upright' (khada), 'horizontal' (leta) ya 'other'."""
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1), max(y2 - y1, 1)
    if w > 1.2 * h:
        return "horizontal"
    if all(kcf[i] > 0.3 for i in (5, 6, 11, 12)):  # kandhe aur kamar dikh rahe
        sh = (kxy[5] + kxy[6]) / 2
        hp = (kxy[11] + kxy[12]) / 2
        # 0 degree = seedha khada, 90 degree = leta hua
        angle = np.degrees(np.arctan2(abs(hp[0] - sh[0]), abs(hp[1] - sh[1]) + 1e-6))
        if angle > 60:
            return "horizontal"
        if angle < 35 and h > 1.3 * w:
            return "upright"
        return "other"
    return "upright" if h > 1.5 * w else "other"


def init_csv():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["time", "event", "track_id", "location", "persons",
                 "confidence", "risk_score", "level", "image", "status"])


def save_alert(frame, event, tid, persons, conf, score, level):
    ts = datetime.now()
    img = f"{OUT_DIR}/{ts.strftime('%Y%m%d_%H%M%S_%f')}_{event}.jpg"
    cv2.imwrite(img, frame)
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(
            [ts.strftime("%Y-%m-%d %H:%M:%S"), event, tid, LOCATION, persons,
             round(conf, 2), score, level, img, "new"])


def main():
    init_csv()
    model = YOLO("yolov8s.pt")
    pose_model = YOLO("yolov8n-pose.pt") if FALL_ENABLED else None
    fall_state = {}           # id -> {"upright_t": time, "horiz_since": time}
    fall_boxes = []
    cap = cv2.VideoCapture(SOURCE)
    is_file = isinstance(SOURCE, str) and not SOURCE.startswith(("rtsp", "http"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0

    loiter = {}               # id -> [start_time, anchor_point]
    bag_alone_since = {}      # bag id -> time
    last_alert = {}           # (event, id) -> time
    banner = ("", (255, 255, 255), 0)  # text, color, expiry

    total_events = 0          # binary system = har event alert
    prioritized = 0           # risk system = sirf High/Critical

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        now = frame_idx / fps if is_file else time.time()
        hour = HOUR_OVERRIDE if HOUR_OVERRIDE is not None else datetime.now().hour
        brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())

        res = model.track(frame, persist=True, verbose=False, conf=0.15,
                          tracker="bytetrack_low.yaml",
                          classes=[PERSON, *BAGS])[0]

        persons, bags = [], []
        if res.boxes is not None and res.boxes.id is not None:
            for box, tid, cls, cf in zip(res.boxes.xyxy.cpu().numpy(),
                                         res.boxes.id.cpu().numpy(),
                                         res.boxes.cls.cpu().numpy(),
                                         res.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                item = (int(tid), ((x1 + x2) // 2, y2), float(cf), (x1, y1, x2, y2))
                (persons if int(cls) == PERSON else bags).append(item)

        events = []  # (event, id, conf)

        # 5) Fall detection (pose keypoints, rule-based)
        fall_boxes = []
        if FALL_ENABLED:
            pres = pose_model.track(frame, persist=True, verbose=False,
                                    conf=0.2, classes=[PERSON])[0]
            if (pres.boxes is not None and pres.boxes.id is not None
                    and pres.keypoints is not None):
                kxy_all = pres.keypoints.xy.cpu().numpy()
                kcf_all = (pres.keypoints.conf.cpu().numpy()
                           if pres.keypoints.conf is not None
                           else np.ones(kxy_all.shape[:2]))
                for pbox, ptid, pcf, kxy, kcf in zip(
                        pres.boxes.xyxy.cpu().numpy(),
                        pres.boxes.id.cpu().numpy(),
                        pres.boxes.conf.cpu().numpy(), kxy_all, kcf_all):
                    ptid = int(ptid)
                    pbox = list(map(int, pbox))
                    st = fall_state.setdefault(
                        ptid, {"upright_t": None, "horiz_since": None})
                    state = pose_state(pbox, kxy, kcf)
                    if state == "upright":
                        st["upright_t"] = now
                        st["horiz_since"] = None
                    elif state == "horizontal":
                        if st["horiz_since"] is None:
                            st["horiz_since"] = now
                        was_up = (st["upright_t"] is not None
                                  and now - st["upright_t"] <= FALL_RECENT_UPRIGHT)
                        if was_up and now - st["horiz_since"] >= FALL_HOLD_SECONDS:
                            events.append(("fall", ptid, float(pcf)))
                            fall_boxes.append(pbox)
                            st["upright_t"] = None  # ek fall = ek alert
                    else:
                        st["horiz_since"] = None

        # 1) Restricted zone + 2) Loitering
        for tid, foot, cf, _ in persons:
            if cv2.pointPolygonTest(ZONE, foot, False) >= 0:
                events.append(("restricted_zone", tid, cf))
            if tid not in loiter:
                loiter[tid] = [now, foot]
            else:
                start, anchor = loiter[tid]
                if np.hypot(foot[0] - anchor[0], foot[1] - anchor[1]) > LOITER_RADIUS:
                    loiter[tid] = [now, foot]
                elif now - start > LOITER_SECONDS:
                    events.append(("loitering", tid, cf))

        # 3) Crowd
        if len(persons) > CROWD_LIMIT:
            avg = float(np.mean([p[2] for p in persons]))
            events.append(("crowd", 0, avg))

        # 4) Abandoned object
        for tid, pos, cf, _ in bags:
            near = any(np.hypot(pos[0] - p[1][0], pos[1] - p[1][1]) < ABANDON_DIST
                       for p in persons)
            if near:
                bag_alone_since.pop(tid, None)
            else:
                bag_alone_since.setdefault(tid, now)
                if now - bag_alone_since[tid] > ABANDON_SECONDS:
                    events.append(("abandoned_object", tid, cf))

        # Risk score + alert
        for event, tid, cf in events:
            key = (event, tid)
            if now - last_alert.get(key, -1e9) < COOLDOWN:
                continue
            last_alert[key] = now
            score = risk_score(event, LOCATION, len(persons), brightness, cf, hour)
            level = risk_level(score)
            total_events += 1
            if level in ("High", "Critical"):
                prioritized += 1
            save_alert(frame, event, tid, len(persons), cf, score, level)
            banner = (f"{level.upper()} RISK: {event} (score {score})",
                      COLORS[level], now + 4)
            print(banner[0])

        # Draw
        cv2.polylines(frame, [ZONE], True, (0, 0, 255), 2)
        for tid, _, cf, (x1, y1, x2, y2) in persons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"person {tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for tid, _, cf, (x1, y1, x2, y2) in bags:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
            cv2.putText(frame, f"bag {tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
        for (x1, y1, x2, y2) in fall_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "FALL", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"{CAMERA_ID} | Persons: {len(persons)} | Location: {LOCATION}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if now < banner[2]:
            cv2.rectangle(frame, (0, frame.shape[0] - 40),
                          (frame.shape[1], frame.shape[0]), banner[1], -1)
            cv2.putText(frame, banner[0], (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow("AI Surveillance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n--- COMPARISON ---")
    print(f"Binary system alerts (sab events): {total_events}")
    print(f"Risk-based priority alerts (High/Critical): {prioritized}")
    if total_events:
        red = 100 * (total_events - prioritized) / total_events
        print(f"Operator ko dikhane wale alerts me kami: {red:.0f}%")


if __name__ == "__main__":
    main()
