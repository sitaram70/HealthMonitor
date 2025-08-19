#!/usr/bin/env python3
import argparse, os, cv2

def extract_frames(video_path, out_dir, fps=10, resize=(224,224)):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(input_fps / fps)))
    idx, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            if resize:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            out_path = os.path.join(out_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"Saved {saved} frames to {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input .mp4/.avi")
    ap.add_argument("--out", required=True, help="Output frames directory")
    ap.add_argument("--fps", type=int, default=10, help="Target FPS for sampling")
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--height", type=int, default=224)
    args = ap.parse_args()
    extract_frames(args.video, args.out, fps=args.fps, resize=(args.width, args.height))

if __name__ == "__main__":
    main()
