"""Practical 9: Basic person detection in CCTV/video footage using OpenCV HOG.

Usage:
  python 09_cctv_object_detection.py --source video.mp4
  python 09_cctv_object_detection.py --source 0
Press q to quit. This detects people only; it is a simple demonstration.
"""
import argparse
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Video file path or camera index (default: 0)")
    parser.add_argument("--output", default="", help="Optional output video path, e.g. detected.mp4")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Error: cannot open video source: {args.source}")

    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    writer = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes, weights = detector.detectMultiScale(
                frame, winStride=(8, 8), padding=(8, 8), scale=1.05
            )
            for (x, y, w, h), confidence in zip(boxes, weights):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {float(confidence):.2f}", (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            if args.output and writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                width, height = frame.shape[1], frame.shape[0]
                writer = cv2.VideoWriter(
                    args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
            if writer is not None:
                writer.write(frame)
            cv2.imshow("CCTV Person Detection - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
