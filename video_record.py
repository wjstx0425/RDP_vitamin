import argparse
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def list_realsense_profiles():
    ctx = rs.context()
    if len(ctx.devices) == 0:
        print("No RealSense device found.")
        return

    for dev in ctx.devices:
        print(f"Device: {dev.get_info(rs.camera_info.name)}")
        print(f"Serial: {dev.get_info(rs.camera_info.serial_number)}")
        for sensor in dev.sensors:
            print(f"  Sensor: {sensor.get_info(rs.camera_info.name)}")
            for profile in sensor.get_stream_profiles():
                try:
                    video_profile = profile.as_video_stream_profile()
                except RuntimeError:
                    continue
                print(
                    "    "
                    f"{profile.stream_type()} "
                    f"{video_profile.width()}x{video_profile.height()} "
                    f"{profile.fps()}fps "
                    f"{profile.format()}"
                )


def record_d415(
    output_path: str,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    color_format: str = "rgb8",
    timeout_ms: int = 15000,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    rs_format = {
        "rgb8": rs.format.rgb8,
        "bgr8": rs.format.bgr8,
    }[color_format]

    # D415 彩色视频流
    config.enable_stream(rs.stream.color, width, height, rs_format, fps)

    try:
        pipeline.start(config)
    except RuntimeError as exc:
        raise RuntimeError(
            f"RealSense 不支持当前彩色流配置: {width}x{height}@{fps} {color_format}。"
            "你的 D415 常用可选配置包括 640x480@30 或 1280x720@15。"
            "也可以运行 `python video_record.py --list-profiles` 查看完整支持列表。"
        ) from exc

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )
    if not writer.isOpened():
        pipeline.stop()
        raise RuntimeError(f"无法打开视频写入器: {output_path}")

    print(f"Recording to: {output_path}")
    print("Press 'q' in preview window or Ctrl+C in terminal to stop.")

    try:
        print("Waiting for first frame...")
        pipeline.wait_for_frames(timeout_ms)

        while True:
            frames = pipeline.wait_for_frames(timeout_ms)
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            if color_format == "rgb8":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            writer.write(frame)

            cv2.imshow("RealSense D415", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except RuntimeError as exc:
        raise RuntimeError(
            "RealSense 已启动，但等待彩色帧超时。"
            "请尝试重新插拔相机、确认使用 USB3 口，或者运行 "
            "`python video_record.py --width 1280 --height 720 --fps 15`。"
        ) from exc
    finally:
        writer.release()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Recording stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="recordings/pick_tube_0805_r1.mp4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--format", choices=["rgb8", "bgr8"], default="rgb8")
    parser.add_argument("--timeout-ms", type=int, default=15000)
    parser.add_argument("--list-profiles", action="store_true")
    args = parser.parse_args()

    if args.list_profiles:
        list_realsense_profiles()
        raise SystemExit(0)

    record_d415(
        output_path=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        color_format=args.format,
        timeout_ms=args.timeout_ms,
    )