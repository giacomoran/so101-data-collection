#!/usr/bin/env python
"""
Rolling QR code display module.

This module provides functionality to display a rolling QR code with timestamps.
It can be used both as an importable module and as a standalone script.

When used as a module:
    from so101_data_collection.shared.rolling_qr import RollingQRDisplay

    display = RollingQRDisplay()
    display.start()  # Runs in current process
    # ... do other work ...
    display.stop()

When used as a script:
    python -m so101_data_collection.shared.rolling_qr

This will start the QR display in a separate process that can be controlled.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass
from queue import Queue
from threading import Event, Thread

import cv2
import numpy as np
import qrcode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# QR code display settings
QR_WINDOW_NAME = "Rolling QR Code Display"
QR_SIZE = 800  # Size of QR code display in pixels


@dataclass
class RollingQRConfig:
    """Configuration for rolling QR code display."""

    qr_size: int = QR_SIZE
    fps: float = 30.0
    window_name: str = QR_WINDOW_NAME


class RollingQRDisplay:
    """
    Displays a rolling QR code with timestamps.

    The QR code encodes the current timestamp, updating at the specified FPS.
    This can be used for latency measurement by capturing the QR code with a camera.
    """

    def __init__(self, config: RollingQRConfig | None = None):
        """
        Initialize the rolling QR display.

        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or RollingQRConfig()
        self._stop_event = Event()
        self._display_thread: Thread | None = None
        self._window_created = False
        self._thread_error: Exception | None = None
        # Queue for frames generated in background thread to be displayed on main thread
        self._frame_queue: Queue[np.ndarray] = Queue(maxsize=2)

    def generate_qr_image(self, data: str, size: int | None = None) -> np.ndarray:
        """
        Generate a QR code image as a numpy array.

        Args:
            data: String data to encode
            size: Output image size in pixels (defaults to config.qr_size)

        Returns:
            BGR image as numpy array
        """
        size = size or self.config.qr_size
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)

        # Create PIL image
        img = qr.make_image(fill_color="black", back_color="white")

        # Convert to numpy array
        img_array = np.array(img.convert("RGB"))

        # Resize to desired size
        img_resized = cv2.resize(
            img_array, (size, size), interpolation=cv2.INTER_NEAREST
        )

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

        return img_bgr

    def create_display_frame(self, timestamp: float) -> np.ndarray:
        """
        Create a display frame with QR code encoding the timestamp.

        The frame includes:
        - Large QR code with timestamp
        - Text showing the encoded timestamp (for debugging)

        Args:
            timestamp: Timestamp to encode in QR code

        Returns:
            BGR image frame ready for display
        """
        # Generate QR code with timestamp
        timestamp_str = f"{timestamp:.6f}"
        qr_img = self.generate_qr_image(timestamp_str, self.config.qr_size)

        # Create frame with padding for text
        padding = 60
        frame_h = self.config.qr_size + padding
        frame_w = self.config.qr_size
        frame = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 255

        # Place QR code
        frame[: self.config.qr_size, : self.config.qr_size] = qr_img

        # Add timestamp text below QR code
        text = f"t={timestamp_str}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (frame_w - text_size[0]) // 2
        text_y = self.config.qr_size + 35
        cv2.putText(
            frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness
        )

        return frame

    def _create_window(self) -> None:
        """
        Create the display window on the main thread.

        This must be called on the main thread (especially on macOS) before
        starting the background display thread.
        """
        if not self._window_created:
            cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self.config.window_name, self.config.qr_size, self.config.qr_size + 60
            )
            self._window_created = True

    def _generate_loop(self) -> None:
        """
        Background thread loop that generates QR frames.

        Frames are put into a queue to be displayed by the main thread,
        since OpenCV GUI operations must happen on the main thread on macOS.
        """
        try:
            target_frame_time = 1.0 / self.config.fps

            while not self._stop_event.is_set():
                loop_start = time.perf_counter()

                # Generate QR code with current timestamp
                # Use time.time() (wall-clock) instead of perf_counter() so timestamps
                # are valid across processes (perf_counter is only valid within a process)
                t_display = time.time()
                display_frame = self.create_display_frame(t_display)

                # Put frame in queue for main thread to display
                # Use non-blocking put with timeout to avoid blocking if queue is full
                try:
                    self._frame_queue.put(display_frame, timeout=0.1)
                except Exception:
                    # Queue full, skip this frame (main thread will catch up)
                    pass

                # Maintain frame rate
                frame_time = time.perf_counter() - loop_start
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)

        except Exception as e:
            logger.error(f"Error in QR generation thread: {e}", exc_info=True)
            self._thread_error = e

    def update_display(self) -> None:
        """
        Update the display with the latest frame from the queue.

        This must be called periodically from the main thread to:
        1. Display frames generated by the background thread
        2. Process OpenCV window events (required on macOS)

        Call this in your main loop, e.g.:
            while running:
                qr_display.update_display()
                cv2.waitKey(1)  # Process events
                # ... do other work ...
        """
        if not self._window_created:
            return

        # Get latest frame from queue (non-blocking)
        frame = None
        while True:
            try:
                frame = self._frame_queue.get_nowait()
            except Exception:
                break

        # Display the latest frame
        if frame is not None:
            try:
                cv2.imshow(self.config.window_name, frame)
            except Exception as e:
                logger.error(f"Failed to display frame: {e}")
                self._thread_error = e

    def start(self, blocking: bool = False) -> None:
        """
        Start the rolling QR display.

        Args:
            blocking: If True, blocks until stopped. If False, runs in background thread.
        """
        if self._display_thread is not None and self._display_thread.is_alive():
            logger.warning("Display already running")
            return

        self._stop_event.clear()
        self._thread_error = None

        if blocking:
            # Run in current thread - create window here
            self._create_window()
            # In blocking mode, run the full display loop here
            target_frame_time = 1.0 / self.config.fps
            while not self._stop_event.is_set():
                loop_start = time.perf_counter()
                # Use time.time() for cross-process compatibility
                t_display = time.time()
                display_frame = self.create_display_frame(t_display)
                cv2.imshow(self.config.window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                frame_time = time.perf_counter() - loop_start
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)
        else:
            # Create window on main thread before starting background thread
            # This is required on macOS where OpenCV window creation must happen
            # on the main thread
            self._create_window()
            # Run frame generation loop in background thread
            # Main thread will call update_display() to show frames
            self._display_thread = Thread(target=self._generate_loop, daemon=True)
            self._display_thread.start()
            logger.info("Rolling QR display started in background thread")

    def stop(self) -> None:
        """Stop the rolling QR display."""
        self._stop_event.set()
        if self._display_thread is not None:
            self._display_thread.join(timeout=2.0)
            self._display_thread = None
        # Clear frame queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Exception:
                break
        # Destroy window on main thread
        if self._window_created:
            cv2.destroyWindow(self.config.window_name)
            self._window_created = False
        logger.info("Rolling QR display stopped")

    def is_running(self) -> bool:
        """Check if the display is currently running."""
        if self._thread_error is not None:
            return False
        return (
            self._display_thread is not None
            and self._display_thread.is_alive()
            and not self._stop_event.is_set()
        )

    def check_error(self) -> Exception | None:
        """Check if there was an error in the display thread."""
        return self._thread_error

    def __enter__(self):
        """Context manager entry."""
        self.start(blocking=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def main() -> None:
    """Main entry point when run as a script."""
    parser = argparse.ArgumentParser(
        description="Display rolling QR codes with timestamps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Display frame rate",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=QR_SIZE,
        help="QR code size in pixels",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration to run in seconds (default: run until 'q' is pressed)",
    )

    args = parser.parse_args()

    config = RollingQRConfig(qr_size=args.size, fps=args.fps)
    display = RollingQRDisplay(config)

    # Handle SIGINT (Ctrl+C) gracefully
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, stopping...")
        display.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    logger.info("Starting rolling QR code display...")
    logger.info("Press 'q' in the QR window to stop")
    logger.info(f"FPS: {args.fps}, Size: {args.size}px")

    if args.duration:
        # Run for specified duration with proper display loop
        display.start(blocking=False)
        try:
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < args.duration:
                # Must call update_display() to actually show frames from the queue
                display.update_display()
                # Process window events and check for 'q' key
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit key pressed")
                    break
        except KeyboardInterrupt:
            pass
        finally:
            display.stop()
    else:
        # Run until 'q' is pressed
        display.start(blocking=True)


if __name__ == "__main__":
    main()
