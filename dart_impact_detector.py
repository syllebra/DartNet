"""
Dart Impact Detection System

This module provides various dart impact detection methods using accelerometer data,
video analysis, and MQTT communication for real-time dart scoring systems.
"""

from enum import Enum
import time
from queue import Queue

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import playsound
import requests


class DartDetector:
    """
    Base class for dart impact detection systems.

    Provides common interface and basic functionality for different
    dart detection implementations.
    """

    def __init__(self) -> None:
        """Initialize the dart detector with default settings."""
        self.pause_detection = False

    def start(self):
        """Start the detector (placeholder implementation)."""
        pass

    def stop(self):
        """Stop the detector (placeholder implementation)."""
        pass

    def on_new_frame(self, img):
        """
        Process a new video frame.

        Args:
            img: Input video frame (BGR format)

        Returns:
            numpy.ndarray or None: Frame difference if impact detected, None otherwise
        """
        return None


class AccelerometerDartImpactDetector(DartDetector):
    """
    Dart impact detector using accelerometer data via HTTP API and MQTT.

    This detector communicates with an external accelerometer device to detect
    dart impacts on the board. It uses HTTP requests for configuration and
    MQTT for real-time impact notifications.
    https://github.com/syllebra/darts-impact-detector
    """

    def __init__(self, detector_url="http://192.168.31.102:80", mqtt_broker="localhost") -> None:
        """
        Initialize the accelerometer-based dart detector.

        Args:
            detector_url (str): URL of the accelerometer device API
            mqtt_broker (str): MQTT broker hostname for impact notifications
        """
        super().__init__()
        self.url = detector_url
        self.config = {
            "threshold": 0.03,
            "min_delay_between_taps": 150,
            "tap_duration_min": 2,
            "tap_duration_max": 150,
            "enable_filtering": True,
            "filter_alpha": 0.7,
            "sensitivity": 50,
            "debug": False,
            "use_adaptive_threshold": True,
            "noise_window": 0.2,
            "peak_ratio": 1.2,
            "use_derivative": True,
            "derivative_threshold": 0.8,
        }

        self.mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqttc.on_connect = self.on_connect
        self.mqttc.on_message = self.on_message
        # TODO: on disconnect

        self.mqttc.connect("localhost", 1883, 60)
        self.last_impact = None

    def __del__(self):
        """Clean up MQTT connection on object destruction."""
        self.mqttc.loop_stop()

    def start(self):
        """Start the MQTT client loop for receiving impact notifications."""
        self.mqttc.loop_start()

    def stop(self):
        """Stop the MQTT client loop."""
        self.mqttc.loop_stop()

    def get_config(self):
        """
        Retrieve current configuration from the accelerometer device.

        Returns:
            dict: Current device configuration

        Raises:
            requests.HTTPError: If the API request fails
        """
        res = requests.get(f"{self.url}/api/config", headers=None)
        res.raise_for_status()
        self.config = res.json()
        return self.config

    def set_config(self):
        """
        Send current configuration to the accelerometer device.

        Raises:
            requests.HTTPError: If the API request fails
        """
        res = requests.post(f"{self.url}/api/config", headers=None, json=self.config)
        res.raise_for_status()

    def calibrate(self):
        """
        Trigger calibration on the accelerometer device.

        Raises:
            requests.HTTPError: If the API request fails
        """
        res = requests.post(f"{self.url}/api/calibrate", headers=None)
        res.raise_for_status()

    def on_connect(self, client, userdata, flags, reason_code, properties):
        """
        MQTT callback for when the client connects to the broker.

        Args:
            client: MQTT client instance
            userdata: User data passed to callbacks
            flags: Response flags from the broker
            reason_code: Connection result code
            properties: MQTT v5.0 properties
        """
        print(f"Connected with result code {reason_code} {properties}")
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe("#")

    def on_message(self, client, userdata, msg):
        """
        MQTT callback for when a message is received.

        Args:
            client: MQTT client instance
            userdata: User data passed to callbacks
            msg: Message received from the broker
        """
        print(msg.topic + " " + str(msg.payload))
        if "sensors/tap" in msg.topic:
            playsound.playsound("sound/90s-game-ui-1-185094.mp3", False)


Status = Enum("Status", [("DETECTING", 1), ("PAUSE", 2), ("DETECTED", 3)])
"""Enumeration for detector states: DETECTING, PAUSE, DETECTED."""


class DeltaVideoAccelImpactDetector(AccelerometerDartImpactDetector):
    """
    Combined accelerometer and video-based dart impact detector.

    This detector uses both accelerometer data (via MQTT) and video frame
    analysis to detect dart impacts. It maintains a buffer of video frames
    and computes frame differences when an impact is detected.
    """

    def __init__(
        self, detector_url="http://192.168.31.102:80", mqtt_broker="localhost", burst_length=20, extra_wait_frames=5
    ) -> None:
        """
        Initialize the combined detector.

        Args:
            detector_url (str): URL of the accelerometer device API
            mqtt_broker (str): MQTT broker hostname
            burst_length (int): Number of frames to buffer for analysis
            extra_wait_frames (int): Additional frames to wait after detection
        """
        super().__init__()
        self.burst_length = burst_length
        self.burst = Queue(self.burst_length + extra_wait_frames)
        self.state = Status.DETECTING
        self.extra_wait_frames = extra_wait_frames

    def on_pause(self, _):
        """Handle pause events (placeholder implementation)."""
        pass

    def on_new_frame(self, img):
        """
        Process a new video frame.

        Args:
            img: Input video frame (BGR format)

        Returns:
            numpy.ndarray or None: Frame difference if impact detected, None otherwise
        """
        ret = None
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
        max_frames = self.burst_length + self.extra_wait_frames if self.state == Status.DETECTED else self.burst_length

        if self.burst.qsize() >= max_frames:
            self.burst.get()
            if self.state == Status.DETECTED:
                ret = self.compute_delta()
                self.state = Status.DETECTING
        self.burst.put(img_gray)
        return ret

    def on_message(self, client, userdata, msg):
        """
        Override MQTT message handler to trigger video analysis.

        Args:
            client: MQTT client instance
            userdata: User data passed to callbacks
            msg: Message received from the broker
        """
        if "sensors/tap" in msg.topic:
            playsound.playsound("sound/90s-game-ui-1-185094.mp3", False)
            print("DeltaVideo:" + msg.topic + " " + str(msg.payload))
            self.count_down_cpt = self.extra_wait_frames
            self.state = Status.DETECTED

    def compute_delta(self):
        """
        Compute the difference between first and last frames in the buffer.

        Returns:
            numpy.ndarray: Frame difference showing movement/impact
        """
        first = self.burst.get()
        last = None
        while self.burst.qsize() > 0:
            last = self.burst.get()

        delta = cv2.absdiff(first, last)
        cv2.imshow("Delta", delta)
        return delta


class DeltaVideoOnlyDartDetector:
    """
    Video-only dart impact detector using frame difference analysis.

    This detector analyzes consecutive video frames to detect dart impacts
    by looking for specific movement patterns. It uses temporal filtering
    to avoid false positives from darts in flight.
    """

    def __init__(self) -> None:
        """Initialize the video-only dart detector."""
        super().__init__()
        self.img_gray = None
        self.diff = None
        self.last = None
        self.last_diff = None
        self.last_dart_time = -1
        # ms to wait to validate (try to filter dart in flight, not yet landed)
        # TODO: flight detection?
        self.temporal_filter = 200

    def on_pause(self, _):
        """Reset detector state when paused."""
        self.diff = None
        self.last = None
        self.last_diff = None
        self.last_dart_time = -1

    def on_new_frame(self, img):
        """
        Process a new video frame to detect dart impacts.

        Args:
            img: Input video frame (BGR format)

        Returns:
            numpy.ndarray or None: Frame difference if impact detected, None otherwise
        """
        detect = False
        delta = None
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0

        if self.last is not None:
            self.diff = cv2.absdiff(self.img_gray, self.last)

            if self.last_diff is not None:
                delta = cv2.absdiff(self.diff, self.last_diff)

                # Check if right amount of pixels is being modified before inference
                non_null = np.sum(delta > 0.04)
                pct = non_null * 100.0 / (delta.shape[0] * delta.shape[1])

                potential_dart_movement = 0.4 < pct < 10

                cv2.imshow("delta", delta)
                if potential_dart_movement:
                    timestamp = int((time.time()) * 1000)
                    print(f"{timestamp}: potential_dart_movement {pct:.1f}%")

                    # Induce a small delay to let dart land and avoid detect while flying
                    detect = False
                    if self.temporal_filter < 0:
                        detect = True
                    else:
                        if self.last_dart_time < 0:
                            self.last_dart_time = time.time()
                        else:
                            elapsed = time.time() - self.last_dart_time
                            detect = elapsed * 1000 >= self.temporal_filter
                    if detect:
                        self.last_dart_time = -1

            if self.last_diff is None or self.last_dart_time < 0:
                self.last_diff = self.diff

            cv2.imshow("diff", self.diff)
        else:
            self.last = self.img_gray

        return delta if detect else None


if __name__ == "__main__":
    """
    Main execution block for testing the dart impact detector.

    This creates a DeltaVideoAccelImpactDetector instance, initializes video capture,
    and runs a detection loop until the user presses 'q' or ESC.
    """
    detector = DeltaVideoAccelImpactDetector()
    print(detector.get_config())

    print("Initialize video capture...")
    cap = cv2.VideoCapture(0)

    detector.start()
    # detector.config["threshold"] = 0.05
    # detector.config["debug"] = True
    # detector.set_config()
    # detector.calibrate()

    while True:
        success, img = cap.read()
        if not success:
            continue
        detector.on_new_frame(img)
        key = cv2.waitKey(1)
        if key == ord("q") or key == 27:
            break

        if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
            cv2.imshow("Webcam", img)

    detector.stop()
    # detector.mqttc.loop_forever()
