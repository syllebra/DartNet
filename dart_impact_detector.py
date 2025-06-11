from enum import Enum
import time
from queue import Queue

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import playsound
import requests


class DartDetector:
    def __init__(self) -> None:
        self.pause_detection = False


class AccelerometerDartImpactDetector(DartDetector):
    def __init__(self, detector_url="http://192.168.31.102:80", mqtt_broker="localhost") -> None:
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
        # TODO: on dosconnect

        self.mqttc.connect("localhost", 1883, 60)
        self.last_impact = None

    def __del__(self):
        self.mqttc.loop_stop()

    def start(self):
        self.mqttc.loop_start()

    def stop(self):
        self.mqttc.loop_stop()

    def get_config(self):
        res = requests.get(f"{self.url}/api/config", headers=None)
        res.raise_for_status()
        self.config = res.json()
        return self.config

    def set_config(self):
        res = requests.post(f"{self.url}/api/config", headers=None, json=self.config)
        res.raise_for_status()

    def calibrate(self):
        res = requests.post(f"{self.url}/api/calibrate", headers=None)
        res.raise_for_status()

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"Connected with result code {reason_code} {properties}")
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        # client.subscribe("$SYS/#")
        client.subscribe("#")

    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        print(msg.topic + " " + str(msg.payload))
        if "sensors/tap" in msg.topic:
            playsound.playsound("sound/90s-game-ui-1-185094.mp3", False)


Status = Enum("Status", [("DETECTING", 1), ("PAUSE", 2), ("DETECTED", 3)])


class DeltaVideoAccelImpactDetector(AccelerometerDartImpactDetector):
    def __init__(
        self, detector_url="http://192.168.31.102:80", mqtt_broker="localhost", burst_length=20, extra_wait_frames=5
    ) -> None:
        super().__init__()
        self.burst_length = burst_length
        self.burst = Queue(self.burst_length + extra_wait_frames)
        self.state = Status.DETECTING
        self.extra_wait_frames = extra_wait_frames

    def on_pause(self, _):
        pass

    def on_new_frame(self, img):
        ret = None
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
        max = self.burst_length + self.extra_wait_frames if self.state == Status.DETECTED else self.burst_length
        if self.burst.qsize() >= max:
            self.burst.get()
            if self.state == Status.DETECTED:
                ret = self.compute_delta()
                self.state = Status.DETECTING
        self.burst.put(img_gray)
        return ret

    def on_message(self, client, userdata, msg):
        # print(msg.topic + " " + str(msg.payload))
        if "sensors/tap" in msg.topic:
            playsound.playsound("sound/90s-game-ui-1-185094.mp3", False)
            print("DeltaVideo:" + msg.topic + " " + str(msg.payload))
            self.count_down_cpt = self.extra_wait_frames
            self.state = Status.DETECTED

    def compute_delta(self):
        first = self.burst.get()
        last = None
        while self.burst.qsize() > 0:
            last = self.burst.get()

        delta = cv2.absdiff(first, last)

        cv2.imshow("Delta", delta)
        return delta


class DeltaVideoOnlyDartDetector:
    def __init__(self) -> None:
        super().__init__()
        self.img_gray = None
        self.diff = None
        self.last = None
        self.last_diff = None
        self.last_dart_time = -1
        # ms to wait to validate (try to filter dart in flight, not yet landed) TODO: flight detection?
        self.temporal_filter = 200

    def on_pause(self, _):
        self.diff = None
        self.last = None
        self.last_diff = None
        self.last_dart_time = -1

    def start(self):
        pass

    def stop(self):
        pass

    def on_new_frame(self, img):
        detect = False
        delta = None
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
        if self.last is not None:
            self.diff = cv2.absdiff(self.img_gray, self.last)

            if self.last_diff is not None:
                delta = cv2.absdiff(self.diff, self.last_diff)
                # delta = delta*delta

                # Check if right amount of pixels is beeing modified before inference
                non_null = np.sum(delta > 0.04)
                pct = non_null * 100.0 / (delta.shape[0] * delta.shape[1])
                # print(f"PCT:{pct:.2f}")

                potential_dart_movement = pct > 0.4 and pct < 10
                # cv2.displayOverlay('Webcam', f"Potential movement:{pct:.2f} > {potential_dart_movement}")

                # dbg = cv2.cvtColor((delta*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
                # dbg = cv2.copyMakeBorder(dbg, 10, 10, 10, 10, cv2.BORDER_ISOLATED,value=(255,0,0) if potential_dart_movement else (50,0,0))
                # cv2.imshow("delta", dbg)
                cv2.imshow("delta", delta)
                if potential_dart_movement:
                    print(f"{int((time.time())*1000)}: potential_dart_movement {pct:.1f}%")
                    # playsound.playsound("sound/start-13691.mp3",False)
                    # induce a small delay to let dart land and avoid detect while flying
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
                    # cv2.waitKey()
            # cv2.imshow("diff", diff)
            if self.last_diff is None or self.last_dart_time < 0:
                self.last_diff = self.diff

            cv2.imshow("diff", self.diff)
        else:
            self.last = self.img_gray
        return delta if detect else None


if __name__ == "__main__":
    detector = DeltaVideoAccelImpactDetector()
    print(detector.get_config())

    import cv2

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
