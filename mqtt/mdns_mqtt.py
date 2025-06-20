#!/usr/bin/env python3
"""
Robust MQTT Client with Auto-Discovery and Reconnection
Uses mDNS to discover MQTT brokers and maintains persistent connection
"""

import socket
import threading
import time
from typing import Any, Callable, Dict, Optional

import paho.mqtt.client as mqtt
from zeroconf import ServiceListener, Zeroconf


class MQTTServiceListener(ServiceListener):
    """Service listener for MQTT broker discovery"""

    def __init__(self):
        self.broker_ip: Optional[str] = None
        self.broker_port: Optional[int] = None
        self.found = False

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        print(f"MQTT Service discovered: {name}")
        info = zc.get_service_info(type_, name)
        if info and info.addresses:
            self.broker_ip = socket.inet_ntoa(info.addresses[0])
            self.broker_port = info.port
            self.found = True
            print(f"MQTT Broker found at {self.broker_ip}:{self.broker_port}")

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        print(f"MQTT Service removed: {name}")

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        pass


class RobustMQTTClient:
    """
    Robust MQTT client with auto-discovery and reconnection handling
    """

    def __init__(self, client_id: str = "robust_mqtt_client", discovery_timeout: int = 30, reconnect_delay: int = 2):
        """
        Initialize the MQTT client

        Args:
            client_id: MQTT client ID
            discovery_timeout: Timeout for broker discovery in seconds
            reconnect_delay: Delay between reconnection attempts in seconds
        """
        self.client_id = client_id
        self.discovery_timeout = discovery_timeout
        self.reconnect_delay = reconnect_delay

        # Connection state
        self.broker_ip: Optional[str] = None
        self.broker_port: int = 1883
        self.is_connected = False
        self.should_reconnect = True

        # MQTT client
        self.client: Optional[mqtt.Client] = None

        # Callbacks
        self.on_connect_callback: Optional[Callable] = None
        self.on_disconnect_callback: Optional[Callable] = None
        self.on_message_callback: Optional[Callable] = None

        # Thread for background operations
        self._discovery_thread: Optional[threading.Thread] = None
        self._reconnect_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        print(f"Initialized MQTT client: {client_id}")

    def set_callbacks(self, on_connect: Callable = None, on_disconnect: Callable = None, on_message: Callable = None):
        """Set callback functions for MQTT events"""
        self.on_connect_callback = on_connect
        self.on_disconnect_callback = on_disconnect
        self.on_message_callback = on_message

    def discover_broker(self) -> bool:
        """
        Discover MQTT broker using mDNS
        Returns True if broker found, False otherwise
        """
        print("Discovering MQTT broker...")

        zeroconf = Zeroconf()
        listener = MQTTServiceListener()

        try:
            zeroconf.add_service_listener("_mqtt._tcp.local.", listener)

            # Wait for discovery with timeout
            start_time = time.time()
            while not listener.found and (time.time() - start_time) < self.discovery_timeout:
                time.sleep(1)

            if listener.found:
                self.broker_ip = listener.broker_ip
                self.broker_port = listener.broker_port or 1883
                print(f"✓ Broker discovered: {self.broker_ip}:{self.broker_port}")
                return True
            else:
                print("✗ No MQTT broker found within timeout")
                return False

        except Exception as e:
            print(f"Discovery error: {e}")
            return False
        finally:
            zeroconf.remove_service_listener(listener)
            zeroconf.close()

    def _on_connect(self, client, userdata, flags, rc):
        """Internal callback for MQTT connection"""
        if rc == 0:
            self.is_connected = True
            print(f"✓ Connected to MQTT broker at {self.broker_ip}:{self.broker_port}")
            if self.on_connect_callback:
                self.on_connect_callback(client, userdata, flags, rc)
        else:
            print(f"✗ Failed to connect to MQTT broker. Return code: {rc}")
            self.is_connected = False

    def _on_disconnect(self, client, userdata, rc):
        """Internal callback for MQTT disconnection"""
        self.is_connected = False
        print(f"⚠ Disconnected from MQTT broker. Return code: {rc}")

        if self.on_disconnect_callback:
            self.on_disconnect_callback(client, userdata, rc)

        # Add delay before triggering reconnection to let broker handle the disconnection properly
        if self.should_reconnect and rc != 0:  # Unexpected disconnection
            print(f"Waiting {self.reconnect_delay} seconds before attempting reconnection...")
            time.sleep(self.reconnect_delay)
            self._start_reconnect_thread()

    def _on_message(self, client, userdata, msg):
        """Internal callback for MQTT messages"""
        if self.on_message_callback:
            self.on_message_callback(client, userdata, msg)

    def connect(self) -> bool:
        """
        Connect to MQTT broker
        Returns True if connected successfully, False otherwise
        """
        with self._lock:
            # Discover broker if not known
            if not self.broker_ip:
                if not self.discover_broker():
                    return False

            # Create and configure MQTT client
            try:
                self.client = mqtt.Client(client_id=self.client_id)
                self.client.on_connect = self._on_connect
                self.client.on_disconnect = self._on_disconnect
                self.client.on_message = self._on_message

                # Connect to broker
                print(f"Connecting to {self.broker_ip}:{self.broker_port}...")
                self.client.connect(self.broker_ip, self.broker_port, 60)
                self.client.loop_start()

                # Wait for connection
                start_time = time.time()
                while not self.is_connected and (time.time() - start_time) < 10:
                    time.sleep(0.1)

                return self.is_connected

            except Exception as e:
                print(f"Connection error: {e}")
                return False

    def disconnect(self):
        """Disconnect from MQTT broker"""
        with self._lock:
            self.should_reconnect = False
            if self.client and self.is_connected:
                print("Disconnecting from MQTT broker...")
                self.client.loop_stop()
                self.client.disconnect()
                self.is_connected = False

    def _reconnect_worker(self):
        """Background worker for reconnection attempts"""
        while self.should_reconnect and not self.is_connected:
            print("Attempting reconnection...")
            if self.connect():
                break

            if self.should_reconnect and not self.is_connected:
                print(f"Reconnection failed. Waiting {self.reconnect_delay} seconds before next attempt...")
                time.sleep(self.reconnect_delay)

    def _start_reconnect_thread(self):
        """Start reconnection thread if not already running"""
        if self._reconnect_thread is None or not self._reconnect_thread.is_alive():
            self._reconnect_thread = threading.Thread(target=self._reconnect_worker, daemon=True)
            self._reconnect_thread.start()

    def check(self) -> bool:
        """
        Check connection status and handle reconnection if needed
        Call this method regularly in your main loop

        Returns:
            bool: True if connected, False otherwise
        """
        if not self.is_connected and self.should_reconnect:
            # Start reconnection if not already in progress
            self._start_reconnect_thread()

        return self.is_connected

    def publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False) -> bool:
        """
        Publish message to MQTT topic

        Args:
            topic: MQTT topic
            payload: Message payload
            qos: Quality of Service level
            retain: Retain flag

        Returns:
            bool: True if published successfully, False otherwise
        """
        if not self.is_connected:
            print("Cannot publish: Not connected to broker")
            return False

        try:
            result = self.client.publish(topic, payload, qos, retain)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            print(f"Publish error: {e}")
            return False

    def subscribe(self, topic: str, qos: int = 0) -> bool:
        """
        Subscribe to MQTT topic

        Args:
            topic: MQTT topic to subscribe to
            qos: Quality of Service level

        Returns:
            bool: True if subscribed successfully, False otherwise
        """
        if not self.is_connected:
            print("Cannot subscribe: Not connected to broker")
            return False

        try:
            result = self.client.subscribe(topic, qos)
            return result[0] == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            print(f"Subscribe error: {e}")
            return False

    def is_broker_available(self) -> bool:
        """Check if broker is available"""
        return self.is_connected


# Example usage
if __name__ == "__main__":

    def on_connect_handler(client, userdata, flags, rc):
        print("Custom connect handler called")
        # Subscribe to test topic
        client.subscribe("test/topic")

    def on_message_handler(client, userdata, msg):
        print(f"Received message: {msg.topic} -> {msg.payload.decode()}")

    def on_disconnect_handler(client, userdata, rc):
        print("Custom disconnect handler called")

    # Create and configure client
    mqtt_client = RobustMQTTClient("test_client")
    mqtt_client.set_callbacks(
        on_connect=on_connect_handler, on_message=on_message_handler, on_disconnect=on_disconnect_handler
    )

    # Connect to broker
    if mqtt_client.connect():
        print("Connected successfully!")

        # Example main loop
        try:
            for i in range(100):
                # Check connection status (call this regularly)
                if mqtt_client.check():
                    # Publish test message
                    mqtt_client.publish("test/topic", f"Hello World {i}")
                    print(f"Published message {i}")
                else:
                    print("Not connected - waiting for reconnection...")

                time.sleep(5)

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            mqtt_client.disconnect()
    else:
        print("Failed to connect to MQTT broker")
