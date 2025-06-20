# https://github.com/docker/docker-py
# pip install docker==2.1.0

# https://medium.com/@mariovanrooij/adding-https-to-fastapi-ad5e0f9e084e

import time

from fastapi import FastAPI

from granboard import GranboardApi
from mqtt.mdns_mqtt import RobustMQTTClient

HACKED_GRANBOARD_API_URL = "http://192.168.31.192:8822"

app = FastAPI()

gb_api = GranboardApi(HACKED_GRANBOARD_API_URL)

next_button_visible = False

topic = "dartnet/hit"

# MQTT (To send messages to Granboard)


def on_mqtt_connect_handler(client, userdata, flags, rc):
    print(f"Subscribing to {topic}")
    client.subscribe(topic)


def on_mqtt_message_handler(client, userdata, msg):
    print(f"Received message: {msg.topic} -> {msg.payload.decode()}")
    if msg.topic == topic:
        score = msg.payload.decode()
        gb_api.score(score)


def on_mqtt_disconnect_handler(client, userdata, rc):
    # print("Custom disconnect handler called")
    pass


# FAST API (To receive messages from Granboard)
# Test if API is Live
@app.get("/")
async def root():
    return {"message": "Alive"}


@app.get("/gbleds")
async def gbleds(func: str | None = None, params: str | None = None):
    print("Leds message received: func:", func, " params:", params)
    return {"message": "Alive"}


@app.get("/hit")
async def hit(cb: str | None = None, cmd: str | None = None):
    global next_button_visible
    print("Dummy received: cb:", cb, " cmd:", cmd)
    ret = {}
    if cb is not None:
        ret["cb"] = cb
        if cb == "click":
            next_button_visible = not next_button_visible
        ret["button_state"] = next_button_visible

    if cmd is not None:
        ret["cmd"] = cmd

    # print(json.dumps(ret.attrs, indent=4))
    time.sleep(0.3)
    return ret


if __name__ == "__main__":
    import uvicorn

    # Start server on port 8080
    config = uvicorn.Config("hacked_granboard_app_to_mqtt:app", port=8822, host="0.0.0.0", log_level="info")  # ,
    #              ssl_keyfile="./test_key.pem",
    #              ssl_certfile="./test_cert.pem")
    server = uvicorn.Server(config)

    # Create and configure client
    mqtt_client = RobustMQTTClient("Hacked Granboard API MQTT Bridge")
    mqtt_client.set_callbacks(
        on_connect=on_mqtt_connect_handler, on_message=on_mqtt_message_handler, on_disconnect=on_mqtt_disconnect_handler
    )

    mqtt_client.connect()
    server.run()
    gb_api.stop()
