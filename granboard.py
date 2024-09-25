from board import Board
from tools import ask_json_url, add_transparent_image
import threading
import requests
import json

class GranboardApi():
    def __init__(self, granboard_url = "http://192.168.31.191:8822",  button_state_callback=None) -> None:
        self.url = granboard_url
        self.command_map = {
            "S1": "2.3@",
            "S2": "9.1@",
            "S3": "7.1@",
            "S4": "0.1@",
            "S5": "5.1@",
            "S6": "1.0@",
            "S7": "11.1@",
            "S8": "6.2@",
            "S9": "9.3@",
            "S10": "2.0@",
            "S11": "7.3@",
            "S12": "5.0@",
            "S13": "0.0@",
            "S14": "10.3@",
            "S15": "3.0@",
            "S16": "11.0@",
            "S17": "10.1@",
            "S18": "1.2@",
            "S19": "6.1@",
            "S20": "3.3@",
            "S1OUT": "2.5@",
            "S2OUT": "9.2@",
            "S3OUT": "7.2@",
            "S4OUT": "0.5@",
            "S5OUT": "5.4@",
            "S6OUT": "1.3@",
            "S7OUT": "11.4@",
            "S8OUT": "6.5@",
            "S9OUT": "9.5@",
            "S10OUT": "2.2@",
            "S11OUT": "7.5@",
            "S12OUT": "5.5@",
            "S13OUT": "0.4@",
            "S14OUT": "10.5@",
            "S15OUT": "3.2@",
            "S16OUT": "11.5@",
            "S17OUT": "10.2@",
            "S18OUT": "1.5@",
            "S19OUT": "6.3@",
            "S20OUT": "3.5@",
            "D1": "2.6@",
            "D2": "8.2@",
            "D3": "8.4@",
            "D4": "0.6@",
            "D5": "4.6@",
            "D6": "4.4@",
            "D7": "8.6@",
            "D8": "6.6@",
            "D9": "9.6@",
            "D10": "4.3@",
            "D11": "7.6@",
            "D12": "5.6@",
            "D13": "4.5@",
            "D14": "10.6@",
            "D15": "4.2@",
            "D16": "11.6@",
            "D17": "8.3@",
            "D18": "1.6@",
            "D19": "8.5@",
            "D20": "3.6@",
            "T1": "2.4@",
            "T2": "9.0@",
            "T3": "7.0@",
            "T4": "0.3@",
            "T5": "5.2@",
            "T6": "1.1@",
            "T7": "11.2@",
            "T8": "6.4@",
            "T9": "9.4@",
            "T10": "2.1@",
            "T11": "7.4@",
            "T12": "5.3@",
            "T13": "0.2@",
            "T14": "10.4@",
            "T15": "3.1@",
            "T16": "11.3@",
            "T17": "10.0@",
            "T18": "1.4@",
            "T19": "6.0@",
            "T20": "3.4@",
            "SBULL": "8.0@",
            "DBULL": "4.0@",
            "MISS": "MISS@",
            "OUT": "OUT@"
        }
        self.button_state = False
        self.button_state_callback = button_state_callback
        def cb(r):
            self.button_state = r["button_state"]
            #print(f"callback function called {r}")
            if(self.button_state_callback is not None):
                self.button_state_callback(self.button_state)


    def _ask_button_state(self):
        ret = requests.get(f"{self.url}/hit?cb")
        if(ret.status_code >=200 and ret.status_code<300):
            bs = json.loads(ret.content.decode("utf8"))
            #print(bs)
            if(bs["button_state"] != self.button_state):
                self.button_state = bs["button_state"]
                if(self.button_state_callback is not None):
                    self.button_state_callback(self.button_state)
    
    def ask_button_state(self):
        threading.Thread(target=self._ask_button_state, args=()).start()

    def request_task(self, url, data, headers):
        try:
            requests.get(url, json=data, headers=headers)
        except:
            pass

    def fire_and_forget(self, url, json=None, headers=None):
        threading.Thread(target=self.request_task, args=(url, json, headers)).start()

    def score(self, sc):
        if(sc == "0"):
            sc = "OUT"
        elif(sc == "DB"):
            sc = "DBULL"
        elif(sc == "B"):
            sc = "SBULL"
        elif("D" not in sc and "T" not in sc):
            sc = f"S{sc}OUT"
        command = self.command_map[sc]
        print(sc,"=>",command)
        self.fire_and_forget(self.url+"/hit?cmd="+command)

    def stop(self):
        self.thread.wait_stop()
        if(self.thread_poll):
            self.thread_poll.wait_stop()

    def click_button(self):
        self.fire_and_forget(self.url+"/hit?cb=click")

