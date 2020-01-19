#!/usr/bin/env python3
"""
Exercise app listening for class intents and executing yoga flows accordingly.
"""

import time
import paho.mqtt.client as mqtt
from flow import Flow

HOST = 'localhost'
PORT = 1883

def on_connect(client, userdata, flags, rc):
    print("Connected to {0} with result code {1}".format(HOST, rc))
    # Subscribe to the hotword detected topic
    client.subscribe("hermes/hotword/default/detected")
    # Subscribe to intent topic
    client.subscribe('hermes/intent/#')
    
def on_message(client, userdata, msg):
    current_pose = None
    f = ['tree_pose', 'warrior_1', 'crescent_lunge', 'warrior_2', 'triangle', 'half_moon', 'plank', 'chaturanga_dandasana', 'cobra']
    if msg.topic == 'hermes/intent/smayorquin:BeginClass':
        #Sample flow
        flow = Flow(f)
        flow.run()
    elif msg.topic == 'hermes/intent/smayorquin:StopClass':
        print("StopClass Intent detected!")
        try:
            del flow
        except:
            pass
    elif msg.topic == 'hermes/intent/smayorquin:PauseClass':
        print("PauseClass Intent detected!")
        current_pose = flow.current_pose
        try:
            del flow
        except:
            pass
    elif msg.topic == 'hermes/intent/smayorquin:RestartClass':
        print("RestartClass Intent detected!")
        flow = Flow(f, current_pose=current_pose) 
        flow.run()


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(HOST, PORT, 60)
client.loop_forever()
