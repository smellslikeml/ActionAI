import pygame

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
pygame.joystick.init()
pygame.joystick.Joystick(0).init()

JoyStick = pygame.joystick.Joystick(0)
JoyName = pygame.joystick.Joystick(0).get_name()

def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys

def getButton():
    pygame.event.pump()
    button_states = {'left':0, 'right':0, 'up':0, 'down':0, 'd1L': 0, 'd1R':0, 'select':0, 'start':0, 'cross':0, 'circle':0, 'triangle':0, 'square':0, 'jLbutton':0, 'jRbutton':0}
    button_states['left'] = JoyStick.get_button(7)
    button_states['right'] = JoyStick.get_button(5)
    button_states['up'] = JoyStick.get_button(4)
    button_states['down'] = JoyStick.get_button(6)
    button_states['d1L'] = JoyStick.get_button(10)
    button_states['d1R'] = JoyStick.get_button(11)
    button_states['select'] = JoyStick.get_button(0)
    button_states['start'] = JoyStick.get_button(3)
    button_states['cross'] = JoyStick.get_button(14)
    button_states['circle'] = JoyStick.get_button(13)
    button_states['triangle'] = JoyStick.get_button(12)
    button_states['square'] = JoyStick.get_button(15)
    button_states['jLbutton'] = JoyStick.get_button(1)
    button_states['jRbutton'] = JoyStick.get_button(2)
    button = getKeysByValue(button_states, 1)
    return button

