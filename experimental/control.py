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
    button_states = {'cross':0, 'circle':0, 'triangle':0, 'square':0}
    button_states['cross'] = JoyStick.get_button(14)
    button_states['circle'] = JoyStick.get_button(13)
    button_states['triangle'] = JoyStick.get_button(12)
    button_states['square'] = JoyStick.get_button(15)
    button = getKeysByValue(button_states, 1)
    return button

