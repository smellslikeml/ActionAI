#!/usr/bin/env python3
import datetime
import time
import cv2
import subprocess
import numpy as np
from utils.inference import motionClassifier

class Flow():
    def __init__(self, flow):
        self.flow = flow
        self.cam = cv2.VideoCapture(0)
        self.motion = motionClassifier()
        self.image = None
        self.current_pose = None
        self.intro_phrase = ["Hello and welcome to Yohg A I.", "Lets start a flow.", "Take a moment to take 3 deep breaths."]
        self.flow_dict = {'chaturanga dandasana': ['Place your forearms firmly on the ground.', 'Make sure your body is straight'], 
                          'cobra': ['Push yourself up from the ground with strong arms.', 'Arc your back.'], 
                          'cow': ['Stabilize yourself with your hands and knees.', 'Slightly arc your back.'], 
                          'crescent lunge': ['Throw your arms back into a lunge.', 'Keep your legs steady.'], 
                          'half moon': ['Keep your backleg straight and out.', 'Keep your top arm pointing towards the sky.'], 
                          'plank': ['Steady your arms and keep your body straight.'], 
                          'tree': ['Secure your foot on your knee.', 'Palms together near your heart.'], 
                          'triangle': ['Make a triangle with your left arm and left leg.'],
                          'warrior 1': ['Plant your feet firmly on the ground.'],
                          'warrior 2': ['Keep your arms straight across.'],
                          'warrior 3': ['Point forward with your arms like superman.', 'Kick your backleg straight.']}

    def getPose(self):
        #run inference
        ret_val, self.image = self.cam.read()
        img = cv2.resize(self.image, (self.motion.inputDim, self.motion.inputDim), 3)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        self.motion.ft_ext(self.image)
        return self.motion.pose()

    def sayPhrase(self, phrase, tm=0.7):
        for p in phrase:
            subprocess.call(['flite', '-voice', 'slt', '-t', p])
            time.sleep(tm)

    def formatPose(self, pose):
        return pose.replace('_', ' ').replace('pose', '').strip()
    
    def run(self, S=7):
        start_idx = 0
        self.sayPhrase(self.intro_phrase)
        time.sleep(S)
        if self.current_pose:
            start_idx = self.flow.index(self.current_pose)
        for i in range(start_idx, len(self.flow)):
            # Get next pose and give instruction
            pose_name = self.formatPose(self.flow[i]) 
            phrase = "Lets go into %s pose" % pose_name
            self.sayPhrase([phrase])
            time.sleep(S)

            # Check user pose
            pose = self.getPose()
            self.current_pose = self.formatPose(pose)
            if self.current_pose != pose_name:
                corr_phrase = "I think you are in %s pose. Try to get into %s pose." % (self.current_pose, pose_name)
                corr_phrase = [corr_phrase]+self.flow_dict[pose_name]
                self.sayPhrase(corr_phrase)
                time.sleep(S)
                self.current_pose = self.formatPose(self.getPose())
            else:
                # Label and store new training sample
                dt = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") 
                flname = './data/yoga/%s/%s.jpg' % (pose, dt)
                cv2.imwrite(flname, self.image)


        end_phrase = ['Great job today!', 'You are becoming an excellent yogi.']
        self.sayPhrase(end_phrase)
