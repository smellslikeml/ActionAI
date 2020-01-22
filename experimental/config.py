#w = 1024
#h = 768
w = 2448
h = 2048
fps = 30
window = 3
input_size = (224, 224)
log = True
video = True
display = False
learning_rate = 1e-4

body_dict = {0:'nose', 1: 'lEye', 2: 'rEye', 3:'lEar', 4:'rEar', 5:'lShoulder', 6:'rShoulder', 
               7:'lElbow', 8:'rElbow', 9:'lWrist', 10:'rWrist', 11:'lHip', 12:'rHip', 13:'lKnee', 14:'rKnee',
              15:'lAnkle', 16:'rAnkle', 17:'neck'}
body_idx = dict([[v,k] for k,v in body_dict.items()])
pose_vec_dim = 2 * len(body_dict)

button_list = ['circle', 'cross', 'square', 'triangle']
#activity_list = ['loiter', 'run', 'walk', 'wave']
activity_list = ['curl', 'extension', 'press', 'raise']
#activity_list = ['standing', 'throwing']

activity_idx = {idx : activity for idx, activity in enumerate(sorted(activity_list))}
activity_dict = {tup[0] : tup[1] for tup in zip(button_list, activity_list)}
idx_dict = {x:idx for idx,x in enumerate(sorted(activity_dict.values()))}


