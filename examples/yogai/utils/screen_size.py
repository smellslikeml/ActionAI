import subprocess
 
cmd = ['xrandr']
cmd2 = ['grep', '*']

def getScreenDims():
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(cmd2, stdin=p.stdout, stdout=subprocess.PIPE)
    p.stdout.close()

    resolution_string, junk = p2.communicate()
    resolution = resolution_string.split()[0].decode('utf8')
    return list(map(int, resolution.split('x')))

if __name__ == '__main__':
    print(getScreenDims())
