import os
import time
import random
import subprocess

p1 = random.randint(10000,60000)
p2 = random.randint(10000,60000)
p3 = random.randint(10000,60000)

cmd1 = "printf \"%s\" | python3.5 mental_poker.py %d 127.0.0.1:%d 127.0.0.1:%d &"%("\\n"*3 + "s" + "\\n"*6,p1,p2,p3)
cmd2 = "printf \"%s\" | python3.5 mental_poker.py %d 127.0.0.1:%d 127.0.0.1:%d &"%("\\n"*2 + "f" + "\\n"*6,p2,p1,p3)
cmd3 = "printf \"%s\" | python3.5 mental_poker.py %d 127.0.0.1:%d 127.0.0.1:%d &"%("\\n"*10,p3,p2,p1)

f1 = subprocess.Popen(cmd1, shell=True)
time.sleep(2*random.random())
f2 = subprocess.Popen(cmd2, shell=True)
time.sleep(2*random.random())
f3 = subprocess.Popen(cmd3, shell=True)

f1.wait()
f2.wait()
f3.wait()

