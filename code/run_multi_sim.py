
from subprocess import Popen
import sys

num_trials = 20
offset = 20

prcs = []
max_prcs_count = 4
# script = 'run_reaching.py'
script = 'run_adaptation_iCub.py'

try:
    idx = offset
    while(idx < num_trials+offset):
        if len(prcs) < max_prcs_count:
            prcs.append(Popen(['python3', script, str(idx)]))
            idx += 1
        else:
            ret = prcs[0].wait()
            if type(ret) == int:
                prcs.pop(0)

    for process in prcs:
        process.wait()

except KeyboardInterrupt:
    print("interrupted")
    for process in prcs:
        process.terminate()
