
from subprocess import Popen
import sys

num_trials = 50

prcs = []
max_prcs_count = 25
try:
    idx = 0
    while(idx < num_trials):
        if len(prcs) < max_prcs_count:
            prcs.append(Popen(['python3', 'run_reaching.py', str(idx)]))
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
