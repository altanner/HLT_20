import sys #! unused
import os #! unused
import re #! unused
import numpy as np #! unused

import json
import subprocess as sp
from tqdm import tqdm #! I can see where this goes


tf_kf_py_file = "tf_kalman_filter_2d.py"

#* eight batches, 2^24 - 2^16 #? why
batch_sizes = [2 ** x for x in range(16, 24)][::-1]

print(batch_sizes)

timingsGPU = {}
timingsGPUNoSkip = {}

for batch_size in batch_sizes:

    name = f"{batch_size}"

    #! this line used to refer to "kfTFVec.py", file absent.
    #! changed to the tf file, "tf_kalman_filter_2d.py" - is this right?
    commandGPU = f"python {tf_kf_py_file} -n {batch_size}"
    #! this file is absent
    commandGPUNoSkip = f"python kfTFVecNoSkip.py -n {batch_size}"

    sp.check_output(commandGPU, shell=True)

    tot = 0
    for i in range(10):

        resultGPU = sp.check_output(commandGPU, shell=True)
        resultGPU = float(str(resultGPU, "utf-8").strip("\n"))
        tot += resultGPU

    timingsGPU[name] = tot / 10.0

    commandGPU = f"python kfTFVec.py -n {batch_size}"

    sp.check_output(commandGPU, shell=True)

    tot = 0
    for i in range(10):

        resultGPU = sp.check_output(commandGPUNoSkip, shell=True)
        resultGPU = float(str(resultGPU, "utf-8").strip("\n"))
        tot += resultGPU

    timingsGPUNoSkip[name] = tot / 10.0


json.dump(timingsGPU, open("kf_bench_tf_skip_gpu.json", "w"))
json.dump(timingsGPUNoSkip, open("kf_bench_tf_noSkip_gpu.json", "w"))
