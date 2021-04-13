import sys
import subprocess
import os

runScript = sys.argv[1]
cw_summary_file = sys.argv[2]
summary_folder = sys.argv[3]
args = sys.argv[4:]

upperEps = float(open(cw_summary_file).read().split()[1])

eps_high = upperEps

while eps_high > 0.00001:
    eps_high -= 0.02 * upperEps
    currentEps = eps_high
    summary_file = summary_folder + "/{}.summary".format(currentEps)
    command = "{} {} --epsilon {} --summary-file {}".format(runScript, " ".join(args), currentEps, summary_file).split()
    if not os.path.isfile(summary_file):
        subprocess.run(command)
    assert(os.path.isfile(summary_file))
    result = open(summary_file, 'r').readlines()[0].split()[0]
    if (result == 'sat'):
        continue
    else:
        assert(result == "unsat")
        exit(0)
