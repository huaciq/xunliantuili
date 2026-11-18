"""Remove a single SDK container if its conversion owner exits or is killed."""
from pathlib import Path
import subprocess
import sys
import time

pid, identity, name = sys.argv[1:]
stat = Path('/proc') / pid / 'stat'
while True:
    try:
        current = stat.read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        current = None
    if current != identity:
        subprocess.run(['sudo', '-n', 'docker', 'rm', '-f', name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        break
    time.sleep(1)
