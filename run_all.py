#!/usr/bin/env python3
import subprocess,sys
subprocess.run([sys.executable,'run_memory.py'],check=True)
subprocess.run([sys.executable,'run_actions.py'],check=True)
print('Generated memory_answers.jsonl and actions_train.jsonl')
