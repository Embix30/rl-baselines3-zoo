import subprocess

command = [
    "python", "train.py",
    "--algo", "q_learning",
    "--env", "ConstructedMaxBias",
    "--eval-freq", "1",
    "--eval-episodes", "1",
    "--env-kwargs", "number_arms:20",
    "-n","1000"
]

for i in range(100):
    subprocess.run(command)
    print(f'run {i+1} done')