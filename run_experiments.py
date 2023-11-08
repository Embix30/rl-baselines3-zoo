import subprocess

command = [
    "python", "train.py",
    "--algo", "q_learning",
    "--env", "FrozenLake-v1",
    "--hyperparams", "n_timesteps:30000",
    "--env-kwargs", "is_slippery:False",
    "--eval-freq", "50",
    "--eval-episodes", "1"
]

for i in range(10):
    subprocess.run(command)
    print(f'run {i+1} done')