import subprocess


for algo in ['q_learning','double_q_learning','truncated_double_q']:
    # env_kwargs = "number_arms:50 mean:-0.1 variance:1 strict_mask:True"
    command = [
        "python", "train.py",
        "--algo", algo,
        "--env", "ConstructedMaxBias",
        "--eval-freq", "1",
        "--eval-episodes", "1",
        "--env-kwargs", "number_arms:50", "mean:-0.1", "variance:1", "strict_mask:True",
        "--hyperparams", "robbins_monro:True",
        "-n","2000",
        "-f","logs_exp/exp1",
        "-tb","monitor_tb/exp1",
        "--verbose","0"
    ]

    # C_upper and Clower = 30 for truncated Q, epsilon greedy behaviour policy mit eps=0.1

    for i in range(1000):
        subprocess.run(command)
        print(f'run {i+1} done')