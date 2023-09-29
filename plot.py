import matplotlib.pyplot as plt
import torch


def plot_graph(xs_td,
               xs_sqt,
               rewards_td,
               rewards_sqt,
               title, xlabel, name, policy="TD7", metric=torch.mean,
               fontsize=14):
    plt.rc('legend', fontsize=fontsize)
    fig, ax = plt.subplots()

    stderr_td = torch.std(rewards_td, dim=0) / torch.sqrt(torch.tensor(rewards_td.shape[0]))
    stderr_sqt = torch.std(rewards_sqt, dim=0) / torch.sqrt(torch.tensor(rewards_sqt.shape[0]))

    ax.plot(metric(xs_td, dim=0), metric(rewards_td, dim=0), label=f"{policy}", color="black")
    ax.fill_between(x=metric(xs_td, dim=0), y1=metric(rewards_td, dim=0) - stderr_td,
                    y2=metric(rewards_td, dim=0) + stderr_td,
                    color="black", alpha=.2)

    ax.plot(metric(xs_sqt, dim=0), metric(rewards_sqt, dim=0), label=f"{policy}+SQT", color="red")
    ax.fill_between(x=metric(xs_sqt, dim=0), y1=metric(rewards_sqt, dim=0) - stderr_sqt,
                    y2=metric(rewards_sqt, dim=0) + stderr_sqt, color="red", alpha=.2)

    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel("Rewards", fontsize=fontsize)
    plt.title(title, fontsize=fontsize)

    plt.legend()
    plt.savefig(f"./results/{name}.png")


def read_results(env_name, seed, policy="TD7", axis=1):
    with open(f"results/{env_name}/{policy}_{seed}", "r") as file:
        lines = file.readlines()
        steps_td = lines[axis]
        rewards_td = lines[2]

    with open(f"results/{env_name}/{policy}+SQT_{seed}", "r") as file:
        lines = file.readlines()
        steps_sqt = lines[axis]
        rewards_sqt = lines[2]


    xs_td = steps_td[1:-2].split(", ")
    xs_sqt = steps_sqt[1:-2].split(", ")

    rewards_td = rewards_td[1:-2].split(", ")
    rewards_sqt = rewards_sqt[1:-2].split(", ")

    xs_td = [float(r) for r in xs_td]
    xs_sqt = [float(r) for r in xs_sqt]

    rewards_td = [float(r) for r in rewards_td]
    rewards_sqt = [float(r) for r in rewards_sqt]

    return xs_td, xs_sqt, rewards_td, rewards_sqt


time = False
table = False

policies = [
        "TD3",
        "TD7",
        "MaxMin",
        "REDQ"
    ]
env_names = [
    "Humanoid-v2",
    "Walker2d-v2",
    "Ant-v2",
    "HalfCheetah-v2",
    "Hopper-v2",
    "HumanoidStandup-v2",
    "Swimmer-v2",
    "walker2d-medium-v2",
    "ant-medium-v2",
    "halfcheetah-medium-v2",
    "hopper-medium-v2"
]

for env_name in env_names:

    for policy in policies:

        xlabel = "Time (m)" if time else "Timesteps"
        title = env_name.split("-")[0] + (f" (MuJoCo)" if env_name.split("-")[0][0].upper() == env_name.split("-")[0][0] else " (D4RL)")
        name = title.replace(" ", "_") + ("_Times" if time else "_Timesteps") + f"_{policy}"
        seeds = [0, 1, 2, 3, 4]

        total_points = 200

        xs_td, xs_sqt, rewards_td, rewards_sqt = [], [], [], []

        for seed in seeds:
            x_td, x_sqt, reward_td, reward_sqt = read_results(env_name=env_name, seed=seed, axis=0 if time else 1, policy=policy)

            xs_td.append(x_td[:total_points])
            xs_sqt.append(x_sqt[:total_points])

            rewards_td.append(reward_td[:total_points])
            rewards_sqt.append(reward_sqt[:total_points])

        if table:
            td = torch.tensor(rewards_td)
            sqt = torch.tensor(rewards_sqt)

            color_td = 'red' if td.mean(dim=0)[-1] > sqt.mean(dim=0)[-1] else 'black'
            color_sqt = 'black' if td.mean(dim=0)[-1] > sqt.mean(dim=0)[-1] else 'red'

            line = f"{title} & \\textcolor{{{color_td}}}{{{td.mean(dim=0)[-1]:,.1f}}} " \
                   f"\\textcolor{{gray}}{{$\pm$ {td.std(dim=0)[-1] / td.shape[0]:,.1f}}} & " \
                   f"\\textcolor{{{color_sqt}}}{{{sqt.mean(dim=0)[-1]:,.1f}}} " \
                   f"\\textcolor{{gray}}{{$\pm$ {sqt.std(dim=0)[-1] / sqt.shape[0]:,.1f}}} \\\\"
            print(line)

        else:
            plot_graph(xs_td=torch.tensor(xs_td),
                       rewards_td=torch.tensor(rewards_td),
                       xs_sqt=torch.tensor(xs_sqt),
                       rewards_sqt=torch.tensor(rewards_sqt),
                       title=title, xlabel=xlabel, name=name, policy=policy)