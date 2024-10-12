import json
import matplotlib.pyplot as plt

# 设置全局字体大小
plt.rcParams.update({'font.size': 18})

proj_list = ["Accuracy", "Cohesion", "Coupling", "KRR"]
ylim_list = [[0.4, 1.0], [0.3, 0.8], [0.2, 0.6], [0.2, 0.7]]

for proj, ylim_each in zip(proj_list, ylim_list):

    file_names = [
        f'./{proj}/lr_0.05_0.05_a1.4_t0.1_bs_128.json',
        f'./{proj}/lr_0.05_0.05_a1.4_t0.2_bs_128.json',
        f'./{proj}/lr_0.05_0.05_a1.4_t0.3_bs_128.json',
        f'./{proj}/lr_0.05_0.05_a1.4_t0.4_bs_128.json',
        f'./{proj}/lr_0.05_0.05_a1.4_t0.5_bs_128.json',
    ]

    colors = ['#4878D0', '#EE854A', '#6ACC64', '#D65F5F', '#956CB4']
    labels = ['temperature=0.1', 'temperature=0.2', 'temperature=0.3', 'temperature=0.4', 'temperature=0.5']

    plt.figure(figsize=(6, 5))

    for i, file_name in enumerate(file_names):
        with open(file_name, 'r') as file:
            data = json.load(file)

        epochs = [entry[1] for entry in data]
        accuracies = [entry[2] for entry in data]

        plt.plot(epochs, accuracies, color=colors[i], label=labels[i], linewidth=2)

    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel(f'{proj}', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=16, loc='best')

    plt.xticks(range(0, max(max(epochs) for file_name in file_names)+1, 20), fontsize=12)
    plt.yticks(fontsize=12)

    plt.ylim(ylim_each[0], ylim_each[1])

    plt.tight_layout()
    plt.savefig(f"./{proj}.png", dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，避免内存泄漏