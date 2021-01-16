import matplotlib.pyplot as plt
import os

bg_col = ['bisque', 'lightgreen', 'mediumspringgreen', 'lightcyan', 'lavender']
ln_col = ['orangered', 'blueviolet', 'fuchsia', 'crimson', 'indianred']

for txt in os.listdir('./'):
    method, dataset, cut_idx = txt[:-4].split('_')
    f = open('./'+txt, 'r')
    acc_results = {}
    for line in f.readlines():
        task, iteration, loss, device, test_task, acc = map(int, line[:-1].split('\t'))
        if task not in acc_results:
            acc_results[task] = ([], [])
        if device == test_task:
            acc_results[test_task][0].append(task*100+iteration)
            acc_results[test_task][1].append(acc*100)
    for key, value in acc_results.items():
        plt.plot(value[0], value[1], label='Task {}'.format(key), color=ln_col[key])
        plt.axvspan(key*100, (key+1)*100, facecolor=bg_col[key], alpha=0.5)
    plt.legend()
    plt.show()