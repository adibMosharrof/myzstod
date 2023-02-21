import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def line_plot():

    data =  [
                [
                    52.09,
                    49.45,
                    45.68,
                    44.31,
                    43.78,
                    43.7,
                ], 
                [
                    67.05,
                    60.22,
                    56.15,
                    54.2,
                    53.51,
                    53.49,
                ], 
                [
                    78.85,
                    76.03,
                    73.73,
                    72.57,
                    72.28,
                    72.38,
                ]
            ]
    labels  = ["SimpleToD", "SimpleToD\n w/ Schema", "ZS-ToD (this work)"]
    x_labels = ["(0,4]","(4-8]","(8-12]","(12-16]","(16-20]","(20-26]"]
    colors = ["#3B498E", "#277DA1", "#43AA8B"]

    plt.style.use("bmh")
    fig, ax = plt.subplots(dpi=140)
    plt.xlabel("Dialog Turns")
    plt.ylabel("Average Goal Accuracy (%)")
    for d , l, c in zip(data, labels, colors):
        ax.plot(x_labels, d, label=l, marker='s', color=c)
    legends = plt.legend()
    ax.set_ylim([40,85])
    for i,text in enumerate(legends.get_texts()):
        text.set_fontsize(9)
        if not i == 2:
            continue
        text.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(base_dir+ 'dialog_turns.pdf', format='pdf',bbox_inches='tight')


def grouped_bar():
    models_names  = ["SimpleToD", "SimpleToD w/ Schema", "ZS-ToD (this work)"]
    legends = ['Two Step Training', 'w/o Two Step Training']
    colors = ["#43AA8B", "#277DA1"]
    x = np.arange(len(models_names))*0.5
    width = 0.15
    two_step = [47.20,58.03,73.18 ]
    no_two_step = [47.85, 53.49,63.54]

    plt.style.use("bmh")
    fig, ax = plt.subplots(dpi=140)
    ax.set_ylabel("Average Goal Accuracy (%)")
    ax.set_xlabel("Models")
    ax.bar(x- width/2, two_step, width, label=legends[0], color=colors[0])
    ax.bar(x + width/2, no_two_step, width, label=legends[1], color=colors[1])
    legends = ax.legend()

    ax.set_xticks(x, models_names)
    for i,text in enumerate(ax.get_xticklabels()):
        if not i == 2:
            continue
        text.set_fontweight('bold')
    fig.tight_layout()
    plt.savefig(base_dir+ 'two_step_training.pdf', format='pdf',bbox_inches='tight')

def grouped_bar_with_sd_error_bars():
    metric_names = ["Intent Acc",  "Avg Goal Acc", "Joint Goal Acc", "Req Slot F1","Inform","Success", "Avg Action Acc","Joint Action Acc", "Response GLEU", "Combined"]    
    our_mean = np.array([55.63,50.23,29.46,89.67,38.90,26.47,38.03,33.60,15.60,48.29])
    our_sd = np.array([7.7,3.84,1.31,0.29,2.9,2.62,2.12,1.37,0.91,3.53])
    stod_mean = np.array([23.37,19.99,10.75,88.23,28.76,16.21,28.21,24.19,14.14,36.62])
    stod_sd = np.array([6.03,4.29,2.89,1.42,7.7,6.79,5.2,4.4,3.39,10.14])

    plt.style.use("bmh")
    fig, ax = plt.subplots(dpi=144)
    # ax.plot(metric_names, our_mean, marker='s', color="#43AA8B", label="Ours")
    # ax.fill_between(metric_names, our_mean - our_sd, our_mean + our_sd, alpha=0.2, color="#43AA8B")
    # ax.plot(metric_names, stod_mean, marker='s', color="#277DA1", label="SimpleTOD w/ Schema")
    # ax.fill_between(metric_names, stod_mean - stod_sd, stod_mean + stod_sd, alpha=0.2, color="#277DA1")
    width = 0.25
    x = np.arange(len(metric_names))*0.75
    ax.bar(x-width/2, our_mean, width, yerr=our_sd , color="#43AA8B", capsize=3, align='center', alpha=1, label="ZS-ToD (this work)", error_kw=dict(lw=1, capsize=2, capthick=1))
    # ax.errorbar(x-width/2, our_mean , width, yerr=our_sd.tolist(), alpha=0.2, color="#43AA8B")
    ax.bar(x+width/2, stod_mean, width, yerr=stod_sd, color="#277DA1", capsize=3, alpha=1, align='center', label="SimpleToD w/ Schema",error_kw=dict(lw=1, capsize=2, capthick=1))
    # ax.errorbar(x-width/2, stod_mean , yerr=stod_sd, alpha=0.2, color="#277DA1")
    ax.set_ylabel('Scores')
    ax.set_xlabel('Metrics')
    ax.plot()
    legends = ax.legend()
    for i,text in enumerate(legends.get_texts()):
        if not i == 0:
            continue
        text.set_fontweight('bold')
    plt.xticks(rotation=25)
    ax.set_xticks(x, metric_names)

    fig.tight_layout()
    plt.savefig(base_dir+ 'sgdx_results.pdf', format='pdf',bbox_inches='tight')



base_dir = "/u/amo-d0/grad/adibm/data/projects/ZSToD/data_exploration/plots/"

if __name__ == "__main__" :
    line_plot()
    grouped_bar()
    grouped_bar_with_sd_error_bars()



