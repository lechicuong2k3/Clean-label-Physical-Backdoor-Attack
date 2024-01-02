import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
src_dir = "/vinserver_user/21thinh.dd/FedBackdoor/source/dataset/facial_recognition_triggers"
trigger_lst = os.listdir(src_dir)
trigger_lst.remove('additional_image')
trigger_lst.remove('real_beard+red_hat')
trigger_lst.remove('clean_image')
labels = os.listdir(os.path.join(src_dir, 'clean_image', 'train'))
clean_data = [len(os.listdir(os.path.join(src_dir, 'clean_image', 'train', label))) for label in labels]

colors = ['green', 'blue', 'purple', 'brown', 'teal', 'magenta', 'red', 'cyan']
widths = [1, 1]
heights = [2, 1, 1, 1, 1, 1, 1, 1, 1]

n_rows = len(trigger_lst) // 2 + 1
fig = plt.figure(figsize=(30,40))
fig.tight_layout()
gs = fig.add_gridspec(nrows=n_rows, ncols=2, width_ratios=widths, height_ratios=heights, wspace=0.15, hspace=0.5)
ax = fig.add_subplot(gs[0, :]) 
# ipdb.set_trace()
ax.bar(labels, clean_data, color=colors)
ax.set_title("Clean Image")

for i, trigger in enumerate(trigger_lst):
    data = [len(os.listdir(os.path.join(src_dir, trigger, label))) for label in labels]
    if (i + 1) % 2 == 0:
        ax = fig.add_subplot(gs[(i+2) // 2, 0])
    else:
        ax = fig.add_subplot(gs[(i+2) // 2, 1])
    # ipdb.set_trace()
    ax.bar(labels, data, color=colors)
    ax.set_title(trigger)

plt.savefig('/vinserver_user/21thinh.dd/FedBackdoor/source/plots', bbox_inches='tight')
