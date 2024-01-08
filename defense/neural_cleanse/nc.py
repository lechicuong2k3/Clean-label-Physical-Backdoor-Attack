import sys

import numpy as np
from config import get_argument
from detection import *


sys.path.insert(0, "../..")


def outlier_detection(l1_norm_list, idx_mapping, opt):
    print("-" * 30)
    print("Determining whether model is backdoor")
    consistency_constant = 1.4826
    median = torch.median(l1_norm_list)
    mad = consistency_constant * torch.median(torch.abs(l1_norm_list - median))
    min_mad = torch.abs(torch.min(l1_norm_list) - median) / mad

    print("Median: {}, MAD: {}".format(median, mad))
    print("Anomaly index: {}".format(min_mad))

    if min_mad < 2:
        result = f"Model is clean. Anomaly Index: {min_mad}"
    else:
        result = f"Model is backdoored! Anomaly Index: {min_mad}"
    print(result)
    if opt.to_file:
        result_path = os.path.join(opt.result, opt.defense_set, opt.attack_mode)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        output_path = os.path.join(result_path, "{}.txt".format(opt.trigger))
        with open(output_path, "a+") as f:
            f.write(
                "Median: " + str(median.cpu().numpy()) + ", " + "MAD: " + str(mad.cpu().numpy()) + ", " "Min_MAD: "+ str(min_mad.cpu().numpy()) + "\n"
            )
            l1_norm_list_to_save = [str(value) for value in l1_norm_list.cpu().numpy()]
            f.write(", ".join(l1_norm_list_to_save) + "\n")
            f.write(result + "\n")

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if torch.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print(
        "Flagged label list: {}".format(",".join(["{}: {}".format(y_label, l_norm) for y_label, l_norm in flag_list]))
    )


def main():
    opt = config.get_argument().parse_args()
    trigger = 'sunglasses'
    result_path = os.path.join(opt.result, opt.defense_set, opt.attack_mode)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output_path = os.path.join(result_path, "{}.txt".format(trigger))
    if opt.to_file:
        with open(output_path, "w+") as f:
            f.write("Output for neural cleanse: {} - {}".format(opt.attack_mode, opt.defense_set) + "\n")

    # init_mask = np.random.randn(1, opt.input_height, opt.input_width).astype(np.float32)
    # init_pattern = np.random.randn(opt.input_channel, opt.input_height, opt.input_width).astype(np.float32)

    init_mask = np.ones((1, opt.input_height, opt.input_width)).astype(np.float32)
    init_pattern = np.ones((opt.input_channel, opt.input_height, opt.input_width)).astype(np.float32)
    for test in range(opt.n_times_test):
        print("Test {}:".format(test))
        if opt.to_file:
            with open(output_path, "a+") as f:
                f.write("-" * 30 + "\n")
                f.write("Test {}:".format(str(test)) + "\n")

        masks = []
        idx_mapping = {}

        for target_label in range(opt.total_label):
            print("----------------- Analyzing label: {} -----------------".format(target_label))
            opt.target_label = target_label
            recorder, opt = train(opt, init_mask, init_pattern)

            mask = recorder.mask_best
            masks.append(mask)
            idx_mapping[target_label] = len(masks) - 1

        l1_norm_list = torch.stack([torch.sum(torch.abs(m)) for m in masks])
        print("{} labels found".format(len(l1_norm_list)))
        print("Norm values: {}".format(l1_norm_list))
        outlier_detection(l1_norm_list, idx_mapping, opt)


if __name__ == "__main__":
    main()