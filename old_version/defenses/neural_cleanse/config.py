import argparse

def list_of_str(values):
    if isinstance(values, list):
        return values
    return values.split(',')

def get_argument():
    parser = argparse.ArgumentParser()

    # Directory option
    parser.add_argument("--checkpoints", type=str, default="../../checkpoints/")
    parser.add_argument("--data_root", type=str, default="../../dataset/facial_recognition_rescale_split")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--result", type=str, default="./results")
    parser.add_argument("--defense_set", type=str, default="testset")
    parser.add_argument("--attack_mode", type=str, default="all2one")
    parser.add_argument("--scenario", type=str, default="random-poison")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--triggers", type=list_of_str, default=["sunglasses", "real_beard", "fake_beard", "white_earings", "black_face_mask", "red_hat", "big_sticker", "blue_sticker"])

    # ---------------------------- For Neural Cleanse --------------------------
    # Model hyperparameters
    parser.add_argument("--model", type=str, default='ResNet50')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--input_height", type=int, default=224)
    parser.add_argument("--input_width", type=int, default=224)
    parser.add_argument("--input_channel", type=int, default=3)
    parser.add_argument("--init_cost", type=float, default=1e-3)
    parser.add_argument("--atk_succ_threshold", type=float, default=99.0)
    parser.add_argument("--early_stop", type=bool, default=True)
    parser.add_argument("--early_stop_threshold", type=float, default=99.0)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cost_multiplier", type=float, default=2)
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--num_workers", type=int, default=8)
    
    # LR scheduler
    parser.add_argument("--lr_decay_step", type=int, default=5)
    parser.add_argument("--lr_decay_factor", type=float, default=0.95)

    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--total_label", type=int, default=8)
    parser.add_argument("--EPSILON", type=float, default=1e-7)

    parser.add_argument("--to_file", type=bool, default=True)
    parser.add_argument("--n_times_test", type=int, default=1)

    return parser
