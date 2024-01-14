# Experiment for trigger selection

# Naive method
python main.py --trigger=real_beard --recipe=naive --poisonkey=0-1 --alpha=0.0 --beta=0.2 --scenario=finetuning  --threatmodel=clean-multi-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=sunglasses --recipe=naive --poisonkey=0-1 --alpha=0.0 --beta=0.2 --scenario=finetuning  --threatmodel=clean-multi-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=red_headband --recipe=naive --poisonkey=0-1 --alpha=0.0 --beta=0.2 --scenario=finetuning  --threatmodel=clean-multi-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=yellow_sticker --recipe=naive --poisonkey=0-1 --alpha=0.0 --beta=0.2 --scenario=finetuning  --threatmodel=clean-multi-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=yellow_hat --recipe=naive --poisonkey=0-1 --alpha=0.0 --beta=0.2 --scenario=finetuning  --threatmodel=clean-multi-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=white_face_mask --recipe=naive --poisonkey=0-1 --alpha=0.0 --beta=0.2 --scenario=finetuning  --threatmodel=clean-multi-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=white_earings --recipe=naive --poisonkey=0-1 --alpha=0.0 --beta=0.2 --scenario=finetuning  --threatmodel=clean-multi-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection

# poisonkey 0-1
python main.py --trigger=real_beard --recipe=gradient-matching --poisonkey=0-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=sunglasses --recipe=gradient-matching --poisonkey=0-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=red_headband --recipe=gradient-matching --poisonkey=0-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=yellow_sticker --recipe=gradient-matching --poisonkey=0-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=yellow_hat --recipe=gradient-matching --poisonkey=0-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=white_face_mask --recipe=gradient-matching --poisonkey=0-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=white_earings --recipe=gradient-matching --poisonkey=0-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection

python main.py --trigger=real_beard --recipe=hidden-trigger --poisonkey=0-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=sunglasses --recipe=hidden-trigger --poisonkey=0-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=red_headband --recipe=hidden-trigger --poisonkey=0-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=yellow_sticker --recipe=hidden-trigger --poisonkey=0-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=yellow_hat --recipe=hidden-trigger --poisonkey=0-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=white_face_mask --recipe=hidden-trigger --poisonkey=0-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=white_earings --recipe=hidden-trigger --poisonkey=0-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection

# poisonkey 7-1
python main.py --trigger=real_beard --recipe=gradient-matching --poisonkey=7-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=sunglasses --recipe=gradient-matching --poisonkey=7-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=red_headband --recipe=gradient-matching --poisonkey=7-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=yellow_sticker --recipe=gradient-matching --poisonkey=7-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=yellow_hat --recipe=gradient-matching --poisonkey=7-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=white_face_mask --recipe=gradient-matching --poisonkey=7-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=white_earings --recipe=gradient-matching --poisonkey=7-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection

python main.py --trigger=real_beard --recipe=hidden-trigger --poisonkey=7-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=sunglasses --recipe=hidden-trigger --poisonkey=7-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=red_headband --recipe=hidden-trigger --poisonkey=7-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=yellow_sticker --recipe=hidden-trigger --poisonkey=7-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=yellow_hat --recipe=hidden-trigger --poisonkey=7-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=white_face_mask --recipe=hidden-trigger --poisonkey=7-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=white_earings --recipe=hidden-trigger --poisonkey=7-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection

# poisonkey 9-1
python main.py --trigger=real_beard --recipe=gradient-matching --poisonkey=9-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=sunglasses --recipe=gradient-matching --poisonkey=9-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=red_headband --recipe=gradient-matching --poisonkey=9-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=yellow_sticker --recipe=gradient-matching --poisonkey=9-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=yellow_hat --recipe=gradient-matching --poisonkey=9-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=white_face_mask --recipe=gradient-matching --poisonkey=9-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=white_earings --recipe=gradient-matching --poisonkey=9-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection

python main.py --trigger=real_beard --recipe=hidden-trigger --poisonkey=9-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=sunglasses --recipe=hidden-trigger --poisonkey=9-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=red_headband --recipe=hidden-trigger --poisonkey=9-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=yellow_sticker --recipe=hidden-trigger --poisonkey=9-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=yellow_hat --recipe=hidden-trigger --poisonkey=9-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=white_face_mask --recipe=hidden-trigger --poisonkey=9-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection
python main.py --trigger=white_earings --recipe=hidden-trigger --poisonkey=9-1 --alpha=0.2 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source --exp_name=trigger_selection --vruns=1 --deterministic --exp_name=trigger_selection

