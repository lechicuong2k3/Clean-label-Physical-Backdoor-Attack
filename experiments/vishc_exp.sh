
# poisonkey 9-1
python main.py --trigger=real_beard --recipe=gradient-matching --poisonkey=9-1 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=1 --deterministic --exp_name=trigger_selection --load_trained_model
python main.py --trigger=red_headband --recipe=gradient-matching --poisonkey=9-1 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source --vruns=1 --deterministic --exp_name=trigger_selection --load_trained_model
python main.py --trigger=white_earings --recipe=gradient-matching --poisonkey=9-1 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=1 --deterministic --exp_name=trigger_selection --load_trained_model

python main.py --trigger=real_beard --recipe=hidden-trigger --poisonkey=9-1 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source --vruns=1 --deterministic --exp_name=trigger_selection --load_trained_model
python main.py --trigger=red_headband --recipe=hidden-trigger --poisonkey=9-1 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=1 --deterministic --exp_name=trigger_selection --load_trained_model
python main.py --trigger=white_earings --recipe=hidden-trigger --poisonkey=9-1 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=1 --deterministic --exp_name=trigger_selection --load_trained_model