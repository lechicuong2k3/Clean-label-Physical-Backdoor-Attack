python main.py --trigger=sunglasses --recipe=gradient-matching --poisonkey=9-1 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=3 --deterministic --exp_name=trigger_selection --load_trained_model --source_gradient_batch=64
python main.py --trigger=yellow_sticker --recipe=gradient-matching --poisonkey=9-1 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=3 --deterministic --exp_name=trigger_selection --load_trained_model --source_gradient_batch=64

