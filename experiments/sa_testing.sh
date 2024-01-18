python main.py --trigger=sunglasses --recipe=gradient-matching --poisonkey=0-1 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=3 --deterministic --exp_name=sa_bdfinetune_update --backdoor_finetuning --target_train_rate=0.5 --load_trained_model
python main.py --trigger=sunglasses --recipe=gradient-matching --poisonkey=0-1 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=3 --deterministic --exp_name=sa_update --target_train_rate=0.5 --load_trained_model
python main.py --trigger=sunglasses --recipe=gradient-matching --poisonkey=0-1 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=3 --deterministic --exp_name=sa_bdfinetune --backdoor_finetuning --load_trained_model
python main.py --trigger=sunglasses --recipe=gradient-matching --poisonkey=0-1 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=3 --deterministic --exp_name=sa --load_trained_model

python main.py --trigger=red_headband --recipe=gradient-matching --poisonkey=8-5 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=3 --deterministic --exp_name=sa_bdfinetune_update --backdoor_finetuning --target_train_rate=0.5 --load_trained_model
python main.py --trigger=red_headband --recipe=gradient-matching --poisonkey=8-5 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=3 --deterministic --exp_name=sa_update --target_train_rate=0.5 --load_trained_model
python main.py --trigger=red_headband --recipe=gradient-matching --poisonkey=8-5 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=3 --deterministic --exp_name=sa_bdfinetune --backdoor_finetuning --load_trained_model
python main.py --trigger=red_headband --recipe=gradient-matching --poisonkey=8-5 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=3 --deterministic --exp_name=sa --load_trained_model

python main.py --trigger=white_facemask --recipe=gradient-matching --poisonkey=8-0 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=3 --deterministic --exp_name=sa_bdfinetune_update --backdoor_finetuning --target_train_rate=0.5 --load_trained_model
python main.py --trigger=white_facemask --recipe=gradient-matching --poisonkey=8-0 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=3 --deterministic --exp_name=sa_update --target_train_rate=0.5 --load_trained_model
python main.py --trigger=white_facemask --recipe=gradient-matching --poisonkey=8-0 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=3 --deterministic --exp_name=sa_bdfinetune --backdoor_finetuning --load_trained_model
python main.py --trigger=white_facemask --recipe=gradient-matching --poisonkey=8-0 --alpha=0.0 --beta=0.2 --poison_triggered_sample --scenario=finetuning  --threatmodel=clean-single-source  --vruns=3 --deterministic --exp_name=sa --load_trained_model