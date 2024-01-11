# Additional experiments

python main.py --trigger=sunglasses --recipe=gradient-matching --poisonkey=6-3 --alpha=0.1 --beta=0.0 --scenario=finetuning --threatmodel=clean-single-source &
python main.py --trigger=sunglasses --recipe=hidden-trigger --poisonkey=6-3 --alpha=0.1 --beta=0.0 --scenario=finetuning --threatmodel=clean-single-source 


python main.py --trigger=red_headband --recipe=gradient-matching --poisonkey=7-4 --alpha=0.1 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source &
python main.py --trigger=red_headband --recipe=hidden-trigger --poisonkey=7-4 --alpha=0.1 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 

python main.py --trigger=real_beard --recipe=gradient-matching --poisonkey=1-0 --alpha=0.1 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source &
python main.py --trigger=real_beard --recipe=hidden-trigger --poisonkey=1-0 --alpha=0.1 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 