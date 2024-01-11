# Poison samples with trigger
python main.py --alpha=0.2 --beta=0.0 --exp_name=clean_poison --deterministic --vruns=3
python main.py --alpha=0.0 --beta=0.2 --poison_triggered_sample --exp_name=trigger_poison --deterministic --vruns=3
# python main.py --recipe=naive --threatmodel=clean-multi-source --alpha=0.0 --beta=0.2 --exp_name=no_poison --deterministic --vruns=3 

# python main.py --alpha=0.1 --beta=0.1 --exp_name=clean_poison --deterministic --vruns=3
# python main.py --alpha=0.1 --beta=0.1 --poison_triggered_sample --exp_name=hybrid_poison --deterministic --vruns=3