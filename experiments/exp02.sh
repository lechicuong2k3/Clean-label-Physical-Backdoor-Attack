# Experiment 1: HTBA for one-to-one attack for real_beard trigger and target class 1 
# We use signAdam for comparison
CUDA_VISIBLE_DEVICES=0,1
python main.py --poisonkey=0-1 --recipe=hidden-trigger --attackoptim=signAdam --trigger=real_beard --alpha=0.4 --beta=0.1 & 
python main.py --poisonkey=0-1 --recipe=hidden-trigger --attackoptim=signAdam --trigger=real_beard --alpha=0.3 --beta=0.2 & 
wait
python main.py --poisonkey=2-1 --recipe=hidden-trigger --attackoptim=signAdam --trigger=real_beard --alpha=0.4 --beta=0.1 & 
python main.py --poisonkey=3-1 --recipe=hidden-trigger --attackoptim=signAdam --trigger=real_beard --alpha=0.4 --beta=0.1 & 
wait
python main.py --poisonkey=4-1 --recipe=hidden-trigger --attackoptim=signAdam --trigger=real_beard --alpha=0.4 --beta=0.1 & 
python main.py --poisonkey=5-1 --recipe=hidden-trigger --attackoptim=signAdam --trigger=real_beard --alpha=0.4 --beta=0.1 & 
wait
python main.py --poisonkey=6-1 --recipe=hidden-trigger --attackoptim=signAdam --trigger=real_beard --alpha=0.4 --beta=0.1 & 
python main.py --poisonkey=7-1 --recipe=hidden-trigger --attackoptim=signAdam --trigger=real_beard --alpha=0.4 --beta=0.1 & 