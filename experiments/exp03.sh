# Test naive attack with target 1 and real_beard trigger
CUDA_VISIBLE_DEVICES=0,1
python main.py --recipe=naive --poisonkey=0-1 --trigger=real_beard --alpha=0.0 --beta=0.5 
python main.py --recipe=naive --poisonkey=2-1 --trigger=real_beard --alpha=0.0 --beta=0.5
wait
python main.py --recipe=naive --poisonkey=3-1 --trigger=real_beard --alpha=0.0 --beta=0.5
python main.py --recipe=naive --poisonkey=4-1 --trigger=real_beard --alpha=0.0 --beta=0.5
wait
python main.py --recipe=naive --poisonkey=5-1 --trigger=real_beard --alpha=0.0 --beta=0.5
python main.py --recipe=naive --poisonkey=6-1 --trigger=real_beard --alpha=0.0 --beta=0.5
wait
python main.py --recipe=naive --poisonkey=7-1 --trigger=real_beard --alpha=0.0 --beta=0.5