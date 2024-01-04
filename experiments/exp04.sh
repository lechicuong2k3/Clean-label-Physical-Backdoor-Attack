export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py --recipe=naive --poisonkey=0-1 --alpha=0.0 --beta=0.05 
wait
python main.py --recipe=naive --poisonkey=0-4 --alpha=0.0 --beta=0.05 
wait
python main.py --recipe=naive --poisonkey=0-5 --alpha=0.0 --beta=0.05 