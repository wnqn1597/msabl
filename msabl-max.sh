for (( i=0; i<=4; i++ ))
do
#  python main.py \
#    --task mnist-max \
#    --pretrain_epochs 20 \
#    --epochs 15 \
#    --first_level_epochs 5 \
#    --label_smoothing 0.0 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --use_level \
#    --loop_list 1 2 3 8 \
#    --nbpt_list 1 1 1 2 \
#    --seed $i
#
#  python main.py \
#    --task mnist-max \
#    --pretrain_epochs 20 \
#    --epochs 15 \
#    --first_level_epochs 5 \
#    --label_smoothing 0.0 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --use_level \
#    --use_rejection \
#    --loop_list 1 2 3 8 \
#    --nbpt_list 1 1 1 2 \
#    --seed $i
#
#  python main.py \
#    --task fashion-max \
#    --pretrain_epochs 20 \
#    --epochs 15 \
#    --first_level_epochs 5 \
#    --label_smoothing 0.0 \
#    --train_batch_size 256 \
#    --eval_batch_size 256 \
#    --use_level \
#    --loop_list 1 3 5 16 \
#    --nbpt_list 1 1 1 2 \
#    --seed $i
#
#  python main.py \
#    --task fashion-max \
#    --pretrain_epochs 20 \
#    --epochs 15 \
#    --first_level_epochs 5 \
#    --label_smoothing 0.0 \
#    --train_batch_size 256 \
#    --eval_batch_size 256 \
#    --use_level \
#    --use_rejection \
#    --reweight \
#    --loop_list 1 3 5 16 \
#    --nbpt_list 1 1 1 2 \
#    --seed $i

  python main.py \
    --task cifar-max \
    --pretrain_epochs 20 \
    --epochs 15 \
    --first_level_epochs 5 \
    --label_smoothing 0.0 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_workers 4 \
    --use_level \
    --loop_list 1 3 5 16 \
    --nbpt_list 1 2 4 8 \
    --seed $i

  python main.py \
    --task cifar-max \
    --pretrain_epochs 20 \
    --epochs 15 \
    --first_level_epochs 5 \
    --label_smoothing 0.0 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_workers 4 \
    --use_level \
    --use_rejection \
    --loop_list 1 3 5 16 \
    --nbpt_list 1 2 4 8 \
    --seed $i
done
