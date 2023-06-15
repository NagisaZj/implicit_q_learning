#!/bin/bash
   # Script to reproduce results


   # 10 20 150 100
   # 30 400
 Foldername="logs"
 mkdir out_logs/${Foldername} &> /dev/null
 declare -a tasks=("antmaze-medium-diverse-v1"   ) #  "halfcheetah-medium-expert-v2" "antmaze-medium-play-v0""maze2d-umaze-dense-v1"
 declare -a seeds=( 7 8 9 )
 declare -a reward_model_paths=( "ensemble_walker2d-umaze-dense-v1_initial_pairs_5_num_queries_5_num_iter_20_retrain_num_iter_20_voi_greedy_seed_345_round_num_10.npy"   ) #"ensemble_antmaze-medium-play-v0_initial_pairs_50_num_queries_10_num_iter_20_retrain_num_iter_20_voi_dis_seed_5_round_num_10.npy"
 #"ensemble_maze2d-umaze-dense-v1_initial_pairs_1_num_queries_1_num_iter_20_retrain_num_iter_20_voi_myucb_seed_315_round_num_30.npy"
 #   "ensemble_maze2d-umaze-dense-v1_initial_pairs_5_num_queries_5_num_iter_20_retrain_num_iter_20_voi_dis_seed_355_round_num_10.npy"
# 41 baseline
# 51 61 71 550 300 150


#maze2d 8 9 baseline 20

# 12 oracle 13 dis

# 14-16 tau 0.3-0.7
# 17 walker med exp baseline
# 18 walker med exp 0
# 19 walker med exp negative
# 20 walker med exp mean
# 21 walker med exp random
# 22 walker2d-medium-expert-v2
# 23 walker2d-medium-expert-v2 zero
# 24 walker2d-medium-expert-v2 random
# 25 10%+random
# 26 10% 1 other -1
# 27 original revision


# 28 antmaze-medium-diverse-v1
# 29 antmaze-medium-diverse-v1 zero
# 30 antmaze-medium-diverse-v1 random
# 31 10%+random
# 32 10% 1 other -1
# 33 original revision


# 34 top10 true reward
# 35 top10 zero
# 36 top10 random
# 37 1 -1
# 38 top30 random
# 39 top 50 random
# 40 top 80 random
# 41 top30 true
# 42 top 50  true
# 43 top 80 true
# 44 baseline
# 45 top 30 +-1
# 46 top 50 +-1
# 47 top 80 +-1
# 48 top 10 +-1
# 49 top 200 +-1
# 50 top 300 +-1

# 51 antmaze-medium-diverse-v0 ori
#

# --reward_model_path ${path_offset}${reward_model_path} \
path_offset="/data3/zj/oprl/reward_learning/rewards/"  #
 n=3
 gpunum=8
 init_seed=50  # 44
 for task in "${tasks[@]}"
 do
 for reward_model_path in "${reward_model_paths[@]}"
 do
 for seed in "${seeds[@]}"
 do
 true_seed=$[($init_seed*10+seed) ]
  XLA_PYTHON_CLIENT_PREALLOCATE=false  CUDA_VISIBLE_DEVICES=${n}  nohup python train_offline.py \
 --env_name ${task} \
 --seed ${true_seed} \
 --config=configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000 \
 >& out_logs/${Foldername}/${task}_${reward_model_path}_${seed}_${true_seed}_${n}.txt &
 n=$[($n+1) % ${gpunum}]
 done
 init_seed=$[($init_seed+1) ]
 done
 done
