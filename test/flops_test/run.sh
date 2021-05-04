# cifar10 & cifar100, when flops range is 30, 50, 100
# tuner_list=('tpe' 'evolution' 'ppo')
# for tuner in ${tuner_list[@]}
# do
#     echo "begin $tuner......"
#     flops_list=(30 50 100)
#     arr=(3 7 11)
#     for flops in ${flops_list[@]}
#     do
#     echo "generate $flops file..."
#     python3 generator.py --flops $flops
#         for i in "${!arr[@]}";   
#         do
#             echo "run $[$i+1] begin, the rand seed is ${arr[$i]}, the flops is $flops, the tuner is $tuner"
#             CUDA_VISIBLE_DEVICES=1 python3 launch.py --situation aware --tuner $tuner --main_file "trial_code.py --tuner $tuner --dataset cifar10" --seed ${arr[$i]} --search_space_path search.json --flops $flops --version $[$i+1] --dataset cifar10 --port 8745
        
#             CUDA_VISIBLE_DEVICES=1 python3 launch.py --situation nonested --tuner $tuner --main_file "total_trial_code.py --dataset cifar10 --flops $flops" --seed ${arr[$i]} --search_space_path total_search.json --flops $flops --version $[$i+1] --dataset cifar10  --port 8745
#         done 
#     done
# done

# imagenet, when flops range is 5, 11, 20
tuner_list=('tpe' 'evolution' 'ppo')
for tuner in ${tuner_list[@]}
do
    echo "begin $tuner......"
    flops_list=(5 11 20)
    arr=(3 7 26)
    for flops in ${flops_list[@]}
    do
    echo "generate $flops file..."
    python3 generator.py --flops $flops
        for i in "${!arr[@]}";   
        do
            echo "run $[$i+1] begin, the rand seed is ${arr[$i]}, the flops is $flops, the tuner is $tuner"
            CUDA_VISIBLE_DEVICES=1 python3 launch.py --situation aware --tuner $tuner --main_file "trial_code.py --tuner $tuner --dataset imagenet16-120" --seed ${arr[$i]} --search_space_path search.json --flops $flops --version $[$i+1] --dataset imagenet16-120 --port 8745
        
            CUDA_VISIBLE_DEVICES=1 python3 launch.py --situation nonested --tuner $tuner --main_file "total_trial_code.py --dataset imagenet16-120 --flops $flops" --seed ${arr[$i]} --search_space_path total_search.json --flops $flops --version $[$i+1] --dataset imagenet16-120  --port 8745
        done 
    done
done
