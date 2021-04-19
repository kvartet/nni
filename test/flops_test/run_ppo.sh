flops_list=(100 50 30)
arr=(7 11 16)
for flops in ${flops_list[@]}
do
echo "generate $flops file..."
python3 generator.py --flops $flops
    for i in "${!arr[@]}";   
    do
        echo "run $[$i+1] begin, the rand seed is ${arr[$i]}, the flops is $flops"
        python3 launch_ppo.py --main_file trial_code_nni.py --seed ${arr[$i]} --search_space_path search.json --flops $flops --version $[$i+1] --situation aware --port 8745
        # python3 launch_ppo.py --main_file "total_trial_code.py --flops $flops" --seed ${arr[$i]} --search_space_path total_search.json --flops $flops --version $[$i+1] --situation nonested --port 8742
        # python3 launch.py --main_file trial_code.py --seed ${arr[$i]} --search_space_path search.json --flops $flops --version $[$i+1] --situation noaware
    done 
done
