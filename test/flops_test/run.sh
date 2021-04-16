flops_list=(100 50 30)
arr=(3 7 11)
for flops in ${flops_list[@]}
do
echo "generate $flops file..."
python3 generator.py --flops $flops
    for i in "${!arr[@]}";   
    do
        echo "run $[$i+1] begin, the rand seed is ${arr[$i]}, the flops is $flops"
        # python3 launch.py --main_file trial_code.py --seed ${arr[$i]} --search_space_path search.json --flops $flops --version $[$i+1] --situation aware
        # python3 launch.py --main_file "total_trial_code.py --flops $flops" --seed ${arr[$i]} --search_space_path total_search.json --flops $flops --version $[$i+1] --situation nonested
        python3 launch.py --main_file trial_code.py --seed ${arr[$i]} --search_space_path search.json --flops $flops --version $[$i+1] --situation noaware
    done 
done
