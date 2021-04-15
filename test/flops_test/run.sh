# parser.add_argument('--main_file', type=str)
# parser.add_argument('--seed', type=int)
# parser.add_argument('--search_space_path', type=str)
# parser.add_argument('--flops', type=str)
# parser.add_argument('--version', type=str)
# parser.add_argument('--situation', type=str)

# arr=(3 7 11)

# for i in "${!arr[@]}";   
# do
#     echo "run $[$i+1] begin..."
#     python3 launch.py --main_file trial_code.py --seed ${arr[$i]} --search_space_path search.json --flops 100 --version $[$i+1] --situation aware

#     python3 launch.py --main_file total_trial_code.py --seed ${arr[$i]} --search_space_path total_search.json --flops 100 --version $[$i+1] --situation nonested
#     # printf "%s\t%s\n" "$[$i+1]" "${arr[$i]}"  
# done 

flops_list=(100 50 30)
arr=(3 7 11)
for flops in ${flops_list[@]}
do
echo "generate $flops file..."
python3 generator.py --flops $flops
    for i in "${!arr[@]}";   
    do
        echo "run $[$i+1] begin, the rand seed is ${arr[$i]}, the flops is $flops"
        python3 launch.py --main_file trial_code.py --seed ${arr[$i]} --search_space_path search.json --flops $flops --version $[$i+1] --situation aware
        python3 launch.py --main_file "total_trial_code.py --flops $flops" --seed ${arr[$i]} --search_space_path total_search.json --flops 100 --version $[$i+1] --situation nonested
    done 
done



python3 launch.py --main_file trial_code.py --seed 7 --search_space_path search.json --flops 100 --version 1 --situation aware

python3 launch.py --main_file total_trial_code.py --seed 7 --search_space_path total_search.json --flops 100 --version 1 --situation nonested


# python3 launch.py --main_file trial_code.py --seed 9 --search_space_path search.json --flops 100 --version 2 --situation aware

# python3 launch.py --main_file total_trial_code.py --seed 9 --search_space_path total_search.json --flops 100 --version 2 --situation nonested

# python3 launch.py --main_file trial_code.py --seed 13 --search_space_path search.json --flops 100 --version 3 --situation aware

# python3 launch.py --main_file total_trial_code.py --seed 9 --search_space_path total_search.json --flops 100 --version 3 --situation nonested