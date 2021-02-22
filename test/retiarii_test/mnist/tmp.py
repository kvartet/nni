def func1(arg=None):
    print(arg)

print('func1')
func1({'arg1': 1,'arg2': 2,'arg3': [10,20]})

print('func2')
func1()

