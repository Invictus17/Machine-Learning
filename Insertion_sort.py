from builtins import bool, range
import random
import time

def Insertion_sort(Input_file_path):

    Input = read_Inputfile(Input_file_path)
    #Citation: Referred https://docs.python.org/3/library/time.html#time.process_time for time.process_time() function
    start_time = time.process_time()  #In seconds
    for j in range(1,len(Input)):
        key = Input[j]
        i = j-1
        while(i>-1 and Input[i]>key):
            Input[i+1]=Input[i]
            i=i-1
        Input[i+1]=key

    end_time = time.process_time()
    run_time = end_time - start_time
    # Using raw string literal by specifying 'r' before the string(path)
    print("Run time for Insertion sort is:", run_time)
    print("Sorted list is:", Input)
