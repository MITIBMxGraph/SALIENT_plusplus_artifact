
import argparse
from argparse import Namespace
import os
import pathlib
from pathlib import Path
import statistics
from typing import List

class ExperimentResult:
    def __init__(self, job_args, performance_info, accuracy_info):
        self.job_args = job_args
        self.performance_info = performance_info
        self.accuracy_info = accuracy_info
        self.attr_sortkey_map = dict()


    def set_attr_sortkey_map(self, sortkey_dict):
        self.attr_sortkey_map = sortkey_dict

    def filter(self, filter_job_args):
        for x in filter_job_args.keys():
            if self.job_args[x] != filter_job_args[x]:
                return False
        return True

    #def mean_valid_accuracy(self):
    #def mean_test_accuracy()
    def get_key(self, job_arg_key_list):
        key = []
        for x in job_arg_key_list:
            if x not in self.attr_sortkey_map:
                key.append(self.job_args[x])
            else:
                key.append(self.attr_sortkey_map[x][self.job_args[x]])
        return tuple(key)

    def get_attribute(self, attr):
        if attr == 'replication_factor':
            return float(self.job_args[attr])/100.0
        return self.job_args[attr]

    def get_test_accuracy(self):
        accuracies = [0,0]
        for x in self.accuracy_info:
            accuracies[0] += x[0]
            accuracies[1] += x[1]
        accuracies[0] /= len(self.accuracy_info)
        accuracies[1] /= len(self.accuracy_info)
        if accuracies[1] == 0:
            return "N/A"
        else:
            return accuracies[1]

    def get_valid_accuracy(self):
        accuracies = [0,0]
        for x in self.accuracy_info:
            accuracies[0] += x[0]
            accuracies[1] += x[1]
        accuracies[0] /= len(self.accuracy_info)
        accuracies[1] /= len(self.accuracy_info)
        if accuracies[0] == 0:
            return "N/A"
        else:
            return accuracies[0]
    def get_time(self, key):
        return self.performance_info[key][0]
    def get_time_stdev(self, key):
        return self.performance_info[key][1]

class ExperimentResultGroup:
    def __init__(self, experiment_result_list : List[ExperimentResult], *pos_args, filter_by=None, sort_by, attr_sortkey_map = None):
        filtered_list = experiment_result_list
        if attr_sortkey_map != None:
            for x in filtered_list:
                x.set_attr_sortkey_map(attr_sortkey_map)
        if filter_by != None:
            filtered_list = list(filter(lambda x: x.filter(filter_by), filtered_list))
        self.result_list = sorted(filtered_list, key=lambda x : x.get_key(sort_by))
    def size(self):
        return len(self.result_list)
    def get_attribute(self, attr, i):
        return self.result_list[i].get_attribute(attr)
    def get_valid_accuracy(self, i):
        return self.result_list[i].get_valid_accuracy()
    def get_test_accuracy(self, i):
        return self.result_list[i].get_test_accuracy()
    def get_total_time(self, i):
        return "{:.2f}".format(self.result_list[i].get_time('Total')) +  " " + u"\u00B1" +" "+ "{:.2f}".format(self.result_list[i].get_time_stdev('Total')) 
    def get_time_csv(self, i):
        return "{:.2f}".format(self.result_list[i].get_time('Total')/1000) 
    def get_time_stdev_csv(self, i):
        return "{:.2f}".format(self.result_list[i].get_time_stdev('Total')/1000) 



def parse_logfile(main_file, job_args_dict, output_data):
    #print("trying to parse file " + str(main_file))
    lines = open(str(main_file)).readlines()

    # get total
    best_line = None
    accuracy_results_line = None
    preamble_times = []
    #('End results for all trials', '[(0.6686384864088133, 0.6381043025501778)]')
    for l in lines:
        if l.startswith("('performance_breakdown_stats'"):
            best_line = l
        if l.startswith("('End results for all trials'"):
            accuracy_results_line = l
    if best_line == None or accuracy_results_line == None:
        #if not args.silent_errors or not args.ignore_errors:
        #print("Error could not find runtime info in logfile at mainfile " + str(main_file))
        #if not args.ignore_errors:
        #    quit()
        return False

    info = eval(eval(best_line.strip())[1])
    info_accuracy = eval(eval(accuracy_results_line.strip())[1])

    #best_line = best_line.replace("PARSE_STATS_DICT:", "").strip()
    #info = eval(best_line)
    mean_preamble = 0
    mean = 0

    performance_info = dict()
    for x in info:
        performance_info[x[0]] = (x[1],x[2])

    output_data.append(ExperimentResult(job_args_dict, performance_info, info_accuracy))
    return True

def parse_joblist(job_list):
   lines = open(job_list).readlines()
   output_data = [] 
   for l in lines:
       if len(l.strip()) == 0:
           continue
       info = eval(l.strip())
   
       log_dir = Path(info[-2])
       log_dir = log_dir / '..' / 'experiment_testbed'

       job_args = info[1]
       #print(log_dir)
       job_args_dict = vars(job_args)
       found_main_file = False
       main_file = None
       for x in log_dir.iterdir():
           if str(x).endswith("logs.txt"):
               main_file = x
               if parse_logfile(main_file, job_args_dict, output_data):
                   found_main_file = True
                   break
       if not found_main_file:
           #if not args.silent_errors or not args.ignore_errors:
           print("Error failed to find main log file for job with logdir " + str(log_dir))
           #if not args.ignore_errors:
           #    quit()
       #parse_logfile(main_file, job_args_dict, output_data)
   return output_data




def parse_logfile_legacy(main_file, args, job_args_dict, output_data):
    lines = open(str(main_file)).readlines()

    # get total
    best_line = None
    preamble_times = []
    for l in lines:
        if l.startswith("PARSE_STATS_DICT"):
            best_line = l
        

    if best_line == None:
        if not args.silent_errors or not args.ignore_errors:
            print("Error could not find runtime info in logfile at mainfile " + str(main_file))
        if not args.ignore_errors:
            quit()
        return
    best_line = best_line.replace("PARSE_STATS_DICT:", "").strip()
    info = eval(best_line)

    if "preamble" in info:
        mean_preamble = statistics.mean(info["preamble"])
    else:
        mean_preamble = 0

    mean = statistics.mean(info["total"])
    stdev = "N/A"
    if len(info["total"]) > 1:
        stdev = statistics.stdev(info["total"])
    #print(args.varied_parameter+"="+str(job_args_dict[args.varied_parameter])+" Mean: " + str(mean) + " stdev: " + str(stdev))
    output_data.append({'param_value':job_args_dict[args.varied_parameter], 'mean':mean/1000, 'stdev':stdev/1000, 'preamble': mean_preamble/1000, 'args': job_args_dict})
    

def parse_joblist_legacy(job_list, args):
   lines = open(job_list).readlines()
   output_data = [] 
   for l in lines:
       try:
           if len(l.strip()) == 0:
               continue
           info = eval(l.strip())
   
           log_dir = Path(info[-2])
           print(log_dir / '..' / 'experiment_testbed')

           job_args = info[1]
           #print(log_dir)
           job_args_dict = vars(job_args)
           found_main_file = False
           main_file = None
           for x in log_dir.iterdir():
               if str(x).endswith(".1.out"):
                   found_main_file = True
                   main_file = x
           if not found_main_file:
               #if not args.silent_errors or not args.ignore_errors:
               print("Error failed to find main log file for job with logdir " + str(log_dir))
               #if not args.ignore_errors:
               #    quit()
           parse_logfile(main_file, args, job_args_dict, output_data)
       except:
           print(f"[Error] Exception occurred when parsing logfile line ({l}).")
   return output_data
  
def print_output_data(output_data, args):
    distinct_params = dict()
    for x in output_data:
        distinct_params[x["param_value"]] = True
    grouped_by_param = dict()
    for x in sorted(distinct_params.keys()):
        grouped_by_param[x] = [] 

    for x in output_data:
        grouped_by_param[x["param_value"]].append((x["mean"], x["stdev"], x["preamble"]))

    print(args.varied_parameter + ", mean, stdev, preamble")
    for x in distinct_params:
        for y in grouped_by_param[x]: 
           if args.varied_parameter == "replication_factor":
               x = x/100.0
           print(str(x) + "," + str(y[0]) + "," + str(y[1]) + "," + str(y[2]))
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run distributed experiment")
    parser.add_argument("--joblist", help="path to a job list file produced by exp_driver.py script", type=str, required=True)
    parser.add_argument("--varied_parameter", help="choose the parameter that you are varying in the experiment.", type=str, required=True)

    parser.add_argument("--ignore_errors", help="ignore files that have missing data.", type=bool, default=False)
    parser.add_argument("--silent_errors", help="if ignoring errors, do you want any prints warning about those errors?", type=bool, default=False)
    
    args = parser.parse_args()

    output_data = parse_joblist(args.joblist, args)

    print(output_data)


    group = ExperimentResultGroup(output_data, filter_by={'replication_factor': 16}, sort_by=[args.varied_parameter])

    for i in range(group.size()):
        print(str(group.get_attribute(args.varied_parameter, i)) +"\t"+ str(group.get_total_time(i)))

    quit()

    print_output_data(output_data, args)









