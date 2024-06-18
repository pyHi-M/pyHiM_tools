#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 19:28:21 2021

@author: marcnol

Launches slurm, srun or bash jobs from a number of target folders

In the command line, run as

$ runHiM_cluster.py


"""
import argparse
import glob
import os
import subprocess


# =============================================================================
# MAIN
# =============================================================================
def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-F", "--dataFolder", help="Folder with data. Default: ~/scratch"
    )
    parser.add_argument("-S", "--singleDataset", help="Folder for single Dataset.")
    parser.add_argument(
        "-A", "--account", type=str, default = 'episcope',help="Provide your account name. Default: episcope."
    )
    parser.add_argument(
        "-P", "--partition", type=str, default='defq',help="Provide partition name. Default: tests"
    )
    parser.add_argument("-N", "--nCPU", type=int, default=1, help="Number of CPUs/Task")
    parser.add_argument("--memPerCPU", help="Memory required per allocated CPU in Mb")
    parser.add_argument(
        "--nodelist", help="Specific host names to include in job allocation."
    )
    parser.add_argument("-T1", "--nTasksNode", help="Number of tasks per node.")
    parser.add_argument("-T2", "--nTasksCPU", help="Number of tasks per CPU")
    parser.add_argument(
        "-C",
        "--cmd",
        help="Comma-separated list of routines to run: \
                     project  register_global register_local  \
                     mask_2d localize_2d \
                     mask_3d localize_3d \
                     filter_localizations register_localizations \
                     build_traces build_matrix",
    )
    parser.add_argument(
        "--threads",
        help="Number of threads for parallel mode. None: sequential execution",
    )
    parser.add_argument("--srun", help="Runs using srun", action="store_true")
    parser.add_argument("--sbatch", help="Runs using sbatch", action="store_true")
    parser.add_argument("--bash", help="Runs using bash", action="store_true")
    parser.add_argument("--parallel", help="Runs all processes in parallel", action="store_true")

    args = parser.parse_args()

    run_parameters = {}
    run_parameters["HOME"] = os.environ["HOME"]
    run_parameters["parallel"] =  args.parallel
    run_parameters["bash"] = args.bash
    run_parameters["srun"] = args.srun
    run_parameters["sbatch"] = args.sbatch
    run_parameters["account"] = args.account
    run_parameters["partition"] = args.partition
    run_parameters["nCPU"] = 1
    
    if args.memPerCPU:
        run_parameters["memPerCPU"] = args.memPerCPU
    else:
        run_parameters["memPerCPU"] = None

    if args.cmd:
        run_parameters["cmd"] = args.cmd
    else:
        run_parameters["cmd"] = None

    if args.dataFolder:
        run_parameters["dataFolder"] = args.dataFolder.rstrip("/")
    else:
        run_parameters["dataFolder"] = run_parameters["HOME"] + os.sep + "scratch"

    if args.singleDataset:
        run_parameters["singleDataset"] = args.singleDataset
    else:
        run_parameters["singleDataset"] = None

    if args.nodelist:
        run_parameters["nodelist"] = args.nodelist
    else:
        run_parameters["nodelist"] = None

    if args.nTasksNode:
        run_parameters["nTasksNode"] = args.nTasksNode
    else:
        run_parameters["nTasksNode"] = None

    if args.nTasksCPU:
        run_parameters["nTasksCPU"] = args.nTasksCPU
    else:
        run_parameters["nTasksCPU"] = None

    if args.threads:
        run_parameters["threads"] = args.threads
    else:
        run_parameters["threads"] = None

    return run_parameters


def main():
    run_parameters = read_arguments()

    print(f"Parameters loaded: {run_parameters}\n")

    root_folder = run_parameters["dataFolder"] 

    run_parameters["dataset"] = root_folder.split("/")[-1]

    if run_parameters["singleDataset"] is None:
        folders = glob.glob(root_folder + os.sep + "*")
        folders0 = [x for x in folders if os.path.isdir(x)]  # keeps only folders
        folders = [
            x for x in folders0 if os.path.exists(x + os.sep + "parameters.json")
        ]
    else:
        folders0 = folders = [run_parameters["singleDataset"]]
        # run_parameters["dataset"] = os.path.basename(run_parameters["singleDataset"])

    folders.sort()

    print("*" * 50)
    print("$ Dataset: {}".format(run_parameters['dataset']))
    print(f"$ Folder: {root_folder}")
    print(f"$ Number of CPUs: {run_parameters['nCPU']}")
    print(f"$ Command: {run_parameters['cmd']}")
    print(f"$ Account: {run_parameters['account']}")
    print(f"$ Partition: {run_parameters['partition']}")

    print("*" * 50)

    print(f"\n\n$ Found {len(folders0)} folders in {root_folder}")
    print(
        f"$ Of these, {len(folders)} contained a parameters.json file and will be processed"
    )
    print(f"Folders to process: {folders}")
    print(f"$ Scheduling {len(folders)} jobs...")
    print("-" * 50)


    if run_parameters["memPerCPU"] is None:
        memPerCPU = ""
    else:
        memPerCPU = " --mem_per_cpu=" + run_parameters["memPerCPU"]

    if run_parameters["nodelist"] is None:
        nodelist = ""
    else:
        nodelist = " --nodelist=" + run_parameters["nodelist"]

    if run_parameters["nCPU"] == 1:
        CPUsPerTask = ""
    else:
        CPUsPerTask = " --cpus-per-task " + str(run_parameters["nCPU"])

    if run_parameters["nTasksCPU"] is None:
        nTasksCPU = ""
    else:
        nTasksCPU = " --ntasks-per-core=" + run_parameters["nTasksCPU"]

    if run_parameters["nTasksNode"] is None:
        nTasksNode = ""
    else:
        nTasksNode = " --ntasks-per-node=" + run_parameters["nTasksNode"]

    if run_parameters["threads"] is None:
        threads = ""
    else:
        threads = " --threads " + run_parameters["threads"]

    if run_parameters["cmd"] is None:
        cmdName = ""
        CMD = ""
        jobNameExt = "_completePipeline"
    else:
        cmdName = run_parameters["cmd"]
        CMD = " -C " + cmdName
        jobNameExt = "_" + cmdName

    if run_parameters["sbatch"]:
        BATCH_file = ["#!/bin/bash"]
        SBATCH_header = [
            [
                "#!/bin/bash",
                "#SBATCH " + memPerCPU,
                "#SBATCH " + CPUsPerTask,
                "#SBATCH " + nTasksCPU,
                "#SBATCH --account=" + run_parameters["account"],
                "#SBATCH --partition=" + run_parameters["partition"],
                "#SBATCH --mail-user=marcnol@gmail.com ",
            ]
        ]
        SBATCH_header.append(
            [
                "",
                "source /trinity/shared/apps/local/Python/Anaconda/3-5.1.0/etc/profile.d/conda.sh",
                "conda activate pyHiM39",
                "",
            ]
        )
    elif run_parameters["bash"]:  
        BATCH_file = ["#!/bin/bash"]
        
        
    for folder in folders:
        output_file = (
            run_parameters["HOME"]
            + os.sep
            + "logs"
            + os.sep
            + run_parameters["dataset"]
            + "_"
            + os.path.basename(folder)
            + "_"
            + cmdName
            + ".log"
        )
        job_name = os.path.basename(folder) + jobNameExt

        print(f"Folder to run: {folder}")
        # print(f"Output logfile: {output_file}")

        pyHiM = "pyHiM.py -F " + folder + CMD + threads 

        if run_parameters["srun"] or run_parameters["parallel"]:
            pyHiM = pyHiM + " &"


        SRUN = (
            "srun --account="
            + run_parameters["account"]
            + " --partition="
            + run_parameters["partition"]
            + " --job-name="
            + job_name
            + CPUsPerTask
            + nodelist
            + nTasksCPU
            + nTasksNode
            + memPerCPU
            + " --mail-user=marcnol@gmail.com "
            + pyHiM
        )

        if run_parameters["sbatch"]:
            SBATCH_list = []
            SBATCH_list = SBATCH_list + SBATCH_header[0]
            SBATCH_list.append(f"#SBATCH --job-name={job_name}")
            SBATCH_list = SBATCH_list + SBATCH_header[1]
            SBATCH_list.append(f"\n# dataset: {job_name}")
            SBATCH_list.append("srun " + pyHiM)
        elif run_parameters["bash"]:
            SBATCH_list = []
            SBATCH_list.append(f"\n# dataset: {job_name}")
            SBATCH_list.append(pyHiM)
            BATCH_file = BATCH_file + SBATCH_list
        elif run_parameters["srun"]:
            print(f"Command to run: {SRUN}")
            print("-" * 50)
            subprocess.run(SRUN)

        if run_parameters["sbatch"]:
            print("$ script: \n{}".format("\n".join(SBATCH_list)))
            print("-" * 80)

            file_name = f"sbatch_script_{job_name}.bash"
            with open(file_name, mode="w", encoding="utf-8") as f:
                for item in SBATCH_list:
                    f.write(f"{item}\n")

            BATCH_file.append(f"sbatch {file_name}")

    if run_parameters["sbatch"]:
        print("*" * 80)
        BATCH_file.append("\n")
        bash_script_name = f"batch_script_{run_parameters['dataset']}.bash"
        with open(bash_script_name, mode="w", encoding="utf-8") as f:
            for item in BATCH_file:
                f.write(f"{item}\n")
    elif run_parameters["bash"]:
        bash_script_name = f"joblist_{run_parameters['dataset']}.bash"
        with open(bash_script_name, mode="w", encoding="utf-8") as f:
            for item in BATCH_file:
                f.write(f"{item}\n")
                        
    print(f"\nTo run master bash script:\n$ bash {bash_script_name}")


if __name__ == "__main__":
    main()
