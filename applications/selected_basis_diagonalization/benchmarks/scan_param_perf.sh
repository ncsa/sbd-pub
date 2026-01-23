#!/usr/bin/env bash

if [ $# -ne 1 ]
then
    echo "Give me the dataset name as argument, please."
    exit 1
else
    DATASET=$1
    dataset_dir=$(dirname ${DATASET})
    dataset_name=$(basename ${dataset_dir})
    dataset_file=$(basename ${DATASET})
fi

kill_job() {
    scancel ${JOBID}    
}
trap kill_job EXIT INT TERM

# scan for best performance for various cmdline parameters

MAX_NODES=32
TPN=8
GPT=1
CPT=8

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

BROKEN_NODES=x1000c5s2b0n0,x1000c3s1b1n0,x1000c3s2b0n0,x1000c1s2b1n0,x1001c5s5b0n0,x1000c7s2b1n0,x1000c1s5b1n0,x1000c6s3b0n0,x1000c6s5b0n0,x1001c1s2b1n0,x1000c0s0b1n0,x1000c1s7b0n0,x1000c5s4b1n0,x1000c7s2b0n0
SUSPECT_NODES=x1000c0s4b1n0 #x1000c0s2b1n0,x1000c4s3b1n0,x1000c4s4b1n0
BROKEN_NODES="${BROKEN_NODES},${SUSPECT_NODES}"

JOBID=$(salloc -x ${BROKEN_NODES} --nodes=${MAX_NODES} --tasks-per-node=${TPN} --cpus-per-task=${CPT} --gpus-per-task=${GPT} --partition=MI250X_A1_COS_OK --time=4:00:00 --no-shell |& grep Granted|cut -f 5 -d " ")

TOTAL_TASKS=$(squeue --job=${JOBID} --noheader --Format=NumTasks|xargs) # xargs eliminates the linefeed after number
NODELIST=$(squeue --job=${JOBID} --noheader --Format=NodeList:-1)

backthen=$(date +%s)

asizes=$(python3 -c 'import numpy; [print(x) for x in 2**numpy.linspace(0,8,9,endpoint=True,dtype="i8")]') 
for adet in 4 8 16 32 # ${asizes} # $(#seq 4 ${TOTAL_TASKS}))
do
    bsizes=$(python3 -c 'import numpy;adet=2**numpy.linspace(0,8,9,endpoint=True,dtype="i8");[print(x) for x in adet if '${adet}'*x <= '${TOTAL_TASKS}']')
    max_bdet=$(awk "BEGIN {printf \"%.0f\", ${TOTAL_TASKS}/${adet}}")
    for bdet in ${bsizes} #$(seq 4 ${max_bdet})
    do
	#if ! python3 -c "import sys; sys.exit(${adet}//${bdet} >= 4 or ${bdet}//${adet} >= 4 or ${adet}*${bdet}<16)"
	if ! python3 -c "import sys; a=${adet}; b=${bdet}; sys.exit((a*b <= 16) or ((a//b > 2 or b//a > 2) and a*b<64) or (a*b >= 64 and (a//b > 4 or b//a > 4)))"
	then
	    continue
	fi
	##max_task=$(awk "BEGIN {printf \"%.0f\", ${TOTAL_TASKS}/(${adet}*${bdet})}")
	#tsizes=$(python3 -c "import numpy;adet=2**numpy.linspace(0,8,9,endpoint=True,dtype=numpy.int64);a=${adet};b=${bdet};[print(x) for x in range(1,${TOTAL_TASKS}//(a*b)+1) if ${TOTAL_TASKS} % (a*b*x) == 0]")
	tsizes=$(python3 -c "import numpy;adet=2**numpy.linspace(0,8,9,endpoint=True,dtype=numpy.int64);a=${adet};b=${bdet};[print(x) for x in range(1,${TOTAL_TASKS}//(a*b)+1) if ${TOTAL_TASKS} == (a*b*x)]")
	for task in ${tsizes} # $(seq 1 ${max_task})
	do
	    #prod=$((${adet}*${bdet}*${task}))
	    rcomm_size=$(python3 -c "tot=${TOTAL_TASKS};a=${adet};b=${bdet};t=${task};print(tot//(a*b*t))")
	    if true # [ "${prod}" -eq "${TOTAL_TASKS}" ]
	    then
		then=$(date +%s)
		LOGFILE="${SCRIPT_DIR}/scan_logs/${dataset_name}/${dataset_file}/${backthen}_adet_${adet}_bdet_${bdet}_task_${task}.out"
		mkdir -p $(dirname ${LOGFILE})
		echo $(date --date=@${then} --iso=seconds) > ${LOGFILE}
		echo "Running ${adet}x${bdet}x${task}x${rcomm_size}" | tee -a ${LOGFILE}
		# The h2o-1em8-alpha.txt requires 64 GiB per rank when runnig 1x16x4, so for that data, we need adet*bdet>=20 or so
		CMD=(srun --nodes=${MAX_NODES} --tasks-per-node=${TPN} --cpus-per-task=${CPT}  --gpus-per-task=${GPT}  --partition=MI250X_A1_COS_OK  --jobid=${JOBID} --label --time=00:45:00 /usr/bin/time -f "RSSHWM:%MkiB" bash -c "source ${SCRIPT_DIR}/../env_setup.sh &> /dev/null; ${SCRIPT_DIR}/../src/diag --fcidump ${dataset_dir}/fcidump.txt --adetfile ${DATASET} --method 0 --block 10 --iteration 1 --tolerance 1.0e-4 --adet_comm_size ${adet} --bdet_comm_size ${bdet} --task_comm_size ${task} --init 0 --shuffle 1 --carryover_ratio 0.5 --rdm 0")
		#srun --nodes=${MAX_NODES} --tasks-per-node=${TPN} --cpus-per-task=${CPT}  --gpus-per-task=${GPT}  --partition=MI250X_A1_COS_OK  --jobid=${JOBID} --label --time=00:45:00 /usr/bin/time -f "RSSHWM:%MkiB" bash -c "source ${SCRIPT_DIR}/../env_setup.sh &> /dev/null; ${SCRIPT_DIR}/../src/diag --fcidump ${SCRIPT_DIR}/../data/h2o/fcidump.txt --adetfile ${SCRIPT_DIR}/../data/h2o/h2o-1em8-alpha.txt --method 0 --block 2 --iteration 1 --tolerance 1.0e-4 --adet_comm_size ${adet} --bdet_comm_size ${bdet} --task_comm_size ${task} --init 0 --shuffle 1 --carryover_ratio 0.5 --rdm 0" >> ${LOGFILE} 2>&1
		echo "Running this:" >> ${LOGFILE}
		echo "${CMD[@]}" >> ${LOGFILE}
		"${CMD[@]}" >> ${LOGFILE} 2>&1
		now=$(date +%s)
		echo $(date --date=@${now} --iso=seconds) >> ${LOGFILE}
		echo "Wall time elapsed: $((${now} - ${then}))" >> ${LOGFILE}
	    fi
	done
    done
done

# kill the allocation
scancel ${JOBID}

# kill the allocation
kill_job


		
#salloc -x x1000c7s2b1n0,x1000c1s5b1n0,x1000c5s2b0n0 --nodes=8 --tasks-per-node=8 --cpus-per-task=8  --gpus-per-task=1  --partition=MI250X_A1_COS_OK /bin/bash -c 'date --iso=seconds; srun /usr/bin/time -f "${TIMEFORMAT}" bash -c "module purge; module restore cray_and_amd_6_4_3; ./src/diag --fcidump data/h2o/fcidump.txt --adetfile data/h2o/h2o-1em8-alpha.txt --method 0 --block 10 --iteration 1 --tolerance 1.0e-4 --adet_comm_size 4 --bdet_comm_size 4 --task_comm_size 4 --init 0 --shuffle 1 --carryover_ratio 0.5 --rdm 0" ; date --iso=seconds'
