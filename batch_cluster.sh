#!/bin/bash
# AUTHOR
# Thomas Gautrais
#
# DESCRIPTION
# mine and display the number of available cores and memory on each node of the cluster
#
# Using of GNU Parallel:
# O. Tange (2011): GNU Parallel - The Command-Line Power Tool,
# ;login: The USENIX Magazine, February 2011:42-47.
#

getname () {
	set -- "${1:-$(</dev/stdin)}" "${@:2}"
	if [ "$#" -eq 0 ]
	then
		(>&2 echo "At least one parameter (uid) is required")
		return 1
	fi
	input=($@)
	#${input//$'\n'/}
	local name
	for ((i=0;i<${#input[@]};i++));
	do
		name=`getent passwd ${input[i]}|cut -d ':' -f 5`
		test -z "$name"  && name="${input[i]}"
		echo -ne "$name" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
	done
}

disp_help () {
	echo 'Usage of batch cluster : batch_cluster [OPTIONS]'
	echo 'List of available options : '
	echo '   -C, --calcul-lahc-only         select only CPU machines calcul-lahc-1 to calcul-lahc-7'
	echo '   -C=LIST                        select only some machines among the calcul-lahc-* CPU machines'
	echo '                                  LIST defines a range or/and particular indexes of CPU machines.'
	echo '                                  AS for example, with the option -C='"'"'7;3-5;1'"'"', only the' 
	echo '                                  machines with indexes 7, 3 to 5 and 1 (i.e. calcul-lahc-7,'
	echo '                                  calcul-lahc-3, calcul-lahc-4, calcul-lahc-5 and calcul-lahc-1)'
	echo '                                  are selected'
	echo '                                  if LIST is empty (i.e. -C='"'"''"'"' or -C=), it is equivalent'
	echo '                                  to -C or --calcul-lahc-only option (select all CPU machines from'
	echo '                                  calcul-lahc-1 to calcul-lahc-7)'
	echo '   -G, --gpu-only                 select only GPU machines calcul-gpu-lahc-1 to calcul-gpu-lahc-3'
	echo '   -H, --arch-haswell             select only CPU machines with Haswell CPU architecture'
	echo '                                  (calcul-lahc-5 to calcul-lahc-7)'
	echo '   -S, --arch-sandybridge         select only CPU machines with Sandy Bridge CPU architecture'
	echo '                                  (calcul-lahc-1 to calcul-lahc-4)'
	echo '   -W, --arch-westmere            select only CPU machines with Westmere CPU architecture (elrond,'
	echo '                                  saroumane)'
	echo '   -H2, --arch-harpertown         select only CPU machines with Harpertown CPU architecture'
	echo '                                  (aragorn, gandalf, legolas, meriadoc, peregrin)'
	echo '   -h, --help                     display usage of this script'

}

add_if_not_already () {
ssh_hosts_ok='ssh  -o "BatchMode=yes" -o "ConnectTimeout=1" -o "StrictHostKeyChecking no" -o "PasswordAuthentication no" ${nodes2add[i]} exit >/dev/null 2>&1'
# add_if_not_already add to the nodes array the values from the nodes2add, which are not already in the nodes array
if [ "${#nodes[@]}" -eq 0 ]
then
	local i
	for ((i=0;i<${#nodes2add[@]};i++))
	do 
		eval "$ssh_hosts_ok"
		if [ "$?" -eq 0 ];then nodes+=("${nodes2add[i]}");fi
	done
else
	local i j
	for ((i=0;i<${#nodes2add[@]};i++))
	do
		for ((j=0;j<${#nodes[@]};j++))
		do
			test "${nodes2add[i]}" = "${nodes[j]}" && break
			if [ "$j" -eq "$((${#nodes[@]} - 1))" ]
			then 
				exit
				test "$?" -eq 0 && nodes+=("${nodes2add[i]}")
			fi
		done
	done
fi
}

rm_el_auto(){ 
	local array suppr
	eval "array=(\"\${$1[@]}\")"
	eval "suppr=(\"\${$2[@]}\")"
	for ((i=0;i<${#suppr[@]};i++))
	do 
		mapfile -t array < <(IFS=$'\n'; echo "${array[*]}" | grep -xv "${suppr[i]}")
	done
	eval "$1=(\"\${array[@]}\")"
}


getuser () {
	local ids_ps=(`ssh  $1 "ps -eLf --no-headers | sort -nrk 5 |head -n 500 | awk '{if( \\\$5 > 8) print \\\$1 }'|sort|uniq"`)
	local ids_who=(`ssh $1 "who|cut -d\" \" -f1|sort|uniq"`)
	local ids_uniq=(`echo ${ids[@]} | tr ' ' '\n' | sort -u | tr '\n' ' '`)
	local users_ps=()
	local users_who=()
	for ((i=0;i<${#ids_ps[@]};i++));do users_ps[i]="`getname ${ids_ps[i]}`";done
	for ((i=0;i<${#ids_who[@]};i++));do users_who[i]="`getname ${ids_who[i]}`";done
	local users_ps_str=`printf " ; %s" "${users_ps[@]}"`
	local users_who_str=`printf " ; %s" "${users_who[@]}"`
	users_ps_str=${users_ps_str:3}
	users_who_str=${users_who_str:3}
	echo "connected_people_ps[$2]=\"${users_ps_str}\"" 
	echo "connected_people_who[$2]=\"${users_who_str}\"" 
}

getcpu () {
	local nb_cores="`ssh  $1 \"grep -c '^processor' /proc/cpuinfo\"`"
	local used_cores="`ssh $1 \"ps -eLf --no-headers | sort -nrk 5 |head -n 500 | awk '{if( \\\$5 > 8) print \\\$5}' |wc -l \"`"
	test "$used_cores" -ge "${nb_cores}" && available_cores="0" || available_cores="$(( ${nb_cores} - $used_cores ))"
	echo "nb_cores[$2]=${nb_cores}"
	echo "available_cores[$2]=${available_cores}"
}

getram () {
	local mem_tot="`ssh $1 \"cat /proc/meminfo |grep '^MemTotal' |awk '{print \\\$2;}'\"`"
	local mem_tot_unit="`ssh $1 \"cat /proc/meminfo |grep '^MemTotal' |awk '{print \\\$3;}'\"`"
	local mem_avail="`ssh $1 \"cat /proc/meminfo |grep '^MemAvailable' |awk '{print \\\$2;}'\"`"
	local mem_avail_unit="`ssh $1 \"cat /proc/meminfo |grep '^MemAvailable' |awk '{print \\\$3;}'\"`"
	if [ -z "${mem_avail}" ]
    then
		mem_avail="`ssh $1 \"cat /proc/meminfo |grep '^MemFree' |awk '{print \\\$2;}'\"`"
		mem_avail_unit="`ssh $1 \"cat /proc/meminfo |grep '^MemFree' |awk '{print \\\$3;}'\"`"
	fi
	
	local ratio
	test "${mem_avail_unit}" = "${mem_tot_unit}" && ratio="$((100*${mem_avail}/${mem_tot}))" || ratio=-100
	echo "total_memory[$2]=${mem_tot}"
	echo "total_memory_unit[$2]=${mem_tot_unit}"
	echo "available_memory[$2]=${mem_avail}"
    echo "available_memory_unit[$2]=${mem_avail_unit}"
	echo "available_memory_percent[$2]=${ratio}"
}


nodes=()
verbose=1
nodes2add=()

if [ "$#" -eq 0 ]
then
	for i in `seq 7`;do nodes2add[$i-1]="calcul-lahc-$i";done
	nodes2add+=('elrond')
	nodes2add+=('saroumane')
	nodes2add+=('aragorn')
	nodes2add+=('gandalf')
	nodes2add+=('meriadoc')
	nodes2add+=('peregrin')
	nodes2add+=('sam')
	nodes2add+=('gimli')
	nodes2add+=('calcul-gpu-lahc-1')
	nodes2add+=('calcul-gpu-lahc-2')
	nodes2add+=('calcul-gpu-lahc-3')
	nodes2add+=('calcul-gpu-lahc-4')
	nodes2add+=('calcul-bigmem-lahc-1')
	nodes2add+=('calcul-bigcpu-lahc-1')
	add_if_not_already 
else
	while [ "$#" -gt 0 ];
	do
		case "$1" in   
			-h|--help)
				disp_help
				exit 0
				;;
			-q|--quiet)
				verbose=0
				shift
				;;
			-C|--calcul-lahc-only)
				nodes2add=()
				for i in `seq 1 7`;do nodes2add+=("calcul-lahc-$i");done
				add_if_not_already 
				shift
				;;	
			-C=*|--calcul-lahc-only=*)
				list="$1"
				range="${list#*=}"
				test "$range" = "" && range='1-7'
				range_started=0
				range_stopped=0
				val_start=0 #is first number of current range set ?
				val_stop=0  #is last numeber of current range set ?
				for ((i=0;i<${#range};i++))
				do 
					char="${range:$i:1}"
					if [[ "$char" =~ ^[1-7]$ ]]
					then
						if [ "$range_started" -eq 0 ]
						then
							val_start="$char"
							range_started=1
						elif [ "$range_stopped" -eq 0 ]
						then
							val_stop="$char"
								range_stopped=1
						else
							echo "Error : wrong argument for --calcul-lahc-only option"
							exit 2
						fi	
					fi
					if [ "$char" = ';' -o "$i" -eq $((${#range} - 1)) ]
					then
						if [ "$range_started" -ne 1 ]
						then
							echo "Error : wrong argument for --calcul-lahc-only option"
							exit 3
						elif [ "$range_stopped" -eq 0 ]
						then
							val_stop=$val_start
						fi
						nodes2add=()
						for ((j=$val_start;j<=$val_stop;j++));do nodes2add+=(calcul-lahc-$j);done
						add_if_not_already	
						range_started=0
						range_stopped=0
						val_start=0
					elif [ "$char" = '-' ]
					then
						if [ "$range_started" -ne 1 -o "$range_stopped" -ne 0 ]
						then
						    echo "Error : wrong argument for --calcul-lahc-only option"
							exit 4
						fi
					fi
	
				done
				shift
				;;
			-H|--arch-haswell)
				nodes2add=()
				for i in `seq 5 7`;do nodes2add+=("calcul-lahc-$i");done
				add_if_not_already 
				shift
				;;
			-S|--arch-sandybridge)
				nodes2add=()
				for i in `seq 1 4`;do nodes2add+=("calcul-lahc-$i");done
				add_if_not_already 
				shift
				;;
			-W|----arch-westmere)
				nodes2add=()
				nodes2add+=('elrond')
				nodes2add+=('saroumane')
				add_if_not_already
				shift
				;;
			-H2|--arch-harpertown)
				nodes2add=()
				nodes2add+=('aragorn')
				nodes2add+=('gandalf')
				nodes2add+=('legolas')
				nodes2add+=('meriadoc')
				nodes2add+=('peregrin')
				#nodes2add+=('sam')
				add_if_not_already 
				shift
				;;
			-G|--gpu-only)
				nodes2add=()
				for i in `seq 1 3`;do nodes2add+=("calcul-gpu-lahc-$i");done
				add_if_not_already 
				shift
				;;
			-A|--amd-processor)
				nodes2add=('gimli')
				add_if_not_already 
				shift
				;;
			-*)
				echo "ERROR : Unknown option: $1" >&2
				disp_help
				exit 1
		esac
	done
fi


NB_NODES=${#nodes[@]}
# CPU
nb_cores=()
available_cores=()
available_cores_total=0
nb_cores_total=0
# RAM
total_memory=()
total_memory_unit=()
available_memory=()
available_memory_unit=()
available_memory_percent=()

connected_people=()

export -f getuser 
export -f getname
#eval $(parallel --ungroup --controlmaster --link --gnu -P 0 "getuser" ::: ${nodes[@]} ::: `seq 0 $((${#nodes[@]}-1))`)
nodes_with_ids=$(for i in $(seq 0 $(( ${#nodes[@]}-1 )) ) ; do printf '"%s %d" ' ${nodes[$i]} $i ; done )
eval set -- $nodes_with_ids # make these the current arguments to be able to use "$@" expansion
#eval $(parallel --colsep ' ' --ungroup --controlmaster --gnu -P 0 "getuser" ::: "$@")
eval $(parallel --colsep ' ' --ungroup --controlmaster --jobs 0 "getuser" ::: "$@")

export -f getcpu
#eval $(parallel --ungroup --controlmaster  --link --gnu -P 0 "getcpu" ::: ${nodes[@]} ::: `seq 0 $((${#nodes[@]}-1))`)
eval $(parallel --colsep ' ' --ungroup --controlmaster --gnu -P 0 "getcpu" ::: "$@")

export -f getram
#eval $(parallel --ungroup --controlmaster  --link --gnu -P 0 "getram" ::: ${nodes[@]} ::: `seq 0 $((${#nodes[@]}-1))`)
eval $(parallel --colsep ' ' --ungroup --controlmaster --gnu -P 0 "getram" ::: "$@")



if [ "$verbose" -ne 0 ]
then
	printf "\e[1m node name : CPU: available cores/total cores ; RAM: available RAM / total RAM (ratio of available RAM) ; connected users \e[0m\n"
	for (( i=0; i<$NB_NODES; i++ ))
	do
		if [ "${available_cores[i]}" -eq "${nb_cores[i]}" -a  "${available_memory_percent[i]}" -ge 90 ]
		then
			printf "\e[1;32m %20s : CPU: %2d/%2d ; RAM: %9ld %s/%9ld %s (%2d%%)\e[0m ;\e[1;36m %s \e[0m;\e[1;35m %s \e[0m\n" "${nodes[i]}" "${available_cores[i]}" "${nb_cores[i]}" "${available_memory[i]}" "${available_memory_unit[i]}" "${total_memory[i]}" "${total_memory_unit[i]}" "${available_memory_percent[i]}" "${connected_people_ps[i]}" "${connected_people_who[i]}"
		elif [ "${available_cores[$i]}" -ge "$((${nb_cores[$i]}/2))" -a  "${available_memory_percent[i]}" -ge 50 ]
		then
			printf "\e[1;33m %20s : CPU: %2d/%2d ; RAM: %9ld %s/%9ld %s (%2d%%)\e[0m ;\e[1;36m %s \e[0m;\e[1;35m %s \e[0m\n" "${nodes[i]}" "${available_cores[i]}" "${nb_cores[i]}" "${available_memory[i]}" "${available_memory_unit[i]}" "${total_memory[i]}" "${total_memory_unit[i]}" "${available_memory_percent[i]}" "${connected_people_ps[i]}" "${connected_people_who[i]}"
		else
			printf "\e[1;31m %20s : CPU: %2d/%2d ; RAM: %9ld %s/%9ld %s (%2d%%)\e[0m ;\e[1;36m %s \e[0m;\e[1;35m %s \e[0m\n" "${nodes[i]}" "${available_cores[i]}" "${nb_cores[i]}" "${available_memory[i]}" "${available_memory_unit[i]}" "${total_memory[i]}" "${total_memory_unit[i]}" "${available_memory_percent[i]}" "${connected_people_ps[i]}" "${connected_people_who[i]}"
		fi
	done
fi

echo ' '
existing_screens=0
for (( i=0; i<$NB_NODES; i++ ))
do
	unset scr_ls
	mapfile -t scr_ls < <(ssh ${nodes[$i]} "screen -ls | sed '/^[[:space:]]*$/d' | grep -v '^No Sockets found'")
	if [ "${#scr_ls[@]}" -ge 3 ]
	then
		echo "On ${nodes[$i]}, screen session(s) are running :"
		for (( j=1; j<${#scr_ls[@]}-1; j++ ))
		do
			echo "   ${scr_ls[j]}"
		done
		((existing_screens++))
	elif [ "${#scr_ls[@]}" -ne 0 ]
	then
		echo "Unexpected number of lines"
		exit 3
	fi
done

if [ "$existing_screens" -gt 0 ]
then
	printf "\n\e[1;31m%s\n\e[1;31;103m%s\e[0m%s\n\e[1;49;39m%s\e[0m%s\n\e[1;49;39m%s\e[0m%s\n" "WARNING : Screen sessions are running on $existing_screens different nodes." "Please CLOSE OBSOLETE SESSIONS" " : " "   ssh <node> \"screen -X -S <session_name> quit\"" " if screen session is alive" "   ssh <node> \"screen -wipe\"" " if screen session is dead"
else
	printf "\n\e[1;32m%s\e[0m\n" "No screen sessions are running"
fi

unset i j unset process_usage used_cores verbose


