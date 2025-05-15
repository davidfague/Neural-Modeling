# usage
# chmod +x setup_slurm_cluster.sh
# ./setup_slurm_cluster.sh 4

#!/bin/bash

# Usage: ./setup_slurm_cluster.sh 4
# This sets up vm0 as head and vm1-vm3 as compute nodes (total 4 nodes)

set -e

NUM_VMS=$1
CONTROL_NODE="vm0"
COMPUTE_NODES=()
for (( i=1; i<$NUM_VMS; i++ )); do
    COMPUTE_NODES+=("vm$i")
done

ALL_NODES=("$CONTROL_NODE" "${COMPUTE_NODES[@]}")

echo "==> Installing SLURM and munge on all nodes..."
for node in "${ALL_NODES[@]}"; do
    ssh "$node" "sudo apt update && sudo apt install -y slurm-wlm munge"
done

echo "==> Setting up munge on $CONTROL_NODE"
ssh "$CONTROL_NODE" "sudo /usr/sbin/create-munge-key && sudo chown -R munge:munge /etc/munge && sudo chmod 0700 /etc/munge && sudo systemctl enable munge && sudo systemctl start munge"

echo "==> Copying munge.key to compute nodes..."
for node in "${COMPUTE_NODES[@]}"; do
    scp "$CONTROL_NODE:/etc/munge/munge.key" "$node:/tmp/"
    ssh "$node" "sudo mv /tmp/munge.key /etc/munge/ && sudo chown munge:munge /etc/munge/munge.key && sudo chmod 400 /etc/munge/munge.key && sudo systemctl enable munge && sudo systemctl start munge"
done

echo "==> Generating slurm.conf"
SLURM_CONF=$(mktemp)
cat <<EOF > "$SLURM_CONF"
ClusterName=vmcluster
ControlMachine=$CONTROL_NODE
SlurmUser=slurm
SlurmctldPort=6817
SlurmdPort=6818
AuthType=auth/munge
StateSaveLocation=/var/spool/slurm-llnl/state
SlurmdSpoolDir=/var/spool/slurmd
SwitchType=switch/none
MpiDefault=none
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmdPidFile=/var/run/slurmd.pid
ProctrackType=proctrack/pgid
ReturnToService=2
SchedulerType=sched/backfill
SlurmctldTimeout=120
SlurmdTimeout=300
EOF

NODELIST="NodeName=vm[1-$(($NUM_VMS-1))] CPUs=2 State=UNKNOWN"
PARTITION="PartitionName=debug Nodes=vm[1-$(($NUM_VMS-1))] Default=YES MaxTime=INFINITE State=UP"

echo "$NODELIST" >> "$SLURM_CONF"
echo "$PARTITION" >> "$SLURM_CONF"

echo "==> Copying slurm.conf to all nodes..."
for node in "${ALL_NODES[@]}"; do
    scp "$SLURM_CONF" "$node:/tmp/slurm.conf"
    ssh "$node" "sudo mkdir -p /etc/slurm && sudo mv /tmp/slurm.conf /etc/slurm/"
done

echo "==> Creating necessary directories on all nodes..."
for node in "${ALL_NODES[@]}"; do
    ssh "$node" "sudo mkdir -p /var/spool/slurm-llnl/state /var/spool/slurmd /var/log/slurm && sudo chown -R slurm: /var/spool/slurm-llnl /var/spool/slurmd /var/log/slurm"
done

echo "==> Starting SLURM services..."
ssh "$CONTROL_NODE" "sudo systemctl enable slurmctld && sudo systemctl restart slurmctld"
for node in "${COMPUTE_NODES[@]}"; do
    ssh "$node" "sudo systemctl enable slurmd && sudo systemctl restart slurmd"
done

echo "==> SLURM cluster setup complete. You can run 'sinfo' on $CONTROL_NODE to check status."
