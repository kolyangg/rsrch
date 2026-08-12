# Independent one-GPU dynamic subject-v2 recovery

Submit `run_worker_01.yaml` through `run_worker_08.yaml` individually. Each MLS
request uses one A100 and runs the same worker against one shared atomic NFS
run-claim directory. The priority order is CL6, CL12, CL13, CL14, then E14-E22.
If CL6 fails, only its one-GPU job exits; the other jobs continue claiming the
remaining runs. Completed step manifests and verified runs are reused.

`monitor_and_refill.sh` polls Running/Pending allocations every 30 seconds and
submits workers 05-08 individually only when observed global capacity and
unclaimed work permit it. It exits on any rejected/unparsed submission and
never retries an allocation rejection.
