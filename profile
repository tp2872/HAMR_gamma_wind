==1603== NVPROF is profiling process 1603, command: harm
==1603== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==1603== Profiling application: harm
==1603== Profiling result:
==1603== Metric result:
Invocations                     Metric Name              Metric Description         Min         Max         Avg
Device "Tesla K20m (0)"
	Kernel: boundprim2(int, int, int, double*, int*, int, int, int, int, int, int)
          4        l1_cache_global_hit_rate              L1 Global Hit Rate       0.00%       0.00%       0.00%
          4         l1_cache_local_hit_rate               L1 Local Hit Rate       0.00%       0.00%       0.00%
          4                   sm_efficiency         Multiprocessor Activity      90.09%      90.55%      90.32%
          4                             ipc                    Executed IPC    0.418813    0.425849    0.420947
          4              achieved_occupancy              Achieved Occupancy    0.406043    0.408602    0.407850
          4        gld_requested_throughput  Requested Global Load Throughp  58.338GB/s  58.673GB/s  58.512GB/s
          4        gst_requested_throughput  Requested Global Store Through  58.338GB/s  58.673GB/s  58.512GB/s
          4          sm_efficiency_instance         Multiprocessor Activity      89.93%      90.78%      90.39%
          4                    ipc_instance                    Executed IPC    0.416721    0.426114    0.422367
          4            inst_replay_overhead     Instruction Replay Overhead    0.502540    0.503221    0.502845
          4          shared_replay_overhead   Shared Memory Replay Overhead    0.000000    0.000000    0.000000
          4          global_replay_overhead   Global Memory Replay Overhead    0.205235    0.205235    0.205235
          4    global_cache_replay_overhead  Global Memory Cache Replay Ove    0.000000    0.000000    0.000000
          4              tex_cache_hit_rate          Texture Cache Hit Rate       0.00%       0.00%       0.00%
          4            tex_cache_throughput        Texture Cache Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4            dram_read_throughput   Device Memory Read Throughput  67.756GB/s  67.980GB/s  67.855GB/s
          4           dram_write_throughput  Device Memory Write Throughput  74.820GB/s  75.189GB/s  75.017GB/s
          4                  gst_throughput         Global Store Throughput  58.338GB/s  58.673GB/s  58.512GB/s
          4                  gld_throughput          Global Load Throughput  58.338GB/s  58.673GB/s  58.512GB/s
          4           local_replay_overhead  Local Memory Cache Replay Over    0.000000    0.000000    0.000000
          4               shared_efficiency        Shared Memory Efficiency       0.00%       0.00%       0.00%
          4                  gld_efficiency   Global Memory Load Efficiency     100.00%     100.00%     100.00%
          4                  gst_efficiency  Global Memory Store Efficiency     100.00%     100.00%     100.00%
          4             l2_l1_read_hit_rate          L2 Hit Rate (L1 Reads)      11.40%      11.57%      11.48%
          4        l2_texture_read_hit_rate           L2 Hit Rate (Texture)       0.00%       0.00%       0.00%
          4           l2_l1_read_throughput        L2 Throughput (L1 Reads)  58.338GB/s  58.673GB/s  58.512GB/s
          4      l2_texture_read_throughput         L2 Throughput (Texture)  0.00000B/s  0.00000B/s  0.00000B/s
          4           local_memory_overhead           Local Memory Overhead       0.00%       0.00%       0.00%
          4       warp_execution_efficiency       Warp Execution Efficiency     100.00%     100.00%     100.00%
          4     nc_gld_requested_throughput  Requested Non-Coherent Global   0.00000B/s  0.00000B/s  0.00000B/s
          4                      issued_ipc                      Issued IPC    0.633381    0.637933    0.635911
          4                   inst_per_warp           Instructions per warp  322.876106  322.876106  322.876106
          4          issue_slot_utilization          Issue Slot Utilization      12.96%      13.12%      13.06%
          4  local_load_transactions_per_re  Local Memory Load Transactions    0.000000    0.000000    0.000000
          4  local_store_transactions_per_r  Local Memory Store Transaction    0.000000    0.000000    0.000000
          4  shared_load_transactions_per_r  Shared Memory Load Transaction    0.000000    0.000000    0.000000
          4  shared_store_transactions_per_  Shared Memory Store Transactio    0.000000    0.000000    0.000000
          4    gld_transactions_per_request  Global Load Transactions Per R    2.582418    2.582418    2.582418
          4    gst_transactions_per_request  Global Store Transactions Per     2.582418    2.582418    2.582418
          4         local_load_transactions         Local Load Transactions           0           0           0
          4        local_store_transactions        Local Store Transactions           0           0           0
          4        shared_load_transactions        Shared Load Transactions           0           0           0
          4       shared_store_transactions       Shared Store Transactions           0           0           0
          4                gld_transactions        Global Load Transactions       36660       36660       36660
          4                gst_transactions       Global Store Transactions       36660       36660       36660
          4        sysmem_read_transactions  System Memory Read Transaction           0           0           0
          4       sysmem_write_transactions  System Memory Write Transactio           1           7           3
          4          tex_cache_transactions      Texture Cache Transactions           0           0           0
          4          dram_read_transactions  Device Memory Read Transaction      128438      128819      128650
          4         dram_write_transactions  Device Memory Write Transactio      142021      142191      142118
          4            l2_read_transactions            L2 Read Transactions      110954      111214      111074
          4           l2_write_transactions           L2 Write Transactions      110880      110886      110882
          4           local_load_throughput    Local Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          local_store_throughput   Local Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          shared_load_throughput   Shared Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         shared_store_throughput  Shared Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4              l2_read_throughput           L2 Throughput (Reads)  58.431GB/s  58.819GB/s  58.617GB/s
          4             l2_write_throughput          L2 Throughput (Writes)  58.350GB/s  58.679GB/s  58.523GB/s
          4          sysmem_read_throughput   System Memory Read Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         sysmem_write_throughput  System Memory Write Throughput  529.23KB/s  2.6454MB/s  1.4517MB/s
          4  warp_nonpred_execution_efficie  Warp Non-Predicated Execution       95.68%      95.68%      95.68%
          4                       cf_issued  Issued Control-Flow Instructio        5136        5148        5142
          4                     cf_executed  Executed Control-Flow Instruct        5080        5080        5080
          4                     ldst_issued  Issued Load/Store Instructions      102876      103020      102953
          4                   ldst_executed  Executed Load/Store Instructio       28392       28392       28392
          4                        flops_sp                   FLOPS(Single)           0           0           0
          4                    flops_sp_add               FLOPS(Single Add)           0           0           0
          4                    flops_sp_mul               FLOPS(Single Mul)           0           0           0
          4                    flops_sp_fma               FLOPS(Single FMA)           0           0           0
          4                        flops_dp                   FLOPS(Double)           0           0           0
          4                    flops_dp_add               FLOPS(Double Add)           0           0           0
          4                    flops_dp_mul               FLOPS(Double Mul)           0           0           0
          4                    flops_dp_fma               FLOPS(Double FMA)           0           0           0
          4                flops_sp_special           FLOPS(Single Special)       65152       65152       65152
          4                stall_inst_fetch  Issue Stall Reasons (Instructi       0.16%       0.17%       0.17%
          4           stall_exec_dependency  Issue Stall Reasons (Execution      88.07%      89.39%      89.04%
          4              stall_data_request  Issue Stall Reasons (Data Requ       4.59%       4.63%       4.61%
          4                   stall_texture   Issue Stall Reasons (Texture)       0.00%       0.00%       0.00%
          4                      stall_sync  Issue Stall Reasons (Synchroni       0.00%       0.00%       0.00%
          4                     stall_other     Issue Stall Reasons (Other)       3.54%       3.65%       3.59%
          4           l1_shared_utilization    L1/Shared Memory Utilization     Low (1)     Low (1)     Low (1)
          4                  l2_utilization            L2 Cache Utilization     Low (3)     Low (3)     Low (3)
          4                 tex_utilization       Texture Cache Utilization    Idle (0)    Idle (0)    Idle (0)
          4                dram_utilization       Device Memory Utilization    High (7)    High (7)    High (7)
          4              sysmem_utilization       System Memory Utilization     Low (1)     Low (1)     Low (1)
          4             ldst_fu_utilization  Load/Store Function Unit Utili     Low (2)     Low (2)     Low (2)
          4              alu_fu_utilization  Arithmetic Function Unit Utili     Low (2)     Low (2)     Low (2)
          4               cf_fu_utilization  Control-Flow Function Unit Uti     Low (1)     Low (1)     Low (1)
          4              tex_fu_utilization  Texture Function Unit Utilizat    Idle (0)    Idle (0)    Idle (0)
          4                   inst_executed           Instructions Executed      218910      218910      218910
          4                     inst_issued             Instructions Issued      328797      329223      329043
          4                     issue_slots                     Issue Slots      271491      271828      271613
	Kernel: fixup(int, int, int, double*, int*, int*, double const *, double const *, double const *, double, double, double, double, double, double, double*, int, int, int, int, int, int)
          4        l1_cache_global_hit_rate              L1 Global Hit Rate       0.00%       0.00%       0.00%
          4         l1_cache_local_hit_rate               L1 Local Hit Rate      99.98%     100.00%      99.99%
          4                   sm_efficiency         Multiprocessor Activity      99.91%      99.93%      99.92%
          4                             ipc                    Executed IPC    1.181314    1.182471    1.182077
          4              achieved_occupancy              Achieved Occupancy    0.185009    0.185153    0.185075
          4        gld_requested_throughput  Requested Global Load Throughp  6.0401GB/s  6.0527GB/s  6.0458GB/s
          4        gst_requested_throughput  Requested Global Store Through  4.6020GB/s  4.6116GB/s  4.6064GB/s
          4          sm_efficiency_instance         Multiprocessor Activity      99.92%      99.93%      99.92%
          4                    ipc_instance                    Executed IPC    1.181601    1.182812    1.181992
          4            inst_replay_overhead     Instruction Replay Overhead    0.434160    0.434630    0.434363
          4          shared_replay_overhead   Shared Memory Replay Overhead    0.000000    0.000000    0.000000
          4          global_replay_overhead   Global Memory Replay Overhead    0.003441    0.003449    0.003445
          4    global_cache_replay_overhead  Global Memory Cache Replay Ove    0.000000    0.000000    0.000000
          4              tex_cache_hit_rate          Texture Cache Hit Rate      87.71%      87.79%      87.75%
          4            tex_cache_throughput        Texture Cache Throughput  9.5762GB/s  9.5864GB/s  9.5814GB/s
          4            dram_read_throughput   Device Memory Read Throughput  6.5865GB/s  6.5982GB/s  6.5924GB/s
          4           dram_write_throughput  Device Memory Write Throughput  5.9808GB/s  5.9933GB/s  5.9863GB/s
          4                  gst_throughput         Global Store Throughput  4.7860GB/s  4.7960GB/s  4.7906GB/s
          4                  gld_throughput          Global Load Throughput  5.2352GB/s  5.2461GB/s  5.2402GB/s
          4           local_replay_overhead  Local Memory Cache Replay Over    0.000000    0.000000    0.000000
          4               shared_efficiency        Shared Memory Efficiency       0.00%       0.00%       0.00%
          4                  gld_efficiency   Global Memory Load Efficiency     115.37%     115.37%     115.37%
          4                  gst_efficiency  Global Memory Store Efficiency      96.15%      96.15%      96.15%
          4             l2_l1_read_hit_rate          L2 Hit Rate (L1 Reads)       2.64%       2.64%       2.64%
          4        l2_texture_read_hit_rate           L2 Hit Rate (Texture)      91.30%      91.95%      91.77%
          4           l2_l1_read_throughput        L2 Throughput (L1 Reads)  5.2352GB/s  5.2461GB/s  5.2402GB/s
          4      l2_texture_read_throughput         L2 Throughput (Texture)  1.1732GB/s  1.1766GB/s  1.1746GB/s
          4           local_memory_overhead           Local Memory Overhead       0.00%       0.00%       0.00%
          4       warp_execution_efficiency       Warp Execution Efficiency      95.93%      95.95%      95.94%
          4     nc_gld_requested_throughput  Requested Non-Coherent Global   9.2039GB/s  9.2231GB/s  9.2127GB/s
          4                      issued_ipc                      Issued IPC    1.693742    1.694423    1.694094
          4                   inst_per_warp           Instructions per warp  4.2753e+03  4.2853e+03  4.2808e+03
          4          issue_slot_utilization          Issue Slot Utilization      37.28%      37.30%      37.29%
          4  local_load_transactions_per_re  Local Memory Load Transactions    0.000000    0.000000    0.000000
          4  local_store_transactions_per_r  Local Memory Store Transaction    1.000000    1.000000    1.000000
          4  shared_load_transactions_per_r  Shared Memory Load Transaction    0.000000    0.000000    0.000000
          4  shared_store_transactions_per_  Shared Memory Store Transactio    0.000000    0.000000    0.000000
          4    gld_transactions_per_request  Global Load Transactions Per R    1.722811    1.722811    1.722811
          4    gst_transactions_per_request  Global Store Transactions Per     1.993865    1.993865    1.993865
          4         local_load_transactions         Local Load Transactions           0           0           0
          4        local_store_transactions        Local Store Transactions       32600       32600       32600
          4        shared_load_transactions        Shared Load Transactions           0           0           0
          4       shared_store_transactions       Shared Store Transactions           0           0           0
          4                gld_transactions        Global Load Transactions      617800      617800      617800
          4                gst_transactions       Global Store Transactions      520000      520000      520000
          4        sysmem_read_transactions  System Memory Read Transaction           0           0           0
          4       sysmem_write_transactions  System Memory Write Transactio           1          10           5
          4          tex_cache_transactions      Texture Cache Transactions     4159360     4161664     4160832
          4          dram_read_transactions  Device Memory Read Transaction     2862547     2864492     2863305
          4         dram_write_transactions  Device Memory Write Transactio     2598982     2599201     2599131
          4            l2_read_transactions            L2 Read Transactions     2793581     2794332     2793977
          4           l2_write_transactions           L2 Write Transactions     2080014     2080030     2080023
          4           local_load_throughput    Local Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          local_store_throughput   Local Memory Store Throughput  300.05MB/s  300.67MB/s  300.32MB/s
          4          shared_load_throughput   Shared Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         shared_store_throughput  Shared Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4              l2_read_throughput           L2 Throughput (Reads)  6.4294GB/s  6.4416GB/s  6.4349GB/s
          4             l2_write_throughput          L2 Throughput (Writes)  4.7861GB/s  4.7969GB/s  4.7909GB/s
          4          sysmem_read_throughput   System Memory Read Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         sysmem_write_throughput  System Memory Write Throughput  4.6110KB/s  18.419KB/s  10.362KB/s
          4  warp_nonpred_execution_efficie  Warp Non-Predicated Execution       92.59%      92.61%      92.60%
          4                       cf_issued  Issued Control-Flow Instructio    20097547    20229792    20170318
          4                     cf_executed  Executed Control-Flow Instruct    19083041    19207508    19151446
          4                     ldst_issued  Issued Load/Store Instructions     1637332     1638563     1637832
          4                   ldst_executed  Executed Load/Store Instructio      652000      652000      652000
          4                        flops_sp                   FLOPS(Single)           0           0           0
          4                    flops_sp_add               FLOPS(Single Add)           0           0           0
          4                    flops_sp_mul               FLOPS(Single Mul)           0           0           0
          4                    flops_sp_fma               FLOPS(Single FMA)           0           0           0
          4                        flops_dp                   FLOPS(Double)  2428609309  2428915880  2428783027
          4                    flops_dp_add               FLOPS(Double Add)   356411600   356411600   356411600
          4                    flops_dp_mul               FLOPS(Double Mul)   300324089   300367866   300348978
          4                    flops_dp_fma               FLOPS(Double FMA)   885936810   886068207   886011224
          4                flops_sp_special           FLOPS(Single Special)    78107440    78107660    78107549
          4                stall_inst_fetch  Issue Stall Reasons (Instructi      12.83%      12.84%      12.84%
          4           stall_exec_dependency  Issue Stall Reasons (Execution      72.60%      72.62%      72.61%
          4              stall_data_request  Issue Stall Reasons (Data Requ       0.87%       0.87%       0.87%
          4                   stall_texture   Issue Stall Reasons (Texture)       0.78%       0.78%       0.78%
          4                      stall_sync  Issue Stall Reasons (Synchroni       0.00%       0.00%       0.00%
          4                     stall_other     Issue Stall Reasons (Other)       5.03%       5.03%       5.03%
          4           l1_shared_utilization    L1/Shared Memory Utilization     Low (1)     Low (1)     Low (1)
          4                  l2_utilization            L2 Cache Utilization     Low (1)     Low (1)     Low (1)
          4                 tex_utilization       Texture Cache Utilization     Low (1)     Low (1)     Low (1)
          4                dram_utilization       Device Memory Utilization     Low (1)     Low (1)     Low (1)
          4              sysmem_utilization       System Memory Utilization     Low (1)     Low (1)     Low (1)
          4             ldst_fu_utilization  Load/Store Function Unit Utili     Low (1)     Low (1)     Low (1)
          4              alu_fu_utilization  Arithmetic Function Unit Utili     Low (3)     Low (3)     Low (3)
          4               cf_fu_utilization  Control-Flow Function Unit Uti     Low (1)     Low (1)     Low (1)
          4              tex_fu_utilization  Texture Function Unit Utilizat     Low (1)     Low (1)     Low (1)
          4                   inst_executed           Instructions Executed   150295393   150645408   150486018
          4                     inst_issued             Instructions Issued   215616526   216051152   215851217
          4                     issue_slots                     Issue Slots   189736942   190111496   189939000
	Kernel: Utoprim(int, int, int, double*, double*, double*, double const *, double*, double*, double const *, double const *, double const *, double const *, double, double, double, double, double, double, double*, int, int, int)
          4        l1_cache_global_hit_rate              L1 Global Hit Rate       0.00%       0.00%       0.00%
          4         l1_cache_local_hit_rate               L1 Local Hit Rate       0.00%       0.00%       0.00%
          4                   sm_efficiency         Multiprocessor Activity      99.82%      99.86%      99.84%
          4                             ipc                    Executed IPC    1.470934    1.475652    1.474120
          4              achieved_occupancy              Achieved Occupancy    0.307611    0.307876    0.307759
          4        gld_requested_throughput  Requested Global Load Throughp  69.366GB/s  69.735GB/s  69.606GB/s
          4        gst_requested_throughput  Requested Global Store Through  10.276GB/s  10.331GB/s  10.312GB/s
          4          sm_efficiency_instance         Multiprocessor Activity      99.83%      99.86%      99.83%
          4                    ipc_instance                    Executed IPC    1.471908    1.475267    1.473987
          4            inst_replay_overhead     Instruction Replay Overhead    0.374215    0.374868    0.374577
          4          shared_replay_overhead   Shared Memory Replay Overhead    0.000000    0.000000    0.000000
          4          global_replay_overhead   Global Memory Replay Overhead    0.029440    0.029631    0.029563
          4    global_cache_replay_overhead  Global Memory Cache Replay Ove    0.000000    0.000000    0.000000
          4              tex_cache_hit_rate          Texture Cache Hit Rate      68.80%      68.82%      68.81%
          4            tex_cache_throughput        Texture Cache Throughput  18.726GB/s  18.801GB/s  18.773GB/s
          4            dram_read_throughput   Device Memory Read Throughput  70.179GB/s  70.571GB/s  70.434GB/s
          4           dram_write_throughput  Device Memory Write Throughput  13.851GB/s  13.922GB/s  13.898GB/s
          4                  gst_throughput         Global Store Throughput  10.688GB/s  10.744GB/s  10.724GB/s
          4                  gld_throughput          Global Load Throughput  72.205GB/s  72.589GB/s  72.455GB/s
          4           local_replay_overhead  Local Memory Cache Replay Over    0.000000    0.000000    0.000000
          4               shared_efficiency        Shared Memory Efficiency       0.00%       0.00%       0.00%
          4                  gld_efficiency   Global Memory Load Efficiency      96.07%      96.07%      96.07%
          4                  gst_efficiency  Global Memory Store Efficiency      96.15%      96.15%      96.15%
          4             l2_l1_read_hit_rate          L2 Hit Rate (L1 Reads)      31.36%      31.37%      31.36%
          4        l2_texture_read_hit_rate           L2 Hit Rate (Texture)      28.72%      28.79%      28.75%
          4           l2_l1_read_throughput        L2 Throughput (L1 Reads)  72.205GB/s  72.589GB/s  72.455GB/s
          4      l2_texture_read_throughput         L2 Throughput (Texture)  5.8379GB/s  5.8627GB/s  5.8548GB/s
          4           local_memory_overhead           Local Memory Overhead       0.00%       0.00%       0.00%
          4       warp_execution_efficiency       Warp Execution Efficiency      96.10%      96.37%      96.30%
          4     nc_gld_requested_throughput  Requested Non-Coherent Global   17.984GB/s  18.079GB/s  18.046GB/s
          4                      issued_ipc                      Issued IPC    2.025304    2.028586    2.026886
          4                   inst_per_warp           Instructions per warp  2.3781e+03  2.3936e+03  2.3836e+03
          4          issue_slot_utilization          Issue Slot Utilization      42.90%      42.96%      42.94%
          4  local_load_transactions_per_re  Local Memory Load Transactions    0.000000    0.000000    0.000000
          4  local_store_transactions_per_r  Local Memory Store Transaction    0.000000    0.000000    0.000000
          4  shared_load_transactions_per_r  Shared Memory Load Transaction    0.000000    0.000000    0.000000
          4  shared_store_transactions_per_  Shared Memory Store Transactio    0.000000    0.000000    0.000000
          4    gld_transactions_per_request  Global Load Transactions Per R    2.259941    2.259941    2.259941
          4    gst_transactions_per_request  Global Store Transactions Per     1.993865    1.993865    1.993865
          4         local_load_transactions         Local Load Transactions           0           0           0
          4        local_store_transactions        Local Store Transactions           0           0           0
          4        shared_load_transactions        Shared Load Transactions           0           0           0
          4       shared_store_transactions       Shared Store Transactions           0           0           0
          4                gld_transactions        Global Load Transactions     3978400     3978400     3978400
          4                gst_transactions       Global Store Transactions      520000      520000      520000
          4        sysmem_read_transactions  System Memory Read Transaction           0           0           0
          4       sysmem_write_transactions  System Memory Write Transactio           4           9           5
          4          tex_cache_transactions      Texture Cache Transactions     3634624     3642464     3638544
          4          dram_read_transactions  Device Memory Read Transaction    13658490    13664588    13661738
          4         dram_write_transactions  Device Memory Write Transactio     2694876     2695926     2695246
          4            l2_read_transactions            L2 Read Transactions    15191926    15194770    15192861
          4           l2_write_transactions           L2 Write Transactions     2239072     2239086     2239079
          4           local_load_throughput    Local Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          local_store_throughput   Local Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          shared_load_throughput   Shared Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         shared_store_throughput  Shared Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4              l2_read_throughput           L2 Throughput (Reads)  78.071GB/s  78.477GB/s  78.334GB/s
          4             l2_write_throughput          L2 Throughput (Writes)  11.505GB/s  11.566GB/s  11.545GB/s
          4          sysmem_read_throughput   System Memory Read Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         sysmem_write_throughput  System Memory Write Throughput  0.00000B/s  56.820KB/s  27.064KB/s
          4  warp_nonpred_execution_efficie  Warp Non-Predicated Execution       91.87%      92.15%      92.07%
          4                       cf_issued  Issued Control-Flow Instructio    13053539    13223860    13118605
          4                     cf_executed  Executed Control-Flow Instruct    12361523    12524473    12424052
          4                     ldst_issued  Issued Load/Store Instructions     4541072     4541651     4541248
          4                   ldst_executed  Executed Load/Store Instructio     2021200     2021200     2021200
          4                        flops_sp                   FLOPS(Single)           0           0           0
          4                    flops_sp_add               FLOPS(Single Add)           0           0           0
          4                    flops_sp_mul               FLOPS(Single Mul)           0           0           0
          4                    flops_sp_fma               FLOPS(Single FMA)           0           0           0
          4                        flops_dp                   FLOPS(Double)   833117440   833253240   833185340
          4                    flops_dp_add               FLOPS(Double Add)    70124352    70124352    70124352
          4                    flops_dp_mul               FLOPS(Double Mul)   109999424   110018824   110009124
          4                    flops_dp_fma               FLOPS(Double FMA)   326496832   326555032   326525932
          4                flops_sp_special           FLOPS(Single Special)    45249280    45249280    45249280
          4                stall_inst_fetch  Issue Stall Reasons (Instructi      10.07%      10.10%      10.08%
          4           stall_exec_dependency  Issue Stall Reasons (Execution      74.99%      75.00%      74.99%
          4              stall_data_request  Issue Stall Reasons (Data Requ       1.66%       1.67%       1.66%
          4                   stall_texture   Issue Stall Reasons (Texture)       0.92%       0.93%       0.93%
          4                      stall_sync  Issue Stall Reasons (Synchroni       0.00%       0.00%       0.00%
          4                     stall_other     Issue Stall Reasons (Other)       4.75%       4.76%       4.75%
          4           l1_shared_utilization    L1/Shared Memory Utilization     Low (1)     Low (1)     Low (1)
          4                  l2_utilization            L2 Cache Utilization     Low (2)     Low (2)     Low (2)
          4                 tex_utilization       Texture Cache Utilization     Low (1)     Low (1)     Low (1)
          4                dram_utilization       Device Memory Utilization     Mid (5)     Mid (5)     Mid (5)
          4              sysmem_utilization       System Memory Utilization     Low (1)     Low (1)     Low (1)
          4             ldst_fu_utilization  Load/Store Function Unit Utili     Low (1)     Low (1)     Low (1)
          4              alu_fu_utilization  Arithmetic Function Unit Utili     Low (3)     Low (3)     Low (3)
          4               cf_fu_utilization  Control-Flow Function Unit Uti     Low (1)     Low (1)     Low (1)
          4              tex_fu_utilization  Texture Function Unit Utilizat     Low (1)     Low (1)     Low (1)
          4                   inst_executed           Instructions Executed    83600886    84144422    83793592
          4                     inst_issued             Instructions Issued   114938114   115625179   115179192
          4                     issue_slots                     Issue Slots    97414331    98012510    97626630
	Kernel: fluxcalc2D2(int, int, int, double*, double const *, double const *, double*, double const *, double const *, double const *, double const *, double const *, double const *, double const *, double, double, double, double, double, int, int, int, int)
         12        l1_cache_global_hit_rate              L1 Global Hit Rate       0.00%       0.00%       0.00%
         12         l1_cache_local_hit_rate               L1 Local Hit Rate      60.55%      61.31%      61.00%
         12                   sm_efficiency         Multiprocessor Activity      99.80%      99.89%      99.85%
         12                             ipc                    Executed IPC    1.281498    1.321286    1.302966
         12              achieved_occupancy              Achieved Occupancy    0.245912    0.246949    0.246564
         12        gld_requested_throughput  Requested Global Load Throughp  28.816GB/s  30.437GB/s  29.960GB/s
         12        gst_requested_throughput  Requested Global Store Through  11.001GB/s  11.619GB/s  11.437GB/s
         12          sm_efficiency_instance         Multiprocessor Activity      99.82%      99.89%      99.85%
         12                    ipc_instance                    Executed IPC    1.293943    1.321179    1.305798
         12            inst_replay_overhead     Instruction Replay Overhead    0.361109    0.377397    0.371113
         12          shared_replay_overhead   Shared Memory Replay Overhead    0.000239    0.000255    0.000250
         12          global_replay_overhead   Global Memory Replay Overhead    0.012906    0.013639    0.013375
         12    global_cache_replay_overhead  Global Memory Cache Replay Ove    0.000000    0.000000    0.000000
         12              tex_cache_hit_rate          Texture Cache Hit Rate      47.98%      48.03%      48.00%
         12            tex_cache_throughput        Texture Cache Throughput  43.758GB/s  45.904GB/s  45.251GB/s
         12            dram_read_throughput   Device Memory Read Throughput  65.959GB/s  68.971GB/s  68.141GB/s
         12           dram_write_throughput  Device Memory Write Throughput  46.602GB/s  48.441GB/s  47.913GB/s
         12                  gst_throughput         Global Store Throughput  28.246GB/s  29.290GB/s  28.920GB/s
         12                  gld_throughput          Global Load Throughput  28.450GB/s  29.730GB/s  29.370GB/s
         12           local_replay_overhead  Local Memory Cache Replay Over    0.010529    0.011081    0.010865
         12               shared_efficiency        Shared Memory Efficiency      97.82%      97.82%      97.82%
         12                  gld_efficiency   Global Memory Load Efficiency     101.25%     102.38%     102.02%
         12                  gst_efficiency  Global Memory Store Efficiency      38.92%      39.92%      39.55%
         12             l2_l1_read_hit_rate          L2 Hit Rate (L1 Reads)       6.26%       6.50%       6.37%
         12        l2_texture_read_hit_rate           L2 Hit Rate (Texture)      12.05%      12.13%      12.10%
         12           l2_l1_read_throughput        L2 Throughput (L1 Reads)  30.570GB/s  31.933GB/s  31.556GB/s
         12      l2_texture_read_throughput         L2 Throughput (Texture)  22.778GB/s  23.817GB/s  23.524GB/s
         12           local_memory_overhead           Local Memory Overhead       6.06%       6.20%       6.14%
         12       warp_execution_efficiency       Warp Execution Efficiency      97.36%      98.28%      97.92%
         12     nc_gld_requested_throughput  Requested Non-Coherent Global   42.538GB/s  44.930GB/s  44.226GB/s
         12                      issued_ipc                      Issued IPC    1.763299    1.817199    1.786588
         12                   inst_per_warp           Instructions per warp  1.9593e+03  2.0899e+03  2.0028e+03
         12          issue_slot_utilization          Issue Slot Utilization      37.78%      38.99%      38.31%
         12  local_load_transactions_per_re  Local Memory Load Transactions    2.000134    2.000359    2.000213
         12  local_store_transactions_per_r  Local Memory Store Transaction    2.264662    2.277752    2.271726
         12  shared_load_transactions_per_r  Shared Memory Load Transaction    1.032258    1.032258    1.032258
         12  shared_store_transactions_per_  Shared Memory Store Transactio    1.000000    1.000000    1.000000
         12    gld_transactions_per_request  Global Load Transactions Per R    1.949486    1.952381    1.951416
         12    gst_transactions_per_request  Global Store Transactions Per     1.935698    1.939072    1.937759
         12         local_load_transactions         Local Load Transactions      402417      406421      403858
         12        local_store_transactions        Local Store Transactions     2429884     2468221     2445318
         12        shared_load_transactions        Shared Load Transactions      562464      562464      562464
         12       shared_store_transactions       Shared Store Transactions      244456      244788      244575
         12                gld_transactions        Global Load Transactions     1373838     1388424     1379024
         12                gst_transactions       Global Store Transactions      553689      559401      555726
         12        sysmem_read_transactions  System Memory Read Transaction           0           0           0
         12       sysmem_write_transactions  System Memory Write Transactio           0           9           4
         12          tex_cache_transactions      Texture Cache Transactions     8290020     8393064     8330258
         12          dram_read_transactions  Device Memory Read Transaction    12500417    12635394    12549407
         12         dram_write_transactions  Device Memory Write Transactio     8757087     8909032     8819744
         12            l2_read_transactions            L2 Read Transactions    10106621    10218319    10146905
         12           l2_write_transactions           L2 Write Transactions     7865351     8009448     7918908
         12           local_load_throughput    Local Memory Load Throughput  8.4905GB/s  8.8870GB/s  8.7718GB/s
         12          local_store_throughput   Local Memory Store Throughput  28.805GB/s  30.612GB/s  30.051GB/s
         12          shared_load_throughput   Shared Memory Load Throughput  23.504GB/s  24.826GB/s  24.436GB/s
         12         shared_store_throughput  Shared Memory Store Throughput  10.229GB/s  10.791GB/s  10.626GB/s
         12              l2_read_throughput           L2 Throughput (Reads)  53.364GB/s  55.794GB/s  55.101GB/s
         12             l2_write_throughput          L2 Throughput (Writes)  41.890GB/s  43.422GB/s  43.003GB/s
         12          sysmem_read_throughput   System Memory Read Throughput  0.00000B/s  0.00000B/s  0.00000B/s
         12         sysmem_write_throughput  System Memory Write Throughput  5.3960KB/s  49.585KB/s  27.630KB/s
         12  warp_nonpred_execution_efficie  Warp Non-Predicated Execution       91.25%      92.90%      92.36%
         12                       cf_issued  Issued Control-Flow Instructio     8369959     9914984     8634711
         12                     cf_executed  Executed Control-Flow Instruct     7903785     9390751     8154665
         12                     ldst_issued  Issued Load/Store Instructions     6023551     6109691     6059451
         12                   ldst_executed  Executed Load/Store Instructio     3053564     3076140     3061678
         12                        flops_sp                   FLOPS(Single)           0           0           0
         12                    flops_sp_add               FLOPS(Single Add)           0           0           0
         12                    flops_sp_mul               FLOPS(Single Mul)           0           0           0
         12                    flops_sp_fma               FLOPS(Single FMA)           0           0           0
         12                        flops_dp                   FLOPS(Double)   809254260   876891312   831845480
         12                    flops_dp_add               FLOPS(Double Add)    50438592    60562944    53813376
         12                    flops_dp_mul               FLOPS(Double Mul)   131498748   137143584   133386858
         12                    flops_dp_fma               FLOPS(Double FMA)   313658460   339592392   322322622
         12                flops_sp_special           FLOPS(Single Special)    31968864    34218753    32718823
         12                stall_inst_fetch  Issue Stall Reasons (Instructi       8.29%       8.81%       8.49%
         12           stall_exec_dependency  Issue Stall Reasons (Execution      66.67%      67.71%      67.18%
         12              stall_data_request  Issue Stall Reasons (Data Requ       3.30%       3.81%       3.55%
         12                   stall_texture   Issue Stall Reasons (Texture)       6.53%       7.35%       6.99%
         12                      stall_sync  Issue Stall Reasons (Synchroni       1.96%       2.26%       2.12%
         12                     stall_other     Issue Stall Reasons (Other)       4.90%       4.99%       4.96%
         12           l1_shared_utilization    L1/Shared Memory Utilization     Low (1)     Low (1)     Low (1)
         12                  l2_utilization            L2 Cache Utilization     Low (3)     Low (3)     Low (3)
         12                 tex_utilization       Texture Cache Utilization     Low (1)     Low (1)     Low (1)
         12                dram_utilization       Device Memory Utilization     Mid (6)     Mid (6)     Mid (6)
         12              sysmem_utilization       System Memory Utilization     Low (1)     Low (1)     Low (1)
         12             ldst_fu_utilization  Load/Store Function Unit Utili     Low (2)     Low (2)     Low (2)
         12              alu_fu_utilization  Arithmetic Function Unit Utili     Low (3)     Low (3)     Low (3)
         12               cf_fu_utilization  Control-Flow Function Unit Uti     Low (1)     Low (1)     Low (1)
         12              tex_fu_utilization  Texture Function Unit Utilizat     Low (1)     Low (1)     Low (1)
         12                   inst_executed           Instructions Executed    68877719    73467308    70407466
         12                     inst_issued             Instructions Issued    94316634    99997073    96533998
         12                     issue_slots                     Issue Slots    80831537    85706991    82807738
	Kernel: Utoprim1(int, int, int, double*, double*, double*, double*, double const *, double*, double const *, double const *, double const *, double const *, double, double*)
          4        l1_cache_global_hit_rate              L1 Global Hit Rate       0.00%       0.00%       0.00%
          4         l1_cache_local_hit_rate               L1 Local Hit Rate       0.00%       0.00%       0.00%
          4                   sm_efficiency         Multiprocessor Activity      99.73%      99.78%      99.75%
          4                             ipc                    Executed IPC    1.628588    1.633929    1.630255
          4              achieved_occupancy              Achieved Occupancy    0.305174    0.305605    0.305370
          4        gld_requested_throughput  Requested Global Load Throughp  22.720GB/s  22.756GB/s  22.736GB/s
          4        gst_requested_throughput  Requested Global Store Through  22.720GB/s  22.756GB/s  22.736GB/s
          4          sm_efficiency_instance         Multiprocessor Activity      99.75%      99.79%      99.77%
          4                    ipc_instance                    Executed IPC    1.629590    1.631956    1.630774
          4            inst_replay_overhead     Instruction Replay Overhead    0.282529    0.282644    0.282599
          4          shared_replay_overhead   Shared Memory Replay Overhead    0.000000    0.000000    0.000000
          4          global_replay_overhead   Global Memory Replay Overhead    0.012326    0.012348    0.012337
          4    global_cache_replay_overhead  Global Memory Cache Replay Ove    0.000000    0.000000    0.000000
          4              tex_cache_hit_rate          Texture Cache Hit Rate      75.82%      75.83%      75.83%
          4            tex_cache_throughput        Texture Cache Throughput  177.07GB/s  177.71GB/s  177.39GB/s
          4            dram_read_throughput   Device Memory Read Throughput  62.209GB/s  62.278GB/s  62.234GB/s
          4           dram_write_throughput  Device Memory Write Throughput  29.378GB/s  29.421GB/s  29.397GB/s
          4                  gst_throughput         Global Store Throughput  23.629GB/s  23.666GB/s  23.646GB/s
          4                  gld_throughput          Global Load Throughput  23.629GB/s  23.666GB/s  23.646GB/s
          4           local_replay_overhead  Local Memory Cache Replay Over    0.000000    0.000000    0.000000
          4               shared_efficiency        Shared Memory Efficiency       0.00%       0.00%       0.00%
          4                  gld_efficiency   Global Memory Load Efficiency      96.15%      96.15%      96.15%
          4                  gst_efficiency  Global Memory Store Efficiency      96.15%      96.15%      96.15%
          4             l2_l1_read_hit_rate          L2 Hit Rate (L1 Reads)       0.00%       0.00%       0.00%
          4        l2_texture_read_hit_rate           L2 Hit Rate (Texture)      41.22%      41.24%      41.23%
          4           l2_l1_read_throughput        L2 Throughput (L1 Reads)  23.629GB/s  23.666GB/s  23.646GB/s
          4      l2_texture_read_throughput         L2 Throughput (Texture)  42.821GB/s  42.891GB/s  42.851GB/s
          4           local_memory_overhead           Local Memory Overhead       0.00%       0.00%       0.00%
          4       warp_execution_efficiency       Warp Execution Efficiency      96.29%      96.30%      96.30%
          4     nc_gld_requested_throughput  Requested Non-Coherent Global   170.40GB/s  170.67GB/s  170.52GB/s
          4                      issued_ipc                      Issued IPC    2.086074    2.093041    2.089588
          4                   inst_per_warp           Instructions per warp  1.1943e+03  1.1963e+03  1.1953e+03
          4          issue_slot_utilization          Issue Slot Utilization      45.48%      45.55%      45.52%
          4  local_load_transactions_per_re  Local Memory Load Transactions    0.000000    0.000000    0.000000
          4  local_store_transactions_per_r  Local Memory Store Transaction    0.000000    0.000000    0.000000
          4  shared_load_transactions_per_r  Shared Memory Load Transaction    0.000000    0.000000    0.000000
          4  shared_store_transactions_per_  Shared Memory Store Transactio    0.000000    0.000000    0.000000
          4    gld_transactions_per_request  Global Load Transactions Per R    1.993865    1.993865    1.993865
          4    gst_transactions_per_request  Global Store Transactions Per     1.993865    1.993865    1.993865
          4         local_load_transactions         Local Load Transactions           0           0           0
          4        local_store_transactions        Local Store Transactions           0           0           0
          4        shared_load_transactions        Shared Load Transactions           0           0           0
          4       shared_store_transactions       Shared Store Transactions           0           0           0
          4                gld_transactions        Global Load Transactions      520000      520000      520000
          4                gst_transactions       Global Store Transactions      520000      520000      520000
          4        sysmem_read_transactions  System Memory Read Transaction           0           0           0
          4       sysmem_write_transactions  System Memory Write Transactio           0           7           3
          4          tex_cache_transactions      Texture Cache Transactions    15589440    15616800    15604200
          4          dram_read_transactions  Device Memory Read Transaction     5471710     5473235     5472289
          4         dram_write_transactions  Device Memory Write Transactio     2585699     2585938     2585815
          4            l2_read_transactions            L2 Read Transactions     5851773     5853222     5852710
          4           l2_write_transactions           L2 Write Transactions     2080018     2080025     2080022
          4           local_load_throughput    Local Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          local_store_throughput   Local Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          shared_load_throughput   Shared Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         shared_store_throughput  Shared Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4              l2_read_throughput           L2 Throughput (Reads)  66.475GB/s  66.598GB/s  66.535GB/s
          4             l2_write_throughput          L2 Throughput (Writes)  23.629GB/s  23.667GB/s  23.646GB/s
          4          sysmem_read_throughput   System Memory Read Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         sysmem_write_throughput  System Memory Write Throughput  11.367KB/s  79.566KB/s  48.315KB/s
          4  warp_nonpred_execution_efficie  Warp Non-Predicated Execution       93.36%      93.37%      93.37%
          4                       cf_issued  Issued Control-Flow Instructio     4807606     4835311     4821986
          4                     cf_executed  Executed Control-Flow Instruct     4576296     4602972     4590033
          4                     ldst_issued  Issued Load/Store Instructions     1322088     1323670     1322886
          4                   ldst_executed  Executed Load/Store Instructio      521600      521600      521600
          4                        flops_sp                   FLOPS(Single)           0           0           0
          4                    flops_sp_add               FLOPS(Single Add)           0           0           0
          4                    flops_sp_mul               FLOPS(Single Mul)           0           0           0
          4                    flops_sp_fma               FLOPS(Single FMA)           0           0           0
          4                        flops_dp                   FLOPS(Double)   466000000   466143500   466092225
          4                    flops_dp_add               FLOPS(Double Add)     7000000     7000000     7000000
          4                    flops_dp_mul               FLOPS(Double Mul)    49000000    49020500    49013175
          4                    flops_dp_fma               FLOPS(Double FMA)   205000000   205061500   205039525
          4                flops_sp_special           FLOPS(Single Special)    15624640    15624640    15624640
          4                stall_inst_fetch  Issue Stall Reasons (Instructi       5.76%       5.78%       5.77%
          4           stall_exec_dependency  Issue Stall Reasons (Execution      60.38%      60.46%      60.42%
          4              stall_data_request  Issue Stall Reasons (Data Requ       0.39%       0.39%       0.39%
          4                   stall_texture   Issue Stall Reasons (Texture)      19.03%      19.12%      19.06%
          4                      stall_sync  Issue Stall Reasons (Synchroni       0.00%       0.00%       0.00%
          4                     stall_other     Issue Stall Reasons (Other)       5.62%       5.65%       5.63%
          4           l1_shared_utilization    L1/Shared Memory Utilization     Low (1)     Low (1)     Low (1)
          4                  l2_utilization            L2 Cache Utilization     Low (2)     Low (2)     Low (2)
          4                 tex_utilization       Texture Cache Utilization     Low (2)     Low (2)     Low (2)
          4                dram_utilization       Device Memory Utilization     Mid (5)     Mid (5)     Mid (5)
          4              sysmem_utilization       System Memory Utilization     Low (1)     Low (1)     Low (1)
          4             ldst_fu_utilization  Load/Store Function Unit Utili     Low (1)     Low (1)     Low (1)
          4              alu_fu_utilization  Arithmetic Function Unit Utili     Mid (4)     Mid (4)     Mid (4)
          4               cf_fu_utilization  Control-Flow Function Unit Uti     Low (1)     Low (1)     Low (1)
          4              tex_fu_utilization  Texture Function Unit Utilizat     Low (1)     Low (1)     Low (1)
          4                   inst_executed           Instructions Executed    41983147    42056322    42021165
          4                     inst_issued             Instructions Issued    53848481    53941317    53896869
          4                     issue_slots                     Issue Slots    46899705    46981093    46942215
	Kernel: fix_flux(int, int, int, int, int, int, int, double*, double*, double*)
          4        l1_cache_global_hit_rate              L1 Global Hit Rate       0.00%       0.00%       0.00%
          4         l1_cache_local_hit_rate               L1 Local Hit Rate       0.00%       0.00%       0.00%
          4                   sm_efficiency         Multiprocessor Activity      76.29%      78.67%      77.46%
          4                             ipc                    Executed IPC    0.630826    0.638628    0.634996
          4              achieved_occupancy              Achieved Occupancy    0.400327    0.407644    0.404249
          4        gld_requested_throughput  Requested Global Load Throughp  20.352GB/s  20.375GB/s  20.368GB/s
          4        gst_requested_throughput  Requested Global Store Through  71.231GB/s  71.313GB/s  71.288GB/s
          4          sm_efficiency_instance         Multiprocessor Activity      75.81%      77.61%      76.63%
          4                    ipc_instance                    Executed IPC    0.630315    0.646156    0.638677
          4            inst_replay_overhead     Instruction Replay Overhead    0.571751    0.597204    0.579321
          4          shared_replay_overhead   Shared Memory Replay Overhead    0.000000    0.000000    0.000000
          4          global_replay_overhead   Global Memory Replay Overhead    0.116807    0.116807    0.116807
          4    global_cache_replay_overhead  Global Memory Cache Replay Ove    0.000000    0.000000    0.000000
          4              tex_cache_hit_rate          Texture Cache Hit Rate       0.00%       0.00%       0.00%
          4            tex_cache_throughput        Texture Cache Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4            dram_read_throughput   Device Memory Read Throughput  34.145GB/s  34.337GB/s  34.219GB/s
          4           dram_write_throughput  Device Memory Write Throughput  96.359GB/s  96.624GB/s  96.504GB/s
          4                  gst_throughput         Global Store Throughput  72.628GB/s  72.711GB/s  72.686GB/s
          4                  gld_throughput          Global Load Throughput  20.751GB/s  20.775GB/s  20.767GB/s
          4           local_replay_overhead  Local Memory Cache Replay Over    0.000000    0.000000    0.000000
          4               shared_efficiency        Shared Memory Efficiency       0.00%       0.00%       0.00%
          4                  gld_efficiency   Global Memory Load Efficiency      98.08%      98.08%      98.08%
          4                  gst_efficiency  Global Memory Store Efficiency      98.08%      98.08%      98.08%
          4             l2_l1_read_hit_rate          L2 Hit Rate (L1 Reads)       0.00%       0.00%       0.00%
          4        l2_texture_read_hit_rate           L2 Hit Rate (Texture)       0.00%       0.00%       0.00%
          4           l2_l1_read_throughput        L2 Throughput (L1 Reads)  20.751GB/s  20.775GB/s  20.767GB/s
          4      l2_texture_read_throughput         L2 Throughput (Texture)  0.00000B/s  0.00000B/s  0.00000B/s
          4           local_memory_overhead           Local Memory Overhead       0.00%       0.00%       0.00%
          4       warp_execution_efficiency       Warp Execution Efficiency      98.71%      98.71%      98.71%
          4     nc_gld_requested_throughput  Requested Non-Coherent Global   0.00000B/s  0.00000B/s  0.00000B/s
          4                      issued_ipc                      Issued IPC    0.983867    1.005821    0.992691
          4                   inst_per_warp           Instructions per warp  174.707965  174.707965  174.707965
          4          issue_slot_utilization          Issue Slot Utilization      20.57%      21.07%      20.90%
          4  local_load_transactions_per_re  Local Memory Load Transactions    0.000000    0.000000    0.000000
          4  local_store_transactions_per_r  Local Memory Store Transaction    0.000000    0.000000    0.000000
          4  shared_load_transactions_per_r  Shared Memory Load Transaction    0.000000    0.000000    0.000000
          4  shared_store_transactions_per_  Shared Memory Store Transactio    0.000000    0.000000    0.000000
          4    gld_transactions_per_request  Global Load Transactions Per R    2.407631    2.407631    2.407631
          4    gst_transactions_per_request  Global Store Transactions Per     2.464951    2.464951    2.464951
          4         local_load_transactions         Local Load Transactions           0           0           0
          4        local_store_transactions        Local Store Transactions           0           0           0
          4        shared_load_transactions        Shared Load Transactions           0           0           0
          4       shared_store_transactions       Shared Store Transactions           0           0           0
          4                gld_transactions        Global Load Transactions        4796        4796        4796
          4                gst_transactions       Global Store Transactions       18004       18004       18004
          4        sysmem_read_transactions  System Memory Read Transaction           0           0           0
          4       sysmem_write_transactions  System Memory Write Transactio           0           7           2
          4          tex_cache_transactions      Texture Cache Transactions           0           0           0
          4          dram_read_transactions  Device Memory Read Transaction       26199       26283       26238
          4         dram_write_transactions  Device Memory Write Transactio       73865       73884       73876
          4            l2_read_transactions            L2 Read Transactions       15995       16038       16011
          4           l2_write_transactions           L2 Write Transactions       58591       58636       58610
          4           local_load_throughput    Local Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          local_store_throughput   Local Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          shared_load_throughput   Shared Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         shared_store_throughput  Shared Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4              l2_read_throughput           L2 Throughput (Reads)  20.882GB/s  21.231GB/s  21.021GB/s
          4             l2_write_throughput          L2 Throughput (Writes)  76.397GB/s  76.530GB/s  76.488GB/s
          4          sysmem_read_throughput   System Memory Read Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         sysmem_write_throughput  System Memory Write Throughput  7.8246MB/s  15.666MB/s  11.421MB/s
          4  warp_nonpred_execution_efficie  Warp Non-Predicated Execution       91.67%      91.67%      91.67%
          4                       cf_issued  Issued Control-Flow Instructio        5348        5351        5349
          4                     cf_executed  Executed Control-Flow Instruct        5348        5348        5348
          4                     ldst_issued  Issued Load/Store Instructions       61426       65640       63476
          4                   ldst_executed  Executed Load/Store Instructio        9296        9296        9296
          4                        flops_sp                   FLOPS(Single)           0           0           0
          4                    flops_sp_add               FLOPS(Single Add)           0           0           0
          4                    flops_sp_mul               FLOPS(Single Mul)           0           0           0
          4                    flops_sp_fma               FLOPS(Single FMA)           0           0           0
          4                        flops_dp                   FLOPS(Double)           0           0           0
          4                    flops_dp_add               FLOPS(Double Add)           0           0           0
          4                    flops_dp_mul               FLOPS(Double Mul)           0           0           0
          4                    flops_dp_fma               FLOPS(Double FMA)           0           0           0
          4                flops_sp_special           FLOPS(Single Special)       43392       43392       43392
          4                stall_inst_fetch  Issue Stall Reasons (Instructi       0.27%       0.28%       0.27%
          4           stall_exec_dependency  Issue Stall Reasons (Execution      56.32%      60.16%      58.54%
          4              stall_data_request  Issue Stall Reasons (Data Requ      26.27%      27.92%      27.49%
          4                   stall_texture   Issue Stall Reasons (Texture)       0.00%       0.00%       0.00%
          4                      stall_sync  Issue Stall Reasons (Synchroni       0.00%       0.00%       0.00%
          4                     stall_other     Issue Stall Reasons (Other)       6.38%       6.62%       6.52%
          4           l1_shared_utilization    L1/Shared Memory Utilization     Low (1)     Low (1)     Low (1)
          4                  l2_utilization            L2 Cache Utilization     Low (3)     Low (3)     Low (3)
          4                 tex_utilization       Texture Cache Utilization    Idle (0)    Idle (0)    Idle (0)
          4                dram_utilization       Device Memory Utilization    High (7)    High (7)    High (7)
          4              sysmem_utilization       System Memory Utilization     Low (1)     Low (1)     Low (1)
          4             ldst_fu_utilization  Load/Store Function Unit Utili     Mid (4)     Mid (4)     Mid (4)
          4              alu_fu_utilization  Arithmetic Function Unit Utili     Low (2)     Low (2)     Low (2)
          4               cf_fu_utilization  Control-Flow Function Unit Uti     Low (1)     Low (1)     Low (1)
          4              tex_fu_utilization  Texture Function Unit Utilizat    Idle (0)    Idle (0)    Idle (0)
          4                   inst_executed           Instructions Executed      118452      118452      118452
          4                     inst_issued             Instructions Issued      184421      185966      185231
          4                     issue_slots                     Issue Slots      156261      156898      156494
	Kernel: fluxcalc2D1(int, int, int, double*, double const *, double const *, double const *, double const *, double const *, double*, double*, double*, double*, int)
         12        l1_cache_global_hit_rate              L1 Global Hit Rate       0.00%       0.00%       0.00%
         12         l1_cache_local_hit_rate               L1 Local Hit Rate      68.20%      68.56%      68.42%
         12                   sm_efficiency         Multiprocessor Activity      99.75%      99.82%      99.79%
         12                             ipc                    Executed IPC    1.415564    1.429945    1.422573
         12              achieved_occupancy              Achieved Occupancy    0.238665    0.239156    0.238948
         12        gld_requested_throughput  Requested Global Load Throughp  0.00000B/s  0.00000B/s  0.00000B/s
         12        gst_requested_throughput  Requested Global Store Through  42.054GB/s  42.650GB/s  42.328GB/s
         12          sm_efficiency_instance         Multiprocessor Activity      99.74%      99.83%      99.79%
         12                    ipc_instance                    Executed IPC    1.415549    1.431206    1.423174
         12            inst_replay_overhead     Instruction Replay Overhead    0.353116    0.354942    0.353941
         12          shared_replay_overhead   Shared Memory Replay Overhead    0.000000    0.000000    0.000000
         12          global_replay_overhead   Global Memory Replay Overhead    0.012962    0.013031    0.012995
         12    global_cache_replay_overhead  Global Memory Cache Replay Ove    0.000000    0.000000    0.000000
         12              tex_cache_hit_rate          Texture Cache Hit Rate      46.34%      59.37%      50.69%
         12            tex_cache_throughput        Texture Cache Throughput  81.321GB/s  82.204GB/s  81.797GB/s
         12            dram_read_throughput   Device Memory Read Throughput  59.273GB/s  60.188GB/s  59.668GB/s
         12           dram_write_throughput  Device Memory Write Throughput  100.52GB/s  101.61GB/s  101.24GB/s
         12                  gst_throughput         Global Store Throughput  54.172GB/s  55.048GB/s  54.674GB/s
         12                  gld_throughput          Global Load Throughput  0.00000B/s  35.321MB/s  9.1742MB/s
         12           local_replay_overhead  Local Memory Cache Replay Over    0.013879    0.014102    0.013995
         12               shared_efficiency        Shared Memory Efficiency       0.00%       0.00%       0.00%
         12                  gld_efficiency   Global Memory Load Efficiency       0.00%       0.00%       0.00%
         12                  gst_efficiency  Global Memory Store Efficiency      76.75%      77.99%      77.42%
         12             l2_l1_read_hit_rate          L2 Hit Rate (L1 Reads)      90.69%      93.76%      91.96%
         12        l2_texture_read_hit_rate           L2 Hit Rate (Texture)      11.24%      19.43%      13.98%
         12           l2_l1_read_throughput        L2 Throughput (L1 Reads)  2.7475GB/s  2.8593GB/s  2.8156GB/s
         12      l2_texture_read_throughput         L2 Throughput (Texture)  43.628GB/s  48.346GB/s  45.335GB/s
         12           local_memory_overhead           Local Memory Overhead       6.51%       6.63%       6.59%
         12       warp_execution_efficiency       Warp Execution Efficiency      97.27%      98.14%      97.82%
         12     nc_gld_requested_throughput  Requested Non-Coherent Global   79.435GB/s  80.560GB/s  79.952GB/s
         12                      issued_ipc                      Issued IPC    1.916677    1.936612    1.926976
         12                   inst_per_warp           Instructions per warp  1.3188e+03  1.3335e+03  1.3246e+03
         12          issue_slot_utilization          Issue Slot Utilization      40.85%      41.48%      41.12%
         12  local_load_transactions_per_re  Local Memory Load Transactions    2.003879    2.010701    2.005904
         12  local_store_transactions_per_r  Local Memory Store Transaction    2.753006    2.765507    2.760164
         12  shared_load_transactions_per_r  Shared Memory Load Transaction    0.000000    0.000000    0.000000
         12  shared_store_transactions_per_  Shared Memory Store Transactio    0.000000    0.000000    0.000000
         12    gld_transactions_per_request  Global Load Transactions Per R    0.000000    0.000000    0.000000
         12    gst_transactions_per_request  Global Store Transactions Per     1.996960    2.000000    1.998987
         12         local_load_transactions         Local Load Transactions      403044      407885      404768
         12        local_store_transactions        Local Store Transactions     2954926     2994274     2970079
         12        shared_load_transactions        Shared Load Transactions           0           0           0
         12       shared_store_transactions       Shared Store Transactions           0           0           0
         12                gld_transactions        Global Load Transactions           0           0           0
         12                gst_transactions       Global Store Transactions     1206252     1219104     1210836
         12        sysmem_read_transactions  System Memory Read Transaction           0           0           0
         12       sysmem_write_transactions  System Memory Write Transactio           0           9           4
         12          tex_cache_transactions      Texture Cache Transactions     9097856     9202168     9136990
         12          dram_read_transactions  Device Memory Read Transaction     6572915     6735591     6665198
         12         dram_write_transactions  Device Memory Write Transactio    11233777    11404562    11303606
         12            l2_read_transactions            L2 Read Transactions     5197329     5749807     5382387
         12           l2_write_transactions           L2 Write Transactions     9867823    10008102     9915433
         12           local_load_throughput    Local Memory Load Throughput  14.392GB/s  14.517GB/s  14.459GB/s
         12          local_store_throughput   Local Memory Store Throughput  66.724GB/s  67.395GB/s  67.045GB/s
         12          shared_load_throughput   Shared Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
         12         shared_store_throughput  Shared Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
         12              l2_read_throughput           L2 Throughput (Reads)  46.422GB/s  51.211GB/s  48.180GB/s
         12             l2_write_throughput          L2 Throughput (Writes)  88.153GB/s  89.195GB/s  88.779GB/s
         12          sysmem_read_throughput   System Memory Read Throughput  0.00000B/s  0.00000B/s  0.00000B/s
         12         sysmem_write_throughput  System Memory Write Throughput  27.055KB/s  90.191KB/s  56.701KB/s
         12  warp_nonpred_execution_efficie  Warp Non-Predicated Execution       93.92%      94.73%      94.44%
         12                       cf_issued  Issued Control-Flow Instructio     4859239     4930138     4897415
         12                     cf_executed  Executed Control-Flow Instruct     4604224     4671790     4640794
         12                     ldst_issued  Issued Load/Store Instructions     4731948     4791278     4757856
         12                   ldst_executed  Executed Load/Store Instructio     1877792     1896384     1884474
         12                        flops_sp                   FLOPS(Single)           0           0           0
         12                    flops_sp_add               FLOPS(Single Add)           0           0           0
         12                    flops_sp_mul               FLOPS(Single Mul)           0           0           0
         12                    flops_sp_fma               FLOPS(Single FMA)           0           0           0
         12                        flops_dp                   FLOPS(Double)   545367276   545517836   545460193
         12                    flops_dp_add               FLOPS(Double Add)    23117688    23117688    23117688
         12                    flops_dp_mul               FLOPS(Double Mul)    93521556    93543038    93534822
         12                    flops_dp_fma               FLOPS(Double FMA)   214364016   214428555   214403841
         12                flops_sp_special           FLOPS(Single Special)    21386700    21386741    21386708
         12                stall_inst_fetch  Issue Stall Reasons (Instructi       7.81%       7.95%       7.88%
         12           stall_exec_dependency  Issue Stall Reasons (Execution      56.43%      57.17%      56.80%
         12              stall_data_request  Issue Stall Reasons (Data Requ       6.09%       6.22%       6.16%
         12                   stall_texture   Issue Stall Reasons (Texture)      13.84%      14.50%      14.20%
         12                      stall_sync  Issue Stall Reasons (Synchroni       0.00%       0.00%       0.00%
         12                     stall_other     Issue Stall Reasons (Other)       6.67%       6.76%       6.71%
         12           l1_shared_utilization    L1/Shared Memory Utilization     Low (1)     Low (1)     Low (1)
         12                  l2_utilization            L2 Cache Utilization     Low (3)     Mid (4)     Low (3)
         12                 tex_utilization       Texture Cache Utilization     Low (1)     Low (1)     Low (1)
         12                dram_utilization       Device Memory Utilization    High (8)    High (8)    High (8)
         12              sysmem_utilization       System Memory Utilization     Low (1)     Low (1)     Low (1)
         12             ldst_fu_utilization  Load/Store Function Unit Utili     Low (2)     Low (2)     Low (2)
         12              alu_fu_utilization  Arithmetic Function Unit Utili     Low (3)     Low (3)     Low (3)
         12               cf_fu_utilization  Control-Flow Function Unit Uti     Low (1)     Low (1)     Low (1)
         12              tex_fu_utilization  Texture Function Unit Utilizat     Low (1)     Low (1)     Low (1)
         12                   inst_executed           Instructions Executed    46360656    46876300    46565443
         12                     inst_issued             Instructions Issued    62787484    63482646    63051690
         12                     issue_slots                     Issue Slots    53599937    54247788    53857399
	Kernel: fluxcalcprep(int, int, int, double*, double*, double const *, int, int)
         12        l1_cache_global_hit_rate              L1 Global Hit Rate       0.00%       0.00%       0.00%
         12         l1_cache_local_hit_rate               L1 Local Hit Rate       0.00%       0.00%       0.00%
         12                   sm_efficiency         Multiprocessor Activity      99.50%      99.75%      99.59%
         12                             ipc                    Executed IPC    0.804282    1.407306    1.199963
         12              achieved_occupancy              Achieved Occupancy    0.481153    0.483409    0.482176
         12        gld_requested_throughput  Requested Global Load Throughp  0.00000B/s  0.00000B/s  0.00000B/s
         12        gst_requested_throughput  Requested Global Store Through  31.409GB/s  55.162GB/s  46.682GB/s
         12          sm_efficiency_instance         Multiprocessor Activity      99.53%      99.74%      99.61%
         12                    ipc_instance                    Executed IPC    0.809813    1.407546    1.199481
         12            inst_replay_overhead     Instruction Replay Overhead    0.176165    0.198588    0.187104
         12          shared_replay_overhead   Shared Memory Replay Overhead    0.000000    0.000000    0.000000
         12          global_replay_overhead   Global Memory Replay Overhead    0.016646    0.017246    0.016945
         12    global_cache_replay_overhead  Global Memory Cache Replay Ove    0.000000    0.000000    0.000000
         12              tex_cache_hit_rate          Texture Cache Hit Rate       0.00%      75.01%      25.03%
         12            tex_cache_throughput        Texture Cache Throughput  95.982GB/s  168.74GB/s  142.75GB/s
         12            dram_read_throughput   Device Memory Read Throughput  79.562GB/s  123.99GB/s  94.828GB/s
         12           dram_write_throughput  Device Memory Write Throughput  40.793GB/s  71.904GB/s  60.813GB/s
         12                  gst_throughput         Global Store Throughput  32.025GB/s  56.244GB/s  47.598GB/s
         12                  gld_throughput          Global Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
         12           local_replay_overhead  Local Memory Cache Replay Over    0.000000    0.000000    0.000000
         12               shared_efficiency        Shared Memory Efficiency       0.00%       0.00%       0.00%
         12                  gld_efficiency   Global Memory Load Efficiency       0.00%       0.00%       0.00%
         12                  gst_efficiency  Global Memory Store Efficiency      98.08%      98.08%      98.08%
         12             l2_l1_read_hit_rate          L2 Hit Rate (L1 Reads)       0.00%       0.00%       0.00%
         12        l2_texture_read_hit_rate           L2 Hit Rate (Texture)       0.39%      65.99%      28.39%
         12           l2_l1_read_throughput        L2 Throughput (L1 Reads)  0.00000B/s  0.00000B/s  0.00000B/s
         12      l2_texture_read_throughput         L2 Throughput (Texture)  68.790GB/s  164.16GB/s  109.68GB/s
         12           local_memory_overhead           Local Memory Overhead       0.00%       0.00%       0.00%
         12       warp_execution_efficiency       Warp Execution Efficiency      97.99%      98.54%      98.35%
         12     nc_gld_requested_throughput  Requested Non-Coherent Global   94.227GB/s  165.49GB/s  140.05GB/s
         12                      issued_ipc                      Issued IPC    0.954667    1.663885    1.424901
         12                   inst_per_warp           Instructions per warp  446.851653  462.947943  454.870465
         12          issue_slot_utilization          Issue Slot Utilization      20.34%      36.00%      30.52%
         12  local_load_transactions_per_re  Local Memory Load Transactions    0.000000    0.000000    0.000000
         12  local_store_transactions_per_r  Local Memory Store Transaction    0.000000    0.000000    0.000000
         12  shared_load_transactions_per_r  Shared Memory Load Transaction    0.000000    0.000000    0.000000
         12  shared_store_transactions_per_  Shared Memory Store Transactio    0.000000    0.000000    0.000000
         12    gld_transactions_per_request  Global Load Transactions Per R    0.000000    0.000000    0.000000
         12    gst_transactions_per_request  Global Store Transactions Per     2.000000    2.000000    2.000000
         12         local_load_transactions         Local Load Transactions           0           0           0
         12        local_store_transactions        Local Store Transactions           0           0           0
         12        shared_load_transactions        Shared Load Transactions           0           0           0
         12       shared_store_transactions       Shared Store Transactions           0           0           0
         12                gld_transactions        Global Load Transactions           0           0           0
         12                gst_transactions       Global Store Transactions      541824      541824      541824
         12        sysmem_read_transactions  System Memory Read Transaction           0           0           0
         12       sysmem_write_transactions  System Memory Write Transactio           0          12           4
         12          tex_cache_transactions      Texture Cache Transactions     6481152     6503232     6492400
         12          dram_read_transactions  Device Memory Read Transaction     3121539     8350733     4878349
         12         dram_write_transactions  Device Memory Write Transactio     2755371     2768720     2763772
         12            l2_read_transactions            L2 Read Transactions     2664032     6494093     5215164
         12           l2_write_transactions           L2 Write Transactions     2329226     2329687     2329393
         12           local_load_throughput    Local Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
         12          local_store_throughput   Local Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
         12          shared_load_throughput   Shared Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
         12         shared_store_throughput  Shared Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
         12              l2_read_throughput           L2 Throughput (Reads)  68.802GB/s  164.15GB/s  109.70GB/s
         12             l2_write_throughput          L2 Throughput (Writes)  34.477GB/s  60.537GB/s  51.234GB/s
         12          sysmem_read_throughput   System Memory Read Throughput  0.00000B/s  0.00000B/s  0.00000B/s
         12         sysmem_write_throughput  System Memory Write Throughput  25.114KB/s  177.12KB/s  101.14KB/s
         12  warp_nonpred_execution_efficie  Warp Non-Predicated Execution       92.56%      93.28%      92.98%
         12                       cf_issued  Issued Control-Flow Instructio     2274465     2311220     2292540
         12                     cf_executed  Executed Control-Flow Instruct     2274410     2311116     2291891
         12                     ldst_issued  Issued Load/Store Instructions      541898      543012      542536
         12                   ldst_executed  Executed Load/Store Instructio      270912      270912      270912
         12                        flops_sp                   FLOPS(Single)           0           0           0
         12                    flops_sp_add               FLOPS(Single Add)           0           0           0
         12                    flops_sp_mul               FLOPS(Single Mul)           0           0           0
         12                    flops_sp_fma               FLOPS(Single FMA)           0           0           0
         12                        flops_dp                   FLOPS(Double)    59427648    59427648    59427648
         12                    flops_dp_add               FLOPS(Double Add)    42448320    42448320    42448320
         12                    flops_dp_mul               FLOPS(Double Mul)    16979328    16979328    16979328
         12                    flops_dp_fma               FLOPS(Double FMA)           0           0           0
         12                flops_sp_special           FLOPS(Single Special)     5624640     5624640     5624640
         12                stall_inst_fetch  Issue Stall Reasons (Instructi       0.90%       2.00%       1.53%
         12           stall_exec_dependency  Issue Stall Reasons (Execution      13.38%      24.12%      20.53%
         12              stall_data_request  Issue Stall Reasons (Data Requ       0.17%       0.31%       0.26%
         12                   stall_texture   Issue Stall Reasons (Texture)      64.55%      81.35%      70.20%
         12                      stall_sync  Issue Stall Reasons (Synchroni       0.00%       0.00%       0.00%
         12                     stall_other     Issue Stall Reasons (Other)       1.84%       3.82%       3.11%
         12           l1_shared_utilization    L1/Shared Memory Utilization     Low (1)     Low (1)     Low (1)
         12                  l2_utilization            L2 Cache Utilization     Low (3)     Mid (5)     Low (3)
         12                 tex_utilization       Texture Cache Utilization     Low (1)     Low (2)     Low (1)
         12                dram_utilization       Device Memory Utilization    High (8)    High (8)    High (8)
         12              sysmem_utilization       System Memory Utilization    Idle (0)     Low (1)    Idle (0)
         12             ldst_fu_utilization  Load/Store Function Unit Utili     Low (1)     Low (1)     Low (1)
         12              alu_fu_utilization  Arithmetic Function Unit Utili     Low (2)     Low (3)     Low (2)
         12               cf_fu_utilization  Control-Flow Function Unit Uti     Low (1)     Low (1)     Low (1)
         12              tex_fu_utilization  Texture Function Unit Utilizat     Low (1)     Low (1)     Low (1)
         12                   inst_executed           Instructions Executed    15708623    16274472    15990516
         12                     inst_issued             Instructions Issued    18477261    19505753    18984052
         12                     issue_slots                     Issue Slots    15798861    16819558    16293588
	Kernel: boundprim3(int, int, int, double*, int*, int, int, int, int, int, int)
          4        l1_cache_global_hit_rate              L1 Global Hit Rate       0.00%       0.00%       0.00%
          4         l1_cache_local_hit_rate               L1 Local Hit Rate       0.00%       0.00%       0.00%
          4                   sm_efficiency         Multiprocessor Activity      87.36%      93.12%      90.98%
          4                             ipc                    Executed IPC    0.022923    0.024295    0.023842
          4              achieved_occupancy              Achieved Occupancy    0.443000    0.446345    0.445244
          4        gld_requested_throughput  Requested Global Load Throughp  3.8931GB/s  3.8982GB/s  3.8950GB/s
          4        gst_requested_throughput  Requested Global Store Through  3.8931GB/s  3.8982GB/s  3.8950GB/s
          4          sm_efficiency_instance         Multiprocessor Activity      89.77%      92.17%      91.48%
          4                    ipc_instance                    Executed IPC    0.023173    0.024680    0.023701
          4            inst_replay_overhead     Instruction Replay Overhead   10.360984   11.502927   10.764388
          4          shared_replay_overhead   Shared Memory Replay Overhead    0.000000    0.000000    0.000000
          4          global_replay_overhead   Global Memory Replay Overhead    4.883375    4.883375    4.883375
          4    global_cache_replay_overhead  Global Memory Cache Replay Ove    0.000000    0.000000    0.000000
          4              tex_cache_hit_rate          Texture Cache Hit Rate       0.00%       0.00%       0.00%
          4            tex_cache_throughput        Texture Cache Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4            dram_read_throughput   Device Memory Read Throughput  61.214GB/s  61.904GB/s  61.463GB/s
          4           dram_write_throughput  Device Memory Write Throughput  31.795GB/s  31.819GB/s  31.807GB/s
          4                  gst_throughput         Global Store Throughput  16.044GB/s  16.104GB/s  16.065GB/s
          4                  gld_throughput          Global Load Throughput  16.044GB/s  16.065GB/s  16.052GB/s
          4           local_replay_overhead  Local Memory Cache Replay Over    0.000000    0.000000    0.000000
          4               shared_efficiency        Shared Memory Efficiency       0.00%       0.00%       0.00%
          4                  gld_efficiency   Global Memory Load Efficiency      24.26%      24.26%      24.26%
          4                  gst_efficiency  Global Memory Store Efficiency      24.26%      24.26%      24.26%
          4             l2_l1_read_hit_rate          L2 Hit Rate (L1 Reads)       3.01%       4.77%       4.26%
          4        l2_texture_read_hit_rate           L2 Hit Rate (Texture)       0.00%       0.00%       0.00%
          4           l2_l1_read_throughput        L2 Throughput (L1 Reads)  16.044GB/s  16.065GB/s  16.052GB/s
          4      l2_texture_read_throughput         L2 Throughput (Texture)  0.00000B/s  0.00000B/s  0.00000B/s
          4           local_memory_overhead           Local Memory Overhead       0.00%       0.00%       0.00%
          4       warp_execution_efficiency       Warp Execution Efficiency     100.00%     100.00%     100.00%
          4     nc_gld_requested_throughput  Requested Non-Coherent Global   0.00000B/s  0.00000B/s  0.00000B/s
          4                      issued_ipc                      Issued IPC    0.258405    0.291931    0.277767
          4                   inst_per_warp           Instructions per warp  215.197640  215.197640  215.197640
          4          issue_slot_utilization          Issue Slot Utilization       5.87%       6.86%       6.46%
          4  local_load_transactions_per_re  Local Memory Load Transactions    0.000000    0.000000    0.000000
          4  local_store_transactions_per_r  Local Memory Store Transaction    0.000000    0.000000    0.000000
          4  shared_load_transactions_per_r  Shared Memory Load Transaction    0.000000    0.000000    0.000000
          4  shared_store_transactions_per_  Shared Memory Store Transactio    0.000000    0.000000    0.000000
          4    gld_transactions_per_request  Global Load Transactions Per R   32.000000   32.000000   32.000000
          4    gst_transactions_per_request  Global Store Transactions Per    32.000000   32.000000   32.000000
          4         local_load_transactions         Local Load Transactions           0           0           0
          4        local_store_transactions        Local Store Transactions           0           0           0
          4        shared_load_transactions        Shared Load Transactions           0           0           0
          4       shared_store_transactions       Shared Store Transactions           0           0           0
          4                gld_transactions        Global Load Transactions      367744      367744      367744
          4                gst_transactions       Global Store Transactions      367744      367744      367744
          4        sysmem_read_transactions  System Memory Read Transaction           0           0           0
          4       sysmem_write_transactions  System Memory Write Transactio           2           6           4
          4          tex_cache_transactions      Texture Cache Transactions           0           0           0
          4          dram_read_transactions  Device Memory Read Transaction     1398263     1429697     1407879
          4         dram_write_transactions  Device Memory Write Transactio      728119      731791      729238
          4            l2_read_transactions            L2 Read Transactions      367872      368655      368246
          4           l2_write_transactions           L2 Write Transactions      698410      700844      699938
          4           local_load_throughput    Local Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          local_store_throughput   Local Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          shared_load_throughput   Shared Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         shared_store_throughput  Shared Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4              l2_read_throughput           L2 Throughput (Reads)  16.060GB/s  16.088GB/s  16.071GB/s
          4             l2_write_throughput          L2 Throughput (Writes)  30.505GB/s  30.618GB/s  30.569GB/s
          4          sysmem_read_throughput   System Memory Read Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         sysmem_write_throughput  System Memory Write Throughput  0.00000B/s  262.12KB/s  163.72KB/s
          4  warp_nonpred_execution_efficie  Warp Non-Predicated Execution       93.74%      93.74%      93.74%
          4                       cf_issued  Issued Control-Flow Instructio        5166        5170        5168
          4                     cf_executed  Executed Control-Flow Instruct        4744        4744        4744
          4                     ldst_issued  Issued Load/Store Instructions     1390654     1847062     1552267
          4                   ldst_executed  Executed Load/Store Instructio       22984       22984       22984
          4                        flops_sp                   FLOPS(Single)           0           0           0
          4                    flops_sp_add               FLOPS(Single Add)           0           0           0
          4                    flops_sp_mul               FLOPS(Single Mul)           0           0           0
          4                    flops_sp_fma               FLOPS(Single FMA)           0           0           0
          4                        flops_dp                   FLOPS(Double)           0           0           0
          4                    flops_dp_add               FLOPS(Double Add)           0           0           0
          4                    flops_dp_mul               FLOPS(Double Mul)           0           0           0
          4                    flops_dp_fma               FLOPS(Double FMA)           0           0           0
          4                flops_sp_special           FLOPS(Single Special)       65152       65152       65152
          4                stall_inst_fetch  Issue Stall Reasons (Instructi       0.01%       0.01%       0.01%
          4           stall_exec_dependency  Issue Stall Reasons (Execution      28.27%      31.16%      29.96%
          4              stall_data_request  Issue Stall Reasons (Data Requ      68.46%      71.72%      70.34%
          4                   stall_texture   Issue Stall Reasons (Texture)       0.00%       0.00%       0.00%
          4                      stall_sync  Issue Stall Reasons (Synchroni       0.00%       0.00%       0.00%
          4                     stall_other     Issue Stall Reasons (Other)       0.27%       0.38%       0.31%
          4           l1_shared_utilization    L1/Shared Memory Utilization     Low (1)     Low (1)     Low (1)
          4                  l2_utilization            L2 Cache Utilization     Low (2)     Low (2)     Low (2)
          4                 tex_utilization       Texture Cache Utilization    Idle (0)    Idle (0)    Idle (0)
          4                dram_utilization       Device Memory Utilization     Mid (5)     Mid (5)     Mid (5)
          4              sysmem_utilization       System Memory Utilization     Low (1)     Low (1)     Low (1)
          4             ldst_fu_utilization  Load/Store Function Unit Utili     Low (3)     Low (3)     Low (3)
          4              alu_fu_utilization  Arithmetic Function Unit Utili     Low (1)     Low (1)     Low (1)
          4               cf_fu_utilization  Control-Flow Function Unit Uti     Low (1)     Low (1)     Low (1)
          4              tex_fu_utilization  Texture Function Unit Utilizat    Idle (0)    Idle (0)    Idle (0)
          4                   inst_executed           Instructions Executed      145904      145904      145904
          4                     inst_issued             Instructions Issued     1539579     1752202     1623473
          4                     issue_slots                     Issue Slots     1587637     1811570     1717753
	Kernel: boundprim1(int, int, int, double*, int*, double const *, double const *, double const *, int, int, int, int, int, int)
          4        l1_cache_global_hit_rate              L1 Global Hit Rate       0.00%       0.00%       0.00%
          4         l1_cache_local_hit_rate               L1 Local Hit Rate       0.00%       0.00%       0.00%
          4                   sm_efficiency         Multiprocessor Activity      77.41%      85.80%      80.87%
          4                             ipc                    Executed IPC    1.656458    1.717280    1.691510
          4              achieved_occupancy              Achieved Occupancy    0.312407    0.338055    0.325034
          4        gld_requested_throughput  Requested Global Load Throughp  17.793GB/s  18.251GB/s  18.059GB/s
          4        gst_requested_throughput  Requested Global Store Through  34.539GB/s  35.428GB/s  35.056GB/s
          4          sm_efficiency_instance         Multiprocessor Activity      77.11%      84.42%      80.85%
          4                    ipc_instance                    Executed IPC    1.668638    1.721269    1.695701
          4            inst_replay_overhead     Instruction Replay Overhead    0.284720    0.294725    0.289331
          4          shared_replay_overhead   Shared Memory Replay Overhead    0.000000    0.000000    0.000000
          4          global_replay_overhead   Global Memory Replay Overhead    0.014716    0.016204    0.015559
          4    global_cache_replay_overhead  Global Memory Cache Replay Ove    0.000000    0.000000    0.000000
          4              tex_cache_hit_rate          Texture Cache Hit Rate      87.67%      87.74%      87.69%
          4            tex_cache_throughput        Texture Cache Throughput  32.391GB/s  33.179GB/s  32.913GB/s
          4            dram_read_throughput   Device Memory Read Throughput  22.776GB/s  23.351GB/s  23.074GB/s
          4           dram_write_throughput  Device Memory Write Throughput  42.839GB/s  43.637GB/s  43.374GB/s
          4                  gst_throughput         Global Store Throughput  34.599GB/s  35.537GB/s  35.122GB/s
          4                  gld_throughput          Global Load Throughput  17.793GB/s  18.251GB/s  18.059GB/s
          4           local_replay_overhead  Local Memory Cache Replay Over    0.000000    0.000000    0.000000
          4               shared_efficiency        Shared Memory Efficiency       0.00%       0.00%       0.00%
          4                  gld_efficiency   Global Memory Load Efficiency     100.00%     100.00%     100.00%
          4                  gst_efficiency  Global Memory Store Efficiency      99.93%     100.00%      99.97%
          4             l2_l1_read_hit_rate          L2 Hit Rate (L1 Reads)       0.00%       0.00%       0.00%
          4        l2_texture_read_hit_rate           L2 Hit Rate (Texture)      90.87%      92.68%      91.91%
          4           l2_l1_read_throughput        L2 Throughput (L1 Reads)  17.793GB/s  18.251GB/s  18.059GB/s
          4      l2_texture_read_throughput         L2 Throughput (Texture)  3.9925GB/s  4.1234GB/s  4.0591GB/s
          4           local_memory_overhead           Local Memory Overhead       0.00%       0.00%       0.00%
          4       warp_execution_efficiency       Warp Execution Efficiency      91.04%      99.71%      95.96%
          4     nc_gld_requested_throughput  Requested Non-Coherent Global   32.446GB/s  33.281GB/s  32.931GB/s
          4                      issued_ipc                      Issued IPC    2.042280    2.197586    2.146718
          4                   inst_per_warp           Instructions per warp  1.4156e+03  1.5587e+03  1.4769e+03
          4          issue_slot_utilization          Issue Slot Utilization      41.83%      46.99%      44.84%
          4  local_load_transactions_per_re  Local Memory Load Transactions    0.000000    0.000000    0.000000
          4  local_store_transactions_per_r  Local Memory Store Transaction    0.000000    0.000000    0.000000
          4  shared_load_transactions_per_r  Shared Memory Load Transaction    0.000000    0.000000    0.000000
          4  shared_store_transactions_per_  Shared Memory Store Transactio    0.000000    0.000000    0.000000
          4    gld_transactions_per_request  Global Load Transactions Per R    1.883436    1.883436    1.883436
          4    gst_transactions_per_request  Global Store Transactions Per     1.935402    1.935402    1.935402
          4         local_load_transactions         Local Load Transactions           0           0           0
          4        local_store_transactions        Local Store Transactions           0           0           0
          4        shared_load_transactions        Shared Load Transactions           0           0           0
          4       shared_store_transactions       Shared Store Transactions           0           0           0
          4                gld_transactions        Global Load Transactions       11052       11052       11052
          4                gst_transactions       Global Store Transactions       21452       21452       21452
          4        sysmem_read_transactions  System Memory Read Transaction           0           0           0
          4       sysmem_write_transactions  System Memory Write Transactio           4           8           6
          4          tex_cache_transactions      Texture Cache Transactions       80576       80848       80690
          4          dram_read_transactions  Device Memory Read Transaction       56072       56654       56357
          4         dram_write_transactions  Device Memory Write Transactio      105422      106779      106007
          4            l2_read_transactions            L2 Read Transactions       56960       57024       56992
          4           l2_write_transactions           L2 Write Transactions       85844       85948       85906
          4           local_load_throughput    Local Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          local_store_throughput   Local Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          shared_load_throughput   Shared Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         shared_store_throughput  Shared Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4              l2_read_throughput           L2 Throughput (Reads)  22.978GB/s  23.539GB/s  23.303GB/s
          4             l2_write_throughput          L2 Throughput (Writes)  34.547GB/s  35.464GB/s  35.091GB/s
          4          sysmem_read_throughput   System Memory Read Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         sysmem_write_throughput  System Memory Write Throughput  1.6447MB/s  3.2204MB/s  2.2407MB/s
          4  warp_nonpred_execution_efficie  Warp Non-Predicated Execution       88.34%      96.84%      93.16%
          4                       cf_issued  Issued Control-Flow Instructio      133624      158873      144439
          4                     cf_executed  Executed Control-Flow Instruct      126292      149224      136120
          4                     ldst_issued  Issued Load/Store Instructions       42199       52851       47022
          4                   ldst_executed  Executed Load/Store Instructio       16952       16952       16952
          4                        flops_sp                   FLOPS(Single)           0           0           0
          4                    flops_sp_add               FLOPS(Single Add)           0           0           0
          4                    flops_sp_mul               FLOPS(Single Mul)           0           0           0
          4                    flops_sp_fma               FLOPS(Single FMA)           0           0           0
          4                        flops_dp                   FLOPS(Double)     9412000     9412000     9412000
          4                    flops_dp_add               FLOPS(Double Add)      208000      208000      208000
          4                    flops_dp_mul               FLOPS(Double Mul)     1674400     1674400     1674400
          4                    flops_dp_fma               FLOPS(Double FMA)     3764800     3764800     3764800
          4                flops_sp_special           FLOPS(Single Special)      605952      605952      605952
          4                stall_inst_fetch  Issue Stall Reasons (Instructi      15.13%      16.21%      15.53%
          4           stall_exec_dependency  Issue Stall Reasons (Execution      51.24%      56.77%      54.20%
          4              stall_data_request  Issue Stall Reasons (Data Requ       4.84%       8.97%       6.87%
          4                   stall_texture   Issue Stall Reasons (Texture)       4.92%       6.66%       5.94%
          4                      stall_sync  Issue Stall Reasons (Synchroni       0.00%       0.00%       0.00%
          4                     stall_other     Issue Stall Reasons (Other)       7.18%       7.47%       7.36%
          4           l1_shared_utilization    L1/Shared Memory Utilization     Low (1)     Low (1)     Low (1)
          4                  l2_utilization            L2 Cache Utilization     Low (2)     Low (2)     Low (2)
          4                 tex_utilization       Texture Cache Utilization     Low (1)     Low (1)     Low (1)
          4                dram_utilization       Device Memory Utilization     Mid (4)     Mid (4)     Mid (4)
          4              sysmem_utilization       System Memory Utilization    Idle (0)     Low (1)    Idle (0)
          4             ldst_fu_utilization  Load/Store Function Unit Utili     Low (1)     Low (1)     Low (1)
          4              alu_fu_utilization  Arithmetic Function Unit Utili     Low (3)     Mid (4)     Low (3)
          4               cf_fu_utilization  Control-Flow Function Unit Uti     Low (1)     Low (1)     Low (1)
          4              tex_fu_utilization  Texture Function Unit Utilizat     Low (1)     Low (1)     Low (1)
          4                   inst_executed           Instructions Executed      959764     1056784     1001344
          4                     inst_issued             Instructions Issued     1241223     1358301     1290979
          4                     issue_slots                     Issue Slots     1040993     1136633     1082402
	Kernel: flux_ct1(int, int, int, double const *, double const *, double const *, double*)
          4        l1_cache_global_hit_rate              L1 Global Hit Rate       0.00%       0.00%       0.00%
          4         l1_cache_local_hit_rate               L1 Local Hit Rate       0.00%       0.00%       0.00%
          4                   sm_efficiency         Multiprocessor Activity      99.06%      99.09%      99.08%
          4                             ipc                    Executed IPC    1.437454    1.442978    1.440263
          4              achieved_occupancy              Achieved Occupancy    0.466113    0.466858    0.466510
          4        gld_requested_throughput  Requested Global Load Throughp  0.00000B/s  0.00000B/s  0.00000B/s
          4        gst_requested_throughput  Requested Global Store Through  42.352GB/s  42.367GB/s  42.357GB/s
          4          sm_efficiency_instance         Multiprocessor Activity      99.01%      99.07%      99.06%
          4                    ipc_instance                    Executed IPC    1.430950    1.439779    1.434592
          4            inst_replay_overhead     Instruction Replay Overhead    0.070090    0.070121    0.070111
          4          shared_replay_overhead   Shared Memory Replay Overhead    0.000000    0.000000    0.000000
          4          global_replay_overhead   Global Memory Replay Overhead    0.012992    0.012992    0.012992
          4    global_cache_replay_overhead  Global Memory Cache Replay Ove    0.000000    0.000000    0.000000
          4              tex_cache_hit_rate          Texture Cache Hit Rate      26.48%      26.52%      26.50%
          4            tex_cache_throughput        Texture Cache Throughput  174.27GB/s  174.76GB/s  174.53GB/s
          4            dram_read_throughput   Device Memory Read Throughput  116.06GB/s  116.36GB/s  116.22GB/s
          4           dram_write_throughput  Device Memory Write Throughput  53.951GB/s  54.008GB/s  53.971GB/s
          4                  gst_throughput         Global Store Throughput  43.610GB/s  43.625GB/s  43.615GB/s
          4                  gld_throughput          Global Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4           local_replay_overhead  Local Memory Cache Replay Over    0.000000    0.000000    0.000000
          4               shared_efficiency        Shared Memory Efficiency       0.00%       0.00%       0.00%
          4                  gld_efficiency   Global Memory Load Efficiency       0.00%       0.00%       0.00%
          4                  gst_efficiency  Global Memory Store Efficiency      97.12%      97.12%      97.12%
          4             l2_l1_read_hit_rate          L2 Hit Rate (L1 Reads)       0.00%       0.00%       0.00%
          4        l2_texture_read_hit_rate           L2 Hit Rate (Texture)      39.12%      39.22%      39.18%
          4           l2_l1_read_throughput        L2 Throughput (L1 Reads)  0.00000B/s  0.00000B/s  0.00000B/s
          4      l2_texture_read_throughput         L2 Throughput (Texture)  148.70GB/s  148.75GB/s  148.72GB/s
          4           local_memory_overhead           Local Memory Overhead       0.00%       0.00%       0.00%
          4       warp_execution_efficiency       Warp Execution Efficiency      99.08%      99.08%      99.08%
          4     nc_gld_requested_throughput  Requested Non-Coherent Global   169.41GB/s  169.47GB/s  169.43GB/s
          4                      issued_ipc                      Issued IPC    1.532204    1.545088    1.539309
          4                   inst_per_warp           Instructions per warp  217.610542  217.610542  217.610542
          4          issue_slot_utilization          Issue Slot Utilization      31.64%      31.76%      31.68%
          4  local_load_transactions_per_re  Local Memory Load Transactions    0.000000    0.000000    0.000000
          4  local_store_transactions_per_r  Local Memory Store Transaction    0.000000    0.000000    0.000000
          4  shared_load_transactions_per_r  Shared Memory Load Transaction    0.000000    0.000000    0.000000
          4  shared_store_transactions_per_  Shared Memory Store Transactio    0.000000    0.000000    0.000000
          4    gld_transactions_per_request  Global Load Transactions Per R    0.000000    0.000000    0.000000
          4    gst_transactions_per_request  Global Store Transactions Per     1.996960    1.996960    1.996960
          4         local_load_transactions         Local Load Transactions           0           0           0
          4        local_store_transactions        Local Store Transactions           0           0           0
          4        shared_load_transactions        Shared Load Transactions           0           0           0
          4       shared_store_transactions       Shared Store Transactions           0           0           0
          4                gld_transactions        Global Load Transactions           0           0           0
          4                gst_transactions       Global Store Transactions      199071      199071      199071
          4        sysmem_read_transactions  System Memory Read Transaction           0           0           0
          4       sysmem_write_transactions  System Memory Write Transactio           0          11           7
          4          tex_cache_transactions      Texture Cache Transactions     3178128     3184800     3181404
          4          dram_read_transactions  Device Memory Read Transaction     2119351     2123657     2121294
          4         dram_write_transactions  Device Memory Write Transactio      984087      985265      984605
          4            l2_read_transactions            L2 Read Transactions     2713205     2713587     2713405
          4           l2_write_transactions           L2 Write Transactions      855838      855848      855843
          4           local_load_throughput    Local Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          local_store_throughput   Local Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          shared_load_throughput   Shared Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         shared_store_throughput  Shared Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4              l2_read_throughput           L2 Throughput (Reads)  148.72GB/s  148.76GB/s  148.73GB/s
          4             l2_write_throughput          L2 Throughput (Writes)  46.908GB/s  46.924GB/s  46.914GB/s
          4          sysmem_read_throughput   System Memory Read Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         sysmem_write_throughput  System Memory Write Throughput  0.00000B/s  328.90KB/s  178.14KB/s
          4  warp_nonpred_execution_efficie  Warp Non-Predicated Execution       91.45%      91.45%      91.45%
          4                       cf_issued  Issued Control-Flow Instructio      368692      368711      368702
          4                     cf_executed  Executed Control-Flow Instruct      368685      368685      368685
          4                     ldst_issued  Issued Load/Store Instructions      199415      199914      199678
          4                   ldst_executed  Executed Load/Store Instructio       99687       99687       99687
          4                        flops_sp                   FLOPS(Single)           0           0           0
          4                    flops_sp_add               FLOPS(Single Add)           0           0           0
          4                    flops_sp_mul               FLOPS(Single Mul)           0           0           0
          4                    flops_sp_fma               FLOPS(Single FMA)           0           0           0
          4                        flops_dp                   FLOPS(Double)    12363612    12363612    12363612
          4                    flops_dp_add               FLOPS(Double Add)     9272709     9272709     9272709
          4                    flops_dp_mul               FLOPS(Double Mul)     3090903     3090903     3090903
          4                    flops_dp_fma               FLOPS(Double FMA)           0           0           0
          4                flops_sp_special           FLOPS(Single Special)     5624640     5624640     5624640
          4                stall_inst_fetch  Issue Stall Reasons (Instructi       0.27%       0.33%       0.29%
          4           stall_exec_dependency  Issue Stall Reasons (Execution      21.06%      21.22%      21.14%
          4              stall_data_request  Issue Stall Reasons (Data Requ       0.09%       0.09%       0.09%
          4                   stall_texture   Issue Stall Reasons (Texture)      69.02%      69.12%      69.04%
          4                      stall_sync  Issue Stall Reasons (Synchroni       0.00%       0.00%       0.00%
          4                     stall_other     Issue Stall Reasons (Other)       6.00%       6.08%       6.04%
          4           l1_shared_utilization    L1/Shared Memory Utilization     Low (1)     Low (1)     Low (1)
          4                  l2_utilization            L2 Cache Utilization     Mid (5)     Mid (5)     Mid (5)
          4                 tex_utilization       Texture Cache Utilization     Low (2)     Low (2)     Low (2)
          4                dram_utilization       Device Memory Utilization    High (9)    High (9)    High (9)
          4              sysmem_utilization       System Memory Utilization     Low (1)     Low (1)     Low (1)
          4             ldst_fu_utilization  Load/Store Function Unit Utili     Low (1)     Low (1)     Low (1)
          4              alu_fu_utilization  Arithmetic Function Unit Utili     Mid (5)     Mid (5)     Mid (5)
          4               cf_fu_utilization  Control-Flow Function Unit Uti     Low (1)     Low (1)     Low (1)
          4              tex_fu_utilization  Texture Function Unit Utilizat     Low (1)     Low (1)     Low (1)
          4                   inst_executed           Instructions Executed     7649881     7649881     7649881
          4                     inst_issued             Instructions Issued     8185941     8186772     8186310
          4                     issue_slots                     Issue Slots     6731644     6731940     6731742
	Kernel: Utoprim2(int, int, int, double*, double*, double*, double const *, double*, double*, double const *, double const *, double const *, double const *, double, double*, int*, int*)
          4        l1_cache_global_hit_rate              L1 Global Hit Rate       0.00%       0.00%       0.00%
          4         l1_cache_local_hit_rate               L1 Local Hit Rate       0.00%       0.00%       0.00%
          4                   sm_efficiency         Multiprocessor Activity      99.85%      99.87%      99.86%
          4                             ipc                    Executed IPC    1.597629    1.606943    1.602416
          4              achieved_occupancy              Achieved Occupancy    0.245969    0.246186    0.246083
          4        gld_requested_throughput  Requested Global Load Throughp  9.1971GB/s  9.2120GB/s  9.2041GB/s
          4        gst_requested_throughput  Requested Global Store Through  10.362GB/s  10.379GB/s  10.370GB/s
          4          sm_efficiency_instance         Multiprocessor Activity      99.83%      99.85%      99.84%
          4                    ipc_instance                    Executed IPC    1.597738    1.605475    1.602384
          4            inst_replay_overhead     Instruction Replay Overhead    0.309933    0.310216    0.310037
          4          shared_replay_overhead   Shared Memory Replay Overhead    0.000000    0.000000    0.000000
          4          global_replay_overhead   Global Memory Replay Overhead    0.005376    0.005385    0.005380
          4    global_cache_replay_overhead  Global Memory Cache Replay Ove    0.000000    0.000000    0.000000
          4              tex_cache_hit_rate          Texture Cache Hit Rate      59.50%      59.54%      59.52%
          4            tex_cache_throughput        Texture Cache Throughput  31.683GB/s  31.748GB/s  31.716GB/s
          4            dram_read_throughput   Device Memory Read Throughput  26.364GB/s  27.387GB/s  26.875GB/s
          4           dram_write_throughput  Device Memory Write Throughput  13.682GB/s  14.106GB/s  13.893GB/s
          4                  gst_throughput         Global Store Throughput  10.776GB/s  10.794GB/s  10.785GB/s
          4                  gld_throughput          Global Load Throughput  10.142GB/s  10.159GB/s  10.150GB/s
          4           local_replay_overhead  Local Memory Cache Replay Over    0.000000    0.000000    0.000000
          4               shared_efficiency        Shared Memory Efficiency       0.00%       0.00%       0.00%
          4                  gld_efficiency   Global Memory Load Efficiency      90.68%      90.68%      90.68%
          4                  gst_efficiency  Global Memory Store Efficiency      96.15%      96.15%      96.15%
          4             l2_l1_read_hit_rate          L2 Hit Rate (L1 Reads)       0.00%       0.00%       0.00%
          4        l2_texture_read_hit_rate           L2 Hit Rate (Texture)      19.21%      19.31%      19.28%
          4           l2_l1_read_throughput        L2 Throughput (L1 Reads)  10.142GB/s  10.159GB/s  10.150GB/s
          4      l2_texture_read_throughput         L2 Throughput (Texture)  12.831GB/s  12.854GB/s  12.841GB/s
          4           local_memory_overhead           Local Memory Overhead       0.00%       0.00%       0.00%
          4       warp_execution_efficiency       Warp Execution Efficiency      95.94%      95.95%      95.95%
          4     nc_gld_requested_throughput  Requested Non-Coherent Global   30.476GB/s  30.525GB/s  30.499GB/s
          4                      issued_ipc                      Issued IPC    2.094346    2.103211    2.100035
          4                   inst_per_warp           Instructions per warp  2.7383e+03  2.7432e+03  2.7408e+03
          4          issue_slot_utilization          Issue Slot Utilization      44.76%      44.89%      44.84%
          4  local_load_transactions_per_re  Local Memory Load Transactions    0.000000    0.000000    0.000000
          4  local_store_transactions_per_r  Local Memory Store Transaction    0.000000    0.000000    0.000000
          4  shared_load_transactions_per_r  Shared Memory Load Transaction    0.000000    0.000000    0.000000
          4  shared_store_transactions_per_  Shared Memory Store Transactio    0.000000    0.000000    0.000000
          4    gld_transactions_per_request  Global Load Transactions Per R    1.993865    1.993865    1.993865
          4    gst_transactions_per_request  Global Store Transactions Per     1.883436    1.883436    1.883436
          4         local_load_transactions         Local Load Transactions           0           0           0
          4        local_store_transactions        Local Store Transactions           0           0           0
          4        shared_load_transactions        Shared Load Transactions           0           0           0
          4       shared_store_transactions       Shared Store Transactions           0           0           0
          4                gld_transactions        Global Load Transactions      520000      520000      520000
          4                gst_transactions       Global Store Transactions      552600      552600      552600
          4        sysmem_read_transactions  System Memory Read Transaction           0           0           0
          4       sysmem_write_transactions  System Memory Write Transactio           3           6           4
          4          tex_cache_transactions      Texture Cache Transactions     6493800     6498200     6496650
          4          dram_read_transactions  Device Memory Read Transaction     5403504     5605747     5505670
          4         dram_write_transactions  Device Memory Write Transactio     2806121     2886762     2846842
          4            l2_read_transactions            L2 Read Transactions     4715283     4717017     4715856
          4           l2_write_transactions           L2 Write Transactions     2273861     2388596     2331736
          4           local_load_throughput    Local Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          local_store_throughput   Local Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          shared_load_throughput   Shared Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         shared_store_throughput  Shared Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4              l2_read_throughput           L2 Throughput (Reads)  22.987GB/s  23.020GB/s  23.005GB/s
          4             l2_write_throughput          L2 Throughput (Writes)  11.088GB/s  11.666GB/s  11.377GB/s
          4          sysmem_read_throughput   System Memory Read Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         sysmem_write_throughput  System Memory Write Throughput  4.8840KB/s  39.036KB/s  19.515KB/s
          4  warp_nonpred_execution_efficie  Warp Non-Predicated Execution       93.04%      93.08%      93.06%
          4                       cf_issued  Issued Control-Flow Instructio    13362475    13418516    13391649
          4                     cf_executed  Executed Control-Flow Instruct    12369129    12424745    12397611
          4                     ldst_issued  Issued Load/Store Instructions     1209416     1213834     1211281
          4                   ldst_executed  Executed Load/Store Instructio      554200      554200      554200
          4                        flops_sp                   FLOPS(Single)           0          10           2
          4                    flops_sp_add               FLOPS(Single Add)           0           0           0
          4                    flops_sp_mul               FLOPS(Single Mul)           0          10           2
          4                    flops_sp_fma               FLOPS(Single FMA)           0           0           0
          4                        flops_dp                   FLOPS(Double)  1111038225  1114695119  1112713360
          4                    flops_dp_add               FLOPS(Double Add)    58912741    59196882    59050264
          4                    flops_dp_mul               FLOPS(Double Mul)   156658082   157202309   156907938
          4                    flops_dp_fma               FLOPS(Double FMA)   447733701   449147964   448377578
          4                flops_sp_special           FLOPS(Single Special)    62865215    63083869    62971054
          4                stall_inst_fetch  Issue Stall Reasons (Instructi      13.75%      13.82%      13.78%
          4           stall_exec_dependency  Issue Stall Reasons (Execution      70.00%      70.13%      70.07%
          4              stall_data_request  Issue Stall Reasons (Data Requ       0.77%       0.94%       0.86%
          4                   stall_texture   Issue Stall Reasons (Texture)       1.71%       1.72%       1.72%
          4                      stall_sync  Issue Stall Reasons (Synchroni       0.00%       0.00%       0.00%
          4                     stall_other     Issue Stall Reasons (Other)       5.31%       5.32%       5.32%
          4           l1_shared_utilization    L1/Shared Memory Utilization     Low (1)     Low (1)     Low (1)
          4                  l2_utilization            L2 Cache Utilization     Low (1)     Low (1)     Low (1)
          4                 tex_utilization       Texture Cache Utilization     Low (1)     Low (1)     Low (1)
          4                dram_utilization       Device Memory Utilization     Low (2)     Low (2)     Low (2)
          4              sysmem_utilization       System Memory Utilization    Idle (0)     Low (1)    Idle (0)
          4             ldst_fu_utilization  Load/Store Function Unit Utili     Low (1)     Low (1)     Low (1)
          4              alu_fu_utilization  Arithmetic Function Unit Utili     Mid (4)     Mid (4)     Mid (4)
          4               cf_fu_utilization  Control-Flow Function Unit Uti     Low (1)     Low (1)     Low (1)
          4              tex_fu_utilization  Texture Function Unit Utilizat     Low (1)     Low (1)     Low (1)
          4                   inst_executed           Instructions Executed    96262659    96433892    96350850
          4                     inst_issued             Instructions Issued   126103446   126330922   126222569
          4                     issue_slots                     Issue Slots   107590513   107786851   107691132
	Kernel: flux_ct2(int, int, int, double*, double*, double*, double const *)
          4        l1_cache_global_hit_rate              L1 Global Hit Rate       0.00%       0.00%       0.00%
          4         l1_cache_local_hit_rate               L1 Local Hit Rate       0.00%       0.00%       0.00%
          4                   sm_efficiency         Multiprocessor Activity      99.36%      99.40%      99.37%
          4                             ipc                    Executed IPC    1.222846    1.229259    1.227136
          4              achieved_occupancy              Achieved Occupancy    0.458363    0.458967    0.458672
          4        gld_requested_throughput  Requested Global Load Throughp  0.00000B/s  0.00000B/s  0.00000B/s
          4        gst_requested_throughput  Requested Global Store Through  83.284GB/s  83.301GB/s  83.292GB/s
          4          sm_efficiency_instance         Multiprocessor Activity      99.36%      99.39%      99.37%
          4                    ipc_instance                    Executed IPC    1.223238    1.230121    1.227727
          4            inst_replay_overhead     Instruction Replay Overhead    0.091768    0.096137    0.093540
          4          shared_replay_overhead   Shared Memory Replay Overhead    0.000000    0.000000    0.000000
          4          global_replay_overhead   Global Memory Replay Overhead    0.030019    0.030019    0.030019
          4    global_cache_replay_overhead  Global Memory Cache Replay Ove    0.000000    0.000000    0.000000
          4              tex_cache_hit_rate          Texture Cache Hit Rate      33.84%      33.90%      33.88%
          4            tex_cache_throughput        Texture Cache Throughput  88.453GB/s  88.566GB/s  88.501GB/s
          4            dram_read_throughput   Device Memory Read Throughput  50.944GB/s  51.021GB/s  50.980GB/s
          4           dram_write_throughput  Device Memory Write Throughput  109.13GB/s  109.21GB/s  109.17GB/s
          4                  gst_throughput         Global Store Throughput  86.330GB/s  86.347GB/s  86.338GB/s
          4                  gld_throughput          Global Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4           local_replay_overhead  Local Memory Cache Replay Over    0.000000    0.000000    0.000000
          4               shared_efficiency        Shared Memory Efficiency       0.00%       0.00%       0.00%
          4                  gld_efficiency   Global Memory Load Efficiency       0.00%       0.00%       0.00%
          4                  gst_efficiency  Global Memory Store Efficiency      96.47%      96.47%      96.47%
          4             l2_l1_read_hit_rate          L2 Hit Rate (L1 Reads)       0.00%       0.00%       0.00%
          4        l2_texture_read_hit_rate           L2 Hit Rate (Texture)      54.90%      54.95%      54.92%
          4           l2_l1_read_throughput        L2 Throughput (L1 Reads)  0.00000B/s  0.00000B/s  0.00000B/s
          4      l2_texture_read_throughput         L2 Throughput (Texture)  70.711GB/s  70.768GB/s  70.729GB/s
          4           local_memory_overhead           Local Memory Overhead       0.00%       0.00%       0.00%
          4       warp_execution_efficiency       Warp Execution Efficiency      98.99%      98.99%      98.99%
          4     nc_gld_requested_throughput  Requested Non-Coherent Global   86.443GB/s  86.461GB/s  86.451GB/s
          4                      issued_ipc                      Issued IPC    1.338991    1.349311    1.343112
          4                   inst_per_warp           Instructions per warp  278.384764  278.384764  278.384764
          4          issue_slot_utilization          Issue Slot Utilization      27.61%      27.70%      27.65%
          4  local_load_transactions_per_re  Local Memory Load Transactions    0.000000    0.000000    0.000000
          4  local_store_transactions_per_r  Local Memory Store Transaction    0.000000    0.000000    0.000000
          4  shared_load_transactions_per_r  Shared Memory Load Transaction    0.000000    0.000000    0.000000
          4  shared_store_transactions_per_  Shared Memory Store Transactio    0.000000    0.000000    0.000000
          4    gld_transactions_per_request  Global Load Transactions Per R    0.000000    0.000000    0.000000
          4    gst_transactions_per_request  Global Store Transactions Per     1.994900    1.994900    1.994900
          4         local_load_transactions         Local Load Transactions           0           0           0
          4        local_store_transactions        Local Store Transactions           0           0           0
          4        shared_load_transactions        Shared Load Transactions           0           0           0
          4       shared_store_transactions       Shared Store Transactions           0           0           0
          4                gld_transactions        Global Load Transactions           0           0           0
          4                gst_transactions       Global Store Transactions      589050      589050      589050
          4        sysmem_read_transactions  System Memory Read Transaction           0           0           0
          4       sysmem_write_transactions  System Memory Write Transactio           1           7           4
          4          tex_cache_transactions      Texture Cache Transactions     2413392     2417480     2414660
          4          dram_read_transactions  Device Memory Read Transaction     1389316     1394678     1392323
          4         dram_write_transactions  Device Memory Write Transactio     2976566     2979487     2977684
          4            l2_read_transactions            L2 Read Transactions     1929228     1931214     1930225
          4           l2_write_transactions           L2 Write Transactions     2535611     2535632     2535618
          4           local_load_throughput    Local Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          local_store_throughput   Local Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4          shared_load_throughput   Shared Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         shared_store_throughput  Shared Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4              l2_read_throughput           L2 Throughput (Reads)  70.726GB/s  70.757GB/s  70.739GB/s
          4             l2_write_throughput          L2 Throughput (Writes)  92.926GB/s  92.945GB/s  92.935GB/s
          4          sysmem_read_throughput   System Memory Read Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          4         sysmem_write_throughput  System Memory Write Throughput  146.61KB/s  366.52KB/s  274.89KB/s
          4  warp_nonpred_execution_efficie  Warp Non-Predicated Execution       91.16%      91.16%      91.16%
          4                       cf_issued  Issued Control-Flow Instructio      661914      665032      663560
          4                     cf_executed  Executed Control-Flow Instruct      651614      651614      651614
          4                     ldst_issued  Issued Load/Store Instructions      715077      724612      719250
          4                   ldst_executed  Executed Load/Store Instructio      295278      295278      295278
          4                        flops_sp                   FLOPS(Single)           0           0           0
          4                    flops_sp_add               FLOPS(Single Add)           0           0           0
          4                    flops_sp_mul               FLOPS(Single Mul)           0           0           0
          4                    flops_sp_fma               FLOPS(Single FMA)           0           0           0
          4                        flops_dp                   FLOPS(Double)    12120000    12120000    12120000
          4                    flops_dp_add               FLOPS(Double Add)     6060000     6060000     6060000
          4                    flops_dp_mul               FLOPS(Double Mul)     6060000     6060000     6060000
          4                    flops_dp_fma               FLOPS(Double FMA)           0           0           0
          4                flops_sp_special           FLOPS(Single Special)     5624640     5624640     5624640
          4                stall_inst_fetch  Issue Stall Reasons (Instructi       0.47%       0.65%       0.55%
          4           stall_exec_dependency  Issue Stall Reasons (Execution      20.52%      20.69%      20.59%
          4              stall_data_request  Issue Stall Reasons (Data Requ       2.61%       3.69%       3.26%
          4                   stall_texture   Issue Stall Reasons (Texture)      64.40%      66.12%      65.45%
          4                      stall_sync  Issue Stall Reasons (Synchroni       0.00%       0.00%       0.00%
          4                     stall_other     Issue Stall Reasons (Other)       6.67%       6.79%       6.72%
          4           l1_shared_utilization    L1/Shared Memory Utilization     Low (1)     Low (1)     Low (1)
          4                  l2_utilization            L2 Cache Utilization     Mid (4)     Mid (4)     Mid (4)
          4                 tex_utilization       Texture Cache Utilization     Low (1)     Low (1)     Low (1)
          4                dram_utilization       Device Memory Utilization    High (8)    High (8)    High (8)
          4              sysmem_utilization       System Memory Utilization    Idle (0)     Low (1)    Idle (0)
          4             ldst_fu_utilization  Load/Store Function Unit Utili     Low (1)     Low (1)     Low (1)
          4              alu_fu_utilization  Arithmetic Function Unit Utili     Mid (4)     Mid (4)     Mid (4)
          4               cf_fu_utilization  Control-Flow Function Unit Uti     Low (1)     Low (1)     Low (1)
          4              tex_fu_utilization  Texture Function Unit Utilizat     Low (1)     Low (1)     Low (1)
          4                   inst_executed           Instructions Executed     9786338     9786338     9786338
          4                     inst_issued             Instructions Issued    10697751    10748024    10720069
          4                     issue_slots                     Issue Slots     8814972     8836901     8826538
