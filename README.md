# Winograd 优化

没有使用GPU优化
因为懒得~没时间~整理代码，请忽略大量代码注释~屎山~

## make
没有依赖，直接运行make
```
make
```

## 运行
```
./winograd conf/small.conf
```

```
sbatch run.sh
```

## 最好结果

```bash
Layer 0 :  Elapse time 81.325293 ms. (  134.04 GFlops) 
Layer 1 :  Elapse time 129.655997 ms. ( 1793.60 GFlops) 
Layer 2 :  Elapse time 49.270709 ms. ( 2317.60 GFlops) 
Layer 3 :  Elapse time 70.273002 ms. ( 3249.89 GFlops) 
Layer 4 :  Elapse time 29.587984 ms. ( 3720.27 GFlops) 
Layer 5 :  Elapse time 46.939294 ms. ( 4690.11 GFlops) 
Layer 6 :  Elapse time 47.312975 ms. ( 4653.07 GFlops) 
Layer 7 :  Elapse time 47.124306 ms. ( 4671.70 GFlops) 
Layer 8 :  Elapse time 20.634333 ms. ( 4946.74 GFlops) 
Layer 9 :  Elapse time 38.180669 ms. ( 5346.82 GFlops) 
Layer 10:  Elapse time 37.981987 ms. ( 5374.79 GFlops) 
Layer 11:  Elapse time 37.907680 ms. ( 5385.32 GFlops) 
Layer 12:  Elapse time 7.792393 ms. ( 5580.64 GFlops) 
Layer 13:  Elapse time 7.856051 ms. ( 5535.42 GFlops) 
Layer 14:  Elapse time 7.876714 ms. ( 5520.90 GFlops) 
Layer 15:  Elapse time 8.124352 ms. ( 5352.62 GFlops) 
Total elapse time: 0.667844. ( 3361.57 GFlops) 
```
```bash

 Performance counter stats for './winograd conf/vgg16.conf':

        140,877.63 msec task-clock                #   53.077 CPUs utilized          
             1,376      context-switches          #    9.767 /sec                   
                 0      cpu-migrations            #    0.000 /sec                   
         2,445,971      page-faults               #   17.362 K/sec                  
   342,576,542,230      cycles                    #    2.432 GHz                      (38.27%)
   641,830,635,865      instructions              #    1.87  insn per cycle           (46.04%)
    92,458,329,196      branches                  #  656.302 M/sec                    (53.80%)
       167,102,534      branch-misses             #    0.18% of all branches          (61.57%)
 1,718,355,823,412      slots                     #   12.198 G/sec                    (69.28%)
   333,238,633,815      topdown-retiring          #     17.4% retiring                (69.28%)
 1,027,585,489,317      topdown-bad-spec          #     53.8% bad speculation         (69.28%)
    11,428,105,361      topdown-fe-bound          #      0.6% frontend bound          (69.28%)
   538,164,217,329      topdown-be-bound          #     28.2% backend bound           (69.28%)
   193,141,026,290      L1-dcache-loads           #    1.371 G/sec                    (69.27%)
    20,111,620,627      L1-dcache-load-misses     #   10.41% of all L1-dcache accesses  (69.38%)
       226,480,127      LLC-loads                 #    1.608 M/sec                    (69.47%)
       140,093,648      LLC-load-misses           #   61.86% of all LL-cache accesses  (69.55%)
   <not supported>      L1-icache-loads                                             
        43,478,916      L1-icache-load-misses                                         (30.81%)
   191,447,631,958      dTLB-loads                #    1.359 G/sec                    (30.94%)
         7,708,746      dTLB-load-misses          #    0.00% of all dTLB cache accesses  (30.77%)
   <not supported>      iTLB-loads                                                  
           156,643      iTLB-load-misses                                              (30.61%)
   <not supported>      L1-dcache-prefetches                                        
   <not supported>      L1-dcache-prefetch-misses                                   

       2.654188234 seconds time elapsed

     124.103938000 seconds user
      16.778565000 seconds sys
```