Notes
1. nv3070 l2 byte mask findings from Nsight compute:
2. 
  *L1 receives 131 sectors/131 reqs of global store but have 103 sector misses to L2. meaning L1 could temporarily store the data for a while, even we write 4 bytes each time.
   
  *L2
  
     -Load: 98 sectors/98 requests with 1.02% hit rate
     
     -Store: 103 sectors/100 requests with 100% hit rate


Summary: Looks like L1 has write combine feature; L2 has byte mask enabled for every cacheline stored.

2. constant mem microbenchmarking from: https://www.stuffedcow.net/research/cudabmk

   *Q1: exec mask for shared TPC/diff TPC/shared SM  is tied to chips, but this may not be correct. Need to figure out thread block distributions

   *Q2: kcicache_interfer kernel. intention of this kernel is to do constatn load plus many dependent ALU ops, which need a lot of instructions to I-cache. However, by checking SASS, I think the compiler optimizes out the ALU ops, which means the kernel intention is not met.

   *Q3: kcbw_8t kernel. I think testing bandwidth should do coalescing access, but each thread is 64 arr ele apart, which means 256byte. not quite sure the reason behind it.

Results on 3070

   latency: 14 clks (constant cache hit) and 78clks (constant cache miss)
   
   i-cache interfere:  mask 0x41 -> smid 0 and smid 12 -> latency 238 clks  (not sure how the block is distributed and what is the SMID mapping) 
   
               -> ans to Q2: the original kernel only has 700 instructions (and my guess is correct.) Need to change the abs to some other arithmetic operations, so we can get large kernel code. 
               
