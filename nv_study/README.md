# NV 3070 Test Results and Analysis
## 1. L2 byte mask findings from Nsight compute (l2_byte_mask_test.cu):
   
### 1.1 Results 

* L1: Receives 131 sectors/131 reqs of global store but have 103 sector misses to L2. meaning L1 could temporarily store the data for a while, even we write 4 bytes each time.

* L2

    * Load： 98 sectors/98 requests with 1.02% hit rate

    * Store: 103 sectors/100 requests with 100% hit rate

### 1.2 Summary

It looks like that L1 has write combine feature and L2 has byte mask enabled for every cacheline stored.

## 2. Constant mem microbenchmarking from: https://www.stuffedcow.net/research/cudabmk

### 2.1 Questions of original code

   * Q1: exec mask for shared TPC/diff TPC/shared SM  is tied to chips, but this may not be correct. Need to figure out thread block distributions

   * Q2: kcicache_interfer kernel. intention of this kernel is to do constatn load plus many dependent ALU ops, which need a lot of instructions to I-cache. However, by checking SASS, I think the compiler optimizes out the ALU ops, which means the kernel intention is not met.

   * Q3: kcbw_8t kernel. I think testing bandwidth should do coalescing access, but each thread is 64 arr ele apart, which means 256byte. not quite sure the reason behind it.

### 2.2 Results

   * latency: 14 clks (constant cache hit) and 78clks (constant cache miss)
   
   * i-cache interfere:  mask 0x41 -> smid 0 and smid 12 -> latency 238 clks  (Q: not sure how the block is distributed and what is the SMID mapping) 
   
ans to Q2: the original kernel only has 700 instructions (and my guess is correct.) Need to change the abs to some other arithmetic operations, so we can get large kernel code. I created a kernel code with 37000 instructions, which is ~290KB.    ~19KB;  

After graduately increasing the kernel size, I found that:  3018 inst(~24KB) -> 78 clk; 4649 inst(~37KB) ->81 clks; (latency keep increasing with increasing # of insts and till) -> 7596 insts(~60KB) -> 238 clks;

**Need to further analyze the reason behind this**

   * bandwidth: changed the kernel code used in example

       * kernel code:
```C++
for (int i = 0; i < CARRAY_SIZE; i = i + blockDim.x) {
    t1 += d_carray[i];
}
```


       * Results: 
       
           * scan through 1 to 64 threads per block. For 1 block, the bandwidth is increasing from 0.068 bytes/clk to 1.997 bytes/clk ; if it accesses the same address (t1 += d_carry[i & 0x1]), the bandwidth can reach 4.558 bytes/clk. It seems matched what I found online: https://forums.developer.nvidia.com/t/constant-memory-bandwidth-program/12574

           * 64 threads per block. Scan through 1 to 64 blocks, the bandwidth is almost fixed to 2 bytes/clk.
       
### 2.3 Summary

* Constant cache latency is 14 clocks and constant memory (LLC) is 78 clocks.
  
* The constant cache has constant data and instruction data. There is some sort of sharing, but it depends on the SM/constant cache/GPC distribution. That needs a more exhausive work to investigate.

* The constant cache BW is slow (2 bytes/clk), and it is better when you have multiple outstanding accesses with the same address. This is different from L1/L2. And since the bandwidth is not increasing with more blocks, it means there is only 1 constant cache. Probably this is a simple blocking cache with low bandwidth.

(But the third conclusion is conflicting with the second one, while I thought there are multiple copies of constant caches and there are some sharing rules between SMs and constant caches. Need further testing and thoughts.)

## 3. Texture

The test uses point sampling of 1D texture, which is 1-to-1 mapping between original 1D array.

### 3.1 Bandwidth Test

In L1 cache, all accesses are 1 sector/req. Meaning that using point sampling on 1D texture, the Texture Pipeline will only coalesce 32Byte.

* For L1 hit, per SM bandwidth is about 61.184bytes/clk

TODO: try 2D linear sampling and see if the sector/req increases.

## 4. Texture and Data Sharing in L1

latency BW
