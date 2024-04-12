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
   
* i-cache interfere:  mask 0x41 -> smid 0 and smid 12 -> latency 238 clks  (Q: not sure how the block is distributed and what is the SMID mapping;)

    * By a scanning of SMIDs, I found that **SMIDs 12, 24, 36, 1, 13**  have worst I-cache interference with SMID 0, >200clks; while **SMIDs 10, 32, 42, 9** have worse I-cache interference with SMID 0, > 100clks
   
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

        * scan through 1 to 64 threads per block. For 1 block, the bandwidth is increasing from 0.068 bytes/clk to 1.997 bytes/clk ; if it accesses the same address (t1 += d_carry[i & 0x1]), the bandwidth can reach 4.558 bytes/clk. It seems matched what I found online: https://forums.developer.nvidia.com/t/constant-memory-bandwidth-program/12574 . The T4 dissect report provided by Citadel mentions that there is a broadcast feature when different threads accessing the same constant memory address.

        * 64 threads per block. Scan through 1 to 64 blocks, the bandwidth is almost fixed to 2 bytes/clk.
       
### 2.3 Summary

* Constant cache latency is 14 clocks and constant cache(L1.5) is 78 clock. 
  
* The constant cache has constant data and instruction data. There is some sort of sharing, but it depends on the SM/constant cache/GPC distribution. That needs a more exhausive work to investigate.

* The constant cache BW is slow (2 bytes/clk), and it is better when you have multiple outstanding accesses with the same address.(Broadcast feature) This is different from L1/L2. And since the bandwidth is not increasing with more blocks, it means there is only 1 constant cache. Probably this is a simple blocking cache with low bandwidth.

(But the third conclusion is conflicting with the second one, while I thought there are multiple copies of constant caches and there are some sharing rules between SMs and constant caches. Need further testing and thoughts.)

* The T4 dissect report mentions that there is a private L1 constant cache and L1.5 constant cache. But after my tests, I think the L1.5 is shared by some of SMs, which needs further investigation. And the bandwidth is not increasing, probably because all L1.5 caches merged into 1 port to L2, so we don't bandwidth increasing. And probably this is only a narrow data bus (16 bits?).

## 3. Texture

The test uses point sampling of 1D texture, which is 1-to-1 mapping between original 1D array. Use linear memory which has larger maximum boundary in CUDA.

### 3.1 Bandwidth Test

In L1 cache, all accesses are 1 sector/req. Meaning that using point sampling on 1D texture, the Texture Pipeline will only coalesce 32Byte.

* For L1 hit, the total bandwidth is ~1940GB/s  (460 blocks and 128 threads per block, freq locked at 1500MHz)

* For L2 hit/L1 miss, how to avoid L1 hit completely?

TODO: 
(not sure why the per SM calculation is wrong)

### 3.2 2D texture access

I used the exampled provided in CUDA Programming Guide, which does 2D texture fetch, and the sectors/req in L1 is 3.57 (935108/262144). Tuning the float angle from 0.5 to 1, the sectors/req reachs 3.82. With angle = 0.5, filter mode changed to Point, the sectors/req is 2.18 (572433/262144)

## 4. Texture and Data Sharing in L1

For global data, L1 hit bandwidth can reach 3097GB/s  (460 blocks and 512 threads per block, freq locked at 1500MHz)  about 
First experiment: 
1. 460 blocks, and 640 threads. when threadId > 127, do global data load; otherwise do texture load. The bandwidth is ~2080GB/s. From Nsight, I can see that tex sector/request becomes 0.5. (wavefronts is 1.5 of requests and 64% of peak. While L1 global load is 21% of peak. ) (peaks should be wavefronts divided by some total number.)
2. 460 blocks, and 256 threads. when threadId > 127, do global data load; otherwise do texture load. From Nsight, I can see that tex sector/request becomes 0.8.

## 5. Texture 1D and 2D Fetch

Nsight results for Texture Load (global stores are 4sectors/requests for all cases)
|Kernel|Filter Mode|L1 Instructions|L1 Requests|L1 Sectors|L1 Sectors/Req|Hit Rate|Bytes|Sectors Miss to L2|Return to SM|
|------|-----------|---------------|-----------|----------|--------------|--------|-----|------------------|------------|
|2D|Point|32768|262,144|262,144|1|50|8,388,608|131,072|262,144|
|1D|Point|32768|131,072|131,072|1|0|4,194,304|131,072|131,072|
|2D|Linear|32768|262,144|783209|2.99|80.18|25,062,688|155,837|262,144|
|1D|Linear|32768|262,144|0|0|0|0|0|262,144|


|Kernel|Filter Mode|L2 Requests|L2 Sectors|L2 Sectors/Req|Hit Rate|Bytes|
|------|-----------|-----------|----------|--------------|--------|-----|
|2D|Point|65523|131,072|2|0|4,194,304|
|1D|Point|32768|131,072|4|0|4,194,304|
|2D|Linear|73195|155,802|2.13|15.89|4,985,664|
|1D|Linear|0|0|0|0|0|

Questions:
1. Linear filter mode for 1D fetch is not correct (no L2 transactions are generated)
2. In linear filter mode for 2D texture read, the results are all 0s.  (In point filter mode, the results are 0 - 1023 for the first row, which is expected.)
     "Linear texture filtering may be done only for textures that are configured to return floating-point data. " From CUDA programming guide.
## 6. Threadfence latency

### 6.1 Results

|API|Latency(cycle)|
|---|-------|
|threadfence_block()|7|
|threadfence()|~327|
|threadfence_system()|~1700|

The order of the API calls in the kernel doesn't affect the latency. I tried to add some global store before threadfence call, and it doesn't affect the results either. Assume the cache flush only takes number of sets cycles:
L1 is 128KB -> 1024 cachelines -> 64 sets (if 16 ways)
L2 is 4MB -> 32K cachelines -> 2K sets (if 16 ways)
But also need to consider the parallelism of different banks/slices. 

## 7. Atomic Sectors per request

### 7.1 Results

|Function|Data Type|L2 sectors per req|L2 request|
|--------|---------|------------------|----------|
|atomicAdd|int|4|2|
|atomicCAS|int|2|4|
|atomicCAS|ull|2|8|
|atomicCAS|short|2|2(load) + 4(CAS)|

## 8. L1 Size and Associativity sharing with Shared Memory

### 8.1 Results

|Number of Int Array Element|Total Size(KB)|Stride(number of array element)|L1 Config|Order of Second Scan|L1 Hit Rate|
|---------------------------|--------------|------|---------|--------------------|-----------|
|96K|384|32|PreferL1|In-Order|0%|
|96K|384|32|PreferL1|Reverse Order|12.97% or 12.94%|
|32K|128|32|PreferL1|In-Order|0.59% or 0.49%|
|32K|128|32|PreferL1|Reverse Order|37.89%|
|30K|120|32|PreferL1|In-Order|6.35%, 6.87%, 6.82%|
|30K|120|32|PreferL1|Reverse Order|39.06%, 39.90%|
|27K|108|32|PreferL1|In-Order|40.97%, 39.99%, 40.91%|
|27K|108|32|PreferL1|Reverse Order|47.45%, 47.28%|
|26K|104|32|PreferL1|In-Order|50%|
|26K|104|32|PreferL1|Reverse Order|50%|
|24K|96|32|PreferL1|In-Order|50%|
|24K|96|32|PreferL1|Reverse Order|50%|
|16K|64|32|PreferL1|In-Order|50%|
|16K|64|32|PreferL1|Reverse Order|50%|
|8K|32|32|PreferL1|In-Order|50%|
|4K|16|32|PreferL1|In-Order|50%|
|32K|128|64|PreferL1|In-Order|50%|
|16K|64|64|PreferL1|In-Order|50%|
|16K|64|32|PreferShared|In-Order|0%|
|16K|64|32|PreferShared|Reverse Order|12.79% or 12.70%|
|8K|32|32|PreferShared|In-Order|17.97% or 16.60%|
|8K|32|32|PreferShared|Reverse Order|24.02% or 22.85%|
|4K|16|32|PreferShared|In-Order|50%|
|4K|16|32|PreferShared|Reverse Order|50%|

### 8.2 Summary
1. Even with PreferL1, users cannot have 128KB L1 total size
2. With PreferShared, users can use 16KB L1 total size

   
Questions:
why in-order is much faster than reverse order access? 21000 cycles vs. 61000 cycles for 8K;  10000 cycles vs. 17000 cycles for 4K

## 9. L2 Data Compression

The official cuda sample has issues and here is the fix, https://www.zhihu.com/question/597437766/answer/3002601515  . Need to use cudaMallocHost and cudaFreeHost for compressible memories.

### 9.1 Results
Default setup:
```
Running saxpy on 167772160 bytes of Compressible memory
Running saxpy with 92 blocks x 768 threads = 0.552 ms 0.912 TB/s
Running saxpy on 167772160 bytes of Non-Compressible memory
Running saxpy with 92 blocks x 768 threads = 1.385 ms 0.363 TB/s
```

Nsights:

||Array Value|L1 load from L2(MB)|L1 store to L2(MB)|L2 load hit rate|L2 load from DDR(MB)|L2 store to DDR(MB)|
|-|----------|------------------|------------------|-----------|--------------------|-------------------|
|With Compression|1.0f|335.54|167.77|62.26%|44.17|165.84|
|Without Compression|1.0f|335.54|167.77|0%|335.58|166.25|
|With Compression|random|335.54|167.77|0%|335.57|166.17|
|Without Compression|random|335.54|167.77|0%|342.97|166.25|
|With Compression|x,z: random; y,w:0|335.54|167.77|0%|335.57|167.97|
|Without Compression|x,z: random; y,w:0|335.54|167.77|0%|335.56|166.17|
|With Compression|h[x].x,h[y].z: random; others:0|335.54|167.77|0%|192.43|165.90|
|Without Compression|h[x].x,h[y].z: random; others:0|335.54|167.77|0%|335.56|166.20|
|With Compression|z,w: random; x,y:0|335.54|167.77|0%|335.57|166.15|
|Without Compression|z,w: random; x,y:0|335.54|167.77|0%|344.13|166.19|

With constant value, the compression rate is really high. And here are some other Nsight statistic for **global load** when changing the number of constant elements:

||length of 1.0f (n = 10485760)|L1 Sectors misses to L2|L2 Requests|L2 Sectors|L2 Sectors per request|L2 Bytes|Hit Rate|
|-|----------------------------|-----------------------|-----------|----------|----------------------|--------|--------|
|With Compression|from 0 to n|10,485,760|2,621,440|3,755,266|1.43|120,168,512|62.26%|
|Without Compression|from 0 to n|10,485,760|2,621,440|10,485,760|4|335,544,320|0%|
|With Compression|from 0 to n/4|10,485,760|2,621,440|8,818,924|3.36|282,205,568|6.31%|
|With Compression|from 0 to n/2|10,485,760|2,621,440|7,051,672|2.69|225,653,504|16.24%|
|With Compression|from 0 to 3n/4|10,485,760|2,621,440|5,385,412|2.05|172,333,184|31.57%|
|With Compression|from 0 to 2(1sector)|10,485,760|2,621,440|10,485,760|4|335,544,320|0%|
|With Compression|from 0 to 8(1line)|10,485,760|2,621,440|10,485,760|4|335,544,320|0%|
|With Compression|from 0 to 16(2lines)|10,485,760|2,621,440|10,485,760|4|335,544,320|0%|
|With Compression|from 0 to 128(16lines)|10,485,760|2,621,440|10,485,760|4|335,544,320|0%|
|With Compression|from 0 to 256(4KB)|10,485,760|2,621,440|10,485,760|4|335,544,320|0%|
|With Compression|from 0 to n - 256|10,485,760|2,621,440|3,756,928/3,759,280|1.43|120,221,696/120,296,960|59.70%|
|With Compression|from 0 to n - 16|10,485,760|2,621,440|3,757,312/3,753,844|1.43|120,233,984/120,123,008|59.69%|
|With Compression|from 0 to n - 8|10,485,760|2,621,440|3,755,884|1.43|120,188,288|59.73%|
|With Compression|from 0 to n - 2|10,485,760|2,621,440|3,757,234|1.43|120,231,488|59.69%|


However, there is no results in L2 Compression in Nsight. We have to see the benefit through DDR traffic.

### 9.2 Summary

1. 


TODO: write compression
