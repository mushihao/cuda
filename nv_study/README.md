1. nv3070 l2 byte mask findings from Nsight compute:
  *L1 receives 131 sectors/131 reqs of global store but have 103 sector misses to L2. meaning L1 could temporarily store the data for a while, even we write 4 bytes each time.
  *L2
     -Load: 98 sectors/98 requests with 1.02% hit rate
     -Store: 103 sectors/100 requests with 100% hit rate

Summary: Looks like L1 has write combine feature; L2 has byte mask enabled for every cacheline stored.
