#include<cuda.h>
#include <stdio.h>

__device__ void haha(uintptr_t *ret)
{
    ret = 0;
}
__global__ void run()
{
    uintptr_t ret;
    haha(&ret);
    printf("ret = %d\n", (void*)ret);
}
main(){
    run<<<1,1>>>();
    cudaDeviceSynchronize();
}