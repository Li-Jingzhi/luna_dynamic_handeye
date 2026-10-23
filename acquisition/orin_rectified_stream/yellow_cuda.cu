#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstdint>

struct YellowContext {
    int width, height;
    uint8_t *mask, *opened;
    unsigned long long *stats;
};

__global__ void threshold_yellow(const uint8_t *bgr, int width, int height, int stride,
                                 int x0, int y0, int x1, int y1, uint8_t *mask) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    bool inside = x >= x0 && x < x1 && y >= y0 && y < y1;
    const uint8_t *p = bgr + y * stride + x * 3;
    float b = p[0], g = p[1], r = p[2];
    float hi = fmaxf(r, fmaxf(g, b)), lo = fminf(r, fminf(g, b)), d = hi - lo;
    float h = 0.f;
    if (d > 0.f) {
        if (hi == r) h = 60.f * fmodf((g-b)/d, 6.f);
        else if (hi == g) h = 60.f * ((b-r)/d + 2.f);
        else h = 60.f * ((r-g)/d + 4.f);
        if (h < 0.f) h += 360.f;
    }
    float s = hi > 0.f ? 255.f * d / hi : 0.f;
    float hue8 = h * .5f;
    mask[y * width + x] = inside && hue8 >= 5.f && hue8 <= 40.f && s >= 100.f && hi >= 100.f;
}

__global__ void erode5(const uint8_t *src, uint8_t *dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint8_t value = 1;
    for (int dy=-2; dy<=2 && value; ++dy) for (int dx=-2; dx<=2; ++dx) {
        int xx=x+dx, yy=y+dy;
        if (xx < 0 || xx >= width || yy < 0 || yy >= height || !src[yy*width+xx]) { value=0; break; }
    }
    dst[y*width+x]=value;
}

__global__ void dilate5_stats(const uint8_t *src, uint8_t *dst, int width, int height,
                              unsigned long long *stats) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint8_t value = 0;
    for (int dy=-2; dy<=2 && !value; ++dy) for (int dx=-2; dx<=2; ++dx) {
        int xx=x+dx, yy=y+dy;
        if (xx >= 0 && xx < width && yy >= 0 && yy < height && src[yy*width+xx]) { value=1; break; }
    }
    dst[y*width+x]=value;
    if (!value) return;
    atomicAdd(&stats[0], 1ULL); atomicAdd(&stats[1], (unsigned long long)x); atomicAdd(&stats[2], (unsigned long long)y);
    atomicMin((unsigned int *)&stats[3], (unsigned int)x); atomicMin((unsigned int *)&stats[4], (unsigned int)y);
    atomicMax((unsigned int *)&stats[5], (unsigned int)x); atomicMax((unsigned int *)&stats[6], (unsigned int)y);
}

extern "C" void *yellow_create(int width, int height) {
    auto *ctx = new YellowContext{width,height,nullptr,nullptr,nullptr};
    size_t pixels=(size_t)width*height;
    if (cudaMalloc(&ctx->mask,pixels) || cudaMalloc(&ctx->opened,pixels) || cudaMalloc(&ctx->stats,7*sizeof(unsigned long long))) return nullptr;
    return ctx;
}

extern "C" void yellow_destroy(void *opaque) {
    auto *ctx=(YellowContext *)opaque; if (!ctx) return;
    cudaFree(ctx->mask); cudaFree(ctx->opened); cudaFree(ctx->stats); delete ctx;
}

extern "C" int yellow_center(void *opaque, uintptr_t device_bgr, int stride, float previous_x,
                             float previous_y, float *out_x, float *out_y, int *out_area) {
    auto *ctx=(YellowContext *)opaque; if (!ctx || !device_bgr) return -1;
    int radius=ctx->width <= 1280 ? 80 : 160, x0=0,y0=0,x1=ctx->width,y1=ctx->height;
    if (previous_x >= 0 && previous_y >= 0) {
        x0=max(0,(int)lrintf(previous_x)-radius); x1=min(ctx->width,(int)lrintf(previous_x)+radius);
        y0=max(0,(int)lrintf(previous_y)-radius); y1=min(ctx->height,(int)lrintf(previous_y)+radius);
    }
    unsigned long long init[7]={0,0,0,UINT_MAX,UINT_MAX,0,0};
    cudaMemcpy(ctx->stats,init,sizeof(init),cudaMemcpyHostToDevice);
    dim3 block(32,8), grid((ctx->width+31)/32,(ctx->height+7)/8);
    threshold_yellow<<<grid,block>>>((const uint8_t *)device_bgr,ctx->width,ctx->height,stride,x0,y0,x1,y1,ctx->mask);
    erode5<<<grid,block>>>(ctx->mask,ctx->opened,ctx->width,ctx->height);
    dilate5_stats<<<grid,block>>>(ctx->opened,ctx->mask,ctx->width,ctx->height,ctx->stats);
    unsigned long long result[7];
    if (cudaMemcpy(result,ctx->stats,sizeof(result),cudaMemcpyDeviceToHost) != cudaSuccess) return -2;
    int area=(int)result[0], bw=(int)result[5]-(int)result[3]+1, bh=(int)result[6]-(int)result[4]+1;
    *out_area=area;
    float ratio=bh>0 ? (float)bw/bh : 0.f, fill=bw>0&&bh>0 ? (float)area/(bw*bh) : 0.f;
    if (area < 500 || area > 50000 || ratio < .70f || ratio > 1.43f || fill < .5f) return 0;
    *out_x=(float)result[1]/area; *out_y=(float)result[2]/area; return 1;
}
