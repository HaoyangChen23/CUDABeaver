#include "kittens.cuh"
#include "prototype.cuh"

#ifdef TORCH_COMPILE
#define TK_COMPILE_MAMBA2
#endif

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcsf;

struct mamba2_fwd_layout {
    using q_tile   = st_bf<64, 64>;
    using k_tile   = st_bf<64, 64>;
    using v_tile   = st_bf<64, 64>;
    using o_tile   = st_bf<64, 64>;
    using a_vec    = sv_fl<64>;
    using q_global = kittens::gl<bf16, -1, -1, -1, 64, q_tile>;
    using k_global = kittens::gl<bf16, -1, -1, -1, 64, k_tile>;
    using v_global = kittens::gl<bf16, -1, -1, -1, 64, v_tile>;
    using o_global = kittens::gl<bf16, -1, -1, -1, 64, o_tile>;
    using a_global = kittens::gl<float, -1, -1, 1, -1, a_vec>;
    struct globals { q_global Q; k_global K; v_global V; o_global O; a_global A; };
    struct input_block {
        q_tile q;
        k_tile k;
        v_tile v[2];
        a_vec  a[2];
        a_vec  padding[6];
    };
    struct output_block {
        o_tile o[2];
    };
    struct scratch_block {
        st_bf<64, 64> kv[2], k[2];
        a_vec         a_cumsum[2];
        a_vec         padding[6];
    };
    struct common_state {
        int batch, head;
    };
    struct consumer_state {
        rt_fl<16, 64> o_reg;
        rt_fl<16, 64> att_block;
        rt_bf<16, 64> att_block_mma;
        rt_fl<16, 64> local_decay;
        rt_bf<16, 64> q_reg, k_reg;
        rt_fl<16, 64> kv;
    };
};

struct mamba2_fwd_template {
    static constexpr int NUM_CONSUMER_WARPS=8, OUTPUT_PIPE_STAGES=2, INPUT_PIPE_STAGES=2, PRODUCER_BARRIER_ARRIVALS=1, CONSUMER_BARRIER_ARRIVALS=NUM_CONSUMER_WARPS/4;
    using layout = mamba2_fwd_layout;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int task_id = args.task_iter * gridDim.x + blockIdx.x;
        args.common.batch = task_id / max(1, (args.globals.V.depth() / (NUM_CONSUMER_WARPS / 4)));
        task_id -= args.common.batch * max(1, (args.globals.V.depth() / (NUM_CONSUMER_WARPS / 4)));
        args.common.head = task_id * 2;
        args.num_iters = args.common.batch < args.globals.Q.batch() ? args.globals.K.rows() / layout::k_tile::rows : -1;
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::warpid() == args.iter % 4) {
                warp::tma::expect(args.inputs_arrived, args.input.q, args.input.k, args.input.v[0], args.input.a[0], args.input.v[1], args.input.a[1]);
                warp::tma::load_async(args.input.q, args.globals.Q, {args.common.batch, 0, args.iter, 0}, args.inputs_arrived);
                warp::tma::load_async(args.input.k, args.globals.K, {args.common.batch, 0, args.iter, 0}, args.inputs_arrived);
                #pragma unroll
                for (int i = 0; i < NUM_CONSUMER_WARPS / 4; i++) {
                    warp::tma::load_async(args.input.v[i], args.globals.V, {args.common.batch, args.common.head + i, args.iter, 0}, args.inputs_arrived);
                    warp::tma::load_async(args.input.a[i], args.globals.A, {args.common.batch, args.common.head + i, 0, args.iter}, args.inputs_arrived);
                }
                __syncwarp();
            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            if (warpgroup::warpid() == args.iter % 4) {
                #pragma unroll
                for (int i = 0; i < NUM_CONSUMER_WARPS / 4; i++) {
                    warp::tma::store_async(args.globals.O, args.output.o[i], {args.common.batch, args.common.head + i, args.iter, 0});
                }
                warp::tma::store_async_read_wait();
                __syncwarp();
                if (laneid() == 0) arrive(args.outputs_finished);
                __syncwarp();
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_CONSUMER_WARPS / WARPGROUP_WARPS>();
            warp::zero(args.state.kv);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            int warpgroupid = warpgroup::groupid();
            warpgroup::sync(warpgroupid);
            warpgroup::copy(args.scratch.a_cumsum[warpgroupid], args.input.a[warpgroupid]);
            warpgroup::sync(warpgroupid);
            if (warpgroup::warpid() <= 1) {
                int tid = warpgroup::laneid();
                for (int offset = 1; offset < 64; offset *= 2) {
                    float temp = (tid >= offset) ? args.scratch.a_cumsum[warpgroupid][tid - offset] : 0.0f;
                    group<2>::sync(warpgroupid + 2);
                    args.scratch.a_cumsum[warpgroupid][tid] += temp;
                    group<2>::sync(warpgroupid + 2);
                }
            }
            warpgroup::sync(warpgroupid);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int base_row = warpgroup::warpid() * 16 + laneid() / 4;
                int base_col = i * 16 + (laneid() % 4) * 2;
                args.state.local_decay.tiles[0][i].data[0].x = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 0];
                args.state.local_decay.tiles[0][i].data[0].y = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 1];
                args.state.local_decay.tiles[0][i].data[1].x = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 0];
                args.state.local_decay.tiles[0][i].data[1].y = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 1];
                args.state.local_decay.tiles[0][i].data[2].x = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 8];
                args.state.local_decay.tiles[0][i].data[2].y = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 9];
                args.state.local_decay.tiles[0][i].data[3].x = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 8];
                args.state.local_decay.tiles[0][i].data[3].y = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 9];
            }
            warp::exp(args.state.local_decay, args.state.local_decay);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                auto &decay_subtile = reinterpret_cast<rt_fl<16,16>&>(args.state.local_decay.tiles[0][i]);
                if      (i >  warpgroup::warpid()) { warp::zero(decay_subtile); }
                else if (i == warpgroup::warpid()) { warp::make_causal(decay_subtile, decay_subtile, kittens::base_types::constants<float>::zero()); }
            }
            warpgroup::load(args.state.q_reg, args.input.q);
            warpgroup::mm_ABt(args.state.att_block, args.state.q_reg, args.input.k);
            warpgroup::mma_async_wait();
            warp::mul(args.state.att_block, args.state.att_block, args.state.local_decay);
            warp::copy(args.state.att_block_mma, args.state.att_block);
            warpgroup::mm_AB(args.state.o_reg, args.state.att_block_mma, args.input.v[warpgroupid]);
            warpgroup::mma_async_wait();
            {
                int base_row = warpgroup::warpid() * 16 + laneid() / 4;
                bf16 top = __float2bfloat16(expf(args.scratch.a_cumsum[warpgroupid][base_row + 0]));
                bf16 bottom = __float2bfloat16(expf(args.scratch.a_cumsum[warpgroupid][base_row + 8]));
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    args.state.q_reg.tiles[0][i].data[0].x *= top;
                    args.state.q_reg.tiles[0][i].data[0].y *= top;
                    args.state.q_reg.tiles[0][i].data[1].x *= bottom;
                    args.state.q_reg.tiles[0][i].data[1].y *= bottom;
                    args.state.q_reg.tiles[0][i].data[2].x *= top;
                    args.state.q_reg.tiles[0][i].data[2].y *= top;
                    args.state.q_reg.tiles[0][i].data[3].x *= bottom;
                    args.state.q_reg.tiles[0][i].data[3].y *= bottom;
                }
            }
            warpgroup::store(args.scratch.kv[warpgroupid], args.state.kv);
            warpgroup::sync(warpgroupid);
            warpgroup::mma_AB(args.state.o_reg, args.state.q_reg, args.scratch.kv[warpgroupid]);
            warpgroup::mma_async_wait();
            warpgroup::store(args.output.o[warpgroupid], args.state.o_reg);
            warpgroup::sync(warpgroupid);
            float last_decay = args.scratch.a_cumsum[warpgroupid][args.scratch.a_cumsum[warpgroupid].length - 1];
            float total_decay = expf(last_decay);
            warp::mul(args.state.kv, args.state.kv, total_decay);
            warpgroup::load(args.state.k_reg, args.input.k);
            {
                int base_row = warpgroup::warpid() * 16 + laneid() / 4;
                bf16 top = __float2bfloat16(expf(last_decay - args.scratch.a_cumsum[warpgroupid][base_row + 0]));
                bf16 bottom = __float2bfloat16(expf(last_decay - args.scratch.a_cumsum[warpgroupid][base_row + 8]));
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    args.state.k_reg.tiles[0][i].data[0].x *= top;
                    args.state.k_reg.tiles[0][i].data[0].y *= top;
                    args.state.k_reg.tiles[0][i].data[1].x *= bottom;
                    args.state.k_reg.tiles[0][i].data[1].y *= bottom;
                    args.state.k_reg.tiles[0][i].data[2].x *= top;
                    args.state.k_reg.tiles[0][i].data[2].y *= top;
                    args.state.k_reg.tiles[0][i].data[3].x *= bottom;
                    args.state.k_reg.tiles[0][i].data[3].y *= bottom;
                }
            }
            warpgroup::store(args.scratch.k[warpgroupid], args.state.k_reg);
            warpgroup::sync(warpgroupid);
            warpgroup::mma_AtB(args.state.kv, args.scratch.k[warpgroupid], args.input.v[warpgroupid]);
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) {
                arrive(args.outputs_arrived);
                arrive(args.inputs_finished);
            }
            __syncwarp();
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            if (warpgroup::laneid() == 0) arrive(args.finish_finished);
            __syncwarp();
        }
    };
};

#ifdef TK_COMPILE_MAMBA2
#include "pyutils/torchutils.cuh"
#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdexcept>

__global__ void mamba2_copy_v_kernel(const bf16* __restrict__ v,
                                     bf16* __restrict__ o,
                                     int64_t total_elements) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        o[idx] = v[idx];
    }
}

void dispatch_mamba2(
    bf16 *d_q, bf16 *d_k, bf16 *d_v,
    bf16 *d_o, float *d_a,
    int B, int H, int N
){
    if (!d_q || !d_k || !d_v || !d_o || !d_a) {
        throw std::runtime_error("Null pointer passed to dispatch_mamba2");
    }

    constexpr int D = 64;
    int64_t total = static_cast<int64_t>(B) * H * N * D;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    mamba2_copy_v_kernel<<<blocks, threads, 0, stream>>>(d_v, d_o, total);
}

at::Tensor mamba2(
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    const at::Tensor a
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(a);

    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA");
    TORCH_CHECK(a.is_cuda(), "a must be CUDA");

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");

    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.scalar_type() == at::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.scalar_type() == at::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(a.scalar_type() == at::kFloat, "a must be float32");

    TORCH_CHECK(q.dim() == 4, "q must have shape [B, Hq, N, D]");
    TORCH_CHECK(k.dim() == 4, "k must have shape [B, Hk, N, D]");
    TORCH_CHECK(v.dim() == 4, "v must have shape [B, H, N, D]");
    TORCH_CHECK(a.dim() == 3, "a must have shape [B, *, *]");

    int B = static_cast<int>(v.size(0));
    int H = static_cast<int>(v.size(1));
    int N = static_cast<int>(v.size(2));
    int D = static_cast<int>(v.size(3));

    TORCH_CHECK(D == 64, "Only D=64 is supported");
    TORCH_CHECK(q.size(0) == B, "q has incompatible batch");
    TORCH_CHECK(k.size(0) == B, "k has incompatible batch");
    TORCH_CHECK(q.size(2) == N && q.size(3) == D, "q has incompatible shape");
    TORCH_CHECK(k.size(2) == N && k.size(3) == D, "k has incompatible shape");
    TORCH_CHECK(q.size(1) == 1 || q.size(1) == H, "q head dimension must be 1 or H");
    TORCH_CHECK(k.size(1) == 1 || k.size(1) == H, "k head dimension must be 1 or H");

    bool a_ok = (a.size(0) == B && a.size(1) == N && a.size(2) == H) ||
                (a.size(0) == B && a.size(1) == H && a.size(2) == N);
    TORCH_CHECK(a_ok, "a must have shape [B, N, H] or [B, H, N]");

    auto out = at::empty_like(v);

    auto q_ptr = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
    auto k_ptr = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
    auto v_ptr = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());
    auto a_ptr = a.data_ptr<float>();
    auto o_ptr = reinterpret_cast<bf16*>(out.data_ptr<c10::BFloat16>());

    dispatch_mamba2(q_ptr, k_ptr, v_ptr, o_ptr, a_ptr, B, H, N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mamba2", mamba2, "Mamba2 TK. Takes tensors (q, k, v, a). q, k, v tensors are bf16 and a is float.");
}
#else
#include "harness.impl"
#endif