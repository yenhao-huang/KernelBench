import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_forward_cuda(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b,
                                            int sd, int sh, int sw, int pd, int ph, int pw,
                                            int opd, int oph, int opw, int groups);
"""

conv_transpose3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int Cin, int Din, int Hin, int Win,
    int Cout, int kD, int kH, int kW,
    int Dout, int Hout, int Wout,
    int sd, int sh, int sw,
    int pd, int ph, int pw,
    int groups,
    int has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Dout * Hout * Wout;
    if (idx >= total) return;

    int ow = idx % Wout;
    int t = idx / Wout;
    int oh = t % Hout;
    t /= Hout;
    int od = t % Dout;
    t /= Dout;
    int oc = t % Cout;
    int n = t / Cout;

    int cout_per_group = Cout / groups;
    int cin_per_group = Cin / groups;
    int g = oc / cout_per_group;
    int ic_start = g * cin_per_group;
    int ic_end = ic_start + cin_per_group;
    int ocg = oc - g * cout_per_group;

    float acc = has_bias ? b[oc] : 0.0f;

    for (int ic = ic_start; ic < ic_end; ++ic) {
        for (int kd = 0; kd < kD; ++kd) {
            int td = od + pd - kd;
            if (td < 0 || td % sd != 0) continue;
            int id = td / sd;
            if (id < 0 || id >= Din) continue;

            for (int kh = 0; kh < kH; ++kh) {
                int th = oh + ph - kh;
                if (th < 0 || th % sh != 0) continue;
                int ih = th / sh;
                if (ih < 0 || ih >= Hin) continue;

                const float* x_base = x + (((n * Cin + ic) * Din + id) * Hin + ih) * Win;
                const float* w_base = w + ((((ic * cout_per_group + ocg) * kD + kd) * kH + kh) * kW);

                for (int kw = 0; kw < kW; ++kw) {
                    int tw = ow + pw - kw;
                    if (tw < 0 || tw % sw != 0) continue;
                    int iw = tw / sw;
                    if (iw >= 0 && iw < Win) {
                        acc += x_base[iw] * w_base[kw];
                    }
                }
            }
        }
    }

    y[idx] = acc;
}

torch::Tensor conv_transpose3d_forward_cuda(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b,
                                            int sd, int sh, int sw, int pd, int ph, int pw,
                                            int opd, int oph, int opw, int groups) {
    int N = x.size(0);
    int Cin = x.size(1);
    int Din = x.size(2);
    int Hin = x.size(3);
    int Win = x.size(4);

    int Cout = w.size(1) * groups;
    int kD = w.size(2);
    int kH = w.size(3);
    int kW = w.size(4);

    int Dout = (Din - 1) * sd - 2 * pd + kD + opd;
    int Hout = (Hin - 1) * sh - 2 * ph + kH + oph;
    int Wout = (Win - 1) * sw - 2 * pw + kW + opw;

    auto y = torch::empty({N, Cout, Dout, Hout, Wout}, x.options());

    const float* bptr = nullptr;
    int has_bias = 0;
    if (b.has_value() && b.value().defined()) {
        bptr = b.value().data_ptr<float>();
        has_bias = 1;
    }

    int total = N * Cout * Dout * Hout * Wout;
    int block = 256;
    int grid = (total + block - 1) / block;

    conv_transpose3d_kernel<<<grid, block>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), bptr, y.data_ptr<float>(),
        N, Cin, Din, Hin, Win, Cout, kD, kH, kW, Dout, Hout, Wout,
        sd, sh, sw, pd, ph, pw, groups, has_bias
    );

    return y;
}
"""

conv_transpose3d_ext = load_inline(
    name="conv_transpose3d_custom_ext",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_cuda_source,
    functions=["conv_transpose3d_forward_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose3d_ext.conv_transpose3d_forward_cuda(
            x,
            self.conv_transpose3d.weight,
            self.conv_transpose3d.bias,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.output_padding[0],
            self.output_padding[1],
            self.output_padding[2],
            self.groups,
        )