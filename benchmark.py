import torch
import triton
import triton.language as tl
from utils import * 


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['vq-causal', 'vq-non-causal',  'flash-attn', 'hyper-attn'],  # Possible values for `line_arg`.
        line_names=['vq-causal', 'vq-non-causal', 'flash-attn', 'hyper-attn'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    config = setup_config()
    config['sequence_len'] = size
    data = build_data(config)
    empty_cache(config)
    q, k, v, init_state, d_model, model = build_test_pipeline(config, data)
    empty_cache(config)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'vq-causal':
        present_z_k, present_z_q = get_short_codes(model, q, k)
        with torch.inference_mode():
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: model.attn(present_z_k=present_z_k,
                        present_z_q=present_z_q,
                        aggcache=init_state['aggcache'],
                        present_v=v,
                        causal=True), quantiles=quantiles)
    if provider == 'vq-non-causal':
        present_z_k, present_z_q = get_short_codes(model, q, k)
        with torch.inference_mode():
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: model.attn(present_z_k=present_z_k,
                        present_z_q=present_z_q,
                        aggcache=init_state['aggcache'],
                        present_v=v,
                        causal=False), quantiles=quantiles)
    return ms, max_ms, min_ms

benchmark.run(show_plots=True, print_data=True)
