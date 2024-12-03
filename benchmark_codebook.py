import torch
import triton
import triton.language as tl
from utils import * 
from matplotlib import pyplot as plt


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['n_code'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(5, 14, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['vq-causal', 'vq-non-causal', 'hyper-attn-causal', 'hyper-attn-non-causal', 'vanilla-causal', 'vanilla-non-causal' ],  # Possible values for `line_arg`.
        line_names=['vq-causal', 'vq-non-causal', 'hyper-attn-causal', 'hyper-attn-non-causal',  'vanilla-causal', 'vanilla-non-causal' ],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('orange', '-'), ('purple', '-'), ('grey', '-')],  # Line styles.
        ylabel='m/s',  # Label name for the y-axis.
        plot_name='Runtime(ms)',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(n_code, provider):
    config = setup_config()
    config['sequence_len'] = 16384
    config['n_code'] =  n_code
    config['n_code_k'] = n_code
    config['n_code_q'] = n_code
    print('Number of codebook entries: ', n_code)
    data = build_data(config)
    empty_cache(config)
    q, k, v, init_state, d_model, model = build_test_pipeline(config, data)
    quantiles = [0.5, 0.2, 0.8]
    del data
    empty_cache(config)
    if provider == 'vq-causal':
        present_z_k, present_z_q = get_short_codes(model, q, k)
        with torch.inference_mode():
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_vq_attn(model, present_z_k, present_z_q, v, init_state, causal=True), quantiles=quantiles)
    if provider == 'vq-non-causal':
        present_z_k, present_z_q = get_short_codes(model, q, k)
        with torch.inference_mode():
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_vq_attn(model, present_z_k, present_z_q, v, init_state, causal=False), quantiles=quantiles)
    if provider == 'hyper-attn-causal':
        del model, init_state
        empty_cache(config)
        hyper_attn = compile_hyper_attn(q.shape[-1], config['device'], block_size=config['block_len'], sample_size=64)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_hyper_attn(hyper_attn, q, k, v, causal=True), quantiles=quantiles)
    if provider == 'hyper-attn-non-causal':
        del model, init_state
        empty_cache(config)
        hyper_attn = compile_hyper_attn(q.shape[-1], config['device'], block_size=config['block_len'], sample_size=64)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_hyper_attn(hyper_attn, q, k, v, causal=True), quantiles=quantiles)
    if provider == 'vanilla-causal':
        del model, init_state
        empty_cache(config)
        vanilla_causal_model = compile_vanilla_causal(config, config['d_model'], config['block_len'])
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_vanilla_attn_causal(q, k, v, vanilla_causal_model) , quantiles=quantiles)
    if provider == 'vanilla-non-causal':
        vanilla_attention_model = compile_vanilla_attn(config['d_model'], config['n_head'], config['device'])
        del model, init_state
        empty_cache(config)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_vanilla_attn_non_causal(q, k, v, vanilla_attention_model), quantiles=quantiles)
    return ms, max_ms, min_ms

benchmark.run(show_plots=True, print_data=True)
plt.savefig('benchmark_codebook_ms.png')
