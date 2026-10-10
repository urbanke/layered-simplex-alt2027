"""Layered simplex architecture: current-paper numerical and reference APIs.

The experiment interface is ``lsa.alt``. Independent reference APIs below are
loaded only when requested; importing the package never compiles a library.
Losses use bits and low-level log evidence uses natural logarithms.
"""
from importlib import import_module

__version__ = "1.1.0"

_EXPORTS = {'ProductSimplexSampler': ('lsa.product_simplex', 'ProductSimplexSampler'), 'fill_product_simplex_sample': ('lsa.product_simplex', 'fill_product_simplex_sample'), 'sample_product_simplex_distribution': ('lsa.product_simplex', 'sample_product_simplex_distribution'), 'entropy': ('lsa.metrics', 'entropy'), 'kl_divergence': ('lsa.metrics', 'kl_divergence'), 'ProductMomentTables': ('lsa.mixture_weights', 'ProductMomentTables'), 'QLambdaResult': ('lsa.mixture_weights', 'QLambdaResult'), 'build_product_moment_tables': ('lsa.mixture_weights', 'build_product_moment_tables'), 'build_selected_product_moment_tables': ('lsa.mixture_weights', 'build_selected_product_moment_tables'), 'compute_log_q_by_partition': ('lsa.mixture_weights', 'compute_log_q_by_partition'), 'log_q_lambda_closed_l1': ('lsa.mixture_weights', 'log_q_lambda_closed_l1'), 'log_q_lambda_grid': ('lsa.mixture_weights', 'log_q_lambda_grid'), 'log_q_lambda_laplace': ('lsa.mixture_weights', 'log_q_lambda_laplace'), 'HybridSaddlepointApproximation': ('lsa.pattern_weights', 'HybridSaddlepointApproximation'), 'SaddlepointApproximation': ('lsa.pattern_weights', 'SaddlepointApproximation'), 'log_a_lambda_hybrid': ('lsa.pattern_weights', 'log_a_lambda_hybrid'), 'log_a_lambda_saddlepoint': ('lsa.pattern_weights', 'log_a_lambda_saddlepoint'), 'partition_multiplicities': ('lsa.pattern_weights', 'partition_multiplicities'), 'largest_entries_first': ('lsa.transforms', 'largest_entries_first'), 'C_STAR': ('lsa.codelength', 'C_STAR'), 'DepthAveragedCodelength': ('lsa.codelength', 'DepthAveragedCodelength'), 'default_l_max': ('lsa.codelength', 'default_l_max'), 'depth_averaged_codelength': ('lsa.codelength', 'depth_averaged_codelength'), 'needed_r_values': ('lsa.codelength', 'needed_r_values'), 'profile_of': ('lsa.codelength', 'profile_of'), 'empirical_conditional_entropy_bits': ('lsa.corpus', 'empirical_conditional_entropy_bits'), 'empirical_entropy_bits': ('lsa.corpus', 'empirical_entropy_bits'), 'load_tokens': ('lsa.corpus', 'load_tokens'), 'prefix_counts': ('lsa.corpus', 'prefix_counts'), 'build_tables_fast': ('lsa.fast_tables', 'build_tables_fast')}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value
