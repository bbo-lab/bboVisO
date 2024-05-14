Each module here is expected to contain the following functions:

objfun_wrapper(free_vars, args, no_ravel=False)
objfun_jacobian_wrapper(free_vars, args)
make_jacobian_sparsity_mask(m, k, n, mask_free_input)
