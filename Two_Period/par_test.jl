addprocs(2)

@everywhere include("par_test_utilities.jl")

function single_proc(;N=100)

  out = solve_problems(N=N)

end

function two_procs(;N=100)

  out_1 = @spawn solve_problems(N=Int(N/2))
  out_2 = @spawn solve_problems(N=Int(N/2))

  res_1 = fetch(out_1)
  res_2 = fetch(out_2)

  minimizer_out = [res_1[1]; res_2[1]]
  minval_out = [res_1[2]; res_2[2]]
  state_out = [res_1[3]; res_2[3]]

  return minimizer_out, minval_out, state_out

end

function five_procs(;N=100)

  out_1 = @spawn solve_problems(N=Int(N/5))
  out_2 = @spawn solve_problems(N=Int(N/5))
  out_3 = @spawn solve_problems(N=Int(N/5))
  out_4 = @spawn solve_problems(N=Int(N/5))
  out_5 = @spawn solve_problems(N=Int(N/5))

  res_1 = fetch(out_1)
  res_2 = fetch(out_2)
  res_3 = fetch(out_3)
  res_4 = fetch(out_4)
  res_5 = fetch(out_5)

  minimizer_out = [res_1[1]; res_2[1]; res_3[1]; res_4[1]; res_5[1]]
  minval_out = [res_1[2]; res_2[2]; res_3[2]; res_4[2]; res_5[2]]
  state_out = [res_1[3]; res_2[3]; res_3[3]; res_4[3]; res_5[3]]

  return minimizer_out, minval_out, state_out

end

single_time = @elapsed test_single = single_proc(N=100000)

double_time = @elapsed test_double = two_procs(N=100000)

ptime = @elapsed p_test = pmap(solve_problems, [50000 50000])

five_time = @elapsed test_five = five_procs(N=100000)

ptime = @elapsed p_test = pmap(solve_problems, [20000 20000 20000 20000 20000])

# test arguments to pass

pass_args = [20000 20000 20000 20000 20000]

ptime = @elapsed p_test = pmap(solve_problems, pass_args)

pass_args2 = [[1, ]]
