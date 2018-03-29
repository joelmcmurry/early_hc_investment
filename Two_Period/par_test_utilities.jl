# PARALLEL TEST

using Optim

# minimize parameterized version of rosenbrock function

function min_rosenbrock_state(state::Float64)

  # define objective
  f(x) = (state - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

  opt_agent = optimize(f, [0., 0.], show_trace=false, iterations=5000)

  return opt_agent.minimizer, opt_agent.minimum

end

# function that solves N optimization problems

function solve_problems(;N=10)

  # create state vector length N
  state_vec = rand(N).*10.

  # intialize storage of solutions
  store_sol = zeros(N,2)
  store_min = zeros(N)

  for n in 1:N
    sol, val = min_rosenbrock_state(state_vec[n])
    store_sol[n,:] = sol
    store_min[n] = val
  end

  return store_sol, store_min, state_vec

end

function solve_problems(N)

  # create state vector length N
  state_vec = rand(N).*10.

  # intialize storage of solutions
  store_sol = zeros(N,2)
  store_min = zeros(N)

  for n in 1:N
    sol, val = min_rosenbrock_state(state_vec[n])
    store_sol[n,:] = sol
    store_min[n] = val
  end

  return store_sol, store_min, state_vec

end

function solve_problems2(N)

  # create state vector length N
  state_vec = rand(N+M).*10.

  # intialize storage of solutions
  store_sol = zeros(N+M,2)
  store_min = zeros(N+M)

  for n in 1:N+M
    sol, val = min_rosenbrock_state(state_vec[n])
    store_sol[n,:] = sol
    store_min[n] = val
  end

  return store_sol, store_min, state_vec

end
