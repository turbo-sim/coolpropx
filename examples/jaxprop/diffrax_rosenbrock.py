import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx

def rosenbrock(x, args):
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2

x0 = jnp.array([-1.2, 1.0])

# solver with reasonable Armijo defaults
solver = optx.BFGS(rtol=1e-8, atol=1e-8)
solver = eqx.tree_at(lambda s: s.search, solver,
                     optx.BacktrackingArmijo(step_init=0.5, slope=1e-8))

sol = optx.minimise(rosenbrock, solver, x0, args=None, max_steps=5000)

# --- Detailed printing ---
print("\n=== Optimistix Solution ===")
print(f"Converged?         {sol.result == optx.RESULTS.successful}")
print(f"Result code:       {sol.result}")
print(f"Argmin (y):        {sol.state.y_eval}")  # <-- fixed
print(f"Final loss value:  {sol.value}")
print(f"Aux output:        {sol.aux}")

print("\n--- Stats ---")
for k, v in sol.stats.items():
    print(f"  {k}: {v}")

state = sol.state
print("\n--- Final solver state ---")
print(f"First step?        {state.first_step}")
print(f"y_eval:            {state.y_eval}")
print(f"Num accepted steps {state.num_accepted_steps}")
print("\nBacktracking line search state:")
print(f"  Last step size:  {state.search_state.step_size}")

print("\nFunction info:")
print(f"  f:               {state.f_info.f}")
print(f"  grad:            {state.f_info.grad}")
print(f"  Hessian inverse: \n{state.f_info.hessian_inv.pytree}")

print("\nDescent info:")
print(f"  Newton step:     {state.descent_state.newton}")
print(f"  Descent result:  {state.descent_state.result}")
