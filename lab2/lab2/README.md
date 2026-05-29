# Lab 2 — Low-Rank Sensitivities and Gradient Computation

This repository contains my implementation for **Lab 2** in  
*Low-Rank and Tensor Network Methods for Differential Equations*.

The goal of the lab is to:
- Solve the variable‑coefficient advection equation using a dense reference solver.
- Compute and analyze the sensitivity matrix \(X(t)\).
- Implement a fixed‑rank step‑and‑truncate RK4 solver for \(X \approx USV^\top\).
- Compare low‑rank and dense solutions in accuracy and cost.
- Compute the gradient of the cost functional using low‑rank factors.
- Use the cheap gradient inside a steepest‑descent loop with Armijo line search.

---