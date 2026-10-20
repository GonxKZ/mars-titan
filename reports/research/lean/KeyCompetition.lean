module

public import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

/-!+Centrado de dos coordenadas reales, sin selección ni normalización posterior.
No verifica el código PyTorch, la aritmética de coma flotante o el predictor completo.
-/

@[expose] public section

namespace MarsTitan.KeyCompetition

noncomputable section

def centered (x y : ℝ) : ℝ × ℝ := ((x - y) / 2, (y - x) / 2)

def energy (x y : ℝ) : ℝ := x^2 + y^2

theorem zero_sum (x y : ℝ) : (centered x y).1 + (centered x y).2 = 0 := by
  dsimp [centered]
  ring

theorem idempotent (x y : ℝ) :
    centered (centered x y).1 (centered x y).2 = centered x y := by
  apply Prod.ext <;> dsimp [centered] <;> ring

theorem energy_identity (x y : ℝ) :
    energy (centered x y).1 (centered x y).2 = energy x y - (x + y)^2 / 2 := by
  dsimp [centered, energy]
  ring

theorem energy_nonincrease (x y : ℝ) :
    energy (centered x y).1 (centered x y).2 ≤ energy x y := by
  have h := energy_identity x y
  nlinarith [sq_nonneg (x + y)]

theorem zero_sum_fixed (x y : ℝ) (h : x + y = 0) : centered x y = (x, y) := by
  apply Prod.ext <;> dsimp [centered] <;> linarith

theorem no_uniform_half_contraction :
    ¬ (∀ x y : ℝ, energy (centered x y).1 (centered x y).2 ≤ energy x y / 2) := by
  intro h
  have counterexample := h 1 (-1)
  norm_num [centered, energy] at counterexample

#print axioms zero_sum
#print axioms idempotent
#print axioms energy_identity
#print axioms energy_nonincrease
#print axioms zero_sum_fixed
#print axioms no_uniform_half_contraction

end
end MarsTitan.KeyCompetition
