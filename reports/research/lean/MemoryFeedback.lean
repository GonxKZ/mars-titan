module

public import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith

/-!
Contraejemplo de realimentación para una escritura escalar con momentum y recorte.
El valor v = 21w se mantiene fijo al derivar parcialmente la pérdida asociativa.
Este archivo verifica la recurrencia resultante, no el código PyTorch ni el MAC completo.
-/

@[expose] public section

namespace MarsTitan.MemoryFeedback

noncomputable section

abbrev State := ℝ × ℝ

def clip (g : ℝ) : ℝ := max (-1) (min 1 g)

def step (z : State) : State :=
  let gradient := 2 * (z.1 - 21 * z.1)
  let momentum := z.2 / 10 - clip gradient / 20
  (z.1 / 2 + momentum, momentum)

def positive : State := (1 / 9, 1 / 18)
def negative : State := (-1 / 9, -1 / 18)

def distance (a b : State) : ℝ := |a.1 - b.1| + |a.2 - b.2|

theorem positive_fixed : step positive = positive := by
  norm_num [step, positive, clip]

theorem negative_fixed : step negative = negative := by
  norm_num [step, negative, clip]

theorem fixed_points_distance : distance positive negative = 1 / 3 := by
  norm_num [distance, positive, negative]

theorem gates_inside_pt1 :
    (1 / 500 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (0 : ℝ) ≤ 1 / 10 ∧ (1 / 10 : ℝ) ≤ 3 / 10 ∧
    (0 : ℝ) ≤ 1 / 20 ∧ (1 / 20 : ℝ) ≤ 1 / 10 := by
  norm_num

theorem positive_branch (w s : ℝ) (hw : (1 : ℝ) / 40 < w) :
    step (w, s) = (w / 2 + s / 10 + 1 / 20, s / 10 + 1 / 20) := by
  have hg : 2 * (w - 21 * w) ≤ (-1 : ℝ) := by linarith
  have hg' : 2 * (w - 21 * w) ≤ (1 : ℝ) := by linarith
  simp only [step, clip, min_eq_right hg', max_eq_left hg]
  apply Prod.ext
  · dsimp
    linarith
  · dsimp
    linarith

theorem negative_branch (w s : ℝ) (hw : w < -(1 : ℝ) / 40) :
    step (w, s) = (w / 2 + s / 10 - 1 / 20, s / 10 - 1 / 20) := by
  have hg : (1 : ℝ) ≤ 2 * (w - 21 * w) := by linarith
  simp only [step, clip, min_eq_left hg, max_eq_right (by norm_num : (-1 : ℝ) ≤ 1)]
  apply Prod.ext
  · dsimp
    linarith
  · rfl

theorem affine_matrix_frobenius_squared :
    (1 / 2 : ℝ)^2 + (1 / 10 : ℝ)^2 + (0 : ℝ)^2 + (1 / 10 : ℝ)^2 = 27 / 100 ∧
    (27 / 100 : ℝ) < 1 := by
  norm_num

theorem not_globally_contractive :
    ¬ ∃ ρ : ℝ, ρ < 1 ∧ ∀ a b : State, distance (step a) (step b) ≤ ρ * distance a b := by
  rintro ⟨ρ, hρ, h⟩
  have pair := h positive negative
  rw [positive_fixed, negative_fixed, fixed_points_distance] at pair
  linarith

#print axioms positive_fixed
#print axioms negative_fixed
#print axioms fixed_points_distance
#print axioms gates_inside_pt1
#print axioms positive_branch
#print axioms negative_branch
#print axioms affine_matrix_frobenius_squared
#print axioms not_globally_contractive

end
end MarsTitan.MemoryFeedback
