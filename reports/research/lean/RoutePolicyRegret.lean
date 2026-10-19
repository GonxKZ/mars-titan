module

public import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith

@[expose] public section

namespace MarsTitan.RoutePolicy

/-- Un máximo de utilidades estimadas pierde como mucho dos veces el error uniforme.
La hipótesis de error se aporta desde fuera. Este lema no la demuestra para ridge.
-/
theorem regret_against_any_route {Action : Type*}
    (utility estimate : Action → ℝ) (chosen alternative : Action) (delta : ℝ)
    (uniform_error : ∀ action, |estimate action - utility action| ≤ delta)
    (chosen_maximizes : ∀ action, estimate action ≤ estimate chosen) :
    utility alternative - utility chosen ≤ 2 * delta := by
  have alternative_error := (abs_le.mp (uniform_error alternative)).1
  have chosen_error := (abs_le.mp (uniform_error chosen)).2
  have selection := chosen_maximizes alternative
  linarith

#print axioms regret_against_any_route

end MarsTitan.RoutePolicy
