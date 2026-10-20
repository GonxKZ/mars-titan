module

public import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
Propiedades de componentes escalares sobre los reales.
No se verifica PyTorch, el redondeo, la identificación de ruido ni la contracción del MAC.
-/

@[expose] public section

namespace MarsTitan.NeuralPlasticity

noncomputable section

def gain (floor signal : ℝ) : ℝ := floor + (1 - floor) * signal

theorem gain_range (floor signal : ℝ)
    (hf : 0 ≤ floor ∧ floor ≤ 1) (hs : 0 ≤ signal ∧ signal ≤ 1) :
    floor ≤ gain floor signal ∧ gain floor signal ≤ 1 := by
  dsimp [gain]
  constructor <;> nlinarith

theorem eligibility_range (decay old activity : ℝ)
    (hd : 0 ≤ decay ∧ decay ≤ 1) (ho : 0 ≤ old ∧ old ≤ 1)
    (ha : 0 ≤ activity ∧ activity ≤ 1) :
    0 ≤ decay * old + (1 - decay) * activity ∧
    decay * old + (1 - decay) * activity ≤ 1 := by
  constructor <;> nlinarith

theorem capture_signal_range (tag coherence : ℝ)
    (ht : 0 ≤ tag ∧ tag ≤ 1) (hc : 0 ≤ coherence ∧ coherence ≤ 1) :
    0 ≤ tag * coherence ∧ tag * coherence ≤ 1 := by
  constructor <;> nlinarith

def exchange (x y coupling : ℝ) : ℝ × ℝ :=
  (x + coupling * (y - x), y + coupling * (x - y) / 2)

def energy (x y : ℝ) : ℝ := x^2 + 2 * y^2

theorem exchange_energy_identity (x y coupling : ℝ) :
    let next := exchange x y coupling
    energy next.1 next.2 - energy x y =
      -(2 * coupling - 3 * coupling^2 / 2) * (x - y)^2 := by
  dsimp [exchange, energy]
  ring

theorem exchange_energy_nonincrease (x y coupling : ℝ)
    (hc : 0 ≤ coupling ∧ coupling ≤ 4 / 3) :
    let next := exchange x y coupling
    energy next.1 next.2 ≤ energy x y := by
  have coefficient : 0 ≤ 2 * coupling - 3 * coupling^2 / 2 := by nlinarith
  have decrement := mul_nonneg coefficient (sq_nonneg (x - y))
  have identity := exchange_energy_identity x y coupling
  dsimp only at identity ⊢
  nlinarith

#print axioms gain_range
#print axioms eligibility_range
#print axioms capture_signal_range
#print axioms exchange_energy_identity
#print axioms exchange_energy_nonincrease

end
end MarsTitan.NeuralPlasticity
