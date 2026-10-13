# Órdenes habituales de MARS-TITAN. Cada objetivo documenta qué ejecuta y qué necesita.
# GNU Make viene con las herramientas de compilación de Linux, así que no añade dependencias.
# CONTRIBUTING.md explica cuándo usar cada uno.

UV ?= uv
RUN ?= $(UV) run --locked
THREADS ?= 2
NATIVE_JOBS ?= 2
BUILD := $(CURDIR)/build/native
DOCS_OUT ?= $(CURDIR)/build/docs

# Etiquetas de CTest que test-native no ejecuta: las pruebas que aplican pasos de optimizador
# (solo fuera del bloqueo de aprendizaje, tests/README.md) y las que necesitan CUDA.
CTEST_EXCLUDED := ^(optimizer-steps|cuda)$$

# Artefactos nativos que leen las pruebas, con las rutas que deja `make native-build`. Solo
# se declaran los que existen: una ruta declarada que no existe hace fallar la prueba.
native_var = $(if $(wildcard $(2)),$(1)=$(firstword $(wildcard $(2))))
NATIVE_ENV := $(call native_var,MARS_TITAN_EPISODIC_NATIVE,$(BUILD)/native-candidate-cuda/_episodic_native*.so) \
	$(call native_var,MARS_TITAN_NATIVE_LIBRARY,$(BUILD)/native-release/libmars_titan_simulation.so) \
	$(call native_var,MARS_TITAN_SIM_EXECUTABLE,$(BUILD)/native-release/mars-titan-sim) \
	$(call native_var,MARS_TITAN_PPO_EXECUTABLE,$(BUILD)/native-ppo-release/mars-titan-ppo) \
	$(call native_var,MARS_TITAN_KLPO_EXECUTABLE,$(BUILD)/native-ppo-release/mars-titan-klpo)
CPU_ENV := OMP_NUM_THREADS=$(THREADS) MKL_NUM_THREADS=$(THREADS) CUDA_VISIBLE_DEVICES=-1
SUITE_PARTS := data models training simulation memory posttraining evaluation environments \
	episodes calibration cm integrity learning_traces native tooling

.DEFAULT_GOAL := help
.PHONY: help setup check lint test-fast test test-native test-cuda native-build docs

help: ## Lista los objetivos
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-13s %s\n", $$1, $$2}'

setup: ## Crea el entorno bloqueado con todos los extras (uv sync --locked --all-extras)
	$(UV) sync --locked --all-extras

lint: ## Ruff, formato y comprobación del repositorio
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) python scripts/check_repository.py

check: lint test-fast ## Comprobaciones obligatorias antes de abrir una PR

test-fast: ## Pruebas rápidas en CPU, sin las lentas, las nativas ni las de CUDA
	$(CPU_ENV) $(RUN) pytest -q -m "not slow and not native and not cuda"

test: ## Suite CPU completa por carpetas, con los binarios de native-build si existen
	@set -e; for part in $(SUITE_PARTS); do \
		echo "== tests/$$part"; \
		$(CPU_ENV) $(NATIVE_ENV) $(RUN) pytest -q -rs tests/$$part; \
	done; $(CPU_ENV) $(RUN) pytest -q -rs tests/test_suite_support.py

test-native: ## Pruebas que usan los binarios de native-build y su CTest, sin CUDA
	$(CPU_ENV) $(NATIVE_ENV) $(RUN) pytest -q -rs -m "native and not cuda"
	@set -e; for preset in native-release native-ppo-release native-candidate-cuda; do \
		$(CPU_ENV) ctest --test-dir $(BUILD)/$$preset --output-on-failure -j 1 \
			-LE '$(CTEST_EXCLUDED)'; \
	done

test-cuda: ## Pruebas marcadas cuda en cuda:0. Reserva la GPU para esta única carga
	CUBLAS_WORKSPACE_CONFIG=:4096:8 MARS_TITAN_REQUIRE_CUDA=1 $(NATIVE_ENV) \
		$(RUN) pytest -q -rs -m cuda

native-build: ## Compila native-release, native-ppo-release y native-candidate-cuda
	cd native && cmake --preset native-release -DMARS_TITAN_BUILD_EPISODIC_PYTHON=ON
	cd native && cmake --build --preset native-release -j $(NATIVE_JOBS)
	cd native && cmake --preset native-ppo-release
	cd native && cmake --build --preset native-ppo-release -j $(NATIVE_JOBS)
	cd native && cmake --preset native-candidate-cuda -DMARS_TITAN_BUILD_EPISODIC_PYTHON=ON \
		-DMARS_TITAN_BUILD_SIMULATION=ON
	cd native && cmake --build --preset native-candidate-cuda -j $(NATIVE_JOBS)

docs: ## Enlaces, órdenes documentadas, web del observatorio y PDF de la memoria
	$(RUN) python scripts/check_repository.py
	node --test site/tests/*.test.mjs
	mkdir -p $(DOCS_OUT)
	cd thesis && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(DOCS_OUT) main.tex
