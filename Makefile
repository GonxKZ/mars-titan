# Órdenes habituales de MARS-TITAN. Cada objetivo documenta qué ejecuta y qué necesita.
# GNU Make viene con las herramientas de compilación de Linux, así que no añade dependencias.
# CONTRIBUTING.md explica cuándo usar cada uno.

UV ?= uv
RUN ?= $(UV) run --locked
THREADS ?= 2
# Trabajadores de pytest-xdist en test-fast. Con 0 corre en un solo proceso (#491).
TEST_WORKERS ?= 3
CUDA_TEST_TIMEOUT ?= 600
NATIVE_JOBS ?= 2
BUILD := $(CURDIR)/build/native
DOCS_OUT ?= $(CURDIR)/build/docs

# Etiquetas de CTest que test-native no ejecuta: las pruebas que aplican pasos de optimizador
# (solo fuera del bloqueo de aprendizaje, tests/README.md) y las que necesitan CUDA.
CTEST_EXCLUDED := ^(optimizer-steps|cuda)$$

# Artefactos nativos que leen las pruebas, con las rutas que deja `make native-build`. Solo
# se declaran los que existen: una ruta declarada que no existe hace fallar la prueba.
native_var = $(shell for artifact in "$(BUILD)"/$(2); do \
	if test -f "$$artifact"; then printf '$(1)="%s"' "$$artifact"; break; fi; done)
NATIVE_ENV := $(call native_var,MARS_TITAN_EPISODIC_NATIVE,native-candidate-cuda/_episodic_native*.so) \
	$(call native_var,MARS_TITAN_NATIVE_LIBRARY,native-release/libmars_titan_simulation.so) \
	$(call native_var,MARS_TITAN_SIM_EXECUTABLE,native-release/mars-titan-sim) \
	$(call native_var,MARS_TITAN_PPO_EXECUTABLE,native-ppo-release/mars-titan-ppo) \
	$(call native_var,MARS_TITAN_KLPO_EXECUTABLE,native-ppo-release/mars-titan-klpo)
CPU_ENV := OMP_NUM_THREADS=$(THREADS) MKL_NUM_THREADS=$(THREADS) CUDA_VISIBLE_DEVICES=-1
CUDA_ENV := OMP_NUM_THREADS=$(THREADS) MKL_NUM_THREADS=$(THREADS) CUDA_VISIBLE_DEVICES=0 \
	CUBLAS_WORKSPACE_CONFIG=:4096:8 MARS_TITAN_REQUIRE_CUDA=1 $(NATIVE_ENV)
CUDA_CHECK_ENV := MARS_TITAN_GRU_CHECK_DEVICE=cuda:0 MARS_TITAN_M3_CHECK_DEVICE=cuda:0 \
	MARS_TITAN_REGENERATION_CHECK_DEVICE=cuda:0 MARS_TITAN_MARS_RUN_CHECK_DEVICE=cuda:0 \
	MARS_TITAN_TITANS_ADAPTERS_CHECK_DEVICE=cuda:0 MARS_TITAN_CM_CONTROL_CHECK_DEVICE=cuda:0 \
	MARS_TITAN_CANDIDATE_RUN_CHECK_DEVICE=cuda:0 MARS_TITAN_READOUT_ADAPTERS_CHECK_DEVICE=cuda:0 \
	MARS_TITAN_CANDIDATE_ADAPTERS_CHECK_DEVICE=cuda:0
SUITE_PARTS := data models training simulation memory posttraining evaluation environments \
	episodes calibration cm integrity learning_traces native tooling hardware
CUDA_CHECKS := $(filter-out tests/models/candidate/cuda_candidate_check.py,\
	$(wildcard tests/*/cuda_*_check.py tests/*/*/cuda_*_check.py))
ifndef CUDA_CHECK_DIR
CUDA_CHECK_DIR := $(CURDIR)/build/checks/cuda-$(shell date -u +%Y%m%dT%H%M%S)
endif

.DEFAULT_GOAL := help
.PHONY: help setup check lint test-fast test test-native test-cuda test-cuda-checks \
	test-cuda-memory native-build docs

help: ## Lista los objetivos
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-13s %s\n", $$1, $$2}'

setup: ## Crea el entorno bloqueado con todos los extras (uv sync --locked --all-extras)
	$(UV) sync --locked --all-extras

lint: ## Ruff, formato y comprobación del repositorio
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) python scripts/check_repository.py

check: lint test-fast ## Comprobaciones obligatorias antes de abrir una PR

test-fast: ## Pruebas rápidas en CPU con pytest-xdist, sin las lentas, las nativas ni las de CUDA
	$(CPU_ENV) $(RUN) pytest -q -n $(TEST_WORKERS) --dist loadfile -m "not slow and not native and not cuda"

test: ## Suite CPU completa por carpetas, con los binarios de native-build si existen
	@set -e; for part in $(SUITE_PARTS); do \
		echo "== tests/$$part"; \
		$(CPU_ENV) $(NATIVE_ENV) $(RUN) pytest -q -rs tests/$$part; \
	done; $(CPU_ENV) $(NATIVE_ENV) $(RUN) pytest -q -rs $(wildcard tests/test_*.py tests/*_test.py)

test-native: ## Pruebas que usan los binarios de native-build y su CTest, sin CUDA
	$(CPU_ENV) $(NATIVE_ENV) $(RUN) pytest -q -rs -m "native and not cuda"
	@set -e; for preset in native-release native-ppo-release native-candidate-cuda; do \
		$(CPU_ENV) ctest --test-dir $(BUILD)/$$preset --output-on-failure -j 1 \
			-LE '$(CTEST_EXCLUDED)'; \
	done

test-cuda: ## Pruebas cuda en procesos nuevos y secuenciales, con recibo por caso
	$(CUDA_ENV) $(RUN) python scripts/run_cuda_tests.py --output "$(CUDA_CHECK_DIR)/cases" \
		--timeout-seconds $(CUDA_TEST_TIMEOUT)

test-cuda-checks: ## Paridades CUDA explícitas, cada archivo en un proceso, sin optimizador
	@set -e; for check in $(CUDA_CHECKS); do \
		$(CUDA_ENV) $(CUDA_CHECK_ENV) $(RUN) pytest -q -rs $$check; \
	done
	mkdir -p "$(CUDA_CHECK_DIR)"
	$(CUDA_ENV) $(CUDA_CHECK_ENV) $(RUN) python tests/models/candidate/cuda_candidate_check.py \
		--output "$(CUDA_CHECK_DIR)/candidate.json"

test-cuda-memory: ## Memoria y recomputación de la GRU en CUDA con 128, 256 y 512 activos
	mkdir -p "$(CUDA_CHECK_DIR)"
	$(CUDA_ENV) MARS_TITAN_CANDIDATE_MEMORY_CHECK_DEVICE=cuda:0 \
		MARS_TITAN_CANDIDATE_MEMORY_CHECK_ASSETS=128,256,512 \
		MARS_TITAN_CANDIDATE_MEMORY_CHECK_REPORT="$(CUDA_CHECK_DIR)/memory.json" \
		$(RUN) pytest -q -rs tests/training/candidate_memory_check.py

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
