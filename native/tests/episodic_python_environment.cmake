if(NOT SDK_ENVIRONMENT)
    set(SDK_ENVIRONMENT "${PROJECT_ROOT}/.venv")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env
    "UV_PROJECT_ENVIRONMENT=${SDK_ENVIRONMENT}" "${UV_EXECUTABLE}"
    run --no-sync --offline --project "${PROJECT_ROOT}" python -c "import sys; print(sys.executable)"
    WORKING_DIRECTORY "${PROJECT_ROOT}"
    RESULT_VARIABLE result OUTPUT_VARIABLE expected ERROR_VARIABLE errors
    OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT 30)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "No se pudo comprobar el intérprete del SDK: ${errors}")
endif()
if(NOT SELECTED_PYTHON STREQUAL expected)
    message(FATAL_ERROR "El enlace seleccionó ${SELECTED_PYTHON}, pero LibTorch usa ${expected}")
endif()
