# Registro de las pruebas de CTest que aplican pasos de optimizador (#567).
#
# Hay dos barreras. La primera es este módulo: cada prueba con la etiqueta optimizer-steps se
# registra con mars_titan_add_optimizer_step_test, que la lanza a través de
# mars-titan-learning-gate. El lanzador lee la protección de aprendizaje y, si bloquea, termina
# con MARS_TITAN_LEARNING_GATE_SKIP antes de ejecutar la prueba, de modo que CTest la marca como
# saltada aunque se llame sin perfil. mars_titan_require_gated_optimizer_step_tests detiene la
# configuración si una prueba etiquetada no pasa por el lanzador. La segunda barrera es el filtro
# de etiquetas del perfil native-test de CMakePresets.json, que ni siquiera las lista.

# Código de salto del lanzador. Debe coincidir con learning_gate_skip de learning_gate.hpp.
set(MARS_TITAN_LEARNING_GATE_SKIP 77)

# mars_titan_add_optimizer_step_test(<nombre> COMMAND <programa> [argumentos...])
# <programa> puede ser un ejecutable del proyecto o una ruta. Las demás propiedades se añaden
# después con set_property(TEST ... APPEND), sin sustituir LABELS ni SKIP_RETURN_CODE.
function(mars_titan_add_optimizer_step_test name)
    cmake_parse_arguments(PARSE_ARGV 1 step "" "" "COMMAND")
    if(step_UNPARSED_ARGUMENTS OR NOT step_COMMAND)
        message(FATAL_ERROR "Uso: mars_titan_add_optimizer_step_test(<nombre> COMMAND <programa> ...)")
    endif()
    if(NOT TARGET mars-titan-learning-gate)
        message(FATAL_ERROR "La prueba ${name} da pasos de optimizador y necesita mars-titan-learning-gate")
    endif()
    list(POP_FRONT step_COMMAND program)
    if(TARGET ${program})
        set(program "$<TARGET_FILE:${program}>")
    endif()
    add_test(NAME ${name} COMMAND mars-titan-learning-gate ${program} ${step_COMMAND})
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE ${MARS_TITAN_LEARNING_GATE_SKIP})
    set_property(TEST ${name} APPEND PROPERTY LABELS optimizer-steps)
    set_property(GLOBAL APPEND PROPERTY MARS_TITAN_GATED_TESTS ${name})
endfunction()

# Se llama una vez registradas todas las pruebas del directorio. Falla si una prueba con la
# etiqueta no se registró con el lanzador, o si una registrada con él perdió la etiqueta o el
# código de salto al cambiar después sus propiedades.
function(mars_titan_require_gated_optimizer_step_tests)
    get_property(gated GLOBAL PROPERTY MARS_TITAN_GATED_TESTS)
    get_directory_property(registered TESTS)
    foreach(test IN LISTS registered)
        get_test_property(${test} LABELS labels)
        if("optimizer-steps" IN_LIST labels AND NOT test IN_LIST gated)
            message(FATAL_ERROR "La prueba ${test} lleva la etiqueta optimizer-steps sin pasar por "
                "mars-titan-learning-gate. Regístrala con mars_titan_add_optimizer_step_test")
        endif()
    endforeach()
    foreach(test IN LISTS gated)
        get_test_property(${test} LABELS labels)
        get_test_property(${test} SKIP_RETURN_CODE skip)
        if(NOT "optimizer-steps" IN_LIST labels OR NOT skip STREQUAL MARS_TITAN_LEARNING_GATE_SKIP)
            message(FATAL_ERROR "La prueba ${test} pasa por mars-titan-learning-gate, pero perdió la "
                "etiqueta optimizer-steps o el código de salto ${MARS_TITAN_LEARNING_GATE_SKIP}")
        endif()
    endforeach()
endfunction()
