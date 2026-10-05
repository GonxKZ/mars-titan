file(MAKE_DIRECTORY "${WORK}")
set(arguments --device cpu --steps 192 --capacity 16 --regime-length 48 --delay 9 --invalid-every 17)
execute_process(COMMAND "${PROGRAM}" ${arguments} --output "${WORK}/full.json"
    RESULT_VARIABLE result ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "La CLI no ejecuta el escenario: ${error}")
endif()
execute_process(COMMAND "${PROGRAM}" ${arguments} --stop-after 101
    --checkpoint "${WORK}/checkpoint.json" --output "${WORK}/partial.json"
    RESULT_VARIABLE result ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "La CLI no publica un checkpoint: ${error}")
endif()
execute_process(COMMAND "${PROGRAM}" --device cpu --resume "${WORK}/checkpoint.json"
    --output "${WORK}/resumed.json" RESULT_VARIABLE result ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "La CLI no recupera el checkpoint: ${error}")
endif()
file(READ "${WORK}/full.json" full)
file(READ "${WORK}/resumed.json" resumed)
string(JSON full REMOVE "${full}" measurements)
string(JSON resumed REMOVE "${resumed}" measurements)
if(NOT full STREQUAL resumed)
    message(FATAL_ERROR "La CLI recuperada cambia el resultado determinista")
endif()
file(WRITE "${WORK}/broken.json" "{\"version\":1,")
foreach(bad IN ITEMS "--unknown" "--capacity;0" "--noise;nan" "--device;cuda:0"
                     "--resume;${WORK}/broken.json" "--stop-after;10"
                     "--output;${WORK}/checkpoint.json;--checkpoint;${WORK}/checkpoint.json"
                     "--resume;${WORK}/checkpoint.json;--steps;192")
    execute_process(COMMAND "${PROGRAM}" --device cpu ${bad}
        RESULT_VARIABLE result OUTPUT_QUIET ERROR_QUIET)
    if(result EQUAL 0)
        message(FATAL_ERROR "La CLI acepta argumentos inválidos: ${bad}")
    endif()
endforeach()
