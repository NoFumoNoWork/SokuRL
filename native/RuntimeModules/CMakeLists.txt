cmake_minimum_required(VERSION 3.20)
project(SokuRLRuntimeModules LANGUAGES C CXX)

if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR "Runtime modules require Win32/x86")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
get_filename_component(MODS "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/SokuMods" ABSOLUTE)

add_library(swrstoys MODULE
    "${MODS}/swrstoys/d3d9.def"
    "${MODS}/swrstoys/d3d9_dummy.cpp"
    "${MODS}/swrstoys/dummy.cpp"
    "${MODS}/swrstoys/SWRSToys.cpp")
set_target_properties(swrstoys PROPERTIES OUTPUT_NAME d3d9)
target_include_directories(swrstoys PRIVATE "${MODS}/include")
target_compile_definitions(swrstoys PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(swrstoys PRIVATE shlwapi user32 advapi32 shell32)

foreach(module IN ITEMS WindowResizer MemoryPatch SkipIntro)
    if(module STREQUAL "SkipIntro")
        set(source "${MODS}/modules/SkipIntro/Soku-SkipIntro/dllmain.cpp")
    else()
        set(source "${MODS}/modules/${module}/${module}.cpp")
    endif()
    add_library(${module} MODULE "${source}")
    target_compile_definitions(${module} PRIVATE _CRT_SECURE_NO_WARNINGS DIRECTINPUT_VERSION=0x0800)
    target_include_directories(${module} PRIVATE "${MODS}/include" "${MODS}/SokuLib/src")
    target_link_libraries(${module} PRIVATE shlwapi user32 advapi32)
endforeach()

add_library(detours STATIC
    "${MODS}/detours/src/creatwth.cpp"
    "${MODS}/detours/src/detours.cpp"
    "${MODS}/detours/src/disasm.cpp"
    "${MODS}/detours/src/disolarm.cpp"
    "${MODS}/detours/src/disolarm64.cpp"
    "${MODS}/detours/src/disolia64.cpp"
    "${MODS}/detours/src/disolx86.cpp"
    "${MODS}/detours/src/disolx64.cpp"
    "${MODS}/detours/src/image.cpp"
    "${MODS}/detours/src/modules.cpp")
target_include_directories(detours PUBLIC "${MODS}/detours/src")
add_library(ReplayDnD MODULE "${MODS}/modules/ReplayDnD/ReplayDnD.cpp")
target_include_directories(ReplayDnD PRIVATE "${MODS}/include")
target_compile_definitions(ReplayDnD PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(ReplayDnD PRIVATE detours shlwapi shell32 user32 advapi32)
