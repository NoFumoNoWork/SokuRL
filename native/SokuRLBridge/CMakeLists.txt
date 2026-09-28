cmake_minimum_required(VERSION 3.20)
project(SokuRLBridge LANGUAGES CXX)

if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR "SokuRLBridge must be configured for Win32/x86")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(SOKUMODS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/SokuMods" ABSOLUTE)
if(NOT EXISTS "${SOKUMODS_ROOT}/SokuLib/CMakeLists.txt")
    message(FATAL_ERROR "SokuLib was not found under ${SOKUMODS_ROOT}")
endif()

add_subdirectory("${SOKUMODS_ROOT}/SokuLib" "${CMAKE_BINARY_DIR}/SokuLib" EXCLUDE_FROM_ALL)

add_library(SokuRLBridge MODULE
    ControlBlock.hpp
    ImageCapture.hpp
    ImageCapture.cpp
    SokuRLBridge.cpp
)
target_link_libraries(SokuRLBridge PRIVATE SokuLib)
target_compile_definitions(SokuRLBridge PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
if(MSVC)
    target_compile_options(SokuRLBridge PRIVATE /W4)
endif()
set_target_properties(SokuRLBridge PROPERTIES PREFIX "" OUTPUT_NAME "SokuRLBridge")
