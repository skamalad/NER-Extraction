cmake_minimum_required(VERSION 3.10)
project(libner)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Create shared library
add_library(ner SHARED src/ner.c)

# Set visibility
if(APPLE)
    target_compile_options(ner PRIVATE "-fvisibility=hidden")
    target_compile_definitions(ner PRIVATE "EXPORT=__attribute__((visibility(\"default\")))")
elseif(WIN32)
    target_compile_definitions(ner PRIVATE "EXPORT=__declspec(dllexport)")
else()
    target_compile_options(ner PRIVATE "-fvisibility=hidden")
    target_compile_definitions(ner PRIVATE "EXPORT=__attribute__((visibility(\"default\")))")
endif()

# Set output name
set_target_properties(ner PROPERTIES 
    PREFIX "lib"
    OUTPUT_NAME "ner")
