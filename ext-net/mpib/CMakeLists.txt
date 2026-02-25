set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugin.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mpib_common.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mpib_init.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mpib_connect.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mpib_p2p.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mpib_gdr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mpib_reg.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mpib_param.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mpib_socket.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mpib_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ibvwrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ibvsymbols.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mlx5dvwrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mlx5dvsymbols.cc
)

# Create shared library
add_library(nccl-net-mpib SHARED ${SRC_FILES})

# Set include directories
target_include_directories(nccl-net-mpib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/nccl
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(nccl-net-mpib PRIVATE NCCL_BUILD_RDMA_CORE=1 NCCL_BUILD_MLX5DV=1)

target_link_libraries(nccl-net-mpib PRIVATE ibverbs rdmacm pthread dl mlx5)

# Set output name to match Makefile
set_target_properties(nccl-net-mpib PROPERTIES
    OUTPUT_NAME "nccl-net-mpib"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/unit/plugins
)
