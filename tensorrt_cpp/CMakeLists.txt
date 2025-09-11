cmake_minimum_required(VERSION 3.12)
project(EigenPlaces_TensorRT)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译选项
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# 设置TensorRT路径
set(TENSORRT_ROOT "/home/xihuidong/usr_lib/TensorRT-8.5.3.1" CACHE PATH "TensorRT root directory")

# 检查TensorRT路径是否存在
if(NOT EXISTS ${TENSORRT_ROOT})
    message(FATAL_ERROR "TensorRT not found at ${TENSORRT_ROOT}")
endif()

# 查找必要的包
find_package(PkgConfig REQUIRED)

# 查找CUDA
find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found")
endif()

# 查找OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found")
endif()

# TensorRT库路径
set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)
set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)

# 检查TensorRT库文件
set(TENSORRT_LIBS 
    ${TENSORRT_LIB_DIR}/libnvinfer.so
    ${TENSORRT_LIB_DIR}/libnvonnxparser.so
    ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so
)

foreach(lib ${TENSORRT_LIBS})
    if(NOT EXISTS ${lib})
        message(FATAL_ERROR "TensorRT library not found: ${lib}")
    endif()
endforeach()

# 包含目录
include_directories(
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# 链接目录
link_directories(
    ${TENSORRT_LIB_DIR}
    ${CUDA_LIBRARY_DIRS}
)

# 源文件
set(SOURCES
    main.cpp
    tensorrt_engine.cpp
    image_processor.cpp
)

# 头文件
set(HEADERS
    tensorrt_engine.h
    image_processor.h
)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# 链接库
target_link_libraries(${PROJECT_NAME}
    # TensorRT库
    nvinfer
    nvonnxparser
    nvinfer_plugin
    
    # CUDA库
    ${CUDA_LIBRARIES}
    cudart
    
    # OpenCV库
    ${OpenCV_LIBS}
    
    # 系统库
    pthread
    dl
)

# 编译信息
message(STATUS "=== 编译配置信息 ===")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "TENSORRT_ROOT: ${TENSORRT_ROOT}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

# 安装规则
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

# 设置运行时库路径
set_target_properties(${PROJECT_NAME} PROPERTIES
    INSTALL_RPATH "${TENSORRT_LIB_DIR}:${CUDA_LIBRARY_DIRS}"
    BUILD_WITH_INSTALL_RPATH TRUE
)
