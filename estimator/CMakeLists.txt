find_package(Ceres REQUIRED)

include_directories(
${EIGEN3_INCLUDE_DIR}
${CERES_INCLUDE_DIRS}
${CMAKE_CURRENT_SOURCE_DIR}
)

ADD_EXECUTABLE(vio_estimator
    main.cpp
    estimator.cpp
    data_loader.cpp
    ${PROJECT_SOURCE_DIR}/src/utilities.cpp
    ${PROJECT_SOURCE_DIR}/src/param.cpp
    ${PROJECT_SOURCE_DIR}/src/imu.cpp
)

TARGET_LINK_LIBRARIES(vio_estimator ${CERES_LIBRARIES})
