set(KENDAMA_ROBOT_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DKENDAMA_ROBOT_FOLDER="${KENDAMA_ROBOT_FOLDER}")

# create an executable
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CS225A_BINARY_DIR}/kendama)
ADD_EXECUTABLE (simviz_kendama simviz.cpp ${CS225A_COMMON_SOURCE})
ADD_EXECUTABLE (controller_kendama controller.cpp ${CS225A_COMMON_SOURCE})

# and link the library against the executable
TARGET_LINK_LIBRARIES (simviz_kendama ${CS225A_COMMON_LIBRARIES})
TARGET_LINK_LIBRARIES (controller_kendama ${CS225A_COMMON_LIBRARIES})
