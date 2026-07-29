cmake_minimum_required(VERSION 3.16)
project(biotransport_package_consumer LANGUAGES CXX)

find_package(biotransport CONFIG REQUIRED)

add_executable(biotransport_package_consumer main.cpp)
target_link_libraries(biotransport_package_consumer PRIVATE biotransport::biotransport)

enable_testing()
add_test(NAME installed_package_smoke COMMAND biotransport_package_consumer)
