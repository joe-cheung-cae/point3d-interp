# CMake generated Testfile for 
# Source directory: /home/joe/repo/point3d_interp/tests
# Build directory: /home/joe/repo/point3d_interp/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(unit_tests "/home/joe/repo/point3d_interp/build/tests/unit_tests")
set_tests_properties(unit_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/joe/repo/point3d_interp/tests/CMakeLists.txt;41;add_test;/home/joe/repo/point3d_interp/tests/CMakeLists.txt;0;")
subdirs("../_deps/googletest-build")
