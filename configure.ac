#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT(motionlights, 0.1, phil@philkingston.com)
AM_INIT_AUTOMAKE(motionlights, 0.1)
AC_CONFIG_SRCDIR([src/lights.cpp])
#AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CXX

# Checks for libraries.
PKG_CHECK_MODULES([opencv], [opencv >= 2.4.9])
AC_SUBST([opencv_CFLAGS])
AC_SUBST([opencv_LIBS])

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL

# Checks for library functions.

AC_OUTPUT(Makefile src/Makefile)
