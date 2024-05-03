CONFIG ?= config.mk
-include $(CONFIG)
COMMON ?= common.mk
-include $(COMMON)

all : bin

# Phony targets
.PHONY : bin lib all clean

ifeq (,$(filter-out undefined default,$(origin CC)))
  CC = gcc
endif
ifeq (,$(filter-out undefined default,$(origin LINK)))
  LINK = $(CC)
endif
ifeq (,$(filter-out undefined default,$(origin AR)))
  AR = ar
endif
ifeq (,$(filter-out undefined default,$(origin ARFLAGS)))
  ARFLAGS = crD
endif

pkgconf-path = $(if $(wildcard $(1)/$(2).pc),$(1)/$(2).pc,$(2))

# Library dependencies
# Note: PETSC_ARCH can be undefined or empty for installations which do not use
#       PETSC_ARCH - for example when using PETSc installed through Spack.
ifneq ($(wildcard ../petsc/lib/libpetsc.*),)
  PETSC_DIR ?= ../petsc
endif
petsc.pc := $(call pkgconf-path,$(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig,petsc)

pkgconf   = $(shell pkg-config $1 | $(SED) -e 's/^"//g' -e 's/"$$//g')

# Error checking flags
PEDANTIC      ?=
PEDANTICFLAGS ?= -Werror -pedantic

# Options from library dependencies
CC       = $(call pkgconf, --variable=ccompiler $(petsc.pc))
CFLAGS   = -std=c99 \
  $(filter-out -fvisibility=hidden, $(call pkgconf, --variable=cflags_extra $(petsc.pc))) \
  $(filter-out -fvisibility=hidden, $(call pkgconf, --cflags-only-other $(petsc.pc))) \
  $(if $(PEDANTIC),$(PEDANTICFLAGS))
CPPFLAGS = $(call pkgconf, --cflags-only-I $(petsc.pc)) \
  $(call pkgconf, --variable=cflags_dep $(petsc.pc)) \
  $(if $(PEDANTIC),$(PEDANTICFLAGS))

LDFLAGS    ?=

AFLAGS  ?= -fsanitize=address
CFLAGS  += $(if $(ASAN),$(AFLAGS))
FFLAGS  += $(if $(ASAN),$(AFLAGS))
LDFLAGS += $(if $(ASAN),$(AFLAGS))

CFLAGS  += $(OPT)

# Library options
BUILDDIR := $(if $(BUILD_ARCH),$(PETSC_ARCH)/,)
INCDIR	 := include
OBJDIR   := $(BUILDDIR)build
BINDIR   := $(BUILDDIR)$(if $(for_install),$(OBJDIR)/bin,bin)
for_install := $(filter install,$(MAKECMDGOALS))
LIBDIR   := $(if $(for_install),$(OBJDIR),$(BUILDDIR)lib)
SRCDIR   := src
DARWIN   := $(filter Darwin,$(shell uname -s))
SO_EXT   := $(if $(DARWIN),dylib,so)

CPPFLAGS += -I./include
LDLIBS    = -lm
PYTHON   ?= python3
SED      ?= sed

# arclength
arclength.pc    := $(LIBDIR)/pkgconfig/arclength.pc
libarclength.so := $(LIBDIR)/libarclength.$(SO_EXT)
libarclength.a  := $(LIBDIR)/libarclength.a
libarclength    := $(if $(STATIC),$(libarclength.a),$(libarclength.so))
arclength_LIB    = -larclength
libarclength.bin := $(SRCDIR)/arch-buckling.c $(SRCDIR)/lee-frame.c
libarclength.c  := $(filter-out $(libarclength.bin),$(sort $(wildcard $(SRCDIR)/*.c)) $(sort $(wildcard $(SRCDIR)/**/*.c)))
libarclength.h = $(sort $(wildcard $(INCDIR)/*.h)) $(sort $(wildcard $(INCDIR)/**/*.h))

BINARIES := $(BINDIR)/arch-buckling $(BINDIR)/lee-frame

$(libarclength.so) : LDFLAGS += $(if $(DARWIN), -install_name @rpath/$(notdir $(libarclength.so)))

# Collect list of libraries and paths for use in linking and pkg-config
RPATH_FLAG   := $(call pkgconf, --variable=ldflag_rpath $(petsc.pc))
PKG.pc       := $(petsc.pc)
PKG_LDLIBS    = $(call pkgconf, --libs-only-l --libs-only-other $(PKG.pc))
PKG_L         = $(call pkgconf, --libs-only-L $(PKG.pc))
PKG_LDFLAGS   = $(PKG_L) $(patsubst -L%,$(RPATH_FLAG)%,$(PKG_L))
arclength_LDFLAGS = -L$(abspath $(LIBDIR)) $(if $(STATIC),,$(RPATH_FLAG)$(if $(for_install),"$(libdir)",$(abspath $(LIBDIR))))

_pkg_ldflags = $(filter -L%,$(PKG_LDFLAGS))
_pkg_ldlibs  = $(filter-out -L%,$(PKG_LDLIBS))
$(libarclength) : LDFLAGS += $(_pkg_ldflags) $(_pkg_ldflags:-L%=$(RPATH_FLAG)%)
$(libarclength) : LDLIBS  += $(_pkg_ldlibs)


.SUFFIXES:
.SECONDEXPANSION:
%/.DIR :
	@mkdir -p $(@D)
	@touch $@
.PRECIOUS: %/.DIR

SYMLINK = ln -sf

libarclength.o = $(libarclength.c:%.c=$(OBJDIR)/%.o)
$(libarclength.so) : $(libarclength.o) | $$(@D)/.DIR
	$(call quiet,LINK) $(LDFLAGS) -shared -o $@ $^ $(LDLIBS)

$(libarclength.a)  : $(libarclength.o) | $$(@D)/.DIR
	$(call quiet,AR) $(ARFLAGS) $@ $^


$(OBJDIR)/%.o : $(CURDIR)/%.c | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $(abspath $<)

$(OBJDIR)/% : %.c $(libarclength) | $$(@D)/.DIR
	$(call quiet,LINK) $(CPPFLAGS) $(CFLAGS) -o $@ $(abspath $<) $(LDFLAGS) $(arclength_LIB) $(LDLIBS)

$(BINDIR)/lee-frame : $(OBJDIR)/src/lee-frame | $$(@D)/.DIR
	$(call quiet,SYMLINK) $(abspath $<) $@

$(BINDIR)/arch-buckling : $(OBJDIR)/src/arch-buckling | $$(@D)/.DIR
	$(call quiet,SYMLINK) $(abspath $<) $@

$(arclength.pc)   : pkgconfig-prefix = $(abspath .)
$(OBJDIR)/arclength.pc : pkgconfig-prefix = $(prefix)
.INTERMEDIATE : $(OBJDIR)/arclength.pc
%/arclength.pc    : arclength.pc.template | $$(@D)/.DIR
	@$(SED) \
	    -e "s:%prefix%:$(pkgconfig-prefix):" \
	    -e "s:%libs_private%:$(pkgconfig-libs-private):" $< > $@

bin : $(BINARIES) $(libarclength) $(OBJ)
lib : $(libarclength)

$(BINARIES) : $(libarclength)
$(BINARIES) : LDFLAGS += $(arclength_LDFLAGS) $(PKG_LDFLAGS)
$(BINARIES) : LDLIBS  += $(PKG_LDLIBS)

clean :
	rm -rf $(BINDIR) $(OBJDIR) $(LIBDIR)

info:
	$(info -----------------------------------------)
	$(info )
	$(info Dependencies:)
	$(info PETSC_DIR      = $(PETSC_DIR))
	$(info PETSC_ARCH     = $(PETSC_ARCH))
	$(info )
	$(info -----------------------------------------)
	$(info )
	$(info Build Options:)
	$(info CC             = $(CC))
	$(info CFLAGS         = $(CFLAGS))
	$(info CPPFLAGS       = $(CPPFLAGS))
	$(info LDFLAGS        = $(LDFLAGS))
	$(info LDLIBS         = $(LDLIBS))
	$(info AR             = $(AR))
	$(info ARFLAGS        = $(ARFLAGS))
	$(info OPT            = $(OPT))
	$(info VERBOSE        = $(or $(V),(empty)) [verbose=$(if $(V),on,off)])
	$(info )
	$(info -----------------------------------------)
	$(info )
	@true


# Dependencies
# Include *.d deps when not -B = --always-make: useful if the paths are wonky in a container
-include $(if $(filter B,$(MAKEFLAGS)),,$(libarclength.c:%.c=$(OBJDIR)/%.d) $(libarclength.bin:%.c=$(OBJDIR)/%.d))
