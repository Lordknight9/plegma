# SPDX-License-Identifier: GPL-3.0-or-later
# Πλέγμα (Plegma) - quad / triangle remesher that keeps each part of a model as its own island.

from . import props, operators, ui

_modules = (props, operators, ui)


def register():
    for m in _modules:
        m.register()


def unregister():
    for m in reversed(_modules):
        m.unregister()
