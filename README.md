Overview
========

![build status](https://github.com/ydewit/lean-lldb/actions/workflows/tests.yml/badge.svg)

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

`lean_lldb` is an LLDB extension for debugging Lean programs.

It can be useful for debugging and troubleshooting stuck threads and crashes in Lean programs. Unlike most popular languages, Lean does not have *yet* (note that this is part of the roadmap) a debugger and without this extension we are stuck inspecting the lower-level Lean runtime in C.

When analyzing the state of a Lean process, normally you would only have
access to Lean "runtime-level" information: most variables would be of type
`lean_object*`, and stack traces would only contain Lean internal calls and
calls to library functions. Unless you are a Lean developer troubleshooting
the a Lean progream, that is typically not very useful. This extension,
however, allows you to view a more compact and accessible representation of the same runtime information about the execution of
a program. At this point the only thing this extension provides is printing the values of variables, but it could also list the source code, display Lean
stack traces, etc.

While Lean already provides a similar extension for gdb [out of the box](
https://github.com/leanprover/lean4/blob/38288ae07a24f469a85fd10e93cbbb130f0e9f6c/src/bin/lean-gdb.py),
LLDB might be the debugger of choice on some operating systems, e.g.
on Mac OS.

This extension requires Lean programs to be built with debugging symbols enabled (see `buildType := .debug` in Lake). For now, a debugging build of the Lean runtime is not required, but that may change when we provide the ability to list Lean source code and Lean stack traces.


Features
========

`lean_lldb` targets CPython 3.5+ and supports the following features:

* pretty-priting of runtime values:
  * `lean_ctor_object`
  * `lean_closure_object`
  * `lean_array_object`
  * `lean_scalararray_object`
  * `lean_string_object`
  * `lean_mpz_object`
  * `lean_thunk_object`
  * `lean_task_object`
  * `lean_ref_object`
  * `lean_external_object`


Some interesting ideas to consider:
* printing of local variables
* printing of Lean-level stack traces
* listing the Lean source code
* walking up and down the Lean call stack

**NOTE**: Although it may be interesting to push this project further and implement these additional features, a better use of time would be to look into generating DWARF symbols for a future, first-class debugger for Lean.

Installation
============

If your version of LLDB is linked against system libpython, it's recommended
that you install the extension to the user site packages directory and allow
it to be loaded automatically on start of a new LLDB session:

```shell
$ python -m pip install --user lean-lldb
$ echo "command script import lean_lldb" >> ~/.lldbinit
$ chmod +x ~/.lldbinit
```

Alternatively, you can install the extension to some other location and tell LLDB
to load it from there, e.g. ~/.lldb:

```shell
$ mkdir -p ~/.lldb/lean_lldb
$ python -m pip install --target ~/.lldb/lean_lldb lean-lldb
$ echo "command script import ~/.lldb/lean_lldb/lean_lldb.py" >> ~/.lldbinit
$ chmod +x ~/.lldbinit
```

MacOS
-----
LLDB bundled with MacOS is linked with the system version of CPython which may not even
be in your PATH. To locate the right version of the interpreter, use:
```shell
$ lldb --print-script-interpreter-info
```
The output of the command above is a JSON with the following structure:
```
{
  "executable":"/Library/.../Python3.framework/Versions/3.9/bin/python3",
  "language":"python",
  "lldb-pythonpath":"/Library/.../LLDB.framework/Resources/Python",
  "prefix":"/Library/.../Python3.framework/Versions/3.9"
}
```
Where the value for "executable" is the CPython version that should be used to install
`lean_lldb` for LLDB to be able to successfully import the script:
```shell
$(lldb --print-script-interpreter-info | jq -r .executable) -m pip install lean_lldb
```

Usage
=====

Start a new LLDB session:

```shell
$ lldb /path/to/lean/program
```

or attach to an existing Lean process:

```shell
$ lldb /path/to/lean/program -p $PID
```

If you've followed the installation steps, the extension will now be automatically
loaded on start of a new LLDB session:

Pretty-printing
---------------

All known `lean_object`'s (i.e. runtime types) are automatically pretty-printed
when encountered, as if you tried to get a `repr()` of something in Python REPL,
e.g.:

```
(lldb) v -P2
(lean_object *) ctor = 0x0000000146458678 (Ctor#0.{1} objs=3, scalars=False) {
  [0] = 0x0000000000000007 (Box 3)
  [1] = 0x000000000000000d (Box 6)
  [2] = 0x0000000000000013 (Box 9)
}
```

Potential issues and how to solve them
======================================

CPython 2.7.x
-------------

CPython 2.7.x is not supported. There are currently no plans to support it in the future.

Missing debugging symbols
-------------------------

Debugging symbols are required only for this extension to work. You can check if they are available as follows:

```shell
$ lldb /usr/bin/python
$ (lldb) type lookup lean_ctor_object
```

If debugging symbols are not available, you'll see something like:

```shell
no type was found matching 'lean_ctor_object'
```

Nix Hardening
-------------

Nix's default hardening and optimization features, while beneficial for security and performance, conflict with lean-lldb's ability to inspect Lean4 variables. This automatic behavior in Nix builds prevents lean-lldb from functioning as intended. To resolve this issue, developers need to manually disable these optimizations by setting `NIX_HARDENING_ENABLE = ""` in their environment or Nix configuration, allowing lean-lldb to effectively view Lean4 runtime objects.

Development
===========

Running tests
-------------

Tests currently require `make` and `docker` to be installed.

To run the tests against the *latest* released Lean version, do:

```
$ make test
```

To run the tests against a specific Lean (or LLDB) version, do:

```
$ LEAN_VERSION=X.Y LLDB_VERSION=Z make test
```

Supported CPython versions are:
* `4.10`

Supported LLDB versions:
* `16`

Contributions
=============

Contributions are welcome! If you have ideas for improvements or encounter any issues, feel free to open a pull request or issue on the GitHub repository.


Acknowledgements
================

I would like to thank [Roman Podoliaka](https://github.com/malor) for the [cpython-lldb project](https://github.com/malor/cpython-lldb). This project was a source of inspiration and a template for what you see here.


References
==========

- [How to create LLDB type summaries and synthetic children for your custom types](https://melatonin.dev/blog/how-to-create-lldb-type-summaries-and-synthetic-children-for-your-custom-types/)
- [Examples from the LLDB repo](https://github.com/llvm/llvm-project/tree/main/lldb/examples/synthetic)
- [JUCE C++ LLDB formatters](https://melatonin.dev/blog/how-to-create-lldb-type-summaries-and-synthetic-children-for-your-custom-types/)
- [Tips for writing LLDB pretty printers](https://offlinemark.com/tips-for-writing-lldb-pretty-printers/)
- [Rust LLDB formatters](https://github.com/vadimcn/codelldb/blob/master/formatters/rust.py)
- [LLDB - Variable Formatting](https://lldb.llvm.org/varformats.html)
- [LLDB - Python Reference](https://lldb.llvm.org/use/python-reference.html)
