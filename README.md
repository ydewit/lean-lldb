# lean-lldb

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

`lean-lldb` is a set of custom LLDB formatters for Lean4 runtime objects. These formatters are designed to make debugging Lean4 programs more intuitive by providing meaningful representations of Lean4 objects in the LLDB debugger.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ydewit/lean-lldb.git
   ```

2. Navigate to the project directory:

   ```bash
   cd lean-lldb
   ```

3. Source the formatters in your LLDB session:

   ```bash
   command script import lean_lldb.py
   ```

   Alternatively, you can add this command to your `~/.lldbinit` file for automatic loading:

   ```bash
   echo "command script import /path/to/lean_lldb.py" >> ~/.lldbinit
   ```

## Usage

Once the formatters are loaded, LLDB will automatically use them when displaying Lean4 runtime objects. You can inspect the values of variables and expressions in your Lean4 programs as usual, but with the enhanced, readable output provided by `lean-lldb`.

For example, when you inspect a Lean4 object:

```lldb
(lldb) p myLeanObject
```

The output will now show a more user-friendly representation of `myLeanObject`, making it easier to understand the state of your program.

## Contributing

Contributions are welcome! If you have ideas for improvements or encounter any issues, feel free to open a pull request or issue on the GitHub repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the Lean4 community for their support and contributions.

## References

- [How to create LLDB type summaries and synthetic children for your custom types](https://melatonin.dev/blog/how-to-create-lldb-type-summaries-and-synthetic-children-for-your-custom-types/)
- [Examples from the LLDB repo](https://github.com/llvm/llvm-project/tree/main/lldb/examples/synthetic)
- [JUCE C++ LLDB formatters](https://melatonin.dev/blog/how-to-create-lldb-type-summaries-and-synthetic-children-for-your-custom-types/)
- [Tips for writing LLDB pretty printers](https://offlinemark.com/tips-for-writing-lldb-pretty-printers/)
- [Rust LLDB formatters](https://github.com/vadimcn/codelldb/blob/master/formatters/rust.py)
