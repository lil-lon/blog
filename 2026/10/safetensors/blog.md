# Why safetensors is safe

I am writing a Rust implementation of a machine learning model ([asanoha](https://github.com/lil-lon/asanoha)), and I decided to read its weights from safetensors files. This post looks at why safetensors is safe and how it differs from loading weights with the pickle-based `torch.load`.

## 1. A pickle is a program

A pickle file is a sequence of opcodes: one-byte instructions that the unpickler executes in order on a stack of Python objects. `pickletools.py` calls a pickle "a program for a virtual pickle machine."

Two opcodes make this dangerous. One imports a module named in the file and pushes one of its attributes onto the stack. Pickle calls it `GLOBAL`. The other, `REDUCE`, pops an argument tuple and calls the callable below it. From CPython 3.13 `pickle.py`:

```python
def find_class(self, module, name):  # used by GLOBAL
    ...
    __import__(module, level=0)
    ...
    return getattr(sys.modules[module], name)

def load_reduce(self):
    stack = self.stack
    args = stack.pop()
    func = stack[-1]
    stack[-1] = func(*args)
```

These opcodes implement `__reduce__`, the hook through which an object tells pickle how to rebuild it as a callable plus arguments. The file stores only the callable's name, so whoever writes the file chooses what runs on load.

For example, the following script creates a pickle that calls `print` when loaded:

```python
import pickle
import pickletools


class Payload:
    def __reduce__(self):
        return (print, ("called during pickle.loads",))


data = pickle.dumps(Payload())
pickletools.dis(data)
print(pickle.loads(data))
```

Output:

```text
    0: \x80 PROTO      4
    2: \x95 FRAME      55
   11: \x8c SHORT_BINUNICODE 'builtins'
   21: \x94 MEMOIZE    (as 0)
   22: \x8c SHORT_BINUNICODE 'print'
   29: \x94 MEMOIZE    (as 1)
   30: \x93 STACK_GLOBAL
   31: \x94 MEMOIZE    (as 2)
   32: \x8c SHORT_BINUNICODE 'called during pickle.loads'
   60: \x94 MEMOIZE    (as 3)
   61: \x85 TUPLE1
   62: \x94 MEMOIZE    (as 4)
   63: R    REDUCE
   64: \x94 MEMOIZE    (as 5)
   65: .    STOP
highest protocol among opcodes = 4
called during pickle.loads
None
```

The disassembly contains `builtins`, `print`, `STACK_GLOBAL` (the variant of `GLOBAL` in pickle format version 4, the default), and `REDUCE`. `pickle.loads` calls `print` and returns its result, `None`. The loader does not need the `Payload` class. With `os.system` in place of `print`, loading runs a shell command.

## 2. torch.save and torch.load

### 2.1 A `.pt` file is a pickle

`torch.save` writes a zip archive. Its `data.pkl` record is a pickle of the saved object. Tensor bytes live in separate `data/<key>` records. The pickle refers to them by persistent ID, a pickle mechanism that replaces an object with an external reference.

A plain float32 tensor like the one below is rebuilt by calling `torch._utils._rebuild_tensor_v2(storage, storage_offset, size, stride, ...)`, which ends in the C++ operator `Tensor.set_(storage, storage_offset, size, stride)`. Other tensor kinds (sparse, quantized, meta, and some newer dtypes) use other `_rebuild_*` functions, but each is still a call. An honest checkpoint should therefore contain `GLOBAL` and `REDUCE`. To check, the following script saves a dict holding one tensor, opens the `.pt` file as a zip, and disassembles `data.pkl`:

```python
import pickletools
import zipfile

import torch

torch.save({"weight": torch.zeros(2, 2)}, "model.pt")
with zipfile.ZipFile("model.pt") as z:
    name = next(n for n in z.namelist() if n.endswith("/data.pkl"))
    pickletools.dis(z.read(name))
```

Output (torch 2.13.0), abridged to the lines that rebuild the tensor, with annotations added:

```text
   18: c    GLOBAL     'torch._utils _rebuild_tensor_v2'   # push the function
  ...
  114: Q        BINPERSID                                  # storage: tensor bytes in data/0
  115: K        BININT1    0                               # storage_offset = 0
  ...                                                      # size = (2, 2)
  ...                                                      # stride = (2, 1)
  ...
  166: R    REDUCE                                         # call the function
```

Loading even an honest tensor is a function call whose name and arguments are read from the file.

### 2.2 Is `weights_only=True` enough?

The obvious fix is to restrict what the unpickler may do. `torch.load(..., weights_only=True)` does this, and it has been the default since torch 2.6. With it, `torch.load` replaces `pickle` with `torch/_weights_only_unpickler.py`, a separate interpreter for the same opcodes.

#### What it protects against

The interpreter never imports. It resolves `GLOBAL` against an allowlist (`OrderedDict`, dtypes, the `torch._utils._rebuild_*` functions, and similar) and runs `REDUCE` only on allowlisted callables. It also accepts only storage persistent IDs. A file that names an arbitrary callable, like the payload in section 1, is rejected:

```python
import torch

# Payload as defined in section 1
torch.save(Payload(), "payload.pt")
torch.load("payload.pt")
```

Output (torch 2.13.0, abridged):

```text
_pickle.UnpicklingError: Weights only load failed. ...
	WeightsUnpickler error: Unsupported global: GLOBAL print was not an allowed global by default. ...
```

#### What it does not protect against

The format itself leaves these risks open:

1. **The allowlist is unavoidable.** Loading a tensor requires `REDUCE` (section 2.1), so `REDUCE` cannot be banned. The loader can only choose which callables to permit.
2. **Arguments are checked only by each callee.** The unpickler checks which function is called, not its arguments. `storage_offset`, `size`, and `stride` reach `set_`, which runs its own checks. Every allowlisted function must be safe for every input.
3. **There is no schema.** A pickle does not declare its result in advance. Checks run per opcode, during execution.
4. **Every entry point must enforce the check.** `weights_only` replaces the unpickler, but `torch.load` accepts several file formats and dispatches on the input. A branch that skips the restricted unpickler skips the check.
5. **The checker is code.** It can have bugs.
6. **The allowlist grows.** New tensor types need new rebuild functions, and users can add entries with `torch.serialization.add_safe_globals`.
7. **Safety depends on the reader.** The torch version, the `weights_only` argument, and the `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` environment variable decide whether the same file is safe to load.

#### Example: torch 2.5.1 and 2.13.0

Comparing the two versions illustrates risks 2, 4, 5, and 6.

**Entry points (risks 2 and 4).** How `torch.load` dispatches with `weights_only=True`:

| Input | 2.5.1 | 2.13.0 |
|---|---|---|
| Path ending in `.safetensors` | no special case | `safetensors.torch.load_file` |
| TorchScript archive (written by `torch.jit.save`, detected by a `constants.pkl` record) | passed to `torch.jit.load`, flag ignored | rejected |
| Other zip | restricted unpickler | restricted unpickler |
| Not a zip: `.tar` archive (obsolete format) | `_legacy_load`, offsets and strides read without the unpickler | rejected |
| Not a zip: legacy pickle (written with `_use_new_zipfile_serialization=False`) | restricted unpickler | restricted unpickler |

In 2.13.0, a path ending in `.safetensors` skips the unpickler and uses the validation described in section 3.

The 2.5.1 `.tar` path read offsets and strides with `struct.unpack` and passed them to `set_` without going through the unpickler. CVE-2025-32434 ("`torch.load` with `weights_only=True` leads to remote code execution") was fixed in 2.6.0 by commit 8d4b8a9, which rejects the `.tar` format under `weights_only=True`.

**Checker bugs (risk 5).** In 2.5.1 the persistent ID check read:

```python
if type(pid) is not tuple and not type(pid) is not int:
```

`not type(pid) is not int` means `type(pid) is int`. The check rejected ints and accepted every other non-tuple type. 2.13.0 drops the extra `not`. 2.5.1 also let `SETITEM` (the opcode that stores a key-value pair) write into any object. 2.13.0 restricts it to dicts.

**Allowlist growth (risk 6).** From 2.5.1 to 2.13.0 the default allowlist gained `set`, `complex`, the 1 to 7 bit integer dtypes, and a new rebuild function.

## 3. safetensors

### 3.1 The format

A safetensors file has three parts:

1. 8 bytes: the header length `N`, an unsigned 64-bit little-endian integer.
2. `N` bytes: a JSON header. Each key is a tensor name. Each value gives the tensor's `dtype`, `shape`, and `data_offsets`, the byte range `[begin, end)` of its data in the buffer. An optional `__metadata__` key holds a string-to-string map.
3. The rest of the file: a byte buffer holding the tensor data.

The following script saves the same dict as in section 2.1 and prints the header:

```python
import struct

import torch
from safetensors.torch import save_file

save_file({"weight": torch.zeros(2, 2)}, "model.safetensors")
with open("model.safetensors", "rb") as f:
    (n,) = struct.unpack("<Q", f.read(8))
    print(n, f.read(n))
```

Output (safetensors 0.8.0):

```text
64 b'{"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}  '
```

The two trailing spaces pad the header to a multiple of 8 bytes. The 16 bytes of tensor data follow.

### 3.2 How it differs from pickle-based loading

`torch.load` with `weights_only=True` could be called safe by filtering: the format can express dangerous operations, and a checker rejects them. Safety then depends on the checker being correct and present on every path. safetensors could be called safe by construction: the format cannot express a dangerous operation at all. SQL prepared statements make a useful analogy. Escaping user input inside a query string is safe by filtering. Passing it as a prepared statement parameter is safe by construction, because the input can never become part of the query.

In safetensors, the header is data, not a program. Every file is loaded by the same fixed steps (`SafeTensors::read_metadata` in `safetensors/src/tensor.rs`, called by the Python loaders):

1. Read `N` and parse the header into a fixed schema. `dtype` must be one of a closed list of names (`F32`, `BF16`, and so on). `shape` and `data_offsets` must be unsigned integers (`usize`).
2. Validate every entry (section 3.3).
3. For each requested tensor, take bytes `[begin, end)` of the buffer, reinterpret them as `dtype`, and reshape them to `shape`.

The loader's code chooses every function it calls. The file supplies only values, and they are checked before any tensor is created. No field in the format can name a function.

Each risk of safe by filtering from section 2.2 changes as follows:

| Risk in section 2.2 (safe by filtering) | safetensors (safe by construction) |
|---|---|
| 1. The allowlist is unavoidable | The format has no callables, so there is nothing to allowlist. |
| 2. Arguments are checked only by each callee | The loader checks every offset against shape and dtype in one place. There is no stride or storage offset field. |
| 3. There is no schema | The schema is fixed, and the whole header is validated before any tensor is created. |
| 4. Every entry point must enforce the check | `safe_open`, `load_file`, and `load` all parse with `read_metadata`. |
| 5. The checker is code | Still true. The parser is code (section 3.3). |
| 6. The allowlist grows | The dtype list grows, but a new dtype adds a bit width, not a callable. |
| 7. Safety depends on the reader | The loaders have no option that disables validation. |

### 3.3 Attacks on the format

A file without code can still carry hostile numbers. The repository's `attacks/README.md` lists ways to abuse the format. `read_metadata` rejects the malformed headers and tensor ranges among them before any tensor is created. Its checks enforce three properties:

- **The header fits in the file.** Its length must not exceed the file size or 100,000,000 bytes.
- **The tensors tile the buffer exactly.** Each tensor starts where the previous one ends, and the last one ends at the end of the file. There are no gaps, overlaps, or trailing bytes.
- **Each byte range matches its shape and dtype.** Its length must equal the element count times the dtype size, computed with overflow checks.

Together they bound what a malicious file can do. Every tensor lies inside the file, and the tensors together are exactly as large as the buffer. The data is not compressed, so a small file cannot declare large tensors.

The list also names two attacks outside these checks. An attacker with write access to the filesystem could modify the file while it is memory-mapped. The JSON parser or the compiled binary could have a bug. Section 4 covers the second.

### 3.4 What it gives up

The format stores only what its schema describes:

- **Only named dense tensors.** A file is a flat map from names to tensors plus string metadata. It cannot hold arbitrary Python objects, such as an instance of a config class.
- **Contiguous, row-major layout.** `save_file` raises an error on a non-contiguous or sparse tensor.
- **No shared tensors.** `save_file` rejects tensors that share memory. `save_model` keeps one name per shared buffer and records the dropped names in `__metadata__`.
- **Little-endian only.**

Only the first limit is tied to safety. Plain values such as integers and dicts could fit a fixed schema, but an instance of an arbitrary class can be rebuilt only by calling a constructor the file names (section 1). The others are choices of the current schema and API. The documentation says shared tensors "do not cause unsafety, or denial of service potential", so the decision could be revisited (`docs/source/torch_shared_tensors.mdx`). The README says endianness and memory order could be added later.

## 4. What Rust adds

Section 3 showed that the format itself rules out code execution. The Rust implementation adds three benefits on top of it. It makes the parser memory safe. A parser bug (risk 5 in section 2.2) can still crash the loader, but it cannot read or write outside the buffer. It checks zero-copy tensor views at compile time. It also provides one parser for every framework.

### 4.1 Memory safety in the parser

In a parser written in C or C++, a missed bounds check can read or write outside the buffer, and memory corruption can lead to code execution. The safetensors README makes this contrast with HDF5, citing its use-after-free vulnerabilities.

In safe Rust, a missed check becomes an error or a crash, not an access outside the buffer. `read_metadata` takes the header with `buffer.get(range)`, which returns `None` for an out-of-range slice. `validate` computes sizes with `checked_mul`, which returns `None` on overflow. `SafeTensors::tensor` slices the buffer with `&data[begin..end]` and relies on that validation. If the validation had a bug, the slice would panic instead of reading past the end.

Rust code outside an `unsafe` block is checked by the compiler for memory safety. Inside one, the programmer may dereference raw pointers or call foreign functions, and correctness becomes the programmer's responsibility. The parsing code in the core crate (`safetensors/src`) contains no `unsafe` block. The only one outside documentation and tests is a macOS `fcntl` call on the write path. The Python binding does use `unsafe`, for example to memory-map the file and to copy tensor bytes in the CUDA and MPS paths. Memory safety covers the parser, not the whole binding.

Parsing is one function from bytes to `Result`, which makes it easy to fuzz. The repository's fuzz target (`safetensors/fuzz/fuzz_targets/fuzz_target_1.rs`) is a single call to `SafeTensors::deserialize` on random bytes. Any out-of-bounds access the fuzzer reaches shows up as a panic.

### 4.2 Zero-copy loading comes from the format

The documentation attributes the CPU speedup over `torch.load` to memory-mapping the file, and notes that the same is possible in pure PyTorch (`docs/source/speed.mdx`). This is a property of the format. The header gives every tensor's byte range up front, so a loader can map the file and hand each range to the framework without copying. The Python binding does this with `torch.UntypedStorage.from_file`. The writer also pads the header to a multiple of 8 bytes so that tensor data stays aligned.

Rust adds one guarantee here, for Rust callers. `SafeTensors<'data>` and `TensorView<'data>` borrow the mapped buffer, so the compiler rejects code that uses a tensor view after the buffer is dropped.

### 4.3 One parser for every framework

The Python package's modules for PyTorch, NumPy, TensorFlow, Flax, Paddle, and MLX all load through two Rust entry points, `safe_open` and `deserialize`. Both call `read_metadata`. The checks in section 3.3 therefore exist once. Compare risk 4 in section 2.2: `torch.load` dispatches over several formats, and each path needs its own check.

## 5. Summary

A pickle is a program. The file names functions and the loader calls them, and even an honest checkpoint does this to rebuild every tensor.

`weights_only=True` restricts which functions may run, but the format can still express calls. Safety then depends on the allowlist, the argument handling, and every entry point being correct. Comparing torch 2.5.1 and 2.13.0 showed examples of each of these going wrong.

A safetensors header is data with a fixed schema. The loader's steps are fixed, and the file's values are validated before any tensor is created. The price is that a file holds only tensors and string metadata, not arbitrary Python objects.

Rust makes the parser memory safe, so a parser bug can crash the loader but cannot corrupt memory. It also gives every framework the same parser.

For weights from an untrusted source, this is the difference that matters: a safetensors file cannot choose what code runs.
