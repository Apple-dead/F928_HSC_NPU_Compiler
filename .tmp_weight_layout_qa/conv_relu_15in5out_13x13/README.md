# Conv + LeakyReLU simulation examples

Both versions share identical input, weights and expected output. Addresses are byte addresses.
aligned64 aligns input, parameters and instructions to 64 bytes; unaligned64 uses 8-byte alignment.

Each valid output channel: all its weights -> int32 little-endian bias -> four ZERO bytes.
Weights: output channel -> input channel group -> kh -> kw -> eight channel lanes (zero padded).
No dummy output-channel parameter records. Output features are still padded to 8 channels.

|Case|Input|Parameters|First bias|Instruction entry|Output|Parameter bytes|Minimum BRAM depth|
|---|---|---|---|---|---|---|---|
|aligned64|0x00000040|0x00000B00|0x00000B90|0x00000E00|0x00000E38|760|78|
|unaligned64|0x00000008|0x00000A98|0x00000B28|0x00000D90|0x00000DC8|760|77|

For 15 inputs / 5 outputs / 3x3: each record is 144 weight bytes + 8 bias-slot bytes = 152 bytes; total 760 bytes.
bias.csv lists actual bias addresses. stride_bytes in memory_plan.json describes interleaved tensors.
all.coe/all.mem/all.bin contain the same 512-bit BRAM image, loaded at address 0: input, parameters and instructions.
expected_output.bin/.mem/.coe contain the reference output. reference.txt shows matrices; reference.csv shows scalar results.
In each 512-bit COE word, the rightmost byte has the lowest address. Instructions are 32-bit little-endian.
Input bytes: channel group -> row -> column -> eight channels. All pixels of channels 1..8 precede channels 9..16.
Conv scale=11; accumulator range=[-190964, 184596]; scaled range=[-94, 90].
Compute acc=sum(input*weight)+bias, arithmetic-shift right by scale, take 8 bits as int8, then shift negative values right by 3.
Reference validated with vector and independent scalar calculations. RTL simulation has not been run.
