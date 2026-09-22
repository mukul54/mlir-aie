# whole_array_w4sh — int4 weights, int32 C, inline per-column pow2 shifts (dead end)

Variant of `whole_array_w4i32` where each streamed B tile carries `n` int8
right-shifts after the nibble blocks, meant to give per-(k-tile, column-block)
power-of-two weight sub-scales for free on accumulator readout.

Status (npu2, 256x2048x16384, 64x128x64 tiles, 8 cols): builds, but
1. `mmul::to_vector<int32>(shift)` on an acc32 accumulator is a plain cast on
   aie2p (the srs shift only exists when narrowing), so the shifts are ignored
   and the output equals the unshifted w4i32 sum;
2. the fresh-accumulator + load/add/store epilogue costs about 21% vs w4i32.
An offline study on the pi0 weights also shows pow2 sub-scales buy nothing on
top of per-channel MSE-clipped scales, so this was not pursued further.
Kept for reference; use `whole_array_w4i32`.
