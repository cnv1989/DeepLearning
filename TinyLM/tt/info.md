## How it works

TinyLM is a character-level language model with its weights hard-wired into
logic. It looks at the last 3 characters, embeds each into 4 signed 4-bit
numbers, runs a 16-neuron hidden layer and a 32-way output layer with ternary
(-1/0/+1) weights, and outputs the most likely next character. One
multiply-accumulate happens per clock, so one character takes 704 cycles
(about 14 us at 50 MHz).

## How to test

1. Reset.
2. For each prompt character: put its token on ui[4:0] (space=0, a..z=1..26,
   . , ' ! ? = 27..31), then pulse PUSH (ui[5]).
3. Set NOISE (ui[7]) to 1 for random sampling or 0 for greedy.
4. Pulse START (ui[6]) and wait for VALID (uo[6]). The predicted character is
   on uio[7:0] as ASCII and uo[4:0] as a token. It has already been appended to
   the context, so pulse START again for the next character.

## External hardware

None. The RP2040 on the demo board can drive the inputs and print the output.
