# TinyPilot: a PID flight controller turned into a hard-wired neural network.
#   make train        behaviour cloning + DAgger (about 2.5 min)  -> model/weights.json
#   make gen          weights -> rtl/generated/, closed-loop vectors -> tb/vectors/
#   make plot         docs/flight.png (needs matplotlib)
#   make test         SPI-level closed-loop replay testbench
#   make synth-tt     generic gate count (Tiny Tapeout estimate)
#   make synth-ice40  LUT count on an iCE40UP5K
PY  ?= python3
SRC := rtl/tt_um_tinypilot.v rtl/pilot_core.v rtl/spi_slave.v rtl/oneshot125.v rtl/generated/pilot_weights.v

.PHONY: train gen plot test synth-tt synth-ice40 clean
train:
	$(PY) model/train.py
gen:
	$(PY) model/gen_verilog.py
plot:
	$(PY) model/plot.py
build:
	mkdir -p build
test: | build
	iverilog -g2012 -Wall -Wno-timescale -I rtl/generated -I tb/vectors -o build/tb_pilot.vvp tb/tb_pilot.v $(SRC)
	vvp -n build/tb_pilot.vvp | grep -v '\$$finish'
synth-tt: | build
	yosys -q -p "read_verilog -I rtl/generated $(SRC); synth -top tt_um_tinypilot -flatten; \
	  abc -g AND,NAND,OR,NOR,XOR,XNOR,MUX; opt_clean; tee -o build/synth_tt.txt stat" > /dev/null
	@grep -E "Number of cells" build/synth_tt.txt
	@grep -E "DFF" build/synth_tt.txt | awk '{s+=$$2} END {print "   flip-flops:", s}'
synth-ice40: | build
	yosys -q -p "read_verilog -I rtl/generated $(SRC); synth_ice40 -top tt_um_tinypilot; \
	  tee -o build/synth_ice40.txt stat" > /dev/null
	@grep -E "SB_LUT4|SB_CARRY" build/synth_ice40.txt
clean:
	rm -f build/*.vvp build/*.txt
