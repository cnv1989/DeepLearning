# TinyLM -- build, test and synthesise.
#   make train            train the model           (CONFIG=tt | fpga)
#   make gen              weights -> Verilog + test vectors
#   make test             run every testbench
#   make examples         run the Verilog refresher examples
#   make wave             dump build/tb_core.vcd for GTKWave / Surfer
#   make synth-tt         gate-count estimate for Tiny Tapeout
#   make synth-ice40      LUT count on an iCE40UP5K
#   make synth-compare    sequential vs fully-parallel core (slow)
#   make bitstream        iCEBreaker bitstream (needs nextpnr-ice40 + icepack)
#   make prog             flash the iCEBreaker (needs iceprog)
#   make tt-bundle        build/tt_submission/ ready to copy into the TT template
CONFIG ?= tt
PY     ?= python3
B      := build
GEN    := rtl/generated
W      := model/weights/$(CONFIG).json
INC    := -I $(GEN) -I tb/vectors
IV     := iverilog -g2012 -Wall -Wno-timescale $(INC)
RTL    := rtl/tinylm_core.v rtl/sync_edge.v rtl/uart_rx.v rtl/uart_tx.v \
          $(GEN)/tinylm_weights.v $(GEN)/tinylm_charmap.v

.PHONY: all train gen test test-core test-parallel test-tt test-uart wave examples \
        synth-tt synth-compare synth-ice40 bitstream prog tt-bundle clean
all: test

train:
	$(PY) model/train.py --config $(CONFIG)

$(W):
	$(PY) model/train.py --config $(CONFIG)

gen: $(W)
	$(PY) model/gen_verilog.py --weights $(W)
	$(PY) model/gen_vectors.py --weights $(W)

$(B):
	mkdir -p $(B)

test: test-core test-parallel test-tt test-uart

test-core: | $(B)
	$(IV) -o $(B)/tb_core.vvp tb/tb_core.v $(RTL)
	vvp -n $(B)/tb_core.vvp | grep -v '\$$finish'

test-parallel: | $(B)
	$(IV) -DPARALLEL -o $(B)/tb_par.vvp tb/tb_core.v $(GEN)/tinylm_core_parallel.v $(RTL)
	vvp -n $(B)/tb_par.vvp | grep -v '\$$finish'

test-tt: | $(B)
	$(IV) -o $(B)/tb_tt.vvp tb/tb_tt.v rtl/tt_um_tinylm.v $(RTL)
	vvp -n $(B)/tb_tt.vvp | grep -v '\$$finish'

test-uart: | $(B)
	$(IV) -o $(B)/tb_uart.vvp tb/tb_uart_top.v rtl/tinylm_uart_top.v $(RTL)
	vvp -n $(B)/tb_uart.vvp | grep -v '\$$finish'

wave: | $(B)
	$(IV) -o $(B)/tb_core.vvp tb/tb_core.v $(RTL)
	vvp -n $(B)/tb_core.vvp +vcd | tail -2
	@echo "open build/tb_core.vcd in GTKWave (gtkwave build/tb_core.vcd) or Surfer"

EX := 01_counter/tb_counter.v:01_counter/counter.v 02_blocking/tb_blocking.v \
      03_fsm/tb_seq_detect.v:03_fsm/seq_detect.v 04_signed/tb_signed.v \
      05_mac/tb_ternary_mac.v:05_mac/ternary_mac.v
examples: | $(B)
	@for e in $(EX); do \
	  files=$$(echo $$e | tr ':' ' ' | sed 's#[^ ]*#examples/&#g'); \
	  iverilog -g2012 -o $(B)/ex.vvp $$files && vvp -n $(B)/ex.vvp | grep -v '\$$finish\|VCD'; \
	done

TT_SRC := rtl/tt_um_tinylm.v rtl/sync_edge.v rtl/tinylm_core.v $(GEN)/tinylm_core_parallel.v \
          $(GEN)/tinylm_weights.v $(GEN)/tinylm_charmap.v
synth-tt: | $(B)
	yosys -q -p "read_verilog -I $(GEN) $(TT_SRC); synth -top tt_um_tinylm -flatten; \
	  abc -g AND,NAND,OR,NOR,XOR,XNOR,MUX; opt_clean; tee -o $(B)/synth_tt.txt stat" > /dev/null
	@grep -E "Number of cells|DFF" $(B)/synth_tt.txt

# Sequential vs fully-parallel core, both mapped to iCE40 LUTs (lesson 10).
# (Generic gate mapping of the parallel core is very slow; LUT mapping takes ~10 min.)
synth-compare: | $(B)
	yosys -q -p "read_verilog -I $(GEN) $(TT_SRC); synth_ice40 -top tt_um_tinylm; \
	  tee -o $(B)/cmp_seq.txt stat" > /dev/null
	yosys -q -p "read_verilog -DTINYLM_PARALLEL -I $(GEN) $(TT_SRC); synth_ice40 -top tt_um_tinylm; \
	  tee -o $(B)/cmp_par.txt stat" > /dev/null
	@echo "sequential:"; grep -E "SB_LUT4|SB_CARRY" $(B)/cmp_seq.txt
	@echo "parallel:";   grep -E "SB_LUT4|SB_CARRY" $(B)/cmp_par.txt

ICE_SRC := rtl/icebreaker_top.v rtl/tinylm_uart_top.v $(RTL)
synth-ice40: | $(B)
	yosys -q -p "read_verilog -I $(GEN) $(ICE_SRC); synth_ice40 -top icebreaker_top -json $(B)/ice.json; \
	  tee -o $(B)/synth_ice40.txt stat" > /dev/null
	@grep -E "Number of cells|SB_LUT4|SB_CARRY" $(B)/synth_ice40.txt

bitstream: synth-ice40
	nextpnr-ice40 --up5k --package sg48 --pcf fpga/icebreaker.pcf --json $(B)/ice.json --asc $(B)/ice.asc --freq 12
	icepack $(B)/ice.asc $(B)/tinylm.bin

prog: bitstream
	iceprog $(B)/tinylm.bin

tt-bundle:
	$(PY) tools/bundle_tt.py

# Generated sim/synth outputs only; trained weights in build/<config>/ are kept.
clean:
	rm -f build/*.vvp build/*.vcd build/*.txt build/*.json build/*.asc build/*.bin
	rm -rf build/tt_submission
