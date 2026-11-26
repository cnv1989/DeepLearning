"""Bundle the design into a Tiny Tapeout-ready src/ folder.

Inlines every `include so the TT flow needs no include paths, and copies
tt/info.yaml + tt/info.md next to it. Then copy build/tt_submission/* into a
repo created from the current Tiny Tapeout Verilog template.
"""
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = ["rtl/tt_um_tinylm.v", "rtl/sync_edge.v", "rtl/tinylm_core.v",
           "rtl/generated/tinylm_weights.v", "rtl/generated/tinylm_charmap.v"]
INC_DIRS = ["rtl/generated"]


def inline(path):
    out = []
    for line in open(os.path.join(ROOT, path)):
        m = re.match(r'\s*`include\s+"([^"]+)"', line)
        if m:
            for d in INC_DIRS:
                p = os.path.join(ROOT, d, m.group(1))
                if os.path.exists(p):
                    out.append(f"// ---- begin {m.group(1)} ----\n")
                    out += [ln for ln in open(p)]
                    out.append(f"// ---- end {m.group(1)} ----\n")
                    break
            else:
                raise FileNotFoundError(m.group(1))
        else:
            out.append(line)
    return "".join(out)


def main():
    dst = os.path.join(ROOT, "build", "tt_submission")
    os.makedirs(os.path.join(dst, "src"), exist_ok=True)
    os.makedirs(os.path.join(dst, "docs"), exist_ok=True)
    with open(os.path.join(dst, "src", "project.v"), "w") as f:
        for s in SOURCES:
            f.write(f"// ======== {s} ========\n{inline(s)}\n")
    shutil.copy(os.path.join(ROOT, "tt", "info.yaml"), os.path.join(dst, "info.yaml"))
    shutil.copy(os.path.join(ROOT, "tt", "info.md"), os.path.join(dst, "docs", "info.md"))
    print(f"wrote {dst}/src/project.v, info.yaml, docs/info.md")


if __name__ == "__main__":
    main()
