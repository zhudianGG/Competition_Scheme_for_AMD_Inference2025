"""Model preprocessing entry point. Hooks the FourOverSix GPTQ path when --quantize is passed."""
import argparse
import shutil
from pathlib import Path


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        target = dst / rel
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)


def quantize_fp4(src: Path, dst: Path) -> None:
    # FourOverSix GPTQ quantization is owned by SpecForge/gptq_custom.py.
    # This is a thin shim: call into that module if available, otherwise fall back to copy.
    try:
        import sys
        sys.path.insert(0, "/sgl-workspace/SpecForge")
        import gptq_custom  # noqa: F401
        # gptq_custom doesn't expose a clean API yet — drop a TODO marker.
        raise NotImplementedError(
            "Wire SpecForge/gptq_custom.py once it exposes a callable quantize(src, dst)."
        )
    except ImportError:
        print("[preprocess_model] SpecForge gptq_custom unavailable — copying instead")
        copy_tree(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--quantize", action="store_true")
    args = ap.parse_args()
    if args.quantize:
        quantize_fp4(args.input, args.output)
    else:
        copy_tree(args.input, args.output)
    print(f"[preprocess_model] wrote {args.output}")


if __name__ == "__main__":
    main()
