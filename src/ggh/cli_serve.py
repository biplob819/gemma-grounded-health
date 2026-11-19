"""`ggh` commands for serving and EvalKit integration (node E3.1): `llama fetch`, `export`,
`serve`, `serve-check`, `tasks`, `evalkit-check` and `behaviour`; X3.1 adds `merge`, `parity`
and `export-records`; X3.2 adds `bench`;
E3.2 adds `serve-hf` and
`probe build`."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

EXIT_ERROR = 2


def _fail(cmd: str, exc: BaseException | str) -> int:
    print(f"{cmd}: FAIL: {exc}")
    return EXIT_ERROR


def _cmd_llama_fetch(_: argparse.Namespace) -> int:
    from ggh.serve import llama

    try:
        root = llama.cache_root()
        for pin in (llama.RELEASE, llama.SOURCE):
            print(f"{pin.name}: verified, extracted to {llama.extract(pin, root)}")
    except (llama.LlamaError, OSError) as exc:
        return _fail("llama fetch", exc)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from ggh.export import gguf, merge
    from ggh.serve import llama
    from ggh.versions import version

    root = Path(args.root)
    try:
        if args.model == "base":
            hf = gguf.hf_snapshot(gguf.BASE_ID, gguf.BASE_REVISION)
            sid, srev = gguf.BASE_ID, gguf.BASE_REVISION
        else:  # a merged fine-tune (v1 ft, v2 ft2): named by the base and the frozen adapter
            hf = root / version(args.model).merged_dir
            sid, srev = f"{gguf.BASE_ID}+adapter", merge.frozen_hash(root, args.model)
            merged = json.loads((hf / "ggh_merge.json").read_text(encoding="utf-8"))
            if merged["adapter_sha256"] != srev:
                raise merge.MergeError(f"{hf} was merged from another adapter")
        m = gguf.export(root, hf, args.model, source_id=sid, source_revision=srev)
    except ImportError as exc:
        return _fail("export", f"{exc}; the converter needs: uv sync --group nli")
    except (llama.LlamaError, OSError, ValueError, KeyError) as exc:
        return _fail("export", exc)
    for name, f in m["files"].items():
        print(f"{name}  {f['bytes']:,} bytes  sha256 {f['sha256'][:12]}…")
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    from ggh.export import gguf, merge
    from ggh.versions import version

    root = Path(args.root)
    v = version(args.model)
    try:
        expected = merge.frozen_hash(root, args.model)
        base = gguf.hf_snapshot(gguf.BASE_ID, gguf.BASE_REVISION)
        rec = merge.merge(base, root / v.adapter_dir, root / v.merged_dir,
                          expected_sha256=expected)  # fmt: skip
    except ImportError as exc:
        return _fail("merge", f"{exc}; merging needs: uv sync --group train")
    except (OSError, ValueError, KeyError) as exc:
        return _fail("merge", exc)
    print(f"merged {rec['lora_layers_merged']} LoRA layers into bf16 -> {v.merged_dir}")
    return 0


def _cmd_parity(args: argparse.Namespace) -> int:
    from ggh.data.build import read_jsonl
    from ggh.export import parity
    from ggh.serve import checks, llama
    from ggh.train.devrun import DEV_EVAL
    from ggh.versions import version

    root = Path(args.root)
    model = args.model
    port = args.port or llama.PORTS[model]
    try:
        rows = read_jsonl(root / DEV_EVAL)[: args.limit]

        def bf16_gen(prompts: Any) -> list[str]:  # pragma: no cover - 10 GB model
            from ggh.train.devrun import release, vocab_size
            from ggh.train.generate import compile_grammar, generate

            model_, tok = parity.load_bf16(root / version(model).merged_dir)
            out = generate(model_, tok, prompts, grammar=compile_grammar(tok, vocab_size(model_)))
            release(model_)
            return out

        def q4_gen(prompts: Any) -> list[str]:  # pragma: no cover - starts llama-server
            log = root / llama.GGUF_DIR / f"serve-{model}-parity.log"
            proc = llama.start(root, model, "Q4_K_M", port, log=log)
            try:
                base = f"http://{llama.HOST}:{port}/v1"
                return parity.server_replies(prompts, checks.evalkit_complete(base),
                                             parity.q4_name(model))  # fmt: skip
            finally:
                proc.terminate()
                proc.wait(timeout=30)

        rep = parity.run(root, rows, bf16_gen=bf16_gen, q4_gen=q4_gen, model=model)
    except ImportError as exc:
        return _fail("parity", f"{exc}; bf16 generation needs: uv sync --group train")
    except (llama.LlamaError, OSError, ValueError) as exc:
        return _fail("parity", exc)
    print(rep.model_dump_json())
    bad = parity.failures(rep)
    out = parity.paths(model)[0]
    print(f"parity: {'PASS' if not bad else 'FAIL: ' + '; '.join(bad)} ({out})")
    return 0 if not bad else EXIT_ERROR


def _cmd_export_records(args: argparse.Namespace) -> int:
    from ggh.export import artifacts, gguf

    root = Path(args.root)
    try:
        base = gguf.hf_snapshot(gguf.BASE_ID, gguf.BASE_REVISION)
        recs = artifacts.write_all(root, base)
    except (OSError, ValueError, KeyError) as exc:
        return _fail("export-records", exc)
    for r in recs:
        q = r.quant or ""
        print(f"{r.kind:12} {r.model:4} {q:7} {r.sha256[:12]}…  parent {r.parent_key[:12]}…")
    print(f"export-records: {len(recs)} artifacts ({artifacts.RECORDS})")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    import datetime as dt

    from ggh.data.build import read_jsonl
    from ggh.serve import bench, llama
    from ggh.train.devrun import DEV_EVAL

    root = Path(args.root)
    v2 = args.model == "ft2"
    variants = bench.V2_VARIANTS if v2 else bench.VARIANTS
    try:
        hw = bench.hardware()
        bench.guard(hw, allow=args.allow_idle_port)
        prompts = bench.select_prompts(read_jsonl(root / DEV_EVAL))
        results, details = {}, {}
        started = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        print("bench: llama-bench, alternating rounds …", flush=True)
        lb, rounds = bench.llama_bench_rounds(root, variants)
        for model, quant in variants:
            print(f"bench: {model} {quant} …", flush=True)
            key = f"{model}-{quant}"
            results[key], details[key] = bench.bench_variant(root, model, quant, prompts, hw,
                                                             lb=lb[key])  # fmt: skip
            details[key]["llama_bench_rounds"] = rounds[key]
            r = results[key]
            print(f"  gen {r.gen_tps_p50} tok/s, ttft {r.ttft_ms_p50} ms, "
                  f"latency p50 {r.answer_latency_ms_p50} ms, peak {r.peak_rss_mb} MB")  # fmt: skip
        cond = ({"idle_servers_listening": sorted(args.allow_idle_port)}
                if args.allow_idle_port else None)  # fmt: skip
        path = bench.write(root, results, details, started,
                           bench.V2_OUT if v2 else bench.OUT, cond)  # fmt: skip
    except (bench.BenchError, llama.LlamaError, OSError, ValueError) as exc:
        return _fail("bench", exc)
    pair = ("ft", "ft2") if v2 else ("base", "ft")
    gaps = {on: bench.per_token_gap(results, on, pair) for on in ("tg128", "request")}
    print(
        f"bench: wrote {path.relative_to(root)}; {pair[0]}/{pair[1]} generation-speed gaps {gaps}"
    )
    return 0


def _cmd_serve_hf(args: argparse.Namespace) -> int:  # pragma: no cover - serves until Ctrl-C
    from ggh.serve import hf
    from ggh.versions import version

    root = Path(args.root)
    model_id = hf.MODEL_IDS[args.model]
    try:
        gen = hf.transformers_generate(root / version(args.model).merged_dir)
        server = hf.serve(gen, args.port, model_id)
    except ImportError as exc:
        return _fail("serve-hf", f"{exc}; it needs: uv sync --group train")
    except OSError as exc:
        return _fail("serve-hf", exc)
    print(f"serving {model_id} (bf16, Transformers) at http://{hf.HOST}:{args.port}/v1", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


def _cmd_probe_build(args: argparse.Namespace) -> int:  # pragma: no cover - paid API calls
    from evalkit.datasets.pubmedqa import load_pqa_l

    from ggh import ledger
    from ggh.evals import probe

    root = Path(args.root)
    spend = probe.Spend()
    try:
        cases = load_pqa_l()
        para, check = probe.live_callers(spend)
        kept, rejected = probe.build(cases, para, check)
        probe.write(root, kept, rejected, cases)
    except (OSError, ValueError, ImportError) as exc:
        return _fail("probe build", exc)
    finally:
        if spend.calls:
            ledger.append(root / "docs/measurements/spend-ledger.jsonl", "probe_build", "spent_usd",
                          round(spend.usd, 6), node="E3.2", calls=spend.calls,
                          models=[probe.PARAPHRASE_MODEL, probe.CHECK_MODEL])  # fmt: skip
    print(f"probe build: {len(kept)} kept, {len(rejected)} rejected, ${spend.usd:.4f}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from ggh.serve import llama

    try:
        log = Path(args.root) / llama.GGUF_DIR / f"serve-{args.model}.log"
        proc = llama.start(Path(args.root), args.model, args.quant, args.port, log=log)
    except (llama.LlamaError, OSError) as exc:
        return _fail("serve", exc)
    port = args.port or llama.PORTS[args.model]
    print(f"serving {args.model} {args.quant} at http://{llama.HOST}:{port}/v1 (Ctrl-C to stop)")
    print(f"server log: {log}")
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        return 0


def _cmd_serve_check(args: argparse.Namespace) -> int:
    from ggh.serve import checks, llama

    root = Path(args.root)
    base = f"http://{llama.HOST}:{llama.PORTS[args.model]}/v1"
    try:
        report = checks.run(
            root,
            args.model,
            args.quant,
            complete=checks.evalkit_complete(base),
            hf=checks.hf_tokenizer(),
        )
    except ImportError as exc:
        return _fail("serve-check", f"{exc}; the tokenizer needs: uv sync --group nli")
    except (llama.LlamaError, OSError, ValueError) as exc:
        return _fail("serve-check", exc)
    out = root / checks.out_path(args.model)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"]))
    print(f"serve-check: {'PASS' if report['passed'] else 'FAIL'} ({checks.out_path(args.model)})")
    return 0 if report["passed"] else EXIT_ERROR


def _cmd_serve_throughput(args: argparse.Namespace) -> int:
    import platform

    from ggh.serve import checks, llama

    root = Path(args.root)
    base = f"http://{llama.HOST}:{llama.PORTS[args.model]}/v1"
    try:
        rep = checks.throughput(
            root, args.model, args.quant, checks.evalkit_async_client(base),
            host=f"{platform.machine()} {platform.system()} {platform.release()}",
        )  # fmt: skip
    except (OSError, ValueError) as exc:
        return _fail("serve-throughput", exc)
    out = root / checks.THROUGHPUT
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep["runs"]))
    return 0


def _cmd_tasks(args: argparse.Namespace) -> int:
    from ggh.evals import tasks

    try:
        for rel in tasks.write(Path(args.root)):
            print(f"wrote {rel}")
    except (OSError, ValueError, KeyError) as exc:
        return _fail("tasks", exc)
    return 0


def _cmd_evalkit_check(args: argparse.Namespace) -> int:
    from evalkit.config import ConfigError, load_run_config

    from ggh.evals import config

    root = Path(args.root)
    try:
        problems = config.config_problems(root) + config.pin_problems(root)
        out: dict[str, Any] = {"config_problems": problems}
        if args.reference:
            if not args.reference_config:
                raise ValueError("--reference needs --reference-config (that run's YAML config)")
            mine = config.ours(load_run_config(root / config.MAIN), config.pinned_commit(root))
            ref = config.reference(Path(args.reference), Path(args.reference_config))
            store = Path(args.store) if args.store else None
            out["reuse_problems"] = config.reuse_problems(mine, ref, store)
            out["reuse_claimable"] = not out["reuse_problems"]
    except (ConfigError, OSError, ValueError, KeyError) as exc:
        return _fail("evalkit-check", exc)
    print(json.dumps(out, indent=2))
    return EXIT_ERROR if problems else 0


def _cmd_behaviour(args: argparse.Namespace) -> int:
    from evalkit.run import read_cases, read_records

    from ggh.evals import behaviour

    run_dir = Path(args.run_dir)
    try:
        rep = behaviour.report(read_records(run_dir), read_cases(run_dir))
    except (OSError, ValueError, KeyError) as exc:
        return _fail("behaviour", exc)
    print(json.dumps(rep, indent=2))
    return 0


def register(sub: Any) -> None:
    from ggh.serve import llama
    from ggh.versions import VERSIONS

    lm = sub.add_parser("llama", help="the pinned llama.cpp release (E3.1)")
    lsub = lm.add_subparsers(dest="llama_command", required=True)
    lf = lsub.add_parser("fetch", help="download, verify and extract the pinned release")
    lf.set_defaults(func=_cmd_llama_fetch)

    ex = sub.add_parser("export", help="HF safetensors → GGUF F16, Q4_K_M, Q8_0")
    ex.add_argument("--model", default="base", choices=sorted(llama.PORTS))
    ex.add_argument("--root", default=".", help="repo root")
    ex.set_defaults(func=_cmd_export)

    mg = sub.add_parser("merge", help="merge the frozen adapter into the bf16 base on CPU (X3.1)")
    mg.add_argument("--model", default="ft", choices=sorted(VERSIONS), help="ft (v1) or ft2 (v2)")
    mg.add_argument("--root", default=".", help="repo root")
    mg.set_defaults(func=_cmd_merge)

    pa = sub.add_parser("parity", help="dev parity: ft Q4_K_M GGUF against merged bf16 (X3.1)")
    pa.add_argument(
        "--model",
        default="ft",
        choices=("ft", "ft2"),
        help="ft (v1) or ft2 (v2, written to docs/measurements/v2/parity.json)",
    )
    pa.add_argument(
        "--port",
        type=int,
        default=None,
        help="llama-server port (default the model's `ggh serve` port)",
    )
    pa.add_argument("--root", default=".", help="repo root")
    pa.add_argument("--limit", type=int, default=None, help="first N dev prompts (smoke)")
    pa.set_defaults(func=_cmd_parity)

    er = sub.add_parser("export-records", help="stage manifests and ExportArtifact records (X3.1)")
    er.add_argument("--root", default=".", help="repo root")
    er.set_defaults(func=_cmd_export_records)

    sh = sub.add_parser("serve-hf", help="OpenAI-compatible bf16 fine-tune server (E3.2 parity)")
    sh.add_argument("--port", type=int, default=8082)
    sh.add_argument("--model", choices=["ft", "ft2"], default="ft", help="ft2: v2 (V2.2)")
    sh.add_argument("--root", default=".", help="repo root")
    sh.set_defaults(func=_cmd_serve_hf)

    pb = sub.add_parser("probe", help="paraphrase probe set (E3.2, D-42)")
    pbs = pb.add_subparsers(dest="probe_command", required=True)
    pbb = pbs.add_parser("build", help="paraphrase and check 100 PQA-L questions (paid)")
    pbb.add_argument("--root", default=".", help="repo root")
    pbb.set_defaults(func=_cmd_probe_build)

    bn = sub.add_parser("bench", help="laptop benchmark under Tech Spec §8 (X3.2)")
    bn.add_argument(
        "--model",
        default="ft",
        choices=("ft", "ft2"),
        help="ft: base and ft (v1's file); ft2: ft and ft2 (docs/measurements/v2)",
    )
    bn.add_argument(
        "--allow-idle-port",
        type=int,
        action="append",
        default=[],
        help="a `ggh serve` port whose server is known idle (recorded in the file)",
    )
    bn.add_argument("--root", default=".", help="repo root")
    bn.set_defaults(func=_cmd_bench)

    sv = sub.add_parser("serve", help="start llama-server: base 8080, ft 8081, ft2 8083")
    sv.add_argument("--model", default="base", choices=sorted(llama.PORTS))
    sv.add_argument("--quant", default="Q4_K_M", choices=llama.QUANTS)
    sv.add_argument("--port", type=int, default=None)
    sv.add_argument("--root", default=".", help="repo root")
    sv.set_defaults(func=_cmd_serve)

    sc = sub.add_parser("serve-check", help="template parity, JSON and determinism checks")
    sc.add_argument("--model", default="base", choices=sorted(llama.PORTS))
    sc.add_argument("--quant", default="Q4_K_M", choices=llama.QUANTS)
    sc.add_argument("--root", default=".", help="repo root")
    sc.set_defaults(func=_cmd_serve_check)

    st = sub.add_parser("serve-throughput", help="seconds per answer through EvalKit's adapter")
    st.add_argument("--model", default="base", choices=sorted(llama.PORTS))
    st.add_argument("--quant", default="Q4_K_M", choices=llama.QUANTS)
    st.add_argument("--root", default=".", help="repo root")
    st.set_defaults(func=_cmd_serve_throughput)

    tk = sub.add_parser("tasks", help="write the dev smoke and off-scope EvalKit tasks")
    tk.add_argument("--root", default=".", help="repo root")
    tk.set_defaults(func=_cmd_tasks)

    ec = sub.add_parser("evalkit-check", help="check configs/evalkit.yaml and cache reuse")
    ec.add_argument("--root", default=".", help="repo root")
    ec.add_argument("--reference", default=None, help="a finished EvalKit run folder")
    ec.add_argument(
        "--reference-config", default=None, help="the YAML config that reference run used"
    )
    ec.add_argument("--store", default=None, help="the EvalKit store holding that run's entries")
    ec.set_defaults(func=_cmd_evalkit_check)

    bh = sub.add_parser("behaviour", help="B1, B3, B4 and the B8 pre-sort over EvalKit records")
    bh.add_argument("run_dir", help="an EvalKit run folder")
    bh.set_defaults(func=_cmd_behaviour)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(EXIT_ERROR)
