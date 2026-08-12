"""
CLI interface for memoria.

Usage:
  memoria remember "We switched auth from JWT to OAuth2 because of the SSO requirement"
  memoria recall "what auth method do we use"
  memoria entity "OAuth2"
  memoria history "auth" --predicate "method"
  memoria consolidate
  memoria stats
  memoria serve   # start MCP server
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .core import Memoria


def make_llm_call():
    """Create an LLM call function using the Anthropic SDK if available."""
    try:
        import anthropic
        client = anthropic.Anthropic()

        def call(prompt: str) -> str:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return call
    except (ImportError, Exception):
        return None


def main():
    parser = argparse.ArgumentParser(
        prog="memoria",
        description="Spectral memory architecture with CAG-informed retrieval",
    )
    parser.add_argument("--db", default="~/.memoria/memoria.db", help="Database path")
    parser.add_argument(
        "--scope",
        default="global",
        help="Memory scope: global, auto, project, or project:/path",
    )
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM extraction (heuristic only)")
    sub = parser.add_subparsers(dest="command")

    # remember
    p_rem = sub.add_parser("remember", help="Store information in memory")
    p_rem.add_argument("text", help="Text to remember")
    p_rem.add_argument("--role", default="user", choices=["user", "assistant", "system"])
    p_rem.add_argument("--session", default=None, help="Session ID")

    # recall
    p_rec = sub.add_parser("recall", help="Retrieve relevant memories")
    p_rec.add_argument("query", help="Query to search for")
    p_rec.add_argument("-k", "--top-k", type=int, default=10, help="Max results")
    p_rec.add_argument("-m", "--mode", default="balanced",
                       choices=["speed", "balanced", "quality"],
                       help="Retrieval mode (default: balanced)")

    # entity
    p_ent = sub.add_parser("entity", help="Get everything about an entity")
    p_ent.add_argument("name", help="Entity name")

    # history
    p_hist = sub.add_parser("history", help="Temporal history of an entity")
    p_hist.add_argument("name", help="Entity name")
    p_hist.add_argument("--predicate", default=None, help="Filter by predicate")

    # consolidate
    p_con = sub.add_parser("consolidate", help="Run memory consolidation")
    p_con.add_argument("--prune-threshold", type=float, default=0.05)
    p_con.add_argument("--half-life", type=float, default=30.0, help="Half-life in days")

    # stats
    sub.add_parser("stats", help="Show system statistics")

    # compress
    p_comp = sub.add_parser("compress", help="Compress memory into a token budget")
    p_comp.add_argument("-b", "--budget", type=int, default=200, help="Token budget")

    # budget
    sub.add_parser("budget", help="Show compression at each tier (L0/L1/L2/L3)")

    # serve
    sub.add_parser("serve", help="Start MCP server (stdio)")

    # install
    p_install = sub.add_parser("install", help="Install an agent integration")
    p_install.add_argument("integration", choices=["claude"])
    p_install.add_argument("--project", action="store_true", help="Install for the current project only")
    p_install.add_argument("--settings", default=None, help=argparse.SUPPRESS)

    # doctor
    p_doctor = sub.add_parser("doctor", help="Check Memoria integration health")
    p_doctor.add_argument("--project", action="store_true", help="Check project-local Claude settings")
    p_doctor.add_argument("--settings", default=None, help=argparse.SUPPRESS)
    p_doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    # daemon
    p_daemon = sub.add_parser("daemon", help="Manage the local hook daemon")
    p_daemon.add_argument("action", choices=["start", "status", "stop"])

    # export
    p_export = sub.add_parser("export", help="Export memory as portable context for Codex/remote agents")
    p_export.add_argument("-o", "--output", default=None,
                          help="Output file path (default: MEMORIA_CONTEXT.md in current dir)")
    p_export.add_argument("-b", "--budget", type=int, default=4000,
                          help="Token budget for export (default: 4000)")
    p_export.add_argument("--format", choices=["markdown", "json"], default="markdown",
                          help="Export format (default: markdown)")

    # serve-http
    p_http = sub.add_parser("serve-http", help="Start HTTP REST API server")
    p_http.add_argument("--port", type=int, default=7437, help="Port (default: 7437)")
    p_http.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")

    # cleanup
    p_clean = sub.add_parser("cleanup", help="Clean up the knowledge graph")
    p_clean.add_argument("action",
                         choices=["list-entities", "list-triples", "delete-entity",
                                  "delete-triple", "merge", "find-duplicates",
                                  "find-orphans", "purge-orphans", "purge-expired"],
                         help="Cleanup action")
    p_clean.add_argument("--name", default=None, help="Entity name (for delete-entity, merge source)")
    p_clean.add_argument("--id", default=None, help="Entity or triple ID")
    p_clean.add_argument("--into", default=None, help="Target entity name for merge")

    # ingest (batch)
    p_ingest = sub.add_parser("ingest", help="Ingest a text file or directory")
    p_ingest.add_argument("path", help="File or directory to ingest")
    p_ingest.add_argument("--session", default=None, help="Session ID for all ingested content")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "serve":
        from .mcp_server import run_mcp_server
        run_mcp_server()
        return

    if args.command == "serve-http":
        from .http_server import run_http_server
        run_http_server(port=args.port, host=args.host)
        return

    if args.command == "install":
        from .integrations import install_claude_hooks

        result = install_claude_hooks(
            settings_path=args.settings,
            project=args.project,
        )
        print(f"Installed Claude Code hooks in {result['settings_path']}")
        if result["backup_path"]:
            print(f"Backup: {result['backup_path']}")
        print("Events: UserPromptSubmit (recall), Stop (async save)")
        return

    if args.command == "doctor":
        from .integrations import doctor_report

        result = doctor_report(
            settings_path=args.settings,
            project=args.project,
            db_path=args.db,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            for check in result["checks"]:
                marker = "ok" if check["ok"] else "FAIL"
                print(f"[{marker}] {check['name']}: {check['detail']}")
        if not result["ok"]:
            raise SystemExit(1)
        return

    if args.command == "daemon":
        from .daemon import daemon_status, start_daemon, stop_daemon

        if args.action == "start":
            ok = start_daemon()
            print("Memoria daemon running" if ok else "Memoria daemon failed to start")
        elif args.action == "status":
            ok = daemon_status()
            print("Memoria daemon running" if ok else "Memoria daemon stopped")
        else:
            was_running = daemon_status()
            ok = stop_daemon() if was_running else True
            print("Memoria daemon stopped" if ok else "Memoria daemon failed to stop")
        if not ok:
            raise SystemExit(1)
        return

    llm_call = None if args.no_llm else make_llm_call()

    with Memoria(db_path=args.db, llm_call=llm_call, scope=args.scope) as m:
        if args.command == "remember":
            result = m.remember(args.text, role=args.role, session_id=args.session)
            print(json.dumps(result, indent=2, default=str))

        elif args.command == "recall":
            result = m.recall_formatted(args.query, top_k=args.top_k, mode=args.mode)
            print(result)

        elif args.command == "entity":
            result = m.entity_context(args.name)
            print(json.dumps(result, indent=2, default=str))

        elif args.command == "history":
            result = m.history(args.name, predicate=args.predicate)
            for r in result:
                status = "ACTIVE" if not r.get("valid_until") else "SUPERSEDED"
                ts = time.strftime("%Y-%m-%d", time.localtime(r.get("created_at", 0)))
                obj = r.get("object_value") or r.get("object_id", "")[:8]
                print(f"  [{status}] {ts}: {r['predicate']} → {obj} (conf={r['confidence']:.2f})")

        elif args.command == "consolidate":
            result = m.consolidate(
                prune_threshold=args.prune_threshold,
                half_life_days=args.half_life,
            )
            print(json.dumps(result, indent=2, default=str))

        elif args.command == "stats":
            result = m.graph_stats()
            print(f"Database: {result['db_path']}")
            print(f"Conversations: {result['conversations']}")
            print(f"Entities: {result['entities']}")
            print(f"Triples: {result['triples_active']} active / {result['triples_total']} total")
            print(f"Superseded: {result['triples_superseded']}")
            print(f"Clusters: {result['clusters']}")
            print(f"Spectral gap: {result['spectral_gap']:.6f}")
            print(f"Screening radius: {result['screening_radius']} hops")
            if result["eigenvalues"]:
                print(f"Eigenvalues: {[f'{e:.4f}' for e in result['eigenvalues'][:6]]}")

        elif args.command == "compress":
            result = m.compress(budget_tokens=args.budget)
            print(f"Tier: {result.tier}")
            print(f"Tokens: ~{result.token_estimate} (budget: {args.budget})")
            print(f"Entities: {result.entities_included}/{result.entities_total} ({result.compression_ratio:.1%})")
            print(f"Eigenvectors used: {result.eigenvectors_used}")
            print(f"\n{result.text}")

        elif args.command == "budget":
            report = m.budget_report()
            for tier, info in report.items():
                print(f"\n{'='*40}")
                print(f"{tier} (budget: {info['budget']} tokens)")
                print(f"  Entities: {info['entities_included']}/{info['entities_total']} ({info['compression_ratio']:.1%})")
                print(f"  Tokens: ~{info['token_estimate']}")
                print(f"  Preview: {info['preview']}")

        elif args.command == "cleanup":
            action_map = {
                "list-entities": "list_entities",
                "list-triples": "list_triples",
                "delete-entity": "delete_entity",
                "delete-triple": "delete_triple",
                "merge": "merge_entities",
                "find-duplicates": "find_duplicates",
                "find-orphans": "find_orphans",
                "purge-orphans": "purge_orphans",
                "purge-expired": "purge_expired",
            }
            result = m.cleanup(
                action=action_map[args.action],
                entity_name=args.name,
                entity_id=args.id,
                triple_id=args.id,
                merge_into=args.into,
            )
            if "error" in result:
                print(f"Error: {result['error']}", file=sys.stderr)
                if "candidates" in result:
                    for c in result["candidates"]:
                        print(f"  {c['id'][:12]}  {c['name']}")
            elif args.action == "list-entities":
                for e in result["entities"]:
                    print(f"  {e['name']:45s} triples={e['active_triples']}  conf={e['confidence']}")
                print(f"  ({result['count']} total)")
            elif args.action == "list-triples":
                for t in result["triples"]:
                    print(f"  {t['subject']:30s} --{t['predicate']:25s}--> {t['object']}")
                print(f"  ({result['count']} total)")
            elif args.action == "find-duplicates":
                for d in result["duplicates"]:
                    print(f"  {d[0]['name']!r} <-> {d[1]['name']!r}")
                print(f"  ({result['count']} duplicate pairs)")
            elif args.action == "find-orphans":
                for o in result["orphans"]:
                    print(f"  {o['name']:45s} conf={o['confidence']:.3f}")
                print(f"  ({result['count']} orphans)")
            else:
                print(json.dumps(result, indent=2, default=str))

        elif args.command == "export":
            from pathlib import Path
            import time as _time

            # Get full L3 compression for the export
            result = m.compress(budget_tokens=args.budget)

            if args.format == "json":
                # Structured export with metadata
                export_data = {
                    "exported_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                    "tier": result.tier,
                    "token_estimate": result.token_estimate,
                    "entities_included": result.entities_included,
                    "entities_total": result.entities_total,
                    "compression_ratio": result.compression_ratio,
                    "content": result.text,
                }
                output_text = json.dumps(export_data, indent=2)
                default_name = "MEMORIA_CONTEXT.json"
            else:
                # Markdown export — ready to drop into CLAUDE.md or Codex context
                lines = [
                    "# Memoria Context (Portable Memory Export)",
                    "",
                    f"*Exported: {_time.strftime('%Y-%m-%d %H:%M UTC', _time.gmtime())}*  ",
                    f"*Entities: {result.entities_included}/{result.entities_total} "
                    f"| Tokens: ~{result.token_estimate} | Tier: {result.tier}*",
                    "",
                    "---",
                    "",
                    result.text,
                    "",
                    "---",
                    "*Generated by `memoria export`. This is a compressed snapshot of the "
                    "full knowledge graph — use it as context for remote/async agents.*",
                ]
                output_text = "\n".join(lines)
                default_name = "MEMORIA_CONTEXT.md"

            out_path = Path(args.output) if args.output else Path.cwd() / default_name
            out_path.write_text(output_text)
            print(f"Exported to {out_path}")
            print(f"  Tier: {result.tier}")
            print(f"  Tokens: ~{result.token_estimate} (budget: {args.budget})")
            print(f"  Entities: {result.entities_included}/{result.entities_total}")

        elif args.command == "ingest":
            from pathlib import Path
            p = Path(args.path).expanduser()
            if p.is_file():
                files = [p]
            elif p.is_dir():
                files = sorted(p.glob("**/*.txt")) + sorted(p.glob("**/*.md")) + sorted(p.glob("**/*.json"))
            else:
                print(f"Not found: {p}", file=sys.stderr)
                return

            total = {"entities_added": 0, "triples_added": 0, "files": 0}
            for f in files:
                text = f.read_text(errors="ignore")
                if not text.strip():
                    continue
                result = m.remember(text, session_id=args.session or f.stem)
                total["entities_added"] += result.get("entities_added", 0)
                total["triples_added"] += result.get("triples_added", 0)
                total["files"] += 1
                print(f"  Ingested {f.name}: +{result.get('entities_added', 0)} entities, +{result.get('triples_added', 0)} triples")

            print(f"\nTotal: {total['files']} files, {total['entities_added']} entities, {total['triples_added']} triples")


if __name__ == "__main__":
    main()
