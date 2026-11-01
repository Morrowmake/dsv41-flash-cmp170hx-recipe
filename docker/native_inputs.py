#!/usr/bin/env python3
"""Which wheel libraries the queue makes stale, by the plugin's native-input rules.

  native_inputs.py --src GIT_DIR --upstream U --queue Q --plugin P

The engine is the queue Q on upstream U with the build products of U's wheel. A change
the queue makes to the wheel's native inputs (CMakeLists.txt, cmake, csrc, rust,
setup.py; paths of other platforms left out) leaves the wheel's binaries stale unless
native_build.py rebuilds them. The plugin commit P's vllm_sm80/launch/data/native_inputs.toml
(schema native-inputs/1) says which native each changed input feeds: a changed header
makes stale every native whose sources include it, directly or through other headers, at
U or Q; any other file goes to the first [[stale]] entry whose paths match it; a CMake file
that differs only in comments and in statements of a [[cmake_option]] makes stale only
that option's natives. The rebuildable natives are the [[extensions]] with wheel = true,
each built by the native_build.py option whose wheel target (WHEEL_TARGETS) it names.

Prints one JSON object {changed, stale: {native glob: [paths]}, rebuild: [native_build.py
options], unbuilt: [paths]} and exits 1 (after printing) when a changed input is unbuilt:
an input of a wheel native no option rebuilds, of a plugin native (not in the engine), or
of no rule.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_build  # noqa: E402

RULES = "vllm_sm80/launch/data/native_inputs.toml"
SCHEMA = "native-inputs/1"
WHEEL_NATIVE = ("CMakeLists.txt", "cmake", "csrc", "rust", "setup.py")
OTHER_PLATFORM_TOKENS = {"cpu", "rocm", "rock", "xpu", "tpu", "amd", "hip", "hipify", "ppc64le", "s390x", "zen",
                         "gfx1250", "neuron", "hpu", "arm"}
# Files whose #include lines are followed (C, C++, CUDA), and among them the headers.
INCLUDING = (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hh", ".hpp", ".inc", ".inl")
HEADERS = (".cuh", ".h", ".hh", ".hpp", ".inc", ".inl")
INCLUDE_RX = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^">\n]+)[">]', re.M)
CMAKE_FILE = re.compile(r"(?:^|/)CMakeLists\.txt$|\.cmake$")
CMAKE_COMMAND = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(")
CMAKE_BRACKET = re.compile(r"\[(=*)\[")
CMAKE_TRUE = {"1", "ON", "YES", "TRUE", "Y"}


class Refusal(Exception):
    pass


def git(src, *args, check=True):
    proc = subprocess.run(["git", "--git-dir", str(src), *args], capture_output=True)
    if check and proc.returncode:
        raise Refusal(f"git {' '.join(args)}: {proc.stderr.decode(errors='replace').strip()}")
    return proc


def blob_text(src, commit, rel):
    """REL's text at COMMIT, '' when it has none."""
    proc = git(src, "cat-file", "blob", f"{commit}:{rel}", check=False)
    return proc.stdout.decode("utf-8", "replace") if proc.returncode == 0 else ""


def other_platform(path):
    return bool(OTHER_PLATFORM_TOKENS & set(re.split(r"[/._\-]+", path.lower())))


def glob_regex(pattern):
    """Tree-path glob: no '/' -> basename at any depth; a leading '/' anchors at the tree root; '**' spans zero or
    more components; '*' and '?' stay inside one component."""
    if pattern.startswith("/"):
        pattern = pattern[1:]
    elif "/" not in pattern.rstrip("/"):
        pattern = "**/" + pattern
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("/**", i) and i + 3 == len(pattern):
            out.append("/.*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def include_graph(src, commit, names, include_dirs):
    """{file: [files it includes]} of the INCLUDING files among NAMES (paths of COMMIT), each #include resolved from
    the including file's directory, then each of INCLUDE_DIRS (a name no file of NAMES answers is a system header)."""
    names = set(names)
    wanted = sorted(n for n in names if n.endswith(INCLUDING))
    raw = subprocess.run(["git", "--git-dir", str(src), "cat-file", "--batch"], check=True, capture_output=True,
                         input="".join(f"{commit}:{n}\n" for n in wanted).encode()).stdout
    graph, at = {}, 0
    for rel in wanted:
        end = raw.index(b"\n", at)
        head = raw[at:end].split()
        at = end + 1
        if head[-1] == b"missing":
            continue
        size = int(head[2])
        text = raw[at:at + size].decode("utf-8", "replace")
        at += size + 1
        out = []
        for name in INCLUDE_RX.findall(text):
            for base in (os.path.dirname(rel), *include_dirs):
                found = os.path.normpath(os.path.join(base, name))
                if found in names:
                    out.append(found)
                    break
        graph[rel] = out
    return graph


def reach(graph, seeds):
    """SEEDS and every file they reach in GRAPH."""
    seen, todo = set(), sorted(seeds)
    while todo:
        rel = todo.pop()
        if rel not in seen:
            seen.add(rel)
            todo.extend(graph.get(rel, ()))
    return seen


def _cmake_bracket_end(text, at):
    """The index after the bracket argument/comment `[=*[ ... ]=*]` starting at AT (len(text) when unclosed)."""
    m = CMAKE_BRACKET.match(text, at)
    end = text.find("]" + m.group(1) + "]", m.end())
    return len(text) if end < 0 else end + len(m.group(1)) + 2


def cmake_statements(text):
    """The command invocations of a CMake file, comments dropped: [(command name lowercased, (argument tokens...))],
    a quoted or bracket argument one token verbatim, unquoted ones split at whitespace and parentheses."""
    out, at, n = [], 0, len(text)
    while at < n:
        if text[at] == "#":
            if CMAKE_BRACKET.match(text, at + 1):
                at = _cmake_bracket_end(text, at + 1)
            else:
                end = text.find("\n", at)
                at = n if end < 0 else end
            continue
        m = CMAKE_COMMAND.match(text, at)
        if not m or (at and (text[at - 1].isalnum() or text[at - 1] == "_")):
            at += 1
            continue
        at, depth, args, word = m.end(), 1, [], ""
        while at < n and depth:
            c = text[at]
            if c == '"':
                end = at + 1
                while end < n and text[end] != '"':
                    end += 2 if text[end] == "\\" else 1
                args.append(text[at:end + 1])
                at = end + 1
                continue
            if c == "[" and not word and CMAKE_BRACKET.match(text, at):
                end = _cmake_bracket_end(text, at)
                args.append(text[at:end])
                at = end
                continue
            if c == "#":
                if CMAKE_BRACKET.match(text, at + 1):
                    at = _cmake_bracket_end(text, at + 1)
                else:
                    end = text.find("\n", at)
                    at = n if end < 0 else end
                continue
            if c.isspace() or c in "()":
                if word:
                    args.append(word)
                    word = ""
                depth += {"(": 1, ")": -1}.get(c, 0)
                if c in "()" and depth:
                    args.append(c)
            else:
                word += c
            at += 1
        out.append((m.group(1).lower(), tuple(args)))
    return out


def cmake_option_split(text, tokens):
    """(kept, removed): the statements of a CMake file, removed those of the fork options TOKENS: each if() block
    without else/elseif whose condition names one (whole words), and each other command whose first argument is one
    (set/option/unset of it) and that sets no true constant (the option stays off)."""
    rx = re.compile(r"(?<![A-Za-z0-9_])(?:%s)(?![A-Za-z0-9_])" % "|".join(map(re.escape, tokens)))

    def names(args):
        return any(rx.search(a) for a in args)

    def defines(args):
        return bool(args) and args[0] in tokens and not any(a.strip('"').upper() in CMAKE_TRUE for a in args[1:])
    statements = cmake_statements(text)
    kept, removed, at = [], [], 0
    while at < len(statements):
        name, args = statements[at]
        if name == "if" and names(args):
            depth, end, branches = 0, at, False
            while end < len(statements):
                inner = statements[end][0]
                depth += {"if": 1, "endif": -1}.get(inner, 0)
                branches |= depth == 1 and inner in ("elseif", "else")
                if not depth:
                    break
                end += 1
            if not branches and end < len(statements):
                removed.extend(statements[at:end + 1])
                at = end + 1
                continue
        elif name not in ("if", "elseif", "else", "endif") and defines(args):
            removed.append(statements[at])
            at += 1
            continue
        kept.append(statements[at])
        at += 1
    return kept, removed


def cmake_option_natives(src, base, cand, rel, options):
    """The natives a change of CMake file REL makes stale by the [[cmake_option]] OPTIONS: [] for a change of
    comments only, the `native` of each entry whose statements differ, or None when other statements differ."""
    tokens = [t for option in options for t in option["tokens"]]
    if not tokens:
        return None
    old, new = blob_text(src, base, rel), blob_text(src, cand, rel)
    if cmake_option_split(old, tokens)[0] != cmake_option_split(new, tokens)[0]:
        return None
    return [option["native"] for option in options
            if cmake_option_split(old, option["tokens"])[1] != cmake_option_split(new, option["tokens"])[1]]


def stale_natives(rule):
    native = rule["native"]
    return [native] if isinstance(native, str) else list(native)


def load_rules(src, plugin):
    text = blob_text(src, plugin, RULES)
    if not text:
        raise Refusal(f"the plugin {plugin[:10]} has no native-input rules ({RULES})")
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise Refusal(f"{plugin[:10]}:{RULES}: not TOML ({error})") from None
    if doc.get("schema") != SCHEMA or not doc.get("stale"):
        raise Refusal(f"{plugin[:10]}:{RULES}: not {SCHEMA} with [[stale]] rules")
    return doc


def native_staleness(src, base, cand, changed, doc):
    """{'natives': {native glob: [paths]}, 'wheel': [paths of wheel natives no option builds], 'unattributed':
    [paths no rule names]} of the CHANGED paths between BASE and CAND (module doc)."""
    rules = doc["stale"]
    rxs = [[glob_regex(p) for p in rule["paths"]] for rule in rules]

    def owner(rel):
        return next((i for i, rx in enumerate(rxs) if any(r.match(rel) for r in rx)), None)

    readers = {}
    if any(rel.endswith(HEADERS) for rel in changed):
        for commit in (base, cand):
            names = [n for n in git(src, "ls-tree", "-r", "--name-only", commit).stdout.decode().split("\n")
                     if n and owner(n) is not None]
            graph = include_graph(src, commit, names, doc.get("include_dirs") or [])
            for i, rule in enumerate(rules):
                seeds = [n for n in names if not n.endswith(HEADERS) and owner(n) == i] + \
                    [n for n in rule.get("generated") or () if n in graph]
                for rel in reach(graph, seeds):
                    readers.setdefault(rel, set()).add(i)
    out = {"natives": {}, "wheel": [], "unattributed": []}

    def stale(glob, rel):
        paths = out["natives"].setdefault(glob, [])
        if rel not in paths:
            paths.append(rel)

    for rel in changed:
        option_natives = (cmake_option_natives(src, base, cand, rel, doc.get("cmake_option") or [])
                          if CMAKE_FILE.search(rel) else None)
        if option_natives is not None:
            for glob in option_natives:
                stale(glob, rel)
            continue
        found = sorted(readers.get(rel, ())) if rel.endswith(HEADERS) else []
        if not found and owner(rel) is not None:
            found = [owner(rel)]
        if not found:
            out["unattributed"].append(rel)
        elif any("wheel" in stale_natives(rules[i]) for i in found):
            out["wheel"].append(rel)
        else:
            for i in found:
                for glob in stale_natives(rules[i]):
                    stale(glob, rel)
    return out


def rebuild_options(doc):
    """{native glob: native_build.py option} of the rules' wheel = true natives; Refusal for one no option builds."""
    options = {}
    for entry in doc.get("extensions") or ():
        if not entry.get("wheel"):
            continue
        rx = glob_regex(entry["artifact"])
        found = [dest for dest, (rel, _) in native_build.WHEEL_TARGETS.items() if rx.match(rel)]
        if len(found) != 1:
            raise Refusal(f"{RULES} marks wheel = true {entry['artifact']}, which no native_build.py option rebuilds")
        options[entry["artifact"]] = "--" + found[0].replace("_", "-")
    return options


def decide(src, upstream, queue, plugin):
    names = git(src, "diff", "--name-only", "-z", upstream, queue, "--", *WHEEL_NATIVE).stdout.decode().split("\0")
    changed = sorted(p for p in set(names) - {""} if not other_platform(p))
    result = {"changed": changed, "stale": {}, "rebuild": [], "unbuilt": []}
    if not changed:
        return result
    doc = load_rules(src, plugin)
    options = rebuild_options(doc)
    stale = native_staleness(src, upstream, queue, changed, doc)
    unbuilt = set(stale["wheel"] + stale["unattributed"])
    unbuilt.update(p for glob, paths in stale["natives"].items() if glob not in options for p in paths)
    result.update(stale=stale["natives"], unbuilt=sorted(unbuilt),
                  rebuild=sorted({options[glob] for glob in stale["natives"] if glob in options}))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("src", "upstream", "queue", "plugin"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args(argv)
    try:
        result = decide(Path(a.src), a.upstream, a.queue, a.plugin)
    except Refusal as error:
        print(f"[native-inputs] FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    if result["unbuilt"]:
        print("[native-inputs] FAIL: the queue changes native inputs of the upstream wheel that no native_build.py "
              "option rebuilds (the wheel's build products would be stale): " + ", ".join(result["unbuilt"]),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
