"""Deterministic entity deduplication.

Strategy (no LLM, fast, conservative):
1. Group by (type, normalised name).
2. Within a type, build a Union-Find over entity_ids:
   - Connect two entities A, B if name(A) is in alias(B) or vice versa.
   - Connect if their normalised names match after aggressive normalisation
     (lowercased, punctuation stripped, ASCII fullwidth folded).
3. The canonical id is the one with the highest mention-count.
4. Rewrite qa_entity_mention.entity_id and relation.{head_id,tail_id}, then
   delete duplicate entity rows. Aliases are merged onto the canonical entity.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from loguru import logger

from .storage import pg_conn


_NORM_PUNCT = re.compile(r"[\s　\-_/\.,，。、（）()『』\[\]【】《》<>{}«»\"'`~!@#$%^&*+=|\\;:?]+")


def _normalise(name: str) -> str:
    if not name:
        return ""
    # Unicode normalize + fold fullwidth to halfwidth
    s = unicodedata.normalize("NFKC", name)
    s = s.lower()
    s = _NORM_PUNCT.sub("", s)
    return s


# --- Union-Find ----------------------------------------------------------

class UF:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != root:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        self.parent[rb] = ra


# --- Main ----------------------------------------------------------------

def dedup(dry_run: bool = False) -> dict:
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT entity_id, name, type, aliases FROM entity")
        rows = cur.fetchall()
        cur.execute(
            "SELECT entity_id, COUNT(*) AS c FROM qa_entity_mention GROUP BY entity_id"
        )
        mention_counts = {r["entity_id"]: int(r["c"]) for r in cur.fetchall()}

    logger.info("loaded {} entities, mention rows for {} of them", len(rows), len(mention_counts))

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    uf = UF()
    # Pre-register every entity_id
    for r in rows:
        uf.find(r["entity_id"])

    # Within each type build union by:
    #  (a) normalised name match
    #  (b) one entity's name appears in another's aliases
    for etype, lst in by_type.items():
        # Index: norm_name -> list of entity_id
        name_to_ids: dict[str, list[str]] = defaultdict(list)
        # Index: alias norm -> list of entity_id where it appears
        alias_to_ids: dict[str, list[str]] = defaultdict(list)

        for r in lst:
            nn = _normalise(r["name"])
            if nn:
                name_to_ids[nn].append(r["entity_id"])
            for a in r.get("aliases") or []:
                an = _normalise(a)
                if an and an != nn:
                    alias_to_ids[an].append(r["entity_id"])

        # (a) merge by exact normalised name
        for nn, ids in name_to_ids.items():
            for i in range(1, len(ids)):
                uf.union(ids[0], ids[i])

        # (b) merge entities whose name (normalised) is an alias of another
        for r in lst:
            nn = _normalise(r["name"])
            if not nn:
                continue
            if nn in alias_to_ids:
                for other in alias_to_ids[nn]:
                    if other != r["entity_id"]:
                        uf.union(r["entity_id"], other)
            # also if r's alias hits another's name
            for a in r.get("aliases") or []:
                an = _normalise(a)
                if not an:
                    continue
                if an in name_to_ids:
                    for other in name_to_ids[an]:
                        if other != r["entity_id"]:
                            uf.union(r["entity_id"], other)

    # Compute canonical id per cluster: highest mention count, tiebreak shortest name
    clusters: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        root = uf.find(r["entity_id"])
        clusters[root].append(r)

    canonical_map: dict[str, str] = {}
    new_aliases: dict[str, list[str]] = {}

    for root, members in clusters.items():
        if len(members) <= 1:
            canonical_map[members[0]["entity_id"]] = members[0]["entity_id"]
            continue
        # pick canonical: max mentions, then shortest name
        members.sort(
            key=lambda r: (
                -mention_counts.get(r["entity_id"], 0),
                len(r["name"]),
                r["entity_id"],
            )
        )
        canon = members[0]
        all_aliases: set[str] = set()
        for m in members:
            canonical_map[m["entity_id"]] = canon["entity_id"]
            all_aliases.update(m.get("aliases") or [])
            all_aliases.add(m["name"])
        new_aliases[canon["entity_id"]] = sorted(all_aliases)

    n_clusters = len(clusters)
    n_merged = sum(1 for r in rows if canonical_map.get(r["entity_id"]) != r["entity_id"])
    logger.info(
        "found {} clusters from {} entities; {} entities will be merged into canonicals",
        n_clusters, len(rows), n_merged,
    )

    if dry_run:
        return {
            "n_entities_before": len(rows),
            "n_clusters": n_clusters,
            "n_merged": n_merged,
            "n_after": n_clusters,
        }

    # Apply: rewrite qa_entity_mention + relation, then delete dup entities + update canon aliases
    with pg_conn() as conn, conn.cursor() as cur:
        # 1. update mentions to point to canonical
        rewrite_pairs = [
            (canon, old) for old, canon in canonical_map.items() if old != canon
        ]
        if rewrite_pairs:
            cur.executemany(
                """
                INSERT INTO qa_entity_mention (qa_id, entity_id, surface)
                SELECT qa_id, %s, surface FROM qa_entity_mention WHERE entity_id = %s
                ON CONFLICT DO NOTHING
                """,
                rewrite_pairs,
            )
            cur.executemany(
                "DELETE FROM qa_entity_mention WHERE entity_id = %s",
                [(old,) for _, old in rewrite_pairs],
            )

            # 2. update relation head/tail
            cur.executemany(
                "UPDATE relation SET head_id = %s WHERE head_id = %s",
                rewrite_pairs,
            )
            cur.executemany(
                "UPDATE relation SET tail_id = %s WHERE tail_id = %s",
                rewrite_pairs,
            )

            # 3. delete dup entities
            cur.executemany(
                "DELETE FROM entity WHERE entity_id = %s",
                [(old,) for _, old in rewrite_pairs],
            )

        # 4. update canonical aliases
        if new_aliases:
            cur.executemany(
                "UPDATE entity SET aliases = %s WHERE entity_id = %s",
                [(aliases, eid) for eid, aliases in new_aliases.items()],
            )
        conn.commit()

    logger.info("dedup committed; entity count after: {}", n_clusters)
    return {
        "n_entities_before": len(rows),
        "n_clusters": n_clusters,
        "n_merged": n_merged,
        "n_after": n_clusters,
    }
