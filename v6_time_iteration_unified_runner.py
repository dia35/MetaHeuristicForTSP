"""
Sezgisel Yontemler TSP GA - V6 Time & Iteration Unified Runner

Bu script, V5 temelinde kritik metodolojik duzeltmeler yapar:
1. Her stage icin ayri sure siniri (maxtime_limit) ve ayri iterasyon siniri (max_iterations_limit).
2. Her run hem sure hem iterasyon siniriyla kontrol edilir; hangisine once ulasirsa durur.
3. stop_reason alani: TIME_LIMIT, ITERATION_LIMIT, BOTH_LIMITS, ERROR.
4. Final stage runlarinda best_tour_sequence ve convergence_history kaydedilir.
5. Gorseller (route, convergence, aggregate rank) tamamen final stage raw runlarindan uretilir.
   Ayri final_visual_runs veya rerun yapilmaz.
6. Eski V5/Fix1 ciktilarina dokunulmaz.

Checkpoint destegi mevcuttur (her 50 run'da bir).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

DATASETS = [
    "berlin52.txt", "ch130.txt", "d493.txt", "eil101.txt", "eil51.txt",
    "eil76.txt", "kroA100.txt", "pcb442.txt", "pr299.txt"
]
SEEDS = [11, 42, 123, 2026, 9999]

CHECKPOINT_INTERVAL = 50
MAX_RETRIES = 3

# Stage bazli sure ve iterasyon limitleri
STAGE_LIMITS = {
    "fast_param_grid_all_dataset":  {"maxtime_limit": 1.0,  "max_iterations_limit": 5000},
    "fast_top10_all_dataset":       {"maxtime_limit": 5.0,  "max_iterations_limit": 20000},
    "fast_selection_all_dataset":   {"maxtime_limit": 5.0,  "max_iterations_limit": 20000},
    "fast_crossover_all_dataset":   {"maxtime_limit": 5.0,  "max_iterations_limit": 20000},
    "fast_mutation_all_dataset":    {"maxtime_limit": 5.0,  "max_iterations_limit": 20000},
    "fast_final_all_dataset":       {"maxtime_limit": 60.0, "max_iterations_limit": 100000},
}

SELECTION_CASES = [
    {"case_id": "SEL_roulette", "selection_operator": "roulette", "tournament_size": 2, "label": "roulette"},
    {"case_id": "SEL_tournament_ts2", "selection_operator": "tournament", "tournament_size": 2, "label": "tournament ts=2"},
    {"case_id": "SEL_tournament_ts3", "selection_operator": "tournament", "tournament_size": 3, "label": "tournament ts=3"},
    {"case_id": "SEL_tournament_ts5", "selection_operator": "tournament", "tournament_size": 5, "label": "tournament ts=5"},
    {"case_id": "SEL_tournament_ts7", "selection_operator": "tournament", "tournament_size": 7, "label": "tournament ts=7"},
    {"case_id": "SEL_tournament_ts10", "selection_operator": "tournament", "tournament_size": 10, "label": "tournament ts=10"},
]

# =============================================================================
# CORE GA
# =============================================================================

def load_dataset(dataset_name: str, dataset_dir: Path) -> np.ndarray:
    path = dataset_dir / dataset_name
    if not path.exists():
        raise FileNotFoundError(f"Veri seti bulunamadi: {path}")
    return np.loadtxt(path)

def distance_matrix(data: np.ndarray) -> np.ndarray:
    xy = data[:, 1:3]
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    return np.rint(dist)

def fitness(seq: np.ndarray, distance: np.ndarray) -> float:
    seq = np.asarray(seq, dtype=int)
    return float(np.sum(distance[seq, np.roll(seq, -1)]))

def is_permutation(arr: np.ndarray) -> bool:
    arr = np.asarray(arr, dtype=int)
    return np.array_equal(np.sort(arr), np.arange(len(arr)))

def tournament_selection(POP: np.ndarray, tournament_size: int) -> np.ndarray:
    ps = len(POP)
    size = min(max(1, tournament_size), ps)
    idx = np.random.choice(ps, size=size, replace=False)
    winner_idx = idx[np.argmin(POP[idx, -1])]
    return POP[winner_idx, :-1].astype(int).copy()

def roulette_selection(POP: np.ndarray) -> np.ndarray:
    ps = len(POP)
    ftn = POP[:, -1].astype(float)
    inv_fitness = 1.0 / (ftn + 1.0)
    total = float(np.sum(inv_fitness))
    if not np.isfinite(total) or total <= 0:
        idx = np.random.choice(ps)
    else:
        probs = inv_fitness / total
        idx = np.random.choice(ps, p=probs)
    return POP[idx, :-1].astype(int).copy()

def select_parent(POP: np.ndarray, sel_op: str, ts: int) -> np.ndarray:
    if sel_op == "tournament": return tournament_selection(POP, ts)
    if sel_op == "roulette": return roulette_selection(POP)
    raise ValueError(f"Bilinmeyen selection: {sel_op}")

def pmx_crossover(p1, p2):
    n = p1.size
    c1, c2 = np.sort(np.random.choice(n, 2, replace=False))
    child = np.full(n, -1, dtype=int)
    child[c1:c2+1] = p2[c1:c2+1]
    map_to_p1 = {int(a): int(b) for a, b in zip(p2[c1:c2+1], p1[c1:c2+1])}
    seg_set = set(map_to_p1.keys())
    for i in range(n):
        if c1 <= i <= c2: continue
        x = int(p1[i])
        while x in seg_set: x = map_to_p1[x]
        child[i] = x
    return child

def ox_crossover(p1, p2):
    n = p1.size
    c1, c2 = np.sort(np.random.choice(n, 2, replace=False))
    child = np.full(n, -1, dtype=int)
    child[c1:c2+1] = p1[c1:c2+1]
    p1_seg = set(p1[c1:c2+1])
    p2_fil = [v for v in p2 if v not in p1_seg]
    fidx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = p2_fil[fidx]
            fidx += 1
    return child

def scx_crossover(p1, p2, distance):
    n = p1.size
    child = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    curr = p1[0]
    child[0] = curr
    visited[curr] = True
    for i in range(1, n):
        idx1 = np.where(p1 == curr)[0][0]
        n1 = p1[(idx1 + 1) % n]
        if visited[n1]: n1 = next(c for c in p1 if not visited[c])
        idx2 = np.where(p2 == curr)[0][0]
        n2 = p2[(idx2 + 1) % n]
        if visited[n2]: n2 = next(c for c in p2 if not visited[c])
        nxt = n1 if distance[curr, n1] < distance[curr, n2] else n2
        child[i] = nxt
        visited[nxt] = True
        curr = nxt
    return child

def edgeEx_crossover(p1, p2):
    p1_l = np.asarray(p1, dtype=int).tolist()
    p2_l = np.asarray(p2, dtype=int).tolist()
    n = len(p1_l)
    off = p1_l.copy()
    u_idx = np.random.randint(0, n - 1)
    u, v1 = off[u_idx], off[u_idx + 1]
    for _ in range(n):
        u_idx_p2 = p2_l.index(u)
        v2 = p2_l[0] if u_idx_p2 == n - 1 else p2_l[u_idx_p2 + 1]
        if v1 == v2: break
        v1_idx, v2_idx = off.index(v1), off.index(v2)
        if v1_idx < v2_idx:
            off[v1_idx:v2_idx+1] = list(reversed(off[v1_idx:v2_idx+1]))
        else:
            temp = off[v1_idx:] + off[:v2_idx+1]
            temp.reverse()
            sp = len(off[v1_idx:])
            off[v1_idx:] = temp[:sp]
            off[:v2_idx+1] = temp[sp:]
        u = v2
        u_idx_off = off.index(u)
        v1 = off[0] if u_idx_off == n - 1 else off[u_idx_off + 1]
    return np.array(off, dtype=int)

def obx_crossover(p1, p2):
    n = len(p1)
    idx = np.random.choice(n, size=max(1, n // 3), replace=False)
    child = np.full(n, -1, dtype=int)
    p2_rem = [x for x in p2 if x not in p1[idx]]
    child[idx] = p1[idx]
    child[child == -1] = p2_rem
    return child

CROSSOVER_FUNCTIONS = {"PMX": pmx_crossover, "OX": ox_crossover, "SCX": scx_crossover, "EdgeEx": edgeEx_crossover, "OBX": obx_crossover}

def apply_crossover(p1, p2, distance, op, mixed_idx=0):
    names = list(CROSSOVER_FUNCTIONS.keys())
    op_name = names[mixed_idx % len(names)] if op == "mixed" else op
    func = CROSSOVER_FUNCTIONS[op_name]
    child = func(p1, p2, distance) if op_name == "SCX" else func(p1, p2)
    return child, op_name

def twors_mutation(seq):
    mut = seq.copy()
    i, j = np.random.choice(mut.size, 2, replace=False)
    mut[i], mut[j] = mut[j], mut[i]
    return mut

def reverseSeq_mutation(seq):
    mut = seq.copy()
    i, j = sorted(np.random.choice(len(mut), 2, replace=False))
    mut[i:j+1] = mut[i:j+1][::-1]
    return mut

def thrors_mutation(seq):
    mut = seq.copy()
    i, j, k = sorted(np.random.choice(len(mut), 3, replace=False))
    mut[j], mut[k], mut[i] = mut[i], mut[j], mut[k]
    return mut

def centreInv_mutation(seq):
    mut = seq.copy()
    sp = np.random.randint(1, len(mut))
    mut[:sp], mut[sp:] = mut[:sp][::-1], mut[sp:][::-1]
    return mut

def throas_mutation(seq):
    mut = seq.copy()
    i = np.random.randint(0, len(mut) - 2)
    a, b, c = mut[i:i+3]
    mut[i:i+3] = [c, a, b]
    return mut

MUTATION_FUNCTIONS = {"TWORS": twors_mutation, "ReverseSeq": reverseSeq_mutation, "THRORS": thrors_mutation, "CentreInv": centreInv_mutation, "THROAS": throas_mutation}

def apply_mutation(child, op, mixed_start):
    if op == "none": return child, "none", mixed_start
    if op != "mixed":
        mut = MUTATION_FUNCTIONS[op](child.copy())
        if is_permutation(mut): return mut, op, mixed_start
        return child, "invalid_"+op, mixed_start
    names = list(MUTATION_FUNCTIONS.keys())
    for k in range(len(names)):
        op_name = names[(mixed_start + k) % len(names)]
        mut = MUTATION_FUNCTIONS[op_name](child.copy())
        if is_permutation(mut):
            return mut, op_name, (mixed_start + k + 1) % len(names)
    return child, "mixed_no_valid", mixed_start

def initialize_population(data, dist, pop_size):
    n = data.shape[0]
    POP = np.zeros((pop_size, n + 1), dtype=float)
    for i in range(pop_size):
        seq = np.random.permutation(n)
        POP[i, 0:n] = seq
        POP[i, n] = fitness(seq, dist)
    return POP[np.argsort(POP[:, -1])]

def run_ga(
    dataset_dir: Path, dataset: str, seed: int,
    population_size: int, crossover_rate: float, elitism_rate: float, mutation_rate: float,
    selection_operator: str, tournament_size: int,
    crossover_operator: str, mutation_operator: str,
    maxtime_limit: float, max_iterations_limit: int,
    record_sequence: bool = False,
    record_convergence: bool = False,
) -> dict[str, Any]:
    """
    GA runner with dual stopping criteria (time AND iteration limits).
    
    Args:
        record_sequence: True ise best_tour_sequence kaydedilir (final stage icin).
        record_convergence: True ise convergence_history kaydedilir (final stage icin).
    
    Returns:
        dict with: best_tour_length, elapsed_seconds, iterations_completed,
                   stop_reason, best_tour_sequence (optional), convergence_history (optional)
    """
    np.random.seed(seed)
    random.seed(seed)
    data = load_dataset(dataset, dataset_dir)
    dist = distance_matrix(data)
    n = data.shape[0]

    ps = max(4, int(population_size))
    esize = max(1, min(int(ps * elitism_rate), ps - 2))
    osize = max(1, min(int(ps * crossover_rate), ps - esize))

    POP = initialize_population(data, dist, ps)
    best_fit = float(POP[0, -1])
    best_seq = POP[0, :-1].astype(int).copy() if record_sequence else None
    
    convergence_history = [] if record_convergence else None
    if record_convergence:
        convergence_history.append({"iteration": 0, "best_tour_length": best_fit})

    start_time = time.monotonic()
    it = 0
    time_hit = False
    iter_hit = False

    while True:
        elapsed = time.monotonic() - start_time
        time_hit = elapsed >= maxtime_limit
        iter_hit = it >= max_iterations_limit
        
        if time_hit or iter_hit:
            break
        
        it += 1
        OFFS = np.empty((osize, n + 1), dtype=float)
        shuf = np.random.permutation(osize)
        mut_start = np.random.randint(len(MUTATION_FUNCTIONS))

        for i in range(osize):
            p1 = select_parent(POP, selection_operator, tournament_size)
            p2 = select_parent(POP, selection_operator, tournament_size)
            child, _ = apply_crossover(p1, p2, dist, crossover_operator, int(shuf[i]))
            if not is_permutation(child): child = np.random.permutation(n)
            if np.random.rand() < mutation_rate:
                child, _, mut_start = apply_mutation(child, mutation_operator, mut_start)
            if not is_permutation(child): child = np.random.permutation(n)
            OFFS[i, 0:n] = child
            OFFS[i, n] = fitness(child, dist)

        POP[0:esize, :] = POP[0:esize, :]
        POP[esize:esize+osize, :] = OFFS
        for j in range(esize+osize, ps):
            seq = np.random.permutation(n)
            POP[j, 0:n] = seq
            POP[j, n] = fitness(seq, dist)

        POP = POP[np.argsort(POP[:, -1])]
        current_best = float(POP[0, -1])
        if current_best < best_fit:
            best_fit = current_best
            if record_sequence:
                best_seq = POP[0, :-1].astype(int).copy()
        
        if record_convergence:
            convergence_history.append({"iteration": it, "best_tour_length": best_fit})

    elapsed_final = round(time.monotonic() - start_time, 4)
    
    # Determine stop_reason
    if time_hit and iter_hit:
        stop_reason = "BOTH_LIMITS"
    elif time_hit:
        stop_reason = "TIME_LIMIT"
    elif iter_hit:
        stop_reason = "ITERATION_LIMIT"
    else:
        stop_reason = "UNKNOWN"

    result = {
        "best_tour_length": best_fit,
        "elapsed_seconds": elapsed_final,
        "iterations_completed": it,
        "stop_reason": stop_reason,
    }
    
    if record_sequence and best_seq is not None:
        result["best_tour_sequence"] = best_seq.tolist()
    
    if record_convergence and convergence_history is not None:
        result["convergence_history"] = convergence_history
    
    return result


# =============================================================================
# RUNNER & RANKING LOGIC
# =============================================================================

class V6UnifiedRunner:
    def __init__(self, dataset_dir: Path, output_dir: Path,
                 stage_limits: dict = None, datasets: list = None, seeds: list = None):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.fig_dir = output_dir / "figures"
        self.chk_dir = output_dir / "checkpoints"
        self.final_data_dir = output_dir / "final_data"
        
        self.chk_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.final_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.stage_limits = stage_limits or STAGE_LIMITS
        self.datasets = datasets or DATASETS
        self.seeds = seeds or SEEDS
        
        self.log_file = output_dir / "pipeline_progress_log.txt"
        self.all_raw = []
        self.completed_runs_map = {}
        self.run_counter = 0
        
        self.chk_file = self.chk_dir / "v6_unified_checkpoint.json"
        if self.chk_file.exists():
            try:
                self.all_raw = json.loads(self.chk_file.read_text(encoding="utf-8"))
                for r in self.all_raw:
                    k = self._get_run_hash(r["stage_name"], r["dataset"], r["seed"], r)
                    self.completed_runs_map[k] = r
                self.log(f"Checkpoint yuklendi: {len(self.all_raw)} runs")
            except Exception as e:
                self.log(f"Checkpoint okuma hatasi: {e}")

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _get_run_hash(self, stage, ds, seed, cfg):
        return f"{stage}|{ds}|{seed}|{cfg['population_size']}|{cfg['crossover_rate']}|{cfg['elitism_rate']}|{cfg['mutation_rate']}|{cfg['selection_operator']}|{cfg.get('tournament_size',2)}|{cfg['crossover_operator']}|{cfg['mutation_operator']}"

    def _execute_run(self, stage: str, exp_group: str, param_group: str, op_label: str,
                     ds: str, seed: int, cfg: dict) -> dict:
        """Execute a single GA run with stage-specific time and iteration limits."""
        
        h = self._get_run_hash(stage, ds, seed, cfg)
        if h in self.completed_runs_map:
            return self.completed_runs_map[h]
        
        self.run_counter += 1
        run_id = f"{stage}_RUN_{len(self.all_raw)+1:06d}"
        
        limits = self.stage_limits.get(stage, {"maxtime_limit": 5.0, "max_iterations_limit": 20000})
        maxtime_limit = limits["maxtime_limit"]
        max_iterations_limit = limits["max_iterations_limit"]
        
        is_final = (stage == "fast_final_all_dataset")
        
        row = {
            "run_id": run_id,
            "stage_name": stage,
            "dataset": ds,
            "seed": seed,
            "experiment_group_id": exp_group,
            "parameter_group_id": param_group,
            "operator_label": op_label,
            "population_size": cfg["population_size"],
            "crossover_rate": cfg["crossover_rate"],
            "elitism_rate": cfg["elitism_rate"],
            "mutation_rate": cfg["mutation_rate"],
            "selection_operator": cfg["selection_operator"],
            "tournament_size": cfg.get("tournament_size", 2),
            "crossover_operator": cfg["crossover_operator"],
            "mutation_operator": cfg["mutation_operator"],
            "initial_solution_type": "random_permutation",
            "maxtime_limit": maxtime_limit,
            "max_iterations_limit": max_iterations_limit,
            "elapsed_seconds": 0.0,
            "iterations_completed": 0,
            "stop_reason": "",
            "best_tour_length": float('inf'),
            "status": "PENDING",
            "error_message": "",
        }
        
        if is_final:
            row["best_tour_sequence_path"] = ""
            row["convergence_history_path"] = ""
        
        try:
            res = run_ga(
                dataset_dir=self.dataset_dir,
                dataset=ds,
                seed=seed,
                population_size=cfg["population_size"],
                crossover_rate=cfg["crossover_rate"],
                elitism_rate=cfg["elitism_rate"],
                mutation_rate=cfg["mutation_rate"],
                selection_operator=cfg["selection_operator"],
                tournament_size=cfg.get("tournament_size", 2),
                crossover_operator=cfg["crossover_operator"],
                mutation_operator=cfg["mutation_operator"],
                maxtime_limit=maxtime_limit,
                max_iterations_limit=max_iterations_limit,
                record_sequence=is_final,
                record_convergence=is_final,
            )
            
            row["elapsed_seconds"] = res["elapsed_seconds"]
            row["iterations_completed"] = res["iterations_completed"]
            row["stop_reason"] = res["stop_reason"]
            row["best_tour_length"] = res["best_tour_length"]
            row["status"] = "OK"
            
            # Final stage: save sequence and convergence to files
            if is_final:
                ds_stem = ds.replace(".txt", "")
                
                seq_path = self.final_data_dir / f"tour_sequence_{ds_stem}_seed{seed}.json"
                seq_path.write_text(json.dumps(res.get("best_tour_sequence", [])), encoding="utf-8")
                row["best_tour_sequence_path"] = str(seq_path)
                
                conv_path = self.final_data_dir / f"convergence_{ds_stem}_seed{seed}.json"
                conv_path.write_text(json.dumps(res.get("convergence_history", [])), encoding="utf-8")
                row["convergence_history_path"] = str(conv_path)
                
        except Exception as e:
            row["status"] = "ERROR"
            row["error_message"] = str(e)
            row["stop_reason"] = "ERROR"
        
        self.all_raw.append(row)
        self.completed_runs_map[h] = row
        
        if self.run_counter % CHECKPOINT_INTERVAL == 0:
            self._save_checkpoint()
        
        return row

    def _save_checkpoint(self):
        """Save checkpoint to disk."""
        try:
            self.chk_file.write_text(json.dumps(self.all_raw, default=str), encoding="utf-8")
        except Exception as e:
            self.log(f"Checkpoint yazma hatasi: {e}")

    def get_aggregate_rank(self, runs: list[dict], group_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.DataFrame(runs)
        df["best_tour_length"] = df["best_tour_length"].astype(float)
        
        ds_ranks = []
        for ds, ds_grp in df.groupby("dataset"):
            g = ds_grp.groupby(group_key).agg(
                mean_tour=("best_tour_length", "mean"),
                best_tour=("best_tour_length", "min"),
                std_tour=("best_tour_length", lambda x: np.std(x, ddof=0)),
                worst_tour=("best_tour_length", "max"),
                n_runs=("best_tour_length", "count")
            ).reset_index()
            
            g = g.sort_values(["mean_tour", "best_tour", "std_tour", "worst_tour"],
                              ascending=[True, True, True, True]).reset_index(drop=True)
            g["rank_in_dataset"] = g.index + 1
            g["dataset"] = ds
            ds_ranks.append(g)
        
        df_ds = pd.concat(ds_ranks, ignore_index=True)
        
        agg = df_ds.groupby(group_key).agg(
            average_rank=("rank_in_dataset", "mean"),
            mean_of_means=("mean_tour", "mean"),
            best_overall=("best_tour", "min"),
            mean_std=("std_tour", "mean"),
            worst_overall=("worst_tour", "max"),
            datasets_tested=("dataset", "count")
        ).reset_index()
        
        agg = agg.sort_values(["average_rank", "mean_of_means", "best_overall", "mean_std", "worst_overall"],
                              ascending=[True, True, True, True, True]).reset_index(drop=True)
        return df_ds, agg

    # =========================================================================
    # STAGES
    # =========================================================================
    
    def run_stage_param_grid(self) -> list[str]:
        stage = "fast_param_grid_all_dataset"
        self.log(f"--- STAGE: {stage} ---")
        runs = []
        for ps in [50, 100, 200]:
            for cr in [0.6, 0.7, 0.8]:
                for er in [0.1, 0.2, 0.3]:
                    if cr + er > 1.0: continue
                    for mr in [0.05, 0.1]:
                        pid = f"PG_ps{ps}_cr{cr}_er{er}_mr{mr}".replace(".", "p")
                        cfg = {
                            "population_size": ps, "crossover_rate": cr, "elitism_rate": er, "mutation_rate": mr,
                            "selection_operator": "tournament", "tournament_size": 2,
                            "crossover_operator": "mixed", "mutation_operator": "mixed"
                        }
                        for ds in self.datasets:
                            for seed in self.seeds:
                                r = self._execute_run(stage, "STAGE_1_PARAM", pid, pid, ds, seed, cfg)
                                runs.append(r)
                                
        self._save_checkpoint()
        ds_ranks, agg_ranks = self.get_aggregate_rank(runs, "parameter_group_id")
        self.stage1_ds = ds_ranks
        self.stage1_agg = agg_ranks
        self.log(f"  Param grid tamamlandi: {len(runs)} run")
        return agg_ranks["parameter_group_id"].head(10).tolist()

    def run_stage_top10(self, top10: list[str]) -> str:
        stage = "fast_top10_all_dataset"
        self.log(f"--- STAGE: {stage} ---")
        runs = []
        for pid in top10:
            cfg = self._parse_cfg(pid)
            cfg.update({"selection_operator": "tournament", "tournament_size": 2,
                        "crossover_operator": "mixed", "mutation_operator": "mixed"})
            for ds in self.datasets:
                for seed in self.seeds:
                    r = self._execute_run(stage, "STAGE_2_TOP10", pid, pid, ds, seed, cfg)
                    runs.append(r)
                    
        self._save_checkpoint()
        ds_ranks, agg_ranks = self.get_aggregate_rank(runs, "parameter_group_id")
        self.stage2_ds = ds_ranks
        self.stage2_agg = agg_ranks
        self.log(f"  Top10 tamamlandi: {len(runs)} run")
        return agg_ranks["parameter_group_id"].iloc[0]

    def _parse_cfg(self, pid: str) -> dict:
        parts = pid.replace("PG_", "").split("_")
        c = {}
        for p in parts:
            k, v = p[:2], p[2:].replace("p", ".")
            if k == "ps": c["population_size"] = int(v)
            elif k == "cr": c["crossover_rate"] = float(v)
            elif k == "er": c["elitism_rate"] = float(v)
            elif k == "mr": c["mutation_rate"] = float(v)
        return c

    def run_stage_selection(self, best_pid: str) -> dict:
        stage = "fast_selection_all_dataset"
        self.log(f"--- STAGE: {stage} ---")
        cfg_base = self._parse_cfg(best_pid)
        cfg_base.update({"crossover_operator": "mixed", "mutation_operator": "mixed"})
        runs = []
        for sc in SELECTION_CASES:
            cfg = dict(cfg_base)
            cfg["selection_operator"] = sc["selection_operator"]
            cfg["tournament_size"] = sc["tournament_size"]
            for ds in self.datasets:
                for seed in self.seeds:
                    r = self._execute_run(stage, "STAGE_3_SEL", best_pid, sc["case_id"], ds, seed, cfg)
                    runs.append(r)
                    
        self._save_checkpoint()
        ds_ranks, agg_ranks = self.get_aggregate_rank(runs, "operator_label")
        self.stage3_ds = ds_ranks
        self.stage3_agg = agg_ranks
        self.log(f"  Selection tamamlandi: {len(runs)} run")
        best_id = agg_ranks["operator_label"].iloc[0]
        return next(c for c in SELECTION_CASES if c["case_id"] == best_id)

    def run_stage_crossover(self, best_pid: str, best_sel: dict) -> str:
        stage = "fast_crossover_all_dataset"
        self.log(f"--- STAGE: {stage} ---")
        cfg_base = self._parse_cfg(best_pid)
        cfg_base.update({
            "selection_operator": best_sel["selection_operator"],
            "tournament_size": best_sel["tournament_size"],
            "mutation_operator": "mixed"
        })
        runs = []
        for op in CROSSOVER_FUNCTIONS.keys():
            cfg = dict(cfg_base)
            cfg["crossover_operator"] = op
            for ds in self.datasets:
                for seed in self.seeds:
                    r = self._execute_run(stage, "STAGE_4_CX", best_pid, op, ds, seed, cfg)
                    runs.append(r)
                    
        self._save_checkpoint()
        ds_ranks, agg_ranks = self.get_aggregate_rank(runs, "operator_label")
        self.stage4_ds = ds_ranks
        self.stage4_agg = agg_ranks
        self.log(f"  Crossover tamamlandi: {len(runs)} run")
        return agg_ranks["operator_label"].iloc[0]

    def run_stage_mutation(self, best_pid: str, best_sel: dict, best_cx: str) -> str:
        stage = "fast_mutation_all_dataset"
        self.log(f"--- STAGE: {stage} ---")
        cfg_base = self._parse_cfg(best_pid)
        cfg_base.update({
            "selection_operator": best_sel["selection_operator"],
            "tournament_size": best_sel["tournament_size"],
            "crossover_operator": best_cx
        })
        runs = []
        for op in MUTATION_FUNCTIONS.keys():
            cfg = dict(cfg_base)
            cfg["mutation_operator"] = op
            for ds in self.datasets:
                for seed in self.seeds:
                    r = self._execute_run(stage, "STAGE_5_MUT", best_pid, op, ds, seed, cfg)
                    runs.append(r)
                    
        self._save_checkpoint()
        ds_ranks, agg_ranks = self.get_aggregate_rank(runs, "operator_label")
        self.stage5_ds = ds_ranks
        self.stage5_agg = agg_ranks
        self.log(f"  Mutation tamamlandi: {len(runs)} run")
        return agg_ranks["operator_label"].iloc[0]

    def run_stage_final(self, final_cfg: dict) -> None:
        stage = "fast_final_all_dataset"
        self.log(f"--- STAGE: {stage} ---")
        runs = []
        for ds in self.datasets:
            for seed in self.seeds:
                r = self._execute_run(stage, "STAGE_6_FINAL", "FINAL", "FINAL_CFG", ds, seed, final_cfg)
                runs.append(r)
        self._save_checkpoint()
        
        df = pd.DataFrame(runs)
        g = df.groupby("dataset").agg(
            n_runs=("best_tour_length", "count"),
            best_tour_length=("best_tour_length", "min"),
            mean_tour_length=("best_tour_length", "mean"),
            worst_tour_length=("best_tour_length", "max"),
            std_tour_length=("best_tour_length", lambda x: np.std(x, ddof=0))
        ).reset_index()
        
        bds = []
        for ds_name, grp in df.groupby("dataset"):
            grp = grp.copy()
            grp["best_tour_length"] = grp["best_tour_length"].astype(float)
            # Tie-break: lowest tour, lowest elapsed, smallest seed
            grp_sorted = grp.sort_values(
                ["best_tour_length", "elapsed_seconds", "seed"],
                ascending=[True, True, True]
            )
            b = grp_sorted.iloc[0]
            bds.append({
                "dataset": ds_name,
                "best_seed": b["seed"],
                "best_tour_length": b["best_tour_length"],
                "best_run_id": b["run_id"],
                "best_elapsed_seconds": b["elapsed_seconds"],
                "best_iterations_completed": b["iterations_completed"],
                "best_stop_reason": b["stop_reason"],
            })
        best_df = pd.DataFrame(bds)
        
        self.final_res = pd.merge(g, best_df, on=["dataset", "best_tour_length"], how="left")
        self.final_runs = runs
        self.log(f"  Final tamamlandi: {len(runs)} run")

    # =========================================================================
    # FIGURE GENERATION (all from final raw runs - no rerun)
    # =========================================================================

    def _generate_aggregate_rank_figure(self, agg_df: pd.DataFrame, title: str, filename: str,
                                        group_col: str) -> dict:
        """Generate aggregate rank bar chart."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = agg_df[group_col].astype(str).tolist()
        ranks = agg_df["average_rank"].tolist()
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
        bars = ax.barh(range(len(labels)), ranks, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Average Rank (daha düşük = daha iyi)", fontsize=11)
        ax.set_title(f"{title}\n(Daha düşük average_rank daha iyidir)", fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        
        for bar, rank in zip(bars, ranks):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{rank:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        fpath = self.fig_dir / filename
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.log(f"  Figure: {filename}")
        return {"filename": filename, "filepath": str(fpath), "type": "aggregate_rank"}

    def _generate_route_figure(self, ds: str, run_row: dict) -> dict:
        """Generate route figure from final run's best_tour_sequence."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        ds_stem = ds.replace(".txt", "")
        
        # Load sequence from saved file
        seq_path = run_row.get("best_tour_sequence_path", "")
        if seq_path and Path(seq_path).exists():
            seq = json.loads(Path(seq_path).read_text(encoding="utf-8"))
        else:
            self.log(f"  WARNING: Sequence file not found for {ds}, seed={run_row['seed']}")
            return None
        
        if not seq:
            self.log(f"  WARNING: Empty sequence for {ds}")
            return None
        
        # Load dataset coordinates
        data = load_dataset(ds, self.dataset_dir)
        coords = data[:, 1:3]  # x, y columns
        
        seq_arr = np.array(seq, dtype=int)
        # Create closed tour: append first city
        route_x = [coords[c, 0] for c in seq_arr] + [coords[seq_arr[0], 0]]
        route_y = [coords[c, 1] for c in seq_arr] + [coords[seq_arr[0], 1]]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(route_x, route_y, 'b-', linewidth=0.8, alpha=0.7)
        ax.plot(coords[:, 0], coords[:, 1], 'ko', markersize=3, alpha=0.5)
        ax.plot(coords[seq_arr[0], 0], coords[seq_arr[0], 1], 'r*', markersize=15,
                label="Başlangıç / Bitiş", zorder=5)
        
        tour_len = run_row["best_tour_length"]
        seed_val = run_row["seed"]
        ax.set_title(f"{ds_stem} - Seed: {seed_val} - Tur Uzunluğu = {tour_len:.0f}",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(fontsize=10, loc='best')
        ax.set_aspect('equal', adjustable='datalim')
        
        plt.tight_layout()
        fname = f"route_{ds_stem}.png"
        fpath = self.fig_dir / fname
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.log(f"  Figure: {fname}")
        return {
            "filename": fname, "filepath": str(fpath), "type": "route",
            "dataset": ds, "run_id": run_row["run_id"],
            "best_tour_length": tour_len, "seed": seed_val
        }

    def _generate_convergence_figure(self, ds: str, final_runs_for_ds: list[dict]) -> dict:
        """Generate convergence figure with all seeds' convergence histories."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        ds_stem = ds.replace(".txt", "")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        has_data = False
        for idx, run_row in enumerate(final_runs_for_ds):
            conv_path = run_row.get("convergence_history_path", "")
            if not conv_path or not Path(conv_path).exists():
                continue
            
            conv_data = json.loads(Path(conv_path).read_text(encoding="utf-8"))
            if not conv_data:
                continue
            
            has_data = True
            iterations = [p["iteration"] for p in conv_data]
            best_vals = [p["best_tour_length"] for p in conv_data]
            
            color = colors[idx % len(colors)]
            ax.plot(iterations, best_vals, '-', color=color, linewidth=1.0,
                    label=f"Seed {run_row['seed']}", alpha=0.8)
        
        if not has_data:
            plt.close(fig)
            self.log(f"  WARNING: No convergence data for {ds}")
            return None
        
        ax.set_xlabel("İterasyon", fontsize=11)
        ax.set_ylabel("En İyi Tur Uzunluğu", fontsize=11)
        ax.set_title(f"Convergence - {ds_stem}", fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fname = f"convergence_{ds_stem}.png"
        fpath = self.fig_dir / fname
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.log(f"  Figure: {fname}")
        return {"filename": fname, "filepath": str(fpath), "type": "convergence", "dataset": ds}

    def generate_all_figures(self) -> list[dict]:
        """Generate all 22 figures from existing data (no reruns)."""
        self.log("--- GORSEL URETIMI (final raw run'lardan) ---")
        figure_index = []
        
        # 1. Aggregate rank figures (4)
        if hasattr(self, 'stage2_agg'):
            fi = self._generate_aggregate_rank_figure(
                self.stage2_agg, "Top10 Parameters - Aggregate Rank",
                "aggregate_rank_top10_parameters.png", "parameter_group_id")
            figure_index.append(fi)
        
        if hasattr(self, 'stage3_agg'):
            fi = self._generate_aggregate_rank_figure(
                self.stage3_agg, "Selection Operators - Aggregate Rank",
                "aggregate_rank_selection.png", "operator_label")
            figure_index.append(fi)
        
        if hasattr(self, 'stage4_agg'):
            fi = self._generate_aggregate_rank_figure(
                self.stage4_agg, "Crossover Operators - Aggregate Rank",
                "aggregate_rank_crossover.png", "operator_label")
            figure_index.append(fi)
        
        if hasattr(self, 'stage5_agg'):
            fi = self._generate_aggregate_rank_figure(
                self.stage5_agg, "Mutation Operators - Aggregate Rank",
                "aggregate_rank_mutation.png", "operator_label")
            figure_index.append(fi)
        
        # 2. Route and Convergence figures from final runs (9+9 = 18)
        if hasattr(self, 'final_runs') and self.final_runs:
            df_final = pd.DataFrame(self.final_runs)
            
            for ds in self.datasets:
                ds_runs = df_final[df_final["dataset"] == ds].copy()
                if ds_runs.empty:
                    continue
                
                # Route: pick best run with tie-break
                ds_runs_sorted = ds_runs.sort_values(
                    ["best_tour_length", "elapsed_seconds", "seed"],
                    ascending=[True, True, True]
                )
                best_run = ds_runs_sorted.iloc[0].to_dict()
                fi = self._generate_route_figure(ds, best_run)
                if fi:
                    figure_index.append(fi)
                
                # Convergence: all 5 seeds
                conv_runs = ds_runs.sort_values("seed").to_dict('records')
                fi = self._generate_convergence_figure(ds, conv_runs)
                if fi:
                    figure_index.append(fi)
        
        self.figure_index = figure_index
        self.log(f"  Toplam {len(figure_index)} gorsel uretildi")
        return figure_index

    # =========================================================================
    # EXCEL & REPORT OUTPUT
    # =========================================================================
    
    def generate_outputs(self, final_cfg: dict):
        """Generate comprehensive Excel file with all required sheets."""
        self.log("--- EXCEL URETIMI ---")
        xls_path = self.output_dir / "v6_time_iteration_unified_results.xlsx"
        
        # Prepare all_raw DataFrame
        df_all = pd.DataFrame(self.all_raw)
        
        # Stage counts
        stage_counts_data = []
        for stage_name, expected in [
            ("fast_param_grid_all_dataset", None),
            ("fast_top10_all_dataset", None),
            ("fast_selection_all_dataset", None),
            ("fast_crossover_all_dataset", None),
            ("fast_mutation_all_dataset", None),
            ("fast_final_all_dataset", None),
        ]:
            mask = df_all["stage_name"] == stage_name
            actual = int(mask.sum())
            limits = self.stage_limits.get(stage_name, {})
            stage_counts_data.append({
                "stage_name": stage_name,
                "actual_runs": actual,
                "maxtime_limit": limits.get("maxtime_limit", ""),
                "max_iterations_limit": limits.get("max_iterations_limit", ""),
            })
        df_stage_counts = pd.DataFrame(stage_counts_data)
        
        # Figure index
        fi_data = []
        if hasattr(self, 'figure_index'):
            for fi in self.figure_index:
                fi_data.append({
                    "filename": fi.get("filename", ""),
                    "filepath": fi.get("filepath", ""),
                    "type": fi.get("type", ""),
                    "dataset": fi.get("dataset", ""),
                    "run_id": fi.get("run_id", ""),
                    "best_tour_length": fi.get("best_tour_length", ""),
                    "seed": fi.get("seed", ""),
                })
        df_figure_index = pd.DataFrame(fi_data) if fi_data else pd.DataFrame(
            columns=["filename", "filepath", "type", "dataset", "run_id", "best_tour_length", "seed"])
        
        # Final run details (45 runs)
        final_runs_df = df_all[df_all["stage_name"] == "fast_final_all_dataset"].copy() if not df_all.empty else pd.DataFrame()
        
        # Validation summary
        validation_checks = self._run_validation_checks(df_all, final_runs_df, df_figure_index)
        df_validation = pd.DataFrame(validation_checks)
        
        # Readme
        readme_data = pd.DataFrame([{
            "info": "V6 Time & Iteration Unified Runner - Coklu Veri Seti Analizi",
            "script": "proje_script/v6_time_iteration_unified_runner.py",
            "total_runs": len(self.all_raw),
            "datasets": len(self.datasets),
            "seeds": len(self.seeds),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Her stage icin ayri sure ve iterasyon siniri. Tek kaynak veri. Rerun yok."
        }])
        
        with pd.ExcelWriter(xls_path, engine='openpyxl') as w:
            readme_data.to_excel(w, sheet_name="00_readme", index=False)
            
            if not df_all.empty:
                # Remove large data columns from raw runs for Excel
                excel_cols = [c for c in df_all.columns 
                              if c not in ("best_tour_sequence", "convergence_history")]
                df_all[excel_cols].to_excel(w, sheet_name="01_all_raw_runs", index=False)
            
            df_stage_counts.to_excel(w, sheet_name="02_stage_counts", index=False)
            
            if hasattr(self, 'stage1_ds'):
                self.stage1_ds.to_excel(w, sheet_name="03_param_grid_by_dataset", index=False)
                self.stage1_agg.to_excel(w, sheet_name="04_param_grid_aggregate_rank", index=False)
            
            if hasattr(self, 'stage2_ds'):
                self.stage2_ds.to_excel(w, sheet_name="05_top10_confirm_by_dataset", index=False)
                self.stage2_agg.to_excel(w, sheet_name="06_top10_confirm_aggregate_rank", index=False)
            
            if hasattr(self, 'stage3_ds'):
                self.stage3_ds.to_excel(w, sheet_name="07_selection_by_dataset", index=False)
                self.stage3_agg.to_excel(w, sheet_name="08_selection_aggregate_rank", index=False)
            
            if hasattr(self, 'stage4_ds'):
                self.stage4_ds.to_excel(w, sheet_name="09_crossover_by_dataset", index=False)
                self.stage4_agg.to_excel(w, sheet_name="10_crossover_aggregate_rank", index=False)
            
            if hasattr(self, 'stage5_ds'):
                self.stage5_ds.to_excel(w, sheet_name="11_mutation_by_dataset", index=False)
                self.stage5_agg.to_excel(w, sheet_name="12_mutation_aggregate_rank", index=False)
            
            if hasattr(self, 'final_res'):
                self.final_res.to_excel(w, sheet_name="13_final_results", index=False)
            
            if not final_runs_df.empty:
                final_runs_df.to_excel(w, sheet_name="14_final_run_details", index=False)
            
            df_figure_index.to_excel(w, sheet_name="15_figure_index", index=False)
            df_validation.to_excel(w, sheet_name="16_validation_summary", index=False)
            
            pd.DataFrame([final_cfg]).to_excel(w, sheet_name="17_final_config", index=False)
        
        self.log(f"Excel uretildi: {xls_path}")
        return xls_path

    def _run_validation_checks(self, df_all: pd.DataFrame, final_runs_df: pd.DataFrame,
                                df_figure_index: pd.DataFrame) -> list[dict]:
        """Run 20 validation checks and return results."""
        checks = []
        
        total = len(df_all)
        
        # 1. Total run count
        checks.append({"check_id": 1, "check": "Toplam run sayisi 3375 mi?",
                       "result": "PASS" if total == 3375 else f"FAIL (actual={total})",
                       "value": str(total)})
        
        # 2. Stage bazli run sayilari
        expected_counts = {
            "fast_param_grid_all_dataset": 2160,
            "fast_top10_all_dataset": 450,
            "fast_selection_all_dataset": 270,
            "fast_crossover_all_dataset": 225,
            "fast_mutation_all_dataset": 225,
            "fast_final_all_dataset": 45,
        }
        if not df_all.empty:
            actual_counts = df_all["stage_name"].value_counts().to_dict()
        else:
            actual_counts = {}
        
        all_stages_ok = True
        stage_detail = []
        for sn, exp in expected_counts.items():
            act = actual_counts.get(sn, 0)
            ok = act == exp
            if not ok: all_stages_ok = False
            stage_detail.append(f"{sn}: {act}/{exp}")
        
        checks.append({"check_id": 2, "check": "Stage bazli run sayilari dogru mu?",
                       "result": "PASS" if all_stages_ok else "FAIL",
                       "value": "; ".join(stage_detail)})
        
        # 3. maxtime_limit dogru mu?
        time_ok = True
        time_detail = []
        if not df_all.empty:
            for sn, limits in STAGE_LIMITS.items():
                stage_rows = df_all[df_all["stage_name"] == sn]
                if stage_rows.empty:
                    continue
                unique_times = stage_rows["maxtime_limit"].unique()
                expected_t = limits["maxtime_limit"]
                ok = len(unique_times) == 1 and float(unique_times[0]) == expected_t
                if not ok: time_ok = False
                time_detail.append(f"{sn}: {unique_times.tolist()} (expected={expected_t})")
        
        checks.append({"check_id": 3, "check": "Her stage icin maxtime_limit dogru mu?",
                       "result": "PASS" if time_ok else "FAIL",
                       "value": "; ".join(time_detail) if time_detail else "N/A"})
        
        # 4. max_iterations_limit dogru mu?
        iter_ok = True
        iter_detail = []
        if not df_all.empty:
            for sn, limits in STAGE_LIMITS.items():
                stage_rows = df_all[df_all["stage_name"] == sn]
                if stage_rows.empty:
                    continue
                unique_iters = stage_rows["max_iterations_limit"].unique()
                expected_i = limits["max_iterations_limit"]
                ok = len(unique_iters) == 1 and int(unique_iters[0]) == expected_i
                if not ok: iter_ok = False
                iter_detail.append(f"{sn}: {unique_iters.tolist()} (expected={expected_i})")
        
        checks.append({"check_id": 4, "check": "Her stage icin max_iterations_limit dogru mu?",
                       "result": "PASS" if iter_ok else "FAIL",
                       "value": "; ".join(iter_detail) if iter_detail else "N/A"})
        
        # 5. elapsed_seconds var mi?
        if not df_all.empty:
            has_elapsed = df_all["elapsed_seconds"].notna().all()
        else:
            has_elapsed = False
        checks.append({"check_id": 5, "check": "Her run'da elapsed_seconds var mi?",
                       "result": "PASS" if has_elapsed else "FAIL",
                       "value": str(has_elapsed)})
        
        # 6. iterations_completed var mi?
        if not df_all.empty:
            has_iter = df_all["iterations_completed"].notna().all()
        else:
            has_iter = False
        checks.append({"check_id": 6, "check": "Her run'da iterations_completed var mi?",
                       "result": "PASS" if has_iter else "FAIL",
                       "value": str(has_iter)})
        
        # 7. stop_reason dolu mu?
        if not df_all.empty:
            has_stop = (df_all["stop_reason"] != "").all() and df_all["stop_reason"].notna().all()
        else:
            has_stop = False
        checks.append({"check_id": 7, "check": "Her run'da stop_reason dolu mu?",
                       "result": "PASS" if has_stop else "FAIL",
                       "value": str(has_stop)})
        
        # 8. Final stage 45 run mi?
        final_count = len(final_runs_df)
        checks.append({"check_id": 8, "check": "Final stage 45 run mi?",
                       "result": "PASS" if final_count == 45 else f"FAIL (actual={final_count})",
                       "value": str(final_count)})
        
        # 9. best_tour_sequence_path var mi?
        if not final_runs_df.empty and "best_tour_sequence_path" in final_runs_df.columns:
            has_seq = (final_runs_df["best_tour_sequence_path"] != "").all()
        else:
            has_seq = False
        checks.append({"check_id": 9, "check": "Final runlarda best_tour_sequence_path var mi?",
                       "result": "PASS" if has_seq else "FAIL",
                       "value": str(has_seq)})
        
        # 10. convergence_history_path var mi?
        if not final_runs_df.empty and "convergence_history_path" in final_runs_df.columns:
            has_conv = (final_runs_df["convergence_history_path"] != "").all()
        else:
            has_conv = False
        checks.append({"check_id": 10, "check": "Final runlarda convergence_history_path var mi?",
                        "result": "PASS" if has_conv else "FAIL",
                        "value": str(has_conv)})
        
        # 11-16. Methodology checks (these are code-level guarantees)
        checks.append({"check_id": 11, "check": "13_final_results final stage raw runlardan mi turetildi?",
                       "result": "PASS (by design)", "value": "Evet - run_stage_final() icerisinde"})
        checks.append({"check_id": 12, "check": "Route gorselleri final stage raw runlardan mi uretildi?",
                       "result": "PASS (by design)", "value": "Evet - generate_all_figures() icerisinde"})
        checks.append({"check_id": 13, "check": "Convergence grafikleri final stage raw runlardan mi uretildi?",
                       "result": "PASS (by design)", "value": "Evet - generate_all_figures() icerisinde"})
        checks.append({"check_id": 14, "check": "Grafik uretimi icin ayri final rerun yapilmadi mi?",
                       "result": "PASS (by design)", "value": "Hayir - tek kaynak veri kullanildi"})
        checks.append({"check_id": 15, "check": "Eski final_visual_runs sistemi kullanilmadi mi?",
                       "result": "PASS (by design)", "value": "Hayir - V6'da boyle bir sistem yok"})
        checks.append({"check_id": 16, "check": "Eski V5 Fix1 gorselleri kullanilmadi mi?",
                       "result": "PASS (by design)", "value": "Hayir - tum gorseller yeni uretildi"})
        
        # 17. figure_index best_tour_length = final_results best
        fig_check_ok = True
        fig_check_detail = []
        if hasattr(self, 'figure_index') and hasattr(self, 'final_res'):
            route_figs = [f for f in self.figure_index if f.get("type") == "route"]
            for rf in route_figs:
                ds = rf.get("dataset", "")
                fig_val = rf.get("best_tour_length", None)
                if ds and fig_val is not None:
                    final_match = self.final_res[self.final_res["dataset"] == ds]
                    if not final_match.empty:
                        final_val = float(final_match["best_tour_length"].iloc[0])
                        if abs(float(fig_val) - final_val) > 0.01:
                            fig_check_ok = False
                            fig_check_detail.append(f"{ds}: fig={fig_val} != final={final_val}")
        
        checks.append({"check_id": 17, "check": "figure_index best_tour_length = final_results best?",
                       "result": "PASS" if fig_check_ok else "FAIL",
                       "value": "; ".join(fig_check_detail) if fig_check_detail else "Tum degerler eslesiyor"})
        
        # 18. Duplicate run_id
        if not df_all.empty:
            dup_count = df_all["run_id"].duplicated().sum()
        else:
            dup_count = 0
        checks.append({"check_id": 18, "check": "Duplicate run_id var mi?",
                       "result": "PASS" if dup_count == 0 else f"FAIL ({dup_count} duplicate)",
                       "value": str(dup_count)})
        
        # 19. Eksik dataset/seed kombinasyonu
        missing = []
        if not final_runs_df.empty:
            for ds in self.datasets:
                for seed in self.seeds:
                    match = final_runs_df[(final_runs_df["dataset"] == ds) & (final_runs_df["seed"] == seed)]
                    if match.empty:
                        missing.append(f"{ds}/seed={seed}")
        checks.append({"check_id": 19, "check": "Eksik dataset/seed kombinasyonu var mi?",
                       "result": "PASS" if len(missing) == 0 else f"FAIL ({len(missing)} eksik)",
                       "value": "; ".join(missing) if missing else "Hicbiri"})
        
        # 20. Hata alan run var mi?
        if not df_all.empty:
            error_runs = df_all[df_all["status"] == "ERROR"]
            err_count = len(error_runs)
        else:
            err_count = 0
        checks.append({"check_id": 20, "check": "Hata alan run var mi?",
                       "result": "PASS" if err_count == 0 else f"WARNING ({err_count} error)",
                       "value": str(err_count)})
        
        return checks

    def generate_validation_report(self, final_cfg: dict, report_path: Path):
        """Generate markdown validation report."""
        self.log("--- DOGRULAMA RAPORU URETIMI ---")
        
        df_all = pd.DataFrame(self.all_raw)
        final_runs_df = df_all[df_all["stage_name"] == "fast_final_all_dataset"].copy() if not df_all.empty else pd.DataFrame()
        df_figure_index = pd.DataFrame(self.figure_index) if hasattr(self, 'figure_index') else pd.DataFrame()
        
        checks = self._run_validation_checks(df_all, final_runs_df, df_figure_index)
        
        lines = [
            "# V6 Time & Iteration Unified - Doğrulama Raporu",
            "",
            f"**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Script:** proje_script/v6_time_iteration_unified_runner.py",
            f"**Toplam Run:** {len(self.all_raw)}",
            "",
            "## Doğrulama Kontrolleri",
            "",
            "| # | Kontrol | Sonuç | Değer |",
            "|---|---------|-------|-------|",
        ]
        
        for c in checks:
            lines.append(f"| {c['check_id']} | {c['check']} | {c['result']} | {c['value']} |")
        
        lines.append("")
        lines.append("## Stage Limitleri")
        lines.append("")
        lines.append("| Stage | maxtime_limit | max_iterations_limit |")
        lines.append("|-------|---------------|---------------------|")
        for sn, lim in STAGE_LIMITS.items():
            lines.append(f"| {sn} | {lim['maxtime_limit']} | {lim['max_iterations_limit']} |")
        
        lines.append("")
        lines.append("## Final Konfigürasyon")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(final_cfg, indent=2))
        lines.append("```")
        
        if not df_all.empty:
            lines.append("")
            lines.append("## Stop Reason Dağılımı")
            lines.append("")
            sr_counts = df_all["stop_reason"].value_counts().to_dict()
            lines.append("| stop_reason | count |")
            lines.append("|-------------|-------|")
            for sr, cnt in sr_counts.items():
                lines.append(f"| {sr} | {cnt} |")
        
        if hasattr(self, 'final_res') and self.final_res is not None:
            lines.append("")
            lines.append("## Final Sonuçlar")
            lines.append("")
            lines.append("| Dataset | Best Tour | Mean Tour | Worst Tour | Std | Best Seed |")
            lines.append("|---------|-----------|-----------|------------|-----|-----------|")
            for _, row in self.final_res.iterrows():
                lines.append(f"| {row['dataset']} | {row['best_tour_length']:.0f} | {row['mean_tour_length']:.0f} | {row['worst_tour_length']:.0f} | {row['std_tour_length']:.1f} | {row.get('best_seed', '')} |")
        
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines), encoding="utf-8")
        self.log(f"Dogrulama raporu: {report_path}")

    def generate_report_data_package(self, final_cfg: dict, package_path: Path):
        """Generate data package for Claude report writing."""
        self.log("--- RAPOR VERI PAKETI URETIMI ---")
        
        lines = [
            "# V6 Time & Iteration Unified - Rapor Veri Paketi",
            "",
            "> **Bu dosya akademik rapor değildir; Claude rapor yazımı için veri paketidir.**",
            "",
            f"## Çalışma Adı",
            "V6 Time & Iteration Unified Runner - TSP GA Çoklu Veri Seti Analizi",
            "",
            f"## Yeni V6 Script Adı",
            "proje_script/v6_time_iteration_unified_runner.py",
            "",
            f"## Toplam Run Sayısı",
            f"{len(self.all_raw)}",
            "",
            "## Stage Süre / İterasyon Limitleri",
            "",
            "| Stage | maxtime_limit (sn) | max_iterations_limit |",
            "|-------|-------------------|---------------------|",
        ]
        
        for sn, lim in STAGE_LIMITS.items():
            lines.append(f"| {sn} | {lim['maxtime_limit']} | {lim['max_iterations_limit']} |")
        
        lines.append("")
        lines.append("## Final Seçilen Konfigürasyon")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(final_cfg, indent=2))
        lines.append("```")
        
        # Top10 parameter result
        if hasattr(self, 'stage2_agg'):
            lines.append("")
            lines.append("## Top10 Parametre Sonucu")
            lines.append("")
            best_param = self.stage2_agg.iloc[0]
            lines.append(f"- **En İyi Parametre:** {best_param['parameter_group_id']}")
            lines.append(f"- **Average Rank:** {best_param['average_rank']:.2f}")
            lines.append(f"- **Mean of Means:** {best_param['mean_of_means']:.0f}")
        
        # Best selection
        if hasattr(self, 'stage3_agg'):
            lines.append("")
            lines.append("## En İyi Selection Sonucu")
            best_sel = self.stage3_agg.iloc[0]
            lines.append(f"- **Operator:** {best_sel['operator_label']}")
            lines.append(f"- **Average Rank:** {best_sel['average_rank']:.2f}")
        
        # Best crossover
        if hasattr(self, 'stage4_agg'):
            lines.append("")
            lines.append("## En İyi Crossover Sonucu")
            best_cx = self.stage4_agg.iloc[0]
            lines.append(f"- **Operator:** {best_cx['operator_label']}")
            lines.append(f"- **Average Rank:** {best_cx['average_rank']:.2f}")
        
        # Best mutation
        if hasattr(self, 'stage5_agg'):
            lines.append("")
            lines.append("## En İyi Mutation Sonucu")
            best_mut = self.stage5_agg.iloc[0]
            lines.append(f"- **Operator:** {best_mut['operator_label']}")
            lines.append(f"- **Average Rank:** {best_mut['average_rank']:.2f}")
        
        # Final 9 dataset results
        if hasattr(self, 'final_res') and self.final_res is not None:
            lines.append("")
            lines.append("## Final 9 Dataset Sonuç Tablosu")
            lines.append("")
            lines.append("| Dataset | Best Tour | Mean Tour | Worst Tour | Std | Best Seed |")
            lines.append("|---------|-----------|-----------|------------|-----|-----------|")
            for _, row in self.final_res.iterrows():
                lines.append(f"| {row['dataset']} | {row['best_tour_length']:.0f} | {row['mean_tour_length']:.0f} | {row['worst_tour_length']:.0f} | {row['std_tour_length']:.1f} | {row.get('best_seed', '')} |")
        
        # Figure list
        if hasattr(self, 'figure_index'):
            lines.append("")
            lines.append("## Üretilen Figürler")
            lines.append("")
            for fi in self.figure_index:
                lines.append(f"- {fi['filename']} ({fi['type']})")
        
        lines.append("")
        lines.append("## Doğrulama Raporu Yolu")
        lines.append("ciktilar/28_v6_time_iteration_unified_dogrulama_raporu.md")
        lines.append("")
        
        package_path.parent.mkdir(parents=True, exist_ok=True)
        package_path.write_text("\n".join(lines), encoding="utf-8")
        self.log(f"Rapor veri paketi: {package_path}")


# =============================================================================
# SMOKE TEST
# =============================================================================

def run_smoke_test(base_dir: Path):
    """Run a small smoke test to verify script functionality."""
    print("=" * 60)
    print("SMOKE TEST BASLIYOR")
    print("=" * 60)
    
    ds_dir = base_dir / "dataSets"
    out_dir = base_dir / "outputs" / "v6_time_iteration_unified" / "smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Smoke test stage limits (very short)
    smoke_limits = {
        "fast_param_grid_all_dataset":  {"maxtime_limit": 0.3, "max_iterations_limit": 50},
        "fast_top10_all_dataset":       {"maxtime_limit": 0.5, "max_iterations_limit": 100},
        "fast_selection_all_dataset":   {"maxtime_limit": 0.5, "max_iterations_limit": 100},
        "fast_crossover_all_dataset":   {"maxtime_limit": 0.5, "max_iterations_limit": 100},
        "fast_mutation_all_dataset":    {"maxtime_limit": 0.5, "max_iterations_limit": 100},
        "fast_final_all_dataset":       {"maxtime_limit": 1.0, "max_iterations_limit": 200},
    }
    
    smoke_datasets = ["eil51.txt", "d493.txt"]
    smoke_seeds = [11, 42]
    
    runner = V6UnifiedRunner(ds_dir, out_dir, 
                             stage_limits=smoke_limits, 
                             datasets=smoke_datasets, 
                             seeds=smoke_seeds)
    runner.log("=== SMOKE TEST BASLIYOR ===")
    
    try:
        # Stage 1: Param grid
        top10 = runner.run_stage_param_grid()
        runner.log(f"Top10 params: {top10[:3]}...")
        
        # Stage 2: Top10
        best_pid = runner.run_stage_top10(top10)
        runner.log(f"Best param: {best_pid}")
        
        # Stage 3: Selection
        best_sel = runner.run_stage_selection(best_pid)
        runner.log(f"Best selection: {best_sel}")
        
        # Stage 4: Crossover
        best_cx = runner.run_stage_crossover(best_pid, best_sel)
        runner.log(f"Best crossover: {best_cx}")
        
        # Stage 5: Mutation
        best_mut = runner.run_stage_mutation(best_pid, best_sel, best_cx)
        runner.log(f"Best mutation: {best_mut}")
        
        # Build final config
        cfg = runner._parse_cfg(best_pid)
        cfg.update({
            "selection_operator": best_sel["selection_operator"],
            "tournament_size": best_sel["tournament_size"],
            "crossover_operator": best_cx,
            "mutation_operator": best_mut
        })
        
        # Stage 6: Final
        runner.run_stage_final(cfg)
        
        # Generate figures
        figure_index = runner.generate_all_figures()
        
        # Generate Excel
        runner.generate_outputs(cfg)
        
        # Print smoke test summary
        df = pd.DataFrame(runner.all_raw)
        print("\n" + "=" * 60)
        print("SMOKE TEST OZET")
        print("=" * 60)
        print(f"Script syntax/import: OK")
        print(f"dataSets bulundu: {ds_dir.exists()}")
        print(f"Toplam smoke run: {len(runner.all_raw)}")
        
        if not df.empty:
            print(f"\nstop_reason dagilimi:")
            for sr, cnt in df["stop_reason"].value_counts().items():
                print(f"  {sr}: {cnt}")
            
            print(f"\nelapsed_seconds ornekleri:")
            sample = df.head(5)[["stage_name", "dataset", "seed", "elapsed_seconds", "iterations_completed", "stop_reason"]]
            print(sample.to_string(index=False))
            
            final_mask = df["stage_name"] == "fast_final_all_dataset"
            final_df = df[final_mask]
            
            if not final_df.empty:
                has_seq_paths = (final_df["best_tour_sequence_path"] != "").all()
                has_conv_paths = (final_df["convergence_history_path"] != "").all()
                print(f"\nbest_tour_sequence kaydi: {'OK' if has_seq_paths else 'FAIL'}")
                print(f"convergence_history kaydi: {'OK' if has_conv_paths else 'FAIL'}")
                
                # Check if actual files exist
                if has_seq_paths:
                    first_seq = final_df.iloc[0]["best_tour_sequence_path"]
                    seq_exists = Path(first_seq).exists()
                    if seq_exists:
                        seq_data = json.loads(Path(first_seq).read_text(encoding="utf-8"))
                        print(f"  Sequence file var: {seq_exists}, len={len(seq_data)}")
                
                if has_conv_paths:
                    first_conv = final_df.iloc[0]["convergence_history_path"]
                    conv_exists = Path(first_conv).exists()
                    if conv_exists:
                        conv_data = json.loads(Path(first_conv).read_text(encoding="utf-8"))
                        print(f"  Convergence file var: {conv_exists}, len={len(conv_data)}")
            
            print(f"\nGrafik uretimi (final raw run'dan): {len(figure_index)} gorsel")
        
        print(f"\nSmoke test: BASARILI")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\nSmoke test HATA: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# FULL RUN
# =============================================================================

def run_full(base_dir: Path):
    """Run the full 3375-run pipeline."""
    ds_dir = base_dir / "dataSets"
    out_dir = base_dir / "outputs" / "v6_time_iteration_unified"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    runner = V6UnifiedRunner(ds_dir, out_dir)
    runner.log("=== V6 FULL RUN BASLIYOR ===")
    
    try:
        # Stage 1: Param grid (48 params × 9 ds × 5 seeds = 2160)
        top10 = runner.run_stage_param_grid()
        runner.log(f"Top10 params: {top10}")
        
        # Stage 2: Top10 confirm (10 × 9 × 5 = 450)
        best_pid = runner.run_stage_top10(top10)
        runner.log(f"Best param: {best_pid}")
        
        # Stage 3: Selection (6 × 9 × 5 = 270)
        best_sel = runner.run_stage_selection(best_pid)
        runner.log(f"Best selection: {best_sel}")
        
        # Stage 4: Crossover (5 × 9 × 5 = 225)
        best_cx = runner.run_stage_crossover(best_pid, best_sel)
        runner.log(f"Best crossover: {best_cx}")
        
        # Stage 5: Mutation (5 × 9 × 5 = 225)
        best_mut = runner.run_stage_mutation(best_pid, best_sel, best_cx)
        runner.log(f"Best mutation: {best_mut}")
        
        # Build final config
        cfg = runner._parse_cfg(best_pid)
        cfg.update({
            "selection_operator": best_sel["selection_operator"],
            "tournament_size": best_sel["tournament_size"],
            "crossover_operator": best_cx,
            "mutation_operator": best_mut
        })
        
        runner.log("--- FINAL CONFIRMED CONFIG ---")
        runner.log(json.dumps(cfg))
        
        # Stage 6: Final (9 × 5 = 45)
        runner.run_stage_final(cfg)
        
        # Generate all 22 figures from final raw runs
        runner.generate_all_figures()
        
        # Generate Excel
        runner.generate_outputs(cfg)
        
        # Generate validation report
        report_path = base_dir / "ciktilar" / "28_v6_time_iteration_unified_dogrulama_raporu.md"
        runner.generate_validation_report(cfg, report_path)
        
        # Generate report data package
        package_path = out_dir / "v6_report_data_package.md"
        runner.generate_report_data_package(cfg, package_path)
        
        runner.log("=== V6 FULL RUN BASARIYLA TAMAMLANDI ===")
        
    except Exception as e:
        runner.log(f"FATAL HATA: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="V6 Time & Iteration Unified Runner")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke",
                        help="Calisma modu: 'smoke' (test) veya 'full' (3375 run)")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Proje kok dizini (dataSets/ iceren)")
    args = parser.parse_args()
    
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        # Try to find dataSets relative to script location
        script_dir = Path(__file__).resolve().parent
        if (script_dir.parent / "dataSets").exists():
            base_dir = script_dir.parent
        elif (script_dir / "dataSets").exists():
            base_dir = script_dir
        elif (Path(".") / "dataSets").exists():
            base_dir = Path(".").resolve()
        else:
            print("HATA: dataSets klasoru bulunamadi. --base-dir ile belirtin.")
            sys.exit(1)
    
    print(f"Base dir: {base_dir}")
    print(f"DataSets: {base_dir / 'dataSets'}")
    print(f"Mode: {args.mode}")
    
    if args.mode == "smoke":
        success = run_smoke_test(base_dir)
        sys.exit(0 if success else 1)
    else:
        run_full(base_dir)


if __name__ == "__main__":
    main()
