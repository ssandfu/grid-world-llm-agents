#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-Auswertung für LLM-Grid-Agent-Experimente.

Ordnerstruktur (Beispiel):
  experiments/
    expl-<difficulty>-<provider>-<model>/
      20250818-202633-llm-grid-agent.log
      20250818-213012-llm-grid-agent.log
      ...

Pro Versuch entspricht ein Ordner "expl-<difficulty>-<provider>-<model>".
Für jedes Modell wird eine CSV erzeugt, die:
  - alle Runs (Logs) enthält und
  - am Ende einen "SUMMARY"-Block mit Aggregaten pro (difficulty, model):
        count, mean, std, median, min, max (auf Basis der Fraction 0..1)

Besonderheiten:
  - Primär wird [RESULT]-Zeile geparst:
        "[RESULT] After 400 Iterations the map was revealed to 65.66 %"
    => explored_frac = 0.6566
  - Falls [RESULT] fehlt:
        Suche letzte Zeile "[Iteration 0399] ... explored: 65.66%"
    => Warnung + Ausgabe "399: 0,6566" ins Log (Konsole)
  - CSV: mit Semikolon als Trennzeichen und Dezimalkomma.

Benutzung:
  - Skript im gleichen Verzeichnis starten, in dem der Ordner "experiments/" liegt.
  - Optional: EXPERIMENTS_DIR unten anpassen.
"""

import os
import re
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict
import pandas as pd

# --- Konfiguration ---
EXPERIMENTS_DIR = "experiments"
OUTPUT_DIR = "experiment_results"
CSV_SEP = ";"          # deutsch üblich
CSV_DECIMAL = ","      # Dezimalkomma

# Logging hübsch formatieren
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# Regexe
RESULT_RE = re.compile(
    r'^INFO:root:\[RESULT\].*?revealed to\s+([0-9]+(?:[.,][0-9]+)?)\s*%.*$',
    re.IGNORECASE
)
ITER_RE = re.compile(
    r'^INFO:root:\[Iteration\s+(\d{1,6})\].*?explored:\s*([0-9]+(?:[.,][0-9]+)?)\s*%.*$',
    re.IGNORECASE
)

# Dateiname: "<date>-llm-grid-agent.log"  -> <date>
RUNFILE_RE = re.compile(r'^(?P<date>.+?)-llm-grid-agent\.log$', re.IGNORECASE)

@dataclass
class RunResult:
    difficulty: str
    provider: str
    model: str
    run_date: str                # aus Dateinamen
    result_type: str             # "RESULT" | "ITERATION_FALLBACK"
    iterations: Optional[int]    # bekannt bei Fallback; bei RESULT meist None
    explored_frac: float         # 0..1
    explored_pct: float          # 0..100
    log_path: str                # relativer Pfad für Nachvollziehbarkeit


def _to_float(num_str: str) -> float:
    """Erlaubt Punkt- und Komma-Notation."""
    return float(num_str.replace(",", ".").strip())


def parse_log_file(log_path: str) -> Tuple[str, Optional[int], float]:
    """
    Parst ein Logfile.
    Rückgabe:
        (result_type, iterations_or_None, explored_frac)
    """
    last_iter_no: Optional[int] = None
    last_iter_explored: Optional[float] = None
    result_explored: Optional[float] = None

    # Wir gehen zeilenweise durch, merken uns letzte Iteration und suchen [RESULT]
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            # [RESULT]
            m_res = RESULT_RE.match(line)
            if m_res:
                val_pct = _to_float(m_res.group(1))
                result_explored = val_pct / 100.0
                # Nicht "break": falls mehrere RESULTs vorhanden sind, nehmen wir den letzten
                continue

            # [Iteration ####] ... explored: NN.NN%
            m_iter = ITER_RE.match(line)
            if m_iter:
                last_iter_no = int(m_iter.group(1))
                val_pct = _to_float(m_iter.group(2))
                last_iter_explored = val_pct / 100.0
                continue

    if result_explored is not None:
        return ("RESULT", None, result_explored)

    # Fallback auf letzte Iteration
    if last_iter_explored is not None:
        # Warnung im geforderten Format: "399: 0,6566"
        explored_str = f"{last_iter_explored:.4f}".replace(".", ",")
        logging.warning(f"{os.path.relpath(log_path)} – Kein [RESULT] gefunden; "
                        f"letzter Iterationseintrag {last_iter_no}: {explored_str}")
        return ("ITERATION_FALLBACK", last_iter_no, last_iter_explored)

    # gar nichts gefunden
    logging.warning(f"{os.path.relpath(log_path)} – Weder [RESULT] noch [Iteration ####] gefunden.")
    return ("ITERATION_FALLBACK", None, float("nan"))


def parse_expl_folder_name(folder_name: str) -> Optional[Tuple[str, str, str]]:
    """
    Erwartet "expl-<difficulty>-<provider>-<model>".
    Achtung: <model> darf Bindestriche enthalten -> maxsplit=2.
    """
    if not folder_name.startswith("expl-"):
        return None
    payload = folder_name[len("expl-"):]
    parts = payload.split("-", 2)
    if len(parts) < 3:
        return None
    difficulty, provider, model = parts[0], parts[1], parts[2]
    return difficulty, provider, model


def collect_runs(base_dir: str) -> List[RunResult]:
    results: List[RunResult] = []

    if not os.path.isdir(base_dir):
        logging.error(f'Basisordner "{base_dir}" nicht gefunden.')
        return results

    for entry in os.scandir(base_dir):
        if not entry.is_dir():
            continue

        parsed = parse_expl_folder_name(entry.name)
        if not parsed:
            # anderen Kram in experiments/ ignorieren
            continue

        difficulty, provider, model = parsed
        exp_dir = entry.path

        # Alle passenden Logfiles im Versuch suchen
        for file in os.scandir(exp_dir):
            if not file.is_file():
                continue
            if not file.name.lower().endswith("-llm-grid-agent.log"):
                continue

            # Run-Date extrahieren
            run_date = file.name
            m = RUNFILE_RE.match(file.name)
            if m:
                run_date = m.group("date")

            result_type, iterations, explored_frac = parse_log_file(file.path)
            explored_pct = explored_frac * 100.0 if explored_frac == explored_frac else float("nan")  # NaN check

            results.append(RunResult(
                difficulty=difficulty,
                provider=provider,
                model=model,
                run_date=run_date,
                result_type=result_type,
                iterations=iterations,
                explored_frac=explored_frac,
                explored_pct=explored_pct,
                log_path=os.path.relpath(file.path)
            ))

    return results


def write_per_model_csv(results: List[RunResult], out_dir: str) -> None:
    if not results:
        logging.info("Keine Ergebnisse gefunden – nichts zu schreiben.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # DataFrame bauen
    df = pd.DataFrame([asdict(r) for r in results])

    # Saubere Sortierung
    df.sort_values(by=["model", "difficulty", "provider", "run_date"], inplace=True)

    # Pro Modell eigene CSV
    for model, df_model in df.groupby("model", sort=True):
        detail_cols = [
            "difficulty", "provider", "model", "run_date",
            "result_type", "iterations", "explored_frac", "explored_pct", "log_path"
        ]
        df_det = df_model.loc[:, detail_cols].copy()

        # Summary pro (difficulty, model)
        # Aggregation über explored_frac (0..1)
        df_sum = (
            df_model
            .groupby(["difficulty", "model"], dropna=False)["explored_frac"]
            .agg(count="count", mean="mean", std="std", median="median", min="min", max="max")
            .reset_index()
        )

        out_path = os.path.join(out_dir, f"results_{model}.csv")

        # Details schreiben
        df_det.to_csv(out_path, sep=CSV_SEP, decimal=CSV_DECIMAL, index=False)

        # Summary-Block anhängen
        with open(out_path, "a", encoding="utf-8") as f:
            f.write("\nSUMMARY\n")
        df_sum.to_csv(out_path, sep=CSV_SEP, decimal=CSV_DECIMAL, index=False, mode="a")

        logging.info(f'CSV geschrieben: {os.path.relpath(out_path)}')

    logging.info("Fertig.")


def main():
    results = collect_runs(EXPERIMENTS_DIR)
    write_per_model_csv(results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
