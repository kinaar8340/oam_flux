"""Per-pulse and cumulative metrics from VQC pulse-train history."""

from __future__ import annotations


def compute_pulse_statistics(history: list[dict]) -> dict:
    """
    Summarize pulse-train dynamics from step history.

    Returns per-pulse η min / end-of-gap η, slip and lattice deposit per pulse,
    plus cumulative totals (phase slip, lattice momentum received).
    """
    if not history or "pulse_index" not in history[0]:
        last = history[-1] if history else {}
        return {
            "pulses": [],
            "cumulative": {
                "total_phase_slip": float(last.get("cumulative_phase_slip", 0.0)),
                "total_lattice_momentum": float(last.get("lattice_received", 0.0)),
                "total_injected": float(last.get("total_injected", last.get("initial_total", 0.0))),
                "mean_eta_pump_min": float(last.get("back_reaction_coupling", 1.0)),
            },
        }

    trackers: dict[int, dict] = {}
    for h in history:
        pid = int(h.get("pulse_index", 0))
        if pid not in trackers:
            trackers[pid] = {
                "eta_min": 1.0,
                "eta_pump_end": 1.0,
                "eta_gap_end": 1.0,
                "slip_start": float(h.get("cumulative_phase_slip", 0.0)),
                "lattice_start": float(h.get("lattice_received", 0.0)),
            }
        tr = trackers[pid]
        eta = float(h.get("back_reaction_coupling", 1.0))
        tr["eta_min"] = min(tr["eta_min"], eta)
        if h.get("pulse_phase_pump", 0) > 0.5:
            tr["eta_pump_end"] = eta
        if h.get("recovery_active", 0) > 0.5:
            tr["eta_gap_end"] = eta

    pulses: list[dict] = []
    for pid in sorted(trackers.keys()):
        tr = trackers[pid]
        rows = [h for h in history if int(h.get("pulse_index", 0)) == pid]
        last_h = rows[-1]
        slip_end = float(last_h.get("cumulative_phase_slip", 0.0))
        lattice_end = float(last_h.get("lattice_received", 0.0))
        pulses.append(
            {
                "pulse": pid + 1,
                "eta_min": tr["eta_min"],
                "eta_after_pump": tr["eta_pump_end"],
                "eta_after_gap": tr["eta_gap_end"],
                "slip": slip_end - tr["slip_start"],
                "lattice_deposited": lattice_end - tr["lattice_start"],
            }
        )

    last = history[-1]
    mean_eta = (
        sum(p["eta_min"] for p in pulses) / len(pulses) if pulses else 1.0
    )
    return {
        "pulses": pulses,
        "cumulative": {
            "total_phase_slip": float(last.get("cumulative_phase_slip", 0.0)),
            "total_lattice_momentum": float(last.get("lattice_received", 0.0)),
            "total_injected": float(last.get("total_injected", 0.0)),
            "mean_eta_pump_min": mean_eta,
        },
    }