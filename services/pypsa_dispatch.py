import contextlib
import io
import logging
import numpy as np
import pandas as pd
import pypsa
from config import LOAD_SHEDDING_COST

# Set logging levels to suppress noisy solver messages
logging.getLogger("pypsa").setLevel(logging.ERROR)
logging.getLogger("linopy").setLevel(logging.WARNING)

def run_pypsa_dispatch(load_24h, snapshots, solar_profile,
                       include_load_shedding=False,
                       load_shedding_cost=LOAD_SHEDDING_COST):
    """
    Solve a 24-hour economic dispatch problem for one demand trajectory.

    For Monte Carlo reliability studies, a high-cost load-shedding generator is
    included so unmet demand appears as measurable shortage energy instead of
    an infeasible optimization.
    """
    load_24h = np.maximum(np.asarray(load_24h, dtype=float), 0.0)

    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.add("Bus", "Bengaluru")

    network.add("Generator", "Coal",
                bus="Bengaluru",
                p_nom=1500,
                p_min_pu=0.4,
                marginal_cost=3.5,
                carrier="coal")
    network.add("Generator", "Gas",
                bus="Bengaluru",
                p_nom=800,
                marginal_cost=6.0,
                carrier="gas")
    network.add("Generator", "Solar",
                bus="Bengaluru",
                p_nom=400,
                marginal_cost=0.0,
                p_max_pu=solar_profile,
                carrier="solar")

    if include_load_shedding:
        network.add("Generator", "Load_Shedding",
                    bus="Bengaluru",
                    p_nom=max(float(load_24h.max()) * 1.5, 1000.0),
                    marginal_cost=load_shedding_cost,
                    carrier="unserved")

    network.add("Load", "City_Load", bus="Bengaluru", p_set=load_24h)

    dispatch_status = "optimal"
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            optimize_result = network.optimize(solver_name="highs", solver_options={"output_flag": False, "log_to_console": False})
        if optimize_result is not None:
            status_text = "_".join(map(str, optimize_result)) if isinstance(optimize_result, tuple) else str(optimize_result)
            dispatch_status = "optimal" if "optimal" in status_text.lower() else status_text
    except Exception:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                network.lopf(pyomo=False)
            dispatch_status = "optimal_fallback"
        except Exception as exc:
            dispatch_status = f"solver_failed: {exc}"
            return network, dispatch_status, merit_order_dispatch(
                load_24h, solar_profile, include_load_shedding, load_shedding_cost
            )

    if not hasattr(network, "generators_t") or network.generators_t.p.empty:
        dispatch_status = "empty_dispatch"
        return network, dispatch_status, merit_order_dispatch(
            load_24h, solar_profile, include_load_shedding, load_shedding_cost
        )

    gen = network.generators_t.p
    coal = gen["Coal"].to_numpy() if "Coal" in gen.columns else np.zeros(24)
    gas = gen["Gas"].to_numpy() if "Gas" in gen.columns else np.zeros(24)
    solar = gen["Solar"].to_numpy() if "Solar" in gen.columns else np.zeros(24)
    shortage = (gen["Load_Shedding"].to_numpy()
                if "Load_Shedding" in gen.columns else np.zeros(24))

    total_cost = (
        coal.sum() * 3.5
        + gas.sum() * 6.0
        + shortage.sum() * load_shedding_cost
    )

    result = {
        "total_cost": total_cost,
        "coal_generation": coal,
        "gas_generation": gas,
        "solar_generation": solar,
        "unserved_energy": shortage,
        "peak_load": float(load_24h.max()),
    }
    return network, dispatch_status, result

def merit_order_dispatch(load_24h, solar_profile,
                          include_load_shedding=False,
                          load_shedding_cost=LOAD_SHEDDING_COST):
    """Deterministic fallback used only when a PyPSA solver is unavailable."""
    load_24h = np.maximum(np.asarray(load_24h, dtype=float), 0.0)
    solar = np.minimum(400 * solar_profile, load_24h)
    remaining = np.maximum(load_24h - solar, 0.0)
    coal = np.minimum(1500, remaining)
    remaining = np.maximum(remaining - coal, 0.0)
    gas = np.minimum(800, remaining)
    remaining = np.maximum(remaining - gas, 0.0)
    shortage = remaining if include_load_shedding else np.zeros_like(load_24h)
    total_cost = coal.sum() * 3.5 + gas.sum() * 6.0 + shortage.sum() * load_shedding_cost
    return {
        "total_cost": total_cost,
        "coal_generation": coal,
        "gas_generation": gas,
        "solar_generation": solar,
        "unserved_energy": shortage,
        "peak_load": float(load_24h.max()),
    }
