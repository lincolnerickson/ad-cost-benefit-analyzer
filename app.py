import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit, minimize_scalar
import pandas as pd
import os
import json

st.set_page_config(page_title="Advertising Cost-Benefit Analyzer", layout="wide")

# --- Password Protection ---
def _check_password():
    """Returns True if the user has entered the correct password."""
    # Skip auth if no password is configured (local development)
    if "password" not in st.secrets:
        return True

    if st.session_state.get("_authenticated"):
        return True

    st.title("Advertising Cost-Benefit Analyzer")
    pwd = st.text_input("Enter password to access the app", type="password", key="_login_pwd")
    if pwd:
        if pwd == st.secrets["password"]:
            st.session_state["_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

if not _check_password():
    st.stop()

st.title("Advertising Cost-Benefit Analyzer")

# --- Persist parameters across browser refreshes ---
_SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".saved_params.json")
_PARAM_KEYS = [
    "price", "cost", "current_spend", "base_demand", "model_type",
    "hill_source", "hill_alpha", "hill_gamma", "hill_max_spend", "hill_effectiveness",
    "use_calculator", "calc_spend", "calc_extra_sales", "effectiveness",
]


def _load_saved_params():
    """Load saved params into session_state on first run."""
    if "params_loaded" in st.session_state:
        return
    st.session_state["params_loaded"] = True
    if os.path.exists(_SETTINGS_PATH):
        try:
            with open(_SETTINGS_PATH, "r") as f:
                saved = json.load(f)
            for k, v in saved.items():
                if k in _PARAM_KEYS and k not in st.session_state:
                    st.session_state[k] = v
        except Exception:
            pass


def _save_current_params():
    """Save current widget values to disk."""
    params = {}
    for k in _PARAM_KEYS:
        if k in st.session_state:
            params[k] = st.session_state[k]
    try:
        with open(_SETTINGS_PATH, "w") as f:
            json.dump(params, f)
    except Exception:
        pass


_load_saved_params()


# --- Multi-channel file system ---
_CHANNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "channels")
os.makedirs(_CHANNELS_DIR, exist_ok=True)
_mult_options = {
    "Actual values (1x)": 1, "In tens (x10)": 10, "In hundreds (x100)": 100,
    "In thousands (x1,000)": 1000, "In millions (x1,000,000)": 1_000_000,
}


def _list_channels():
    """Return sorted list of saved channel names (from settings JSON files)."""
    if not os.path.isdir(_CHANNELS_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0].removesuffix("_settings")
        for f in os.listdir(_CHANNELS_DIR)
        if f.endswith("_settings.json")
    )


def _load_channel(name):
    """Load a channel's source file and return dataframe, or None on failure."""
    settings = _load_channel_settings(name)
    source = settings.get("source_path")
    if not source or not os.path.exists(source):
        return None
    try:
        if source.endswith((".xlsx", ".xls")):
            return pd.read_excel(source)
        return pd.read_csv(source)
    except Exception:
        return None


def _delete_channel(name):
    """Delete a channel's settings JSON."""
    json_path = os.path.join(_CHANNELS_DIR, f"{name}_settings.json")
    if os.path.exists(json_path):
        os.remove(json_path)


def _save_channel_settings(name, settings):
    """Merge per-channel widget settings into existing JSON (preserves source_path etc.)."""
    path = os.path.join(_CHANNELS_DIR, f"{name}_settings.json")
    existing = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(settings)
    try:
        with open(path, "w") as f:
            json.dump(existing, f)
    except Exception:
        pass


def _load_channel_settings(name):
    """Load per-channel widget settings from JSON."""
    path = os.path.join(_CHANNELS_DIR, f"{name}_settings.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

# --- Ensure selected_channel is initialized before precompute ---
if "selected_channel" not in st.session_state:
    _early_channels = _list_channels()
    if _early_channels:
        st.session_state["selected_channel"] = _early_channels[0]
    else:
        st.session_state["selected_channel"] = "(Upload new channel)"

# --- Pre-compute: fit curves for selected channel (needed before sidebar) ---
_selected_channel = st.session_state.get("selected_channel")
if _selected_channel and _selected_channel != "(Upload new channel)" and "_fitted_from_upload" not in st.session_state:
    _preload_df = _load_channel(_selected_channel)
    if _preload_df is not None:
        try:
            _preload_df.columns = [c.strip().lower() for c in _preload_df.columns]
            _pre_numeric = [c for c in _preload_df.columns if pd.api.types.is_numeric_dtype(_preload_df[c])]
            _ch_settings = _load_channel_settings(_selected_channel)

            if len(_pre_numeric) >= 2:
                _pre_spend_col = _ch_settings.get("spend_col", st.session_state.get("upload_spend_col", _pre_numeric[0]))
                _pre_resp_col = _ch_settings.get("response_col", st.session_state.get("upload_response_col", _pre_numeric[min(1, len(_pre_numeric) - 1)]))
                _pre_spend_mult = _mult_options.get(_ch_settings.get("spend_mult", st.session_state.get("upload_spend_mult", "Actual values (1x)")), 1)
                _pre_resp_mult = _mult_options.get(_ch_settings.get("resp_mult", st.session_state.get("upload_resp_mult", "Actual values (1x)")), 1)
                _pre_resp_type = _ch_settings.get("resp_type", st.session_state.get("upload_resp_type", "Revenue ($)"))

                if _pre_spend_col in _preload_df.columns and _pre_resp_col in _preload_df.columns and _pre_spend_col != _pre_resp_col:
                    _pre_spend = _preload_df[_pre_spend_col].dropna().values.astype(float) * _pre_spend_mult
                    _pre_resp = _preload_df[_pre_resp_col].dropna().values.astype(float) * _pre_resp_mult
                    _pre_minlen = min(len(_pre_spend), len(_pre_resp))
                    _pre_spend = _pre_spend[:_pre_minlen]
                    _pre_resp = _pre_resp[:_pre_minlen]
                    _pre_pos = _pre_spend > 0
                    _pre_spend = _pre_spend[_pre_pos]
                    _pre_resp = _pre_resp[_pre_pos]

                    if len(_pre_spend) >= 3:
                        def _pre_sqrt(x, b, e):
                            return b + e * np.sqrt(x)
                        def _pre_log(x, b, e):
                            return b + e * np.log1p(x)
                        _pre_max = float(np.max(_pre_spend))
                        def _pre_hill(x, b, e, alpha, gamma):
                            inflexion = float(gamma) * _pre_max
                            x = np.asarray(x, dtype=float)
                            xa = np.power(np.maximum(x, 0), float(alpha))
                            ga = np.power(max(inflexion, 1e-9), float(alpha))
                            return b + e * (xa / (xa + ga))

                        try:
                            _pp_sq, _ = curve_fit(_pre_sqrt, _pre_spend, _pre_resp, p0=[np.min(_pre_resp), 1])
                            _pp_lg, _ = curve_fit(_pre_log, _pre_spend, _pre_resp, p0=[np.min(_pre_resp), 1])
                            _pp_hl, _ = curve_fit(
                                _pre_hill, _pre_spend, _pre_resp,
                                p0=[np.min(_pre_resp), np.max(_pre_resp) - np.min(_pre_resp), 1.0, 0.5],
                                bounds=([0, 0, 0.1, 0.01], [np.inf, np.inf, 5.0, 1.0]),
                                maxfev=10000,
                            )
                            _psq_b, _psq_e = float(_pp_sq[0]), float(_pp_sq[1])
                            _plg_b, _plg_e = float(_pp_lg[0]), float(_pp_lg[1])
                            _ph_b, _ph_e, _ph_a, _ph_g = [float(v) for v in _pp_hl]

                            def _adj_r2(y_true, y_pred, n_params):
                                n = len(y_true)
                                ss_res = np.sum((y_true - y_pred) ** 2)
                                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                                if ss_tot == 0:
                                    return 0.0
                                r2 = 1 - ss_res / ss_tot
                                if n <= n_params + 1:
                                    return r2
                                return 1 - (1 - r2) * (n - 1) / (n - n_params - 1)

                            _pre_n = len(_pre_spend)
                            _pre_use_adj = _pre_n > 5
                            if _pre_use_adj:
                                _pr2_sq = _adj_r2(_pre_resp, _pre_sqrt(_pre_spend, _psq_b, _psq_e), 2)
                                _pr2_lg = _adj_r2(_pre_resp, _pre_log(_pre_spend, _plg_b, _plg_e), 2)
                                _pr2_hl = _adj_r2(_pre_resp, _pre_hill(_pre_spend, _ph_b, _ph_e, _ph_a, _ph_g), 4)
                            else:
                                _pr2_sq = _adj_r2(_pre_resp, _pre_sqrt(_pre_spend, _psq_b, _psq_e), 0)
                                _pr2_lg = _adj_r2(_pre_resp, _pre_log(_pre_spend, _plg_b, _plg_e), 0)
                                _pr2_hl = _adj_r2(_pre_resp, _pre_hill(_pre_spend, _ph_b, _ph_e, _ph_a, _ph_g), 0)
                            _pr2s = {"sqrt": _pr2_sq, "log": _pr2_lg, "Hill (Robyn)": _pr2_hl}
                            _p_best = max(_pr2s, key=lambda k: (_pr2s[k], -len(k)))

                            st.session_state["_fitted_from_upload"] = {
                                "sqrt_e": _psq_e, "sqrt_b": _psq_b,
                                "log_e": _plg_e, "log_b": _plg_b,
                                "hill_e": _ph_e, "hill_b": _ph_b,
                                "hill_alpha": _ph_a, "hill_gamma": _ph_g,
                                "hill_max_spend": _pre_max,
                                "best_model": _p_best, "best_r2": _pr2s[_p_best],
                                "resp_type": _pre_resp_type,
                            }
                        except Exception:
                            pass
        except Exception:
            pass


# --- Adjusted R² (penalizes overfitting with more parameters) ---
def adjusted_r2(y_true, y_pred, n_params):
    """Adjusted R² = 1 - (1-R²)*(n-1)/(n-p-1). Falls back to raw R² for small samples."""
    n = len(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    r2 = 1 - ss_res / ss_tot
    # Not enough data to compute adjusted R² — return raw R² (flagged elsewhere)
    if n <= n_params + 1:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - n_params - 1)


def show_r2_benchmarks(best_r2=None):
    """Display R² advertising benchmarks as an expander."""
    with st.expander("R² Benchmarks for Advertising"):
        if best_r2 is not None:
            if best_r2 >= 0.80:
                st.success(f"Your best R² of {best_r2:.3f} is **strong** — the model captures most of the variance.")
            elif best_r2 >= 0.60:
                st.info(f"Your best R² of {best_r2:.3f} is **decent** — typical for advertising data.")
            elif best_r2 >= 0.40:
                st.warning(f"Your best R² of {best_r2:.3f} is **weak** — directionally useful, but other factors dominate.")
            else:
                st.error(f"Your best R² of {best_r2:.3f} is **poor** — spend alone may not explain this channel well.")

        st.markdown("""
**What's a good R² for advertising?**

| R² Range | Quality | Interpretation |
|----------|---------|----------------|
| **0.80+** | Strong | Model explains most variation in sales/revenue |
| **0.60 – 0.80** | Decent | Typical for well-tracked digital channels |
| **0.40 – 0.60** | Weak | Directionally useful, other factors at play |
| **< 0.40** | Poor | Spend alone doesn't explain this channel well |

**Typical R² by channel:**

| Channel | Typical R² | Why |
|---------|-----------|-----|
| Search / SEM | 0.50 – 0.80 | Strongest direct response; intent-driven |
| Facebook / Social | 0.30 – 0.60 | Mix of brand + performance; creative varies |
| TV | 0.20 – 0.50 | Delayed effect; hard to isolate from other media |
| Print / OOH | 0.15 – 0.40 | Long-term brand building; weak short-term signal |

**Why R² is never 1.0 in advertising:**
- Ad spend is only one factor — seasonality, pricing, competitors, and organic demand also drive sales
- Creative quality varies across campaigns at the same spend level
- There's often a lag between spending and results (especially brand channels)
- External events (holidays, weather, news) add noise
""")


# --- Hill Saturation Function (from Meta's Robyn) ---
def hill_saturation(x, alpha, gamma_point):
    """Hill function: x^alpha / (x^alpha + gamma_point^alpha). Output in [0, 1]."""
    x = np.asarray(x, dtype=float)
    alpha = float(alpha)
    gamma_point = float(gamma_point)
    xa = np.power(np.maximum(x, 0), alpha)
    ga = np.power(max(gamma_point, 1e-9), alpha)
    return xa / (xa + ga)


# --- Sidebar Inputs ---
st.sidebar.header("Parameters")

price = st.sidebar.number_input("Price per Unit ($)", min_value=0.0, value=50.0, step=0.5, key="price")
cost = st.sidebar.number_input("Cost per Unit ($)", min_value=0.0, value=20.0, step=0.5, key="cost")
current_spend = st.sidebar.number_input("Current Ad Spend ($)", min_value=0, value=5000, step=100, key="current_spend")
with st.sidebar.expander("Base Demand (advanced)"):
    st.caption("Units you'd sell with zero ad spend. Shifts the profit curve up/down but **does not change optimal ad spend**.")
    base_demand = st.number_input("Base Demand (units)", min_value=0, value=100, step=1, key="base_demand")
model_type = st.sidebar.radio("Demand Model", ["sqrt", "log", "Hill (Robyn)"], horizontal=True, key="model_type")

# --- Hill Parameters (only shown when Hill model selected) ---
if model_type == "Hill (Robyn)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Hill Function Parameters")

    _fitted_hill = st.session_state.get("_fitted_from_upload")
    _has_hill_upload = _fitted_hill is not None

    # Source selector for Hill params
    _hill_shape_options = ["Derive from a campaign", "Manual"]
    if _has_hill_upload:
        _hill_shape_options.insert(0, "From uploaded data")

    selected_source = st.sidebar.selectbox(
        "Load parameters from",
        _hill_shape_options,
        index=0,
        help="Load all Hill params from fitted data, derive from a single campaign, or enter manually.",
        key="hill_source",
    )

    # --- Hill params: From uploaded data ---
    if selected_source == "From uploaded data" and _has_hill_upload:
        hill_alpha = round(_fitted_hill["hill_alpha"], 2)
        hill_gamma = round(_fitted_hill["hill_gamma"], 2)
        hill_max_spend = round(_fitted_hill["hill_max_spend"])
        _hem_raw = _fitted_hill["hill_e"]
        _hem_resp = _fitted_hill.get("resp_type", "Revenue ($)")
        if _hem_resp == "Revenue ($)" and price > 0:
            effectiveness = max(round(_hem_raw / price, 2), 0.1)
            _e_note = f"{_hem_raw:,.0f} / ${price:,.2f} = **{effectiveness:,.2f}** (revenue → units)"
        else:
            effectiveness = max(round(_hem_raw, 1), 1.0)
            _e_note = f"**{effectiveness:,.0f}**"
        st.sidebar.success(
            f"**All Hill params from uploaded data:**\n\n"
            f"- Alpha (shape): **{hill_alpha}**\n"
            f"- Gamma (inflection): **{hill_gamma}**\n"
            f"- E (effectiveness): {_e_note}\n"
            f"- Max spend: **${hill_max_spend:,.0f}**\n"
            f"- R²: {_fitted_hill.get('best_r2', 0):.4f}"
        )

    # --- Hill params: Derive from a campaign ---
    elif selected_source == "Derive from a campaign":
        st.sidebar.caption(
            "Answer a few questions about one campaign to derive all Hill parameters."
        )

        st.sidebar.markdown("**Your campaign:**")
        _dc_spend = st.sidebar.number_input(
            "Ad spend in that campaign ($)", min_value=1.0, value=10000.0, step=100.0, key="hill_dc_spend"
        )
        _dc_extra = st.sidebar.number_input(
            "Extra units sold from ads", min_value=1.0, value=400.0, step=10.0, key="hill_dc_extra"
        )
        hill_max_spend = st.sidebar.number_input(
            "Max realistic ad spend ($)", min_value=100, value=50000, step=1000,
            help="The highest you'd ever spend on this channel.",
            key="hill_max_spend",
        )

        st.sidebar.markdown("**Curve shape:**")
        _dc_shape = st.sidebar.radio(
            "How do ads behave at low spend?",
            ["Diminishing returns from the start (C-curve)", "Slow start, then accelerates (S-curve)"],
            key="hill_dc_shape",
            help="C-curve: every dollar helps but with decreasing returns (like sqrt). S-curve: need minimum spend before ads kick in.",
        )
        if _dc_shape.startswith("Diminishing"):
            hill_alpha = st.sidebar.number_input("Alpha", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="hill_alpha",
                                                  help="< 1 for C-curve. Lower = more aggressive diminishing returns.")
        else:
            hill_alpha = st.sidebar.number_input("Alpha", min_value=1.0, max_value=5.0, value=2.0, step=0.1, key="hill_alpha",
                                                  help="> 1 for S-curve. Higher = sharper threshold effect.")

        st.sidebar.markdown("**Saturation estimate:**")
        _dc_sat_pct = st.sidebar.slider(
            "At that spend, what % of maximum ad effect have you captured?",
            min_value=10, max_value=90, value=50, step=5, key="hill_dc_sat_pct",
            help="50% = you're at the midpoint. 80% = nearly saturated. 20% = lots of room to grow.",
        )

        # Derive gamma and E from the inputs
        _dc_sat_frac = _dc_sat_pct / 100.0
        # From: sat(spend) = spend^α / (spend^α + inflexion^α) = sat_frac
        # Solve: inflexion = spend × ((1 - sat_frac) / sat_frac) ^ (1/α)
        _dc_inflexion = _dc_spend * ((1.0 - _dc_sat_frac) / _dc_sat_frac) ** (1.0 / hill_alpha)
        hill_gamma = min(max(round(_dc_inflexion / hill_max_spend, 4), 0.01), 1.0)

        # E = extra_units / sat_frac
        effectiveness = max(round(_dc_extra / _dc_sat_frac, 1), 0.1)

        # Verify by computing actual saturation at the campaign spend
        _dc_infl_actual = hill_gamma * hill_max_spend
        _dc_sat_check = hill_saturation(np.array([_dc_spend]), hill_alpha, _dc_infl_actual)[0]

        st.sidebar.success(
            f"**Derived Hill parameters:**\n\n"
            f"- Alpha: **{hill_alpha}**\n"
            f"- Gamma: **{hill_gamma}** (inflection at **${_dc_infl_actual:,.0f}**)\n"
            f"- E: **{effectiveness:,.1f}** (max extra units from ads)\n"
            f"- At ${_dc_spend:,.0f}: **{_dc_sat_check:.0%}** saturated → **{effectiveness * _dc_sat_check:,.0f}** extra units"
        )

    # --- Hill params: Full manual entry ---
    else:
        st.sidebar.caption(
            "From Meta's Robyn MMM framework. "
            "Alpha controls curve shape, gamma controls the inflection point."
        )
        hill_alpha = st.sidebar.number_input(
            "Alpha (shape: <1 = C-curve, >1 = S-curve)", min_value=0.1, value=1.5, step=0.1, key="hill_alpha"
        )
        hill_gamma = st.sidebar.number_input(
            "Gamma (inflection: 0-1, fraction of max spend)", min_value=0.01, max_value=1.0, value=0.5, step=0.01, key="hill_gamma"
        )
        hill_max_spend = st.sidebar.number_input(
            "Max ad spend in your range ($)", min_value=100, value=50000, step=1000,
            help="Your maximum realistic ad spend. The inflection point = gamma x this value.",
            key="hill_max_spend",
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("Ad Effectiveness (E)")

        _hill_e_options = ["Calculate from a campaign", "Manual"]
        if _has_hill_upload:
            _hill_e_options.insert(0, "From uploaded data")

        _hill_e_source = st.sidebar.radio(
            "E source",
            _hill_e_options,
            horizontal=True,
            key="hill_e_source",
        )

        if _hill_e_source == "From uploaded data" and _has_hill_upload:
            _hem_raw = _fitted_hill["hill_e"]
            _hem_resp = _fitted_hill.get("resp_type", "Revenue ($)")
            if _hem_resp == "Revenue ($)" and price > 0:
                effectiveness = max(round(_hem_raw / price, 2), 0.1)
                st.sidebar.success(f"**E = {_hem_raw:,.0f} / ${price:,.2f} = {effectiveness:,.2f}** (revenue → units)")
            else:
                effectiveness = max(round(_hem_raw, 1), 1.0)
                st.sidebar.success(f"**E = {effectiveness:,.0f}** (from uploaded data Hill fit)")
        elif _hill_e_source == "Calculate from a campaign":
            st.sidebar.caption(
                "Enter a past campaign's numbers to calculate E. "
                "E = extra units sold / Hill(spend)."
            )
            _hill_calc_spend = st.sidebar.number_input("Ad spend in that campaign ($)", min_value=1.0, value=10000.0, step=100.0, key="hill_calc_spend")
            _hill_calc_extra = st.sidebar.number_input("Extra units sold from ads", min_value=1.0, value=400.0, step=10.0, key="hill_calc_extra")
            infl = hill_gamma * hill_max_spend
            hill_val = hill_saturation(np.array([_hill_calc_spend]), hill_alpha, infl)[0]
            if hill_val > 0.001:
                calc_e = _hill_calc_extra / hill_val
                effectiveness = max(round(calc_e, 1), 0.1)
                st.sidebar.markdown(
                    f"**E = {_hill_calc_extra:,.0f} / Hill({_hill_calc_spend:,.0f}) = {_hill_calc_extra:,.0f} / {hill_val:.4f} = {calc_e:,.1f}**"
                )
                st.sidebar.info(f"Using calculated E = {effectiveness}")
            else:
                st.sidebar.warning(f"Hill saturation near zero at ${_hill_calc_spend:,.0f}. Try higher spend or adjust gamma/max spend.")
                effectiveness = 500.0
        else:
            effectiveness = st.sidebar.number_input(
                "E (max incremental units from ads)", min_value=1.0, value=500.0, step=10.0,
                help="The maximum extra units ads could ever generate above base demand.",
                key="hill_effectiveness",
            )

    infl_point = hill_gamma * hill_max_spend
    st.sidebar.info(
        f"Inflection point: **${infl_point:,.0f}** (50% saturation)\n\n"
        f"At max spend (${hill_max_spend:,.0f}): **{base_demand + effectiveness * hill_saturation(np.array([hill_max_spend]), hill_alpha, infl_point)[0]:,.0f}** units"
    )

# --- Ad Effectiveness Calculator (for sqrt/log only) ---
if model_type != "Hill (Robyn)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ad Effectiveness (E)")

    _fitted = st.session_state.get("_fitted_from_upload")
    _has_upload_fit = _fitted is not None

    if _has_upload_fit:
        _fit_e_key = "sqrt_e" if model_type == "sqrt" else "log_e"
        _fit_e_val = _fitted.get(_fit_e_key, 10.0)
        _fit_b_key = "sqrt_b" if model_type == "sqrt" else "log_b"
        _fit_b_val = _fitted.get(_fit_b_key, 0.0)

    e_source = st.sidebar.radio(
        "E source",
        ["From uploaded data", "Calculate from a campaign", "Manual"] if _has_upload_fit else ["Calculate from a campaign", "Manual"],
        horizontal=True,
        key="e_source_sqlog",
    )

    if e_source == "From uploaded data" and _has_upload_fit:
        _fit_resp_type = _fitted.get("resp_type", "Revenue ($)")
        if _fit_resp_type == "Revenue ($)" and price > 0:
            _raw_e = _fit_e_val
            _converted_e = _raw_e / price
            effectiveness = max(round(_converted_e, 4), 0.1)
            st.sidebar.success(
                f"**E = {_raw_e:,.2f} / ${price:,.2f} = {effectiveness:,.4f}**\n\n"
                f"Fitted E was in revenue $ — divided by price to get units.\n\n"
                f"Base intercept from fit: {_fit_b_val:,.1f}"
            )
        else:
            effectiveness = max(round(_fit_e_val, 2), 0.1)
            st.sidebar.success(
                f"**E = {effectiveness:,.2f}** (fitted from uploaded data, {model_type} model)\n\n"
                f"Base intercept from fit: {_fit_b_val:,.1f}"
            )
    elif e_source == "Calculate from a campaign":
        st.sidebar.caption(
            "Enter a past campaign's numbers to calculate E. "
            "E represents extra units sold per $1 of ad spend (via the selected model)."
        )
        calc_spend = st.sidebar.number_input("Ad spend in that campaign ($)", min_value=1.0, value=10000.0, step=100.0, key="calc_spend")
        calc_extra_sales = st.sidebar.number_input("Extra units sold from ads", min_value=1.0, value=400.0, step=10.0, key="calc_extra_sales")

        if model_type == "sqrt":
            calc_e = calc_extra_sales / np.sqrt(calc_spend) if calc_spend > 0 else 0.1
            st.sidebar.markdown(
                f"**E = {calc_extra_sales:,.0f} / sqrt({calc_spend:,.0f}) = {calc_e:.2f}**"
            )
        else:
            calc_e = calc_extra_sales / np.log1p(calc_spend) if calc_spend > 0 else 0.1
            st.sidebar.markdown(
                f"**E = {calc_extra_sales:,.0f} / log(1+{calc_spend:,.0f}) = {calc_e:.2f}**"
            )

        effectiveness = max(round(calc_e, 2), 0.1)
        st.sidebar.info(f"Using calculated E = {effectiveness}")
    else:
        effectiveness = st.sidebar.number_input("Ad Effectiveness (E)", min_value=0.1, value=10.0, step=0.1, key="effectiveness")

margin = price - cost

# Save all parameters after widgets are set
_save_current_params()


# --- Model Functions ---
def units_sold(ad_spend, model=model_type):
    a = np.asarray(ad_spend, dtype=float)
    if model == "sqrt":
        return base_demand + effectiveness * np.sqrt(a)
    elif model == "log":
        return base_demand + effectiveness * np.log1p(a)
    else:  # Hill (Robyn)
        inflexion = hill_gamma * hill_max_spend
        return base_demand + effectiveness * hill_saturation(a, hill_alpha, inflexion)


def revenue(ad_spend):
    return units_sold(ad_spend) * price


def total_cost(ad_spend):
    a = np.asarray(ad_spend, dtype=float)
    return cost * units_sold(a) + a


def profit(ad_spend):
    return revenue(ad_spend) - total_cost(ad_spend)


# --- Optimal Ad Spend ---
if model_type == "sqrt":
    optimal_spend = ((margin * effectiveness) / 2) ** 2 if margin > 0 else 0.0
elif model_type == "log":
    optimal_spend = max(margin * effectiveness - 1, 0.0)
else:  # Hill (Robyn) — grid search + refine (Brent's method fails on S-curves)
    if margin > 0:
        # Coarse grid to find approximate peak (Brent's gets stuck at 0 for S-curves)
        _hill_grid = np.linspace(0, 200000, 2000)
        _hill_profits = profit(_hill_grid)
        _hill_best_idx = int(np.argmax(_hill_profits))
        _hill_lo = _hill_grid[max(_hill_best_idx - 1, 0)]
        _hill_hi = _hill_grid[min(_hill_best_idx + 1, len(_hill_grid) - 1)]
        # Refine around the peak
        result = minimize_scalar(lambda a: -profit(a), bounds=(_hill_lo, _hill_hi), method="bounded")
        optimal_spend = float(result.x)
    else:
        optimal_spend = 0.0

optimal_spend = float(optimal_spend)
max_profit = float(profit(optimal_spend))
current_profit = float(profit(current_spend))
contribution_margin = margin

# --- Diminishing returns threshold: where marginal REVENUE per $1 of ad spend drops below $1 ---
# i.e., the spend level where spending $1 more generates less than $1 extra revenue (marginal ROAS = 1)
# Uses E_rev = effectiveness * price, so it works correctly regardless of sidebar cost/margin settings
_E_rev = effectiveness * price  # revenue-equivalent effectiveness
diminishing_threshold = None
if _E_rev > 0:
    if model_type == "sqrt":
        # marginal revenue = E_rev / (2*sqrt(a)); set = 1 → a = (E_rev/2)^2
        _dr_val = (_E_rev / 2.0) ** 2
        if _dr_val >= 0.01:
            diminishing_threshold = float(_dr_val)
    elif model_type == "log":
        # marginal revenue = E_rev / (1+a); set = 1 → a = E_rev - 1
        _dr_val = _E_rev - 1.0
        if _dr_val >= 0.01:
            diminishing_threshold = float(_dr_val)
    else:  # Hill — numerical with log-spaced grid for precision at low spend
        _hill_infl = hill_gamma * hill_max_spend
        _dr_max = max(optimal_spend * 3, _hill_infl * 10, 50000)
        _dr_range = np.geomspace(0.1, _dr_max, 10000)
        _dr_revenues = revenue(_dr_range)
        _dr_marginal = np.diff(_dr_revenues) / np.diff(_dr_range)
        # For S-curves (alpha>1), marginal revenue goes low→high→low.
        # Find the LAST transition from ≥$1 to <$1 (not the first, which is just the S-curve start).
        _above_1 = _dr_marginal >= 1.0
        _transitions = np.where(np.diff(_above_1.astype(int)) == -1)[0]  # ≥1 → <1
        if len(_transitions) > 0:
            diminishing_threshold = float(_dr_range[_transitions[-1] + 1])
        elif not np.any(_above_1):
            # Marginal revenue is always below $1 — ads never pay for themselves at any level
            diminishing_threshold = None

# --- Hill 50% saturation point ---
hill_half_sat_spend = None
if model_type == "Hill (Robyn)":
    hill_half_sat_spend = float(hill_gamma * hill_max_spend)

# --- Dynamic chart range ---
# Show enough of the curve to see the peak and decline, centered on the interesting region
key_points = [optimal_spend, current_spend, 1000]  # always include at least 1000
chart_max = max(key_points) * 3  # show 3x the largest key point
chart_max = max(chart_max, 5000)  # minimum range of $5,000
chart_max = float(chart_max)

# --- Break-even ad spend (find where profit = 0) ---
a_range_be = np.linspace(0, chart_max * 2, 5000)
profit_vals_be = profit(a_range_be)
break_even_spend = None
if profit_vals_be[0] >= 0:
    crossings = np.where(np.diff(np.sign(profit_vals_be)))[0]
    if len(crossings) > 0:
        break_even_spend = float(a_range_be[crossings[-1]])
        chart_max = max(chart_max, break_even_spend * 1.2)  # show past break-even
else:
    crossings = np.where(np.diff(np.sign(profit_vals_be)))[0]
    if len(crossings) > 0:
        break_even_spend = float(a_range_be[crossings[0]])

# --- Key Metrics Row 1 ---
st.subheader("Key Metrics")
st.caption(
    "All values are **per row** of your uploaded data. "
    "If each row is a day, these are daily figures; if weekly, they're weekly; etc."
)
if optimal_spend < 10 and effectiveness > 0 and margin > 0:
    st.warning(
        f"Optimal ad spend is very low (${optimal_spend:.2f}). This usually means the sidebar **Price** (${price:,.2f}) "
        f"and **Cost** (${cost:,.2f}) don't match your data. "
        f"The profit model uses margin (${margin:,.2f}) × E ({effectiveness:,.4f}) = {margin * effectiveness:.4f}. "
        f"Try adjusting Price/Cost to reflect your actual unit economics."
    )
col1, col2, col3, col4 = st.columns(4)
col1.metric("Optimal Ad Spend", f"${optimal_spend:,.0f}")
col2.metric("Profit at Optimal Ad Spend", f"${max_profit:,.0f}")
col3.metric(
    "Profit at Current Spend",
    f"${current_profit:,.0f}",
    delta=f"${current_profit - max_profit:,.0f} vs optimal",
)
col4.metric("Contribution Margin", f"${contribution_margin:,.2f}/unit")

# --- Key Metrics Row 2 ---
col5, col6, col7, col8 = st.columns(4)
col5.metric("Units at Current Spend", f"{units_sold(current_spend):,.0f}")
col6.metric("Units at Optimal Spend", f"{units_sold(optimal_spend):,.0f}")
col7.metric("Revenue at Optimal", f"${revenue(optimal_spend):,.0f}")
col8.metric(
    "Break-even Ad Spend",
    f"${break_even_spend:,.0f}" if break_even_spend is not None else "N/A",
)

# --- ROAS Metrics ---
# ROAS = Revenue attributable to ads / Ad Spend
# Revenue from ads = (units from ads) * price = (units_sold - base_demand) * price
_ad_units_current = float(units_sold(current_spend)) - base_demand
_ad_revenue_current = _ad_units_current * price
_roas_current = _ad_revenue_current / current_spend if current_spend > 0 else 0

_ad_units_optimal = float(units_sold(optimal_spend)) - base_demand
_ad_revenue_optimal = _ad_units_optimal * price
_roas_optimal = _ad_revenue_optimal / optimal_spend if optimal_spend > 0 else 0

# Marginal ROAS at current spend (revenue from next $1)
_mroas_step = max(current_spend * 0.01, 1)
_mroas_rev1 = float(revenue(current_spend))
_mroas_rev2 = float(revenue(current_spend + _mroas_step))
_mroas_current = (_mroas_rev2 - _mroas_rev1) / _mroas_step if _mroas_step > 0 else 0

st.subheader("ROAS (Return on Ad Spend)")
rc1, rc2, rc3, rc4 = st.columns(4)
rc1.metric(
    "ROAS at Current Spend",
    f"{_roas_current:.2f}x",
    help=f"${_ad_revenue_current:,.0f} ad-driven revenue / ${current_spend:,.0f} spend",
)
rc2.metric(
    "ROAS at Optimal Spend",
    f"{_roas_optimal:.2f}x",
    help=f"${_ad_revenue_optimal:,.0f} ad-driven revenue / ${optimal_spend:,.0f} spend",
)
rc3.metric(
    "Marginal ROAS (current)",
    f"{_mroas_current:.2f}x",
    delta=f"{'profitable' if _mroas_current > 1 else 'unprofitable'} marginal $",
    help="Revenue generated by the next $1 of ad spend at your current level",
)
# E expressed as ROAS context
if current_spend > 0:
    _e_roas_note = (
        f"E = {effectiveness:,.2f} means at ${current_spend:,.0f} spend, "
        f"ads drive {_ad_units_current:,.0f} extra units "
        f"(${_ad_revenue_current:,.0f} revenue)"
    )
else:
    _e_roas_note = f"E = {effectiveness:,.2f} — set current spend to see ROAS"
rc4.metric("E as ROAS", f"{_roas_current:.2f}:1" if current_spend > 0 else "N/A", help=_e_roas_note)

# --- Key Metrics Row 3 ---
col9, col10, col11, col12 = st.columns(4)
col9.metric(
    "Diminishing Returns Threshold",
    f"${diminishing_threshold:,.0f}" if diminishing_threshold is not None else "N/A",
    help="Spend level where each extra $1 of ad spend generates less than $1 of additional revenue (marginal ROAS = 1)",
)
if model_type == "Hill (Robyn)" and hill_half_sat_spend is not None:
    col10.metric(
        "50% Saturation Point",
        f"${hill_half_sat_spend:,.0f}",
        help="Hill inflection: at this spend, you've captured 50% of max ad-driven sales",
    )
else:
    col10.metric("50% Saturation Point", "N/A (Hill only)", help="Only applies to the Hill demand model")
_dr_profit = float(profit(diminishing_threshold)) if diminishing_threshold is not None else 0
col11.metric(
    "Profit at DR Threshold",
    f"${_dr_profit:,.0f}" if diminishing_threshold is not None else "N/A",
)
_headroom = optimal_spend - (diminishing_threshold or 0)
col12.metric(
    "Efficient Headroom",
    f"${_headroom:,.0f}" if diminishing_threshold is not None else "N/A",
    help="The spend range between the DR threshold (where each $1 returns less than $1 in revenue) and optimal spend (where profit is maximized). Spending in this zone still adds profit, just at a declining rate.",
)

# --- Ad Effectiveness (E) Explainer ---
with st.expander("What is Ad Effectiveness (E)?"):
    _e_current_units = float(units_sold(current_spend)) - base_demand if current_spend > 0 else 0
    _e_current_rev = _e_current_units * price

    st.markdown(f"""
**E is the single number that captures how well your ads convert spend into sales.**

It connects your ad spend to the extra units (or revenue) your ads generate, using the selected demand model.
The higher E is, the more effective your advertising.

---

**How E works in each model:**

| Model | Formula | What E means |
|-------|---------|-------------|
| **sqrt** | Extra units = E × √(spend) | Each √dollar of spend produces E extra units |
| **log** | Extra units = E × ln(1 + spend) | Each log-dollar of spend produces E extra units |
| **Hill** | Extra units = E × saturation(spend) | E is the *maximum* extra units ads could ever produce |

---

**Your current E = {effectiveness:,.2f}** ({model_type} model)
""")

    if current_spend > 0:
        st.markdown(f"""
**Concrete example with your numbers:**
- At your current spend of **${current_spend:,.0f}**, ads generate **{_e_current_units:,.0f} extra units** (${_e_current_rev:,.0f} revenue)
- Without ads (spend = $0), you'd sell just the base demand of **{base_demand:,.0f} units**
- The difference — those {_e_current_units:,.0f} extra units — is what E produces via the {model_type} curve
""")

    st.markdown(f"""
---

**Where does E come from?**
- **From uploaded data** — fitted automatically by finding the curve that best matches your spend vs sales data
- **Calculate from a campaign** — enter a past campaign's spend and results, and E is back-calculated
- **Manual** — enter E directly if you know it

**Key insight:** E does *not* change with spend. It's a fixed property of how effective your advertising channel is.
What changes is the *diminishing returns* — the {model_type} curve means each additional dollar of spend produces
less and less additional revenue. E just scales the whole curve up or down.

**Higher E** → ads work better → higher optimal spend, more profit from ads
**Lower E** → ads are less effective → lower optimal spend, less reason to advertise
""")

# --- Chart Data ---
a_range = np.linspace(0, chart_max, 500)
profit_vals = profit(a_range)

# --- Chart 1: Profit vs Ad Spend (full width) ---
st.subheader("Profit vs Ad Spend")
fig1 = go.Figure()

# Shade zones: profitable growth vs overspending
if optimal_spend > 0:
    fig1.add_vrect(
        x0=0, x1=optimal_spend,
        fillcolor="rgba(0,180,0,0.07)", line_width=0,
        annotation_text="Profitable growth", annotation_position="top left",
        annotation_font_size=11,
    )
    fig1.add_vrect(
        x0=optimal_spend, x1=chart_max,
        fillcolor="rgba(255,0,0,0.05)", line_width=0,
        annotation_text="Overspending", annotation_position="top right",
        annotation_font_size=11,
    )

# Diminishing returns threshold (all models)
if diminishing_threshold is not None and diminishing_threshold < chart_max:
    _dr_chart_profit = float(profit(diminishing_threshold))
    fig1.add_vline(
        x=diminishing_threshold, line_dash="dot", line_color="purple", opacity=0.6,
    )
    fig1.add_trace(go.Scatter(
        x=[diminishing_threshold], y=[_dr_chart_profit],
        mode="markers", name=f"DR threshold (${diminishing_threshold:,.0f})",
        marker=dict(symbol="triangle-down", size=10, color="purple"),
    ))

# Hill 50% saturation point
if model_type == "Hill (Robyn)" and hill_half_sat_spend is not None:
    if hill_half_sat_spend < chart_max:
        _half_sat_profit = float(profit(hill_half_sat_spend))
        fig1.add_vline(
            x=hill_half_sat_spend, line_dash="dot", line_color="orange", opacity=0.7,
        )
        fig1.add_trace(go.Scatter(
            x=[hill_half_sat_spend], y=[_half_sat_profit],
            mode="markers", name=f"50% saturation (${hill_half_sat_spend:,.0f})",
            marker=dict(symbol="triangle-up", size=10, color="orange"),
        ))

fig1.add_trace(
    go.Scatter(x=a_range, y=profit_vals, mode="lines", name="Profit", line=dict(color="royalblue", width=2))
)
fig1.add_trace(
    go.Scatter(
        x=[optimal_spend], y=[max_profit],
        mode="markers", name=f"Optimal (${optimal_spend:,.0f})",
        marker=dict(symbol="star", size=14, color="red"),
    )
)
fig1.add_trace(
    go.Scatter(
        x=[current_spend], y=[current_profit],
        mode="markers", name=f"Current (${current_spend:,.0f})",
        marker=dict(symbol="diamond", size=12, color="green"),
    )
)
fig1.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even")
if current_spend > 0:
    fig1.add_vline(x=current_spend, line_dash="dot", line_color="green", opacity=0.5)
fig1.update_layout(
    xaxis_title="Ad Spend ($)", yaxis_title="Profit ($)",
    height=450, margin=dict(t=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)

# Explain the key thresholds
_caption_parts = []
if diminishing_threshold is not None:
    _caption_parts.append(
        f"**DR threshold** (${diminishing_threshold:,.0f}): each extra $1 of ad spend generates less than $1 of additional revenue beyond this point (marginal ROAS < 1)."
    )
if model_type == "Hill (Robyn)" and hill_half_sat_spend is not None:
    hill_at_opt = hill_saturation(
        np.array([optimal_spend]), hill_alpha, hill_gamma * hill_max_spend
    )[0]
    _caption_parts.append(
        f"**50% saturation** (${hill_half_sat_spend:,.0f}): half of all ad-driven sales captured. "
        f"Optimal spend (${optimal_spend:,.0f}) is at **{hill_at_opt:.0%} saturation**."
    )
if optimal_spend > 0:
    _caption_parts.append(
        f"**Optimal** (${optimal_spend:,.0f}): the profit-maximizing spend — beyond here you lose money on each extra dollar."
    )
if _caption_parts:
    st.caption(" | ".join(_caption_parts))

st.plotly_chart(fig1, use_container_width=True)


# --- Summary Table ---
st.subheader("Current vs Optimal Comparison")
current_units = float(units_sold(current_spend))
optimal_units = float(units_sold(optimal_spend))
current_revenue = float(revenue(current_spend))
optimal_revenue = float(revenue(optimal_spend))
current_total_cost = float(total_cost(current_spend))
optimal_total_cost = float(total_cost(optimal_spend))

summary_data = {
    "Metric": ["Ad Spend", "Units Sold", "Revenue", "Total Cost", "Profit", "ROI (Profit / Ad Spend)"],
    "Current": [
        f"${current_spend:,.0f}", f"{current_units:,.0f}", f"${current_revenue:,.0f}",
        f"${current_total_cost:,.0f}", f"${current_profit:,.0f}",
        f"{(current_profit / current_spend * 100):,.1f}%" if current_spend > 0 else "N/A",
    ],
    "Optimal": [
        f"${optimal_spend:,.0f}", f"{optimal_units:,.0f}", f"${optimal_revenue:,.0f}",
        f"${optimal_total_cost:,.0f}", f"${max_profit:,.0f}",
        f"{(max_profit / optimal_spend * 100):,.1f}%" if optimal_spend > 0 else "N/A",
    ],
    "Difference": [
        f"${optimal_spend - current_spend:,.0f}", f"{optimal_units - current_units:,.0f}",
        f"${optimal_revenue - current_revenue:,.0f}", f"${optimal_total_cost - current_total_cost:,.0f}",
        f"${max_profit - current_profit:,.0f}", "",
    ],
}
st.table(summary_data)

# --- Channel Manager ---
st.markdown("---")
st.subheader("Your Ad Channels")
st.caption("Each channel is a separate CSV/Excel file (one channel's spend vs its response). Upload files and switch between them.")

_saved_channels = _list_channels()
_channel_options = _saved_channels + ["(Upload new channel)"]

# Restore selected channel or default
if "selected_channel" not in st.session_state:
    if _saved_channels:
        st.session_state["selected_channel"] = _saved_channels[0]
    else:
        st.session_state["selected_channel"] = "(Upload new channel)"

_chan_choice = st.selectbox(
    "Channel",
    _channel_options,
    index=_channel_options.index(st.session_state.get("selected_channel", _channel_options[-1]))
          if st.session_state.get("selected_channel") in _channel_options else len(_channel_options) - 1,
    key="_chan_selector",
)

# Sync selection to session state; clear fitted data on channel switch
if _chan_choice != st.session_state.get("selected_channel"):
    st.session_state["selected_channel"] = _chan_choice
    if "_fitted_from_upload" in st.session_state:
        del st.session_state["_fitted_from_upload"]
    # Clear widget keys so they re-initialize from the new channel's settings
    for _k in ["upload_spend_col", "upload_response_col", "upload_spend_mult", "upload_resp_mult", "upload_resp_type"]:
        st.session_state.pop(_k, None)
    st.rerun()

upload_df = None

if _chan_choice == "(Upload new channel)":
    _new_name = st.text_input("Channel name (e.g. Facebook, Google, TV)", key="_new_chan_name")

    _upload_tab, _path_tab = st.tabs(["Upload file", "Local file path"])

    with _upload_tab:
        _uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key="_new_chan_upload",
        )
        if _uploaded_file is not None and _new_name.strip():
            _clean_name = _new_name.strip()
            try:
                if _uploaded_file.name.endswith((".xlsx", ".xls")):
                    _test_df = pd.read_excel(_uploaded_file)
                else:
                    _test_df = pd.read_csv(_uploaded_file)
                # Save a copy to channels/ so it persists
                os.makedirs(_CHANNELS_DIR, exist_ok=True)
                _saved_csv = os.path.join(_CHANNELS_DIR, f"{_clean_name}.csv")
                _test_df.to_csv(_saved_csv, index=False)
                _save_channel_settings(_clean_name, {"source_path": _saved_csv})
                st.session_state["selected_channel"] = _clean_name
                if "_fitted_from_upload" in st.session_state:
                    del st.session_state["_fitted_from_upload"]
                for _k in ["upload_spend_col", "upload_response_col", "upload_spend_mult", "upload_resp_mult", "upload_resp_type"]:
                    st.session_state.pop(_k, None)
                st.rerun()
            except Exception as e:
                st.error(f"Could not read file: {e}")
        elif _uploaded_file is not None and not _new_name.strip():
            st.warning("Please enter a channel name above.")

    with _path_tab:
        st.caption("For local use — enter the full path to a file on your computer.")
        _new_path = st.text_input("File path (CSV or Excel)", key="_new_chan_path",
                                   help="Full path to your data file, e.g. C:/Users/you/data/facebook.csv")
        if _new_path.strip() and _new_name.strip():
            _clean_name = _new_name.strip()
            _clean_path = _new_path.strip()
            if os.path.exists(_clean_path):
                try:
                    if _clean_path.endswith((".xlsx", ".xls")):
                        pd.read_excel(_clean_path, nrows=1)
                    else:
                        pd.read_csv(_clean_path, nrows=1)
                    _save_channel_settings(_clean_name, {"source_path": _clean_path})
                    st.session_state["selected_channel"] = _clean_name
                    if "_fitted_from_upload" in st.session_state:
                        del st.session_state["_fitted_from_upload"]
                    for _k in ["upload_spend_col", "upload_response_col", "upload_spend_mult", "upload_resp_mult", "upload_resp_type"]:
                        st.session_state.pop(_k, None)
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not read file: {e}")
            elif _clean_path:
                st.warning(f"File not found: {_clean_path}")
        elif _new_path.strip() and not _new_name.strip():
            st.warning("Please enter a channel name above.")
else:
    # Load existing channel
    upload_df = _load_channel(_chan_choice)
    if upload_df is None:
        st.error(f"Could not load channel '{_chan_choice}'. File may be missing.")
    else:
        # Load per-channel settings into session state before widgets render
        _ch_saved_settings = _load_channel_settings(_chan_choice)
        for _sk, _wk in [("spend_col", "upload_spend_col"), ("response_col", "upload_response_col"),
                          ("spend_mult", "upload_spend_mult"), ("resp_mult", "upload_resp_mult"),
                          ("resp_type", "upload_resp_type")]:
            if _sk in _ch_saved_settings and _wk not in st.session_state:
                st.session_state[_wk] = _ch_saved_settings[_sk]

        # Show source info and channel actions
        _ch_src = _ch_saved_settings.get("source_path", "unknown")
        _is_local_path = _ch_src != "unknown" and not _ch_src.startswith(_CHANNELS_DIR)
        if _is_local_path:
            st.caption(f"Source: `{_ch_src}`")

        _btn_cols = st.columns(3)
        with _btn_cols[0]:
            if _is_local_path:
                if st.button("Reload from disk", key="_reload_chan", help="Re-read the source file to pick up changes"):
                    if "_fitted_from_upload" in st.session_state:
                        del st.session_state["_fitted_from_upload"]
                    st.rerun()
        with _btn_cols[1]:
            _replace_file = st.file_uploader("Replace data", type=["csv", "xlsx", "xls"], key="_replace_chan_file", label_visibility="collapsed")
            if _replace_file is not None:
                try:
                    if _replace_file.name.endswith((".xlsx", ".xls")):
                        _repl_df = pd.read_excel(_replace_file)
                    else:
                        _repl_df = pd.read_csv(_replace_file)
                    os.makedirs(_CHANNELS_DIR, exist_ok=True)
                    _saved_csv = os.path.join(_CHANNELS_DIR, f"{_chan_choice}.csv")
                    _repl_df.to_csv(_saved_csv, index=False)
                    _save_channel_settings(_chan_choice, {"source_path": _saved_csv})
                    if "_fitted_from_upload" in st.session_state:
                        del st.session_state["_fitted_from_upload"]
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not read file: {e}")
        with _btn_cols[2]:
            if st.button("Delete channel", key="_delete_chan"):
                _delete_channel(_chan_choice)
                # Also remove saved CSV copy if it exists
                _csv_copy = os.path.join(_CHANNELS_DIR, f"{_chan_choice}.csv")
                if os.path.exists(_csv_copy):
                    os.remove(_csv_copy)
                st.session_state["selected_channel"] = "(Upload new channel)"
                if "_fitted_from_upload" in st.session_state:
                    del st.session_state["_fitted_from_upload"]
                for _k in ["upload_spend_col", "upload_response_col", "upload_spend_mult", "upload_resp_mult", "upload_resp_type"]:
                    st.session_state.pop(_k, None)
                st.rerun()

if upload_df is not None:
    try:
        upload_df.columns = [c.strip().lower() for c in upload_df.columns]
        st.dataframe(upload_df.head(10), use_container_width=True)

        # Let user pick columns
        numeric_cols = [c for c in upload_df.columns if pd.api.types.is_numeric_dtype(upload_df[c])]

        if len(numeric_cols) >= 2:
            pick_left, pick_right = st.columns(2)
            with pick_left:
                spend_col = st.selectbox(
                    "Select spend column",
                    numeric_cols,
                    index=0,
                    key="upload_spend_col",
                )
            with pick_right:
                default_response = 1 if len(numeric_cols) > 1 else 0
                response_col = st.selectbox(
                    "Select sales/revenue column",
                    numeric_cols,
                    index=default_response,
                    key="upload_response_col",
                )

            # Data format multipliers
            _mult_left, _mult_right = st.columns(2)
            with _mult_left:
                _spend_mult_label = st.selectbox(
                    "Spend column format",
                    list(_mult_options.keys()),
                    index=0,
                    key="upload_spend_mult",
                )
            with _mult_right:
                _resp_mult_label = st.selectbox(
                    "Response column format",
                    list(_mult_options.keys()),
                    index=0,
                    key="upload_resp_mult",
                )
            _spend_mult = _mult_options[_spend_mult_label]
            _resp_mult = _mult_options[_resp_mult_label]

            _resp_type = st.radio(
                "What does the response column measure?",
                ["Revenue ($)", "Units sold"],
                horizontal=True,
                key="upload_resp_type",
                help="If your data is revenue, E will be auto-converted to units using your Price per Unit from the sidebar.",
            )

            # Save per-channel settings whenever widgets are set
            if _chan_choice != "(Upload new channel)":
                _save_channel_settings(_chan_choice, {
                    "spend_col": spend_col,
                    "response_col": response_col,
                    "spend_mult": _spend_mult_label,
                    "resp_mult": _resp_mult_label,
                    "resp_type": _resp_type,
                })

            if spend_col != response_col:
                up_spend = upload_df[spend_col].dropna().values.astype(float) * _spend_mult
                up_response = upload_df[response_col].dropna().values.astype(float) * _resp_mult

                # Use same length
                min_len = min(len(up_spend), len(up_response))
                up_spend = up_spend[:min_len]
                up_response = up_response[:min_len]

                # Filter positive spend
                pos_mask = up_spend > 0
                up_spend = up_spend[pos_mask]
                up_response = up_response[pos_mask]

                if _spend_mult > 1 or _resp_mult > 1:
                    st.caption(
                        f"Spend values multiplied by {_spend_mult:,}x, "
                        f"response values multiplied by {_resp_mult:,}x."
                    )

                if len(up_spend) >= 3:
                    def _up_sqrt(x, b, e):
                        return b + e * np.sqrt(x)

                    def _up_log(x, b, e):
                        return b + e * np.log1p(x)

                    _up_max = float(np.max(up_spend))

                    def _up_hill(x, b, e, alpha, gamma):
                        inflexion = float(gamma) * _up_max
                        return b + e * hill_saturation(x, float(alpha), inflexion)

                    try:
                        up_popt_sq, _ = curve_fit(_up_sqrt, up_spend, up_response, p0=[np.min(up_response), 1])
                        up_popt_lg, _ = curve_fit(_up_log, up_spend, up_response, p0=[np.min(up_response), 1])
                        up_popt_hl, _ = curve_fit(
                            _up_hill, up_spend, up_response,
                            p0=[np.min(up_response), np.max(up_response) - np.min(up_response), 1.0, 0.5],
                            bounds=([0, 0, 0.1, 0.01], [np.inf, np.inf, 5.0, 1.0]),
                            maxfev=10000,
                        )

                        usq_b, usq_e = float(up_popt_sq[0]), float(up_popt_sq[1])
                        ulg_b, ulg_e = float(up_popt_lg[0]), float(up_popt_lg[1])
                        uh_b, uh_e, uh_a, uh_g = [float(v) for v in up_popt_hl]

                        # R² comparison — use adjusted R² only when n is large enough
                        # for ALL models (Hill needs n > 5). Otherwise raw R² for all.
                        _up_n = len(up_spend)
                        _use_adj = _up_n > 5  # Hill has 4 params, needs n > 4+1=5
                        if _use_adj:
                            r2_up_sq = adjusted_r2(up_response, _up_sqrt(up_spend, usq_b, usq_e), 2)
                            r2_up_lg = adjusted_r2(up_response, _up_log(up_spend, ulg_b, ulg_e), 2)
                            r2_up_hl = adjusted_r2(up_response, _up_hill(up_spend, uh_b, uh_e, uh_a, uh_g), 4)
                            _up_r2_label = "Adj. R²"
                        else:
                            r2_up_sq = adjusted_r2(up_response, _up_sqrt(up_spend, usq_b, usq_e), 0)
                            r2_up_lg = adjusted_r2(up_response, _up_log(up_spend, ulg_b, ulg_e), 0)
                            r2_up_hl = adjusted_r2(up_response, _up_hill(up_spend, uh_b, uh_e, uh_a, uh_g), 0)
                            _up_r2_label = "R²"

                        # Results
                        uc1, uc2, uc3 = st.columns(3)
                        with uc1:
                            st.markdown("**sqrt fit:**")
                            st.code(f"{response_col} = {usq_b:.1f} + {usq_e:.2f} * sqrt({spend_col})")
                            st.caption(f"{_up_r2_label} = {r2_up_sq:.4f}")
                        with uc2:
                            st.markdown("**log fit:**")
                            st.code(f"{response_col} = {ulg_b:.1f} + {ulg_e:.2f} * log(1+{spend_col})")
                            st.caption(f"{_up_r2_label} = {r2_up_lg:.4f}")
                        with uc3:
                            st.markdown("**Hill fit (Robyn):**")
                            st.code(f"{response_col} = {uh_b:.1f} + {uh_e:.0f} * Hill({spend_col})\n  alpha={uh_a:.2f}, gamma={uh_g:.2f}")
                            st.caption(f"{_up_r2_label} = {r2_up_hl:.4f}")

                        up_r2s = {"sqrt": r2_up_sq, "log": r2_up_lg, "Hill (Robyn)": r2_up_hl}
                        # Prefer simpler models on ties (sqrt/log have 2 params vs Hill's 4)
                        up_best = max(up_r2s, key=lambda k: (up_r2s[k], -len(k)))
                        st.info(f"Best fit: **{up_best}** model ({_up_r2_label} = {up_r2s[up_best]:.4f})")

                        # Store fitted params so sidebar can use them
                        st.session_state["_fitted_from_upload"] = {
                            "sqrt_e": usq_e, "sqrt_b": usq_b,
                            "log_e": ulg_e, "log_b": ulg_b,
                            "hill_e": uh_e, "hill_b": uh_b,
                            "hill_alpha": uh_a, "hill_gamma": uh_g,
                            "hill_max_spend": _up_max,
                            "best_model": up_best,
                            "best_r2": up_r2s[up_best],
                            "resp_type": _resp_type,
                        }
                        if _up_n <= 10:
                            st.warning(
                                f"Only {_up_n} data points — Hill (4 params) may overfit. "
                                f"Consider sqrt or log for small datasets, or add more data for reliable comparison."
                            )
                        show_r2_benchmarks(up_r2s[up_best])

                        # Chart
                        x_up = np.linspace(0, _up_max * 1.3, 300)
                        fig_up = go.Figure()

                        fig_up.add_trace(go.Scatter(
                            x=up_spend, y=up_response, mode="markers", name="Your Data",
                            marker=dict(size=6, color="black", opacity=0.6),
                        ))
                        fig_up.add_trace(go.Scatter(
                            x=x_up, y=_up_sqrt(x_up, usq_b, usq_e),
                            mode="lines", name=f"sqrt ({_up_r2_label}={r2_up_sq:.3f})",
                            line=dict(color="royalblue", width=2),
                        ))
                        fig_up.add_trace(go.Scatter(
                            x=x_up, y=_up_log(x_up, ulg_b, ulg_e),
                            mode="lines", name=f"log ({_up_r2_label}={r2_up_lg:.3f})",
                            line=dict(color="tomato", width=2, dash="dash"),
                        ))
                        fig_up.add_trace(go.Scatter(
                            x=x_up, y=_up_hill(x_up, uh_b, uh_e, uh_a, uh_g),
                            mode="lines", name=f"Hill ({_up_r2_label}={r2_up_hl:.3f})",
                            line=dict(color="darkgreen", width=3),
                        ))

                        # Hill inflection
                        up_infl_x = uh_g * _up_max
                        up_infl_y = _up_hill(np.array([up_infl_x]), uh_b, uh_e, uh_a, uh_g)[0]
                        fig_up.add_trace(go.Scatter(
                            x=[up_infl_x], y=[up_infl_y], mode="markers",
                            name=f"Hill inflection (${up_infl_x:,.0f})",
                            marker=dict(symbol="x", size=12, color="darkgreen", line=dict(width=2)),
                        ))

                        fig_up.update_layout(
                            xaxis_title=f"{spend_col} ($)", yaxis_title=response_col,
                            height=450, margin=dict(t=30),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        )
                        st.plotly_chart(fig_up, use_container_width=True)

                        # Marginal returns using best-fit model
                        sort_idx = np.argsort(up_spend)
                        up_s_sorted = up_spend[sort_idx]
                        up_r_sorted = up_response[sort_idx]
                        pcts = [0, 25, 50, 75, 100]
                        pct_vals = np.percentile(up_s_sorted, pcts)
                        st.caption(f"**Marginal {response_col} at spend percentiles ({up_best} model):**")
                        mr_rows = []
                        for i in range(len(pct_vals) - 1):
                            s0, s1 = float(pct_vals[i]), float(pct_vals[i + 1])
                            if up_best == "sqrt":
                                r0 = _up_sqrt(s0, usq_b, usq_e)
                                r1 = _up_sqrt(s1, usq_b, usq_e)
                            elif up_best == "log":
                                r0 = _up_log(s0, ulg_b, ulg_e)
                                r1 = _up_log(s1, ulg_b, ulg_e)
                            else:
                                r0 = _up_hill(np.array([s0]), uh_b, uh_e, uh_a, uh_g)[0]
                                r1 = _up_hill(np.array([s1]), uh_b, uh_e, uh_a, uh_g)[0]
                            delta_s = s1 - s0
                            delta_r = r1 - r0
                            mr = delta_r / (delta_s / 1000) if delta_s > 0 else 0
                            mr_rows.append({
                                "Spend Range": f"${s0:,.0f} -> ${s1:,.0f}",
                                f"{response_col} Change": f"{delta_r:,.0f}",
                                "Per $1K Spent": f"{mr:,.0f}",
                            })
                        st.table(mr_rows)

                    except Exception as e:
                        st.error(f"Could not fit curves to uploaded data: {e}")
                else:
                    st.warning(f"Need at least 3 rows with positive {spend_col} values. Found {len(up_spend)}.")
            else:
                st.warning("Please select different columns for spend and sales/revenue.")
        else:
            st.warning("File needs at least 2 numeric columns.")
    except Exception as e:
        st.error(f"Could not read file: {e}")


# --- Expandable: Model Formulas ---
with st.expander("Model Formulas & Derivation"):
    st.markdown(
        r"""
### Demand Models

**Variables:**
- $B$ = base demand, $E$ = ad effectiveness, $A$ = ad spend
- $P$ = price/unit, $C$ = cost/unit, $M = P - C$ (margin)

**Units Sold:**

| Model | Formula | Behavior |
|-------|---------|----------|
| sqrt  | $Q = B + E \sqrt{A}$ | Moderate diminishing returns |
| log   | $Q = B + E \ln(1 + A)$ | Aggressive diminishing returns |
| Hill (Robyn) | $Q = B + E \cdot \frac{A^\alpha}{A^\alpha + (\gamma \cdot A_{max})^\alpha}$ | Configurable S-curve or C-curve |

**Revenue:** $R = Q \times P$

**Total Cost:** $TC = F + C \times Q + A$

**Profit:** $\pi = R - TC = M \times Q - F - A$

---

### Optimal Ad Spend (sqrt model)

$$\frac{d\pi}{dA} = \frac{M \cdot E}{2\sqrt{A}} - 1 = 0 \implies A^* = \left(\frac{M \cdot E}{2}\right)^2$$

### Optimal Ad Spend (log model)

$$\frac{d\pi}{dA} = \frac{M \cdot E}{1 + A} - 1 = 0 \implies A^* = M \cdot E - 1$$

### Optimal Ad Spend (Hill model)

No closed-form solution — found via numerical optimization (scipy bounded minimization).

---

### Hill Function (from Meta's Robyn)

The Hill function originates from biochemistry (dose-response modeling) and is used by
Meta's [Robyn](https://github.com/facebookexperimental/Robyn) Marketing Mix Model framework.

$$\text{Hill}(A) = \frac{A^\alpha}{A^\alpha + (\gamma \cdot A_{max})^\alpha}$$

**Parameters:**
- **$\alpha$ (alpha)** — controls curve **shape**:
  - $\alpha < 1$: **C-shaped** (concave) — diminishing returns from the start, similar to sqrt/log
  - $\alpha > 1$: **S-shaped** (sigmoidal) — low spend has little effect, then accelerates, then saturates
- **$\gamma$ (gamma)** — controls the **inflection point** as a fraction of max spend:
  - At $A = \gamma \cdot A_{max}$, the output is exactly 50% of maximum
  - Low $\gamma$ (0.1–0.3): saturation kicks in early
  - High $\gamma$ (0.7–1.0): still growing across the observed range

**Key advantage:** Unlike sqrt/log, the Hill function is **bounded [0, 1]** and can model
both immediate diminishing returns AND threshold effects (minimum spend needed before ads work).
"""
    )
