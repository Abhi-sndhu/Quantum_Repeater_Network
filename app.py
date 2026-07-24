"""
QNeCT Streamlit GUI - Quantum Repeater Network Simulator
=========================================================
Rebuilt to match the original app (see uploaded screenshot): a sidebar of
simulation parameters (distance, repeaters, topology, distributed state,
noise model) driving a "Simulation" tab (topology diagram, final state,
fidelity), plus "Plots" and "Documentation" tabs. The other package
features (state utilities, entanglement check, raw teleportation, channel
fidelity) are kept as a bonus "Other Tools" tab so nothing from the
previous version of this file is lost.

Run with:
    streamlit run app.py
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from qrnetwork import Q, Qp, Is, Ebit, Teleport, QChannel, eswap, QRep

st.set_page_config(page_title="Quantum Repeater Network Simulator", layout="wide")

BELL_STATES = {
    "phi_plus": Q.Bell.phi_plus,
    "phi_minus": Q.Bell.phi_minus,
    "psi_plus": Q.Bell.psi_plus,
    "psi_minus": Q.Bell.psi_minus,
}
BELL_LABELS = {
    "Bell State |\u03a6+\u27e9": "phi_plus",
    "Bell State |\u03a6-\u27e9": "phi_minus",
    "Bell State |\u03a8+\u27e9": "psi_plus",
    "Bell State |\u03a8-\u27e9": "psi_minus",
}


# ----------------------------------------------------------------------
# Shared display helpers
# ----------------------------------------------------------------------
def to_display_matrix(x):
    """Convert Q / sympy.Matrix / ndarray into a plain complex ndarray."""
    if hasattr(x, "state"):        # Q instance
        return np.asarray(x.state)
    try:
        return np.array(x.tolist(), dtype=complex)   # sympy Matrix
    except AttributeError:
        return np.asarray(x, dtype=complex)


def matrix_to_dataframe(arr, decimals=4):
    arr = to_display_matrix(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    formatted = np.vectorize(
        lambda z: f"{np.real(z):.{decimals}f}{'+' if np.imag(z) >= 0 else '-'}{abs(np.imag(z)):.{decimals}f}i"
    )(arr)
    return pd.DataFrame(formatted)


def to_bmatrix_latex(x, decimals=2):
    """Render a ket/density matrix as a LaTeX bmatrix (like the original app)."""
    arr = to_display_matrix(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    rows = []
    for row in arr:
        cells = []
        for v in np.atleast_1d(row):
            re, im = float(np.real(v)), float(np.imag(v))
            if abs(im) < 10 ** (-decimals):
                cells.append(f"{re:.{decimals}f}")
            else:
                sign = "+" if im >= 0 else "-"
                cells.append(f"{re:.{decimals}f}{sign}{abs(im):.{decimals}f}i")
        rows.append(" & ".join(cells))
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def draw_topology(n_repeaters, total_distance):
    """Draw a linear repeater-chain diagram: End A -- R1 -- R2 -- ... -- End B."""
    n_links = n_repeaters + 1
    n_nodes = n_repeaters + 2
    xs = np.linspace(0, 10, n_nodes)
    labels = ["End A"] + [f"R{i + 1}" for i in range(n_repeaters)] + ["End B"]

    fig, ax = plt.subplots(figsize=(min(10, 1.6 * n_nodes), 2.2))
    for i in range(n_nodes - 1):
        ax.plot([xs[i], xs[i + 1]], [0, 0], color="#9ca3af", zorder=1, linewidth=1.5)

    for i, (x, label) in enumerate(zip(xs, labels)):
        if i == 0 or i == n_nodes - 1:
            ax.add_patch(plt.Circle((x, 0), 0.22, color="#2563eb", zorder=3))
        else:
            ax.add_patch(plt.Rectangle((x - 0.32, -0.22), 0.64, 0.44, fill=False, edgecolor="#374151", zorder=3))
            for dx in (-0.11, 0.11):
                ax.add_patch(plt.Circle((x + dx, 0), 0.09, color="#f59e0b", zorder=4))
        ax.text(x, 0.42, label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlim(xs[0] - 1, xs[-1] + 1)
    ax.set_ylim(-0.7, 0.7)
    ax.axis("off")
    fig.tight_layout()
    return fig, n_links


def run_repeater_simulation(distance, n_repeaters, bell_key, noise_enabled, T_p_us, T2_us, eta, p_d, fiber_loss):
    """Core simulation used by the Simulation tab and the Plots sweeps."""
    n_links = n_repeaters + 1
    link_length = distance / n_links
    base_ket = BELL_STATES[bell_key]

    if noise_enabled:
        shared = [Qp(base_ket).op() for _ in range(n_links)]
        rep = QRep(shared).noise(
            L=link_length, T_p=T_p_us * 1e-6, T_dp=T2_us * 1e-6, eta=eta, p_d=p_d, loss=fiber_loss
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = rep.linear()
    else:
        shared = [base_ket for _ in range(n_links)]
        results = QRep(shared).linear()

    total_p = sum(r["probability"] for r in results) or 1.0
    avg_fidelity = sum(r["probability"] * r["fidelity"] for r in results) / total_p
    best = max(results, key=lambda r: r["probability"])
    return {
        "link_length": link_length,
        "n_links": n_links,
        "results": results,
        "avg_fidelity": avg_fidelity,
        "final_state": best["obtained_entangled_state"],
    }


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding: 1.2rem 0 0.6rem 0;">
        <h1 style="margin-bottom:0;">QNeCT</h1>
        <p style="color:#4b5563; font-size:1.05rem;">Quantum Network and Communication Toolbox</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Sidebar: Simulation Parameters
# ----------------------------------------------------------------------
st.sidebar.header("Simulation Parameters")

distance = st.sidebar.slider("Total Distance (km)", min_value=1, max_value=500, value=100, key="distance")
n_repeaters = st.sidebar.slider("Number of Repeaters", min_value=1, max_value=10, value=5, key="n_repeaters")
topology = st.sidebar.selectbox("Topology", ["Linear"], key="topology")
if topology != "Linear":
    st.sidebar.info("Only Linear topology is currently supported by the underlying `QRep` class.")

with st.sidebar.expander("Distributed State", expanded=True):
    dist_state_label = st.selectbox("Select Distributed State", list(BELL_LABELS.keys()), key="dist_state")

with st.sidebar.expander("Noise Parameters", expanded=False):
    noise_enabled = st.checkbox("Enable Noise Model", value=False, key="noise_enabled")
    T_p_us = st.number_input(
        "Entanglement Preparation Time (\u00b5s)", min_value=0.0, value=1.00, step=0.1, key="T_p_us"
    )
    T2_us = st.number_input("Memory Loss Time T\u2082 (\u00b5s)", min_value=0.01, value=100.00, step=1.0, key="T2_us")
    eta = st.slider("Detector Efficiency", 0.0, 1.0, 0.90, 0.01, key="eta")
    p_d = st.number_input("Dark Count Probability", min_value=0.0, value=1e-6, format="%.2e", key="p_d")
    fiber_loss = st.number_input("Fiber Loss (dB/km)", min_value=0.01, value=0.20, step=0.01, key="fiber_loss")

run_clicked = st.sidebar.button("Run Simulation", use_container_width=True)

if run_clicked:
    try:
        sim = run_repeater_simulation(
            distance, n_repeaters, BELL_LABELS[dist_state_label], noise_enabled, T_p_us, T2_us, eta, p_d, fiber_loss
        )
        st.session_state["last_sim"] = sim
        st.session_state["last_noise_enabled"] = noise_enabled
    except ValueError as e:
        st.session_state["last_sim"] = None
        st.session_state["last_error"] = str(e)
    else:
        st.session_state["last_error"] = None

# ----------------------------------------------------------------------
# Main tabs
# ----------------------------------------------------------------------
tab_sim, tab_plots, tab_docs, tab_tools = st.tabs(["Simulation", "Plots", "Documentation", "Other Tools"])

# ---------------- Simulation tab ----------------
with tab_sim:
    st.subheader("Selected Network Topology")
    fig, n_links = draw_topology(n_repeaters, distance)
    st.pyplot(fig)
    st.caption(
        f"Linear topology: {n_repeaters} repeater(s), {n_links} link(s), "
        f"~{distance / n_links:.2f} km per link"
    )

    st.subheader("Simulation Output")
    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])

    sim = st.session_state.get("last_sim")
    if sim:
        st.success("Simulation Completed")

        state_label = r"\rho_{\text{final}}" if st.session_state.get("last_noise_enabled") else r"|\psi_{\text{final}}\rangle"
        st.markdown("### Final Entangled State")
        st.latex(f"{state_label} = {to_bmatrix_latex(sim['final_state'])}")

        st.markdown("### Fidelity")
        st.latex(rf"\mathcal{{F}} = {sim['avg_fidelity']:.2f}")

        with st.expander("Per-outcome measurement details"):
            df = pd.DataFrame(sim["results"])[["measurement_outcome", "probability", "fidelity"]]
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download results as CSV",
                df.to_csv(index=False).encode(),
                "qrn_results.csv",
                "text/csv",
            )
    else:
        st.info("Configure parameters in the sidebar and click **Run Simulation**.")

# ---------------- Plots tab ----------------
with tab_plots:
    st.subheader("Fidelity vs. Distance")
    st.caption("Sweeps total distance (current repeater count and noise settings held fixed).")
    if st.button("Compute distance sweep"):
        distances = np.linspace(10, 300, 15)
        fids = [
            run_repeater_simulation(
                d, n_repeaters, BELL_LABELS[dist_state_label], noise_enabled, T_p_us, T2_us, eta, p_d, fiber_loss
            )["avg_fidelity"]
            for d in distances
        ]
        df_d = pd.DataFrame({"Distance (km)": distances, "Fidelity": fids}).set_index("Distance (km)")
        st.line_chart(df_d)

    st.subheader("Fidelity vs. Number of Repeaters")
    st.caption("Sweeps repeater count (current distance and noise settings held fixed).")
    if st.button("Compute repeater-count sweep"):
        rep_counts = list(range(1, 11))
        fids_r = [
            run_repeater_simulation(
                distance, r, BELL_LABELS[dist_state_label], noise_enabled, T_p_us, T2_us, eta, p_d, fiber_loss
            )["avg_fidelity"]
            for r in rep_counts
        ]
        df_r = pd.DataFrame({"Number of Repeaters": rep_counts, "Fidelity": fids_r}).set_index("Number of Repeaters")
        st.line_chart(df_r)

    st.subheader("Measurement Outcome Probabilities (last run)")
    sim = st.session_state.get("last_sim")
    if sim:
        df_p = pd.DataFrame(sim["results"])[["measurement_outcome", "probability"]].set_index("measurement_outcome")
        st.bar_chart(df_p)
    else:
        st.info("Run a simulation on the **Simulation** tab first.")

# ---------------- Documentation tab ----------------
with tab_docs:
    st.subheader("About QNeCT")
    st.write(
        "QNeCT is a simulation toolbox for quantum repeater networks, built on NumPy, SciPy, and SymPy. "
        "This simulator models linear repeater chains: an entangled Bell pair (optionally passed through a "
        "physical noise model) is distributed over each link, then entanglement swapping is performed at "
        "each repeater node to connect the two end parties."
    )

    st.markdown("#### Parameters")
    st.markdown(
        """
- **Total Distance (km)** - end-to-end distance between End A and End B.
- **Number of Repeaters** - intermediate nodes; distance is split evenly across `repeaters + 1` links.
- **Topology** - network layout (currently only *Linear* is implemented).
- **Distributed State** - the Bell state shared over each link before swapping.
- **Noise Parameters** - when enabled, each link's state is passed through amplitude damping (fiber loss),
  dephasing (memory decoherence), and detector-click noise before swapping:
    - *Entanglement Preparation Time* - time to generate entanglement on a link.
    - *Memory Loss Time T\u2082* - quantum memory dephasing time.
    - *Detector Efficiency* - probability a real photon click is registered.
    - *Dark Count Probability* - probability of a false click.
    - *Fiber Loss* - attenuation in dB/km, used to compute channel transmissivity.
"""
    )

    st.markdown("#### Function reference")
    st.dataframe(
        pd.DataFrame(
            [
                ["Q(input).d()", "Dagger (Hermitian conjugate) of a state"],
                ["Q(input).n()", "Norm of a state"],
                ["Q(input).u()", "Normalized state"],
                ["Q(input).dm()", "Density matrix from a ket"],
                ["Q.rand.p(n) / Q.rand.m(n)", "Random n-qubit pure / mixed state"],
                ["Q.Bell.phi_plus / phi_minus / psi_plus / psi_minus", "The four Bell states"],
                ["Ebit(state)", "Entanglement (concurrence) of a two-qubit state"],
                ["Teleport(qS, qC)", "Teleport state qS through channel qC"],
                ["QChannel(channel, iteration).teleport()", "Average teleportation fidelity over N trials"],
                ["QRep([states]).linear()", "Chain entanglement swaps across a repeater network"],
                ["QRep([states]).noise(...).linear()", "Same, with a physical noise model applied per link"],
            ],
            columns=["Function", "Description"],
        ),
        use_container_width=True,
        hide_index=True,
    )

# ---------------- Other Tools tab (bonus, from the previous version of this file) ----------------
with tab_tools:
    st.write("Standalone utilities from the `qrnetwork` package, useful outside the repeater-network workflow.")
    tool = st.selectbox(
        "Choose a tool",
        ["State Utilities (Q)", "Entanglement Check (Ebit)", "Teleportation (Teleport)", "Channel Fidelity (QChannel)"],
    )

    if tool == "State Utilities (Q)":
        col1, col2 = st.columns(2)
        with col1:
            kind = st.selectbox("State type", ["Random pure state", "Random mixed state", "Bell state"])
        with col2:
            n_qubits = st.slider("Number of qubits", 1, 3, 1, disabled=(kind == "Bell state"))

        if kind == "Random pure state":
            state = Q.rand.p(n_qubits)
        elif kind == "Random mixed state":
            state = Q.rand.m(n_qubits)
        else:
            bell_name = st.selectbox("Bell state", list(BELL_STATES.keys()))
            state = BELL_STATES[bell_name]

        st.dataframe(matrix_to_dataframe(state))
        q = Q(state)
        c1, c2, c3 = st.columns(3)
        c1.metric("Norm  Q(state).n()", q.n())
        c2.metric("Purity  Q(state).purity()", q.purity())
        is_ket = Is(state).ket()
        c3.metric("Is ket?", "Yes" if is_ket else "No")

        d1, d2 = st.columns(2)
        with d1:
            st.caption("Dagger - `Q(state).d()`")
            st.dataframe(matrix_to_dataframe(q.d()))
        with d2:
            st.caption("Normalized - `Q(state).u()`")
            st.dataframe(matrix_to_dataframe(q.u()))
        if is_ket:
            st.caption("Density matrix - `Q(state).dm()`")
            st.dataframe(matrix_to_dataframe(q.dm()))

    elif tool == "Entanglement Check (Ebit)":
        rep = st.selectbox("Representation", ["Ket vector", "Density matrix (Werner state)"])
        if rep == "Ket vector":
            bell_name = st.selectbox("Base Bell state", list(BELL_STATES.keys()))
            state = BELL_STATES[bell_name]
        else:
            bell_name = st.selectbox("Base Bell state", list(BELL_STATES.keys()))
            p = st.slider("Werner mixing parameter p", 0.0, 1.0, 0.8, 0.01)
            state = Q.Werner(p, state=BELL_STATES[bell_name])
        st.dataframe(matrix_to_dataframe(state))
        st.metric("Concurrence (Ebit)", round(float(Ebit(state)), 4))
        st.caption("0 -> separable, non-zero -> entangled (max 1 for a maximally entangled state).")

    elif tool == "Teleportation (Teleport)":
        mode = st.radio("Input type", ["Pure state (ket)", "Mixed state (density matrix)"], horizontal=True)
        channel_choice = st.selectbox("Entangled channel", list(BELL_STATES.keys()) + ["Werner state"])
        if channel_choice == "Werner state":
            base = st.selectbox("Werner base Bell state", list(BELL_STATES.keys()), key="werner_base_tel")
            p = st.slider("Werner mixing parameter p", 0.5, 1.0, 0.9, 0.01, key="werner_p_tel")
            channel = Q.Werner(p, state=BELL_STATES[base])
        else:
            channel = BELL_STATES[channel_choice]
            if mode == "Mixed state (density matrix)":
                channel = Q(channel).dm()

        if st.button("Run teleportation"):
            try:
                qS = Q.rand.p(1) if mode == "Pure state (ket)" else Q.rand.m(1)
                st.caption("Input state to teleport")
                st.dataframe(matrix_to_dataframe(qS))
                result = Teleport(qS, channel)
                rows = [
                    {"Measurement outcome": r["measurement_outcome"], "Probability": r["probability"], "Fidelity": r["fidelity"]}
                    for r in result.results["results"]
                ]
                st.dataframe(pd.DataFrame(rows))
                st.caption("Local unitary matrix (Bell basis -> computational basis)")
                st.dataframe(matrix_to_dataframe(result.results["Local unitary matrix"]))
            except ValueError as e:
                st.error(str(e))

    elif tool == "Channel Fidelity (QChannel)":
        channel_choice = st.selectbox("Entangled channel", list(BELL_STATES.keys()) + ["Werner state"])
        iterations = st.slider("Iterations", 5, 500, 50, 5)
        if channel_choice == "Werner state":
            base = st.selectbox("Werner base Bell state", list(BELL_STATES.keys()), key="werner_base_qc")
            p = st.slider("Werner mixing parameter p", 0.5, 1.0, 0.9, 0.01, key="werner_p_qc")
            channel = Q.Werner(p, state=BELL_STATES[base])
        else:
            channel = BELL_STATES[channel_choice]

        if st.button("Run channel simulation"):
            try:
                result = QChannel(channel, iteration=iterations).teleport()
                st.metric("Average fidelity", result["Average fidelity"])
            except ValueError as e:
                st.error(str(e))

st.sidebar.markdown("---")
st.sidebar.caption("QNeCT \u00b7 built on NumPy, SciPy, SymPy")
