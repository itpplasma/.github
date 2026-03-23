"""Interactive GUI for RMP/CX/TS signal analysis.

The original analysis arrived as a procedural script. This version keeps the
numerical workflow modular, removes repeated data wrangling, and exposes the
analysis through a Tkinter GUI with embedded Matplotlib plots.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import pickle
import queue
import threading
import traceback
from typing import Any
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit

from eq_lib import get_Rout, map_to_eq

plt.rcParams["lines.linewidth"] = 2.2
plt.rcParams["font.size"] = 12

LINESTYLES = ["-", "--", "-.", ":"]
DEFAULT_DATA_PATH = Path("/home/sistema/Edo/PhD/Work/MAST-U/Freia_data/")
DEFAULT_SAVE_PATH = Path("/home/sistema/Edo/PhD/Work/MAST-U/RT01/Paper/Plots/Analysis/")


@dataclass(slots=True)
class AnalysisFlags:
    eq_mapping: bool = False
    transp_mapping: bool = False
    use_cx: bool = True
    use_ti: bool = False
    use_te: bool = False
    use_ne: bool = False
    use_rsep: bool = False
    use_radius: bool = False
    plot_rmp: bool = True
    plot_rmp_fft: bool = False
    plot_sig: bool = False
    plot_snr_singlef: bool = True
    use_amp: bool = False
    plot_vel_rmp: bool = False
    plot_label: bool = False
    save_plot: bool = False


@dataclass(slots=True)
class AnalysisConfig:
    shot: int = 51330
    src: str = "epm"
    key: str = "main_nomap"
    time_window: tuple[float, float] = (0.4, 1.0)
    r_window: tuple[float, float] = (0.9, 1.41)
    rhop: float = 0.97
    rlim: float = 1.35
    path: Path = DEFAULT_DATA_PATH
    pathsave: Path = DEFAULT_SAVE_PATH


@dataclass(slots=True)
class SignalBundle:
    title: str
    ylabel: str
    namesave: str
    fact: float
    base: np.ndarray
    time: np.ndarray
    values: np.ndarray
    detrended: np.ndarray
    mean: np.ndarray
    x_label: str = "R [m]"
    errors: np.ndarray | None = None
    rtime_masked: np.ndarray | None = None
    rsep_detrend: np.ndarray | None = None


@dataclass(slots=True)
class RmpSignal:
    time: np.ndarray
    signal: np.ndarray
    spectrum: np.ndarray
    freqs: np.ndarray
    dominant_freq: float


@dataclass(slots=True)
class FitSummary:
    frequency: float
    amp_plot: np.ndarray
    amp_err_plot: np.ndarray
    snr: np.ndarray
    peak_index: int


@dataclass(slots=True)
class PlotArtifact:
    title: str
    draw: Any


@dataclass(slots=True)
class AnalysisResult:
    config: AnalysisConfig
    flags: AnalysisFlags
    bundle: SignalBundle
    rmp: RmpSignal
    figures: list[PlotArtifact]
    log_lines: list[str]


class AnalysisError(RuntimeError):
    """Raised when the selected inputs cannot produce a valid analysis."""


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def nearest_slice(values: np.ndarray, window: tuple[float, float]) -> slice:
    start = int(np.argmin(np.abs(values - window[0])))
    stop = int(np.argmin(np.abs(values - window[1])))
    if start == stop:
        stop = min(len(values), start + 1)
    return slice(min(start, stop), max(start, stop))


def drop_nan_pairs(time: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(data)
    return time[mask], data[mask]


def detrend_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_mean = np.nanmean(values, axis=1)
    return values - row_mean[:, np.newaxis], row_mean


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full_like(numerator, np.nan, dtype=float)
    np.divide(numerator, denominator, out=output, where=np.isfinite(denominator) & (denominator != 0))
    return output


def fit_sinusoid(time: np.ndarray, data: np.ndarray, freq: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit a sinusoid at a fixed frequency using non-linear least squares."""
    time, data = drop_nan_pairs(time, data)

    def sinusoid(t: np.ndarray, amp: float, phase: float, offset: float) -> np.ndarray:
        return amp * np.sin(2 * np.pi * freq * t + phase) + offset

    try:
        params, covariance = curve_fit(sinusoid, time, data, maxfev=5000)
    except Exception:
        nan_triplet = np.full(3, np.nan)
        return nan_triplet, nan_triplet

    return params, np.sqrt(np.diag(covariance))


def fit_sinusoid_linear(
    time: np.ndarray,
    data: np.ndarray,
    freq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``a*cos(wt) + b*sin(wt) + c`` and return amplitude/phase/error."""
    time, data = drop_nan_pairs(time, data)
    if time.size < 4:
        return np.array([np.nan, np.nan]), np.array([np.nan])

    omega = 2 * np.pi * freq
    design = np.column_stack((np.cos(omega * time), np.sin(omega * time), np.ones_like(time)))
    beta, *_ = np.linalg.lstsq(design, data, rcond=None)
    a_coef, b_coef, _offset = beta
    fitted = design @ beta

    degrees_of_freedom = len(data) - design.shape[1]
    if degrees_of_freedom <= 0:
        return np.array([np.nan, np.nan]), np.array([np.nan])

    sigma2 = np.sum((data - fitted) ** 2) / degrees_of_freedom
    xtx = design.T @ design
    try:
        covariance = sigma2 * np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan]), np.array([np.nan])

    amplitude = np.hypot(a_coef, b_coef)
    phase = np.arctan2(b_coef, a_coef)
    if amplitude == 0:
        return np.array([amplitude, phase]), np.array([np.nan])

    var_a = covariance[0, 0]
    var_b = covariance[1, 1]
    cov_ab = covariance[0, 1]
    amp_error = np.sqrt(
        (a_coef**2 * var_a + b_coef**2 * var_b + 2 * a_coef * b_coef * cov_ab) / amplitude**2
    )
    return np.array([amplitude, phase]), np.array([amp_error])


def dominant_frequency(time: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if len(time) < 2 or len(signal) < 2:
        raise AnalysisError("RMP signal is too short to compute an FFT.")
    fmax = max(2, (signal.size - 1) // 2)
    spectrum = fft(signal)[1:fmax]
    freqs = fftfreq(signal.size, d=time[1] - time[0])[1:fmax]
    if spectrum.size == 0 or freqs.size == 0:
        raise AnalysisError("RMP FFT produced no positive frequencies.")
    peak = float(freqs[np.argmax(np.abs(spectrum))])
    return spectrum, freqs, peak


def load_rmp_signal(config: AnalysisConfig) -> RmpSignal:
    rmp_data = load_pickle(config.path / str(config.shot) / "rmp_data.pickle")
    time = np.asarray(rmp_data["t"])
    time_slice = nearest_slice(time, config.time_window)
    time_masked = time[time_slice]
    signal = np.asarray(rmp_data["data"])[time_slice]
    spectrum, freqs, peak = dominant_frequency(time_masked, signal)
    return RmpSignal(time=time_masked, signal=signal, spectrum=spectrum, freqs=freqs, dominant_freq=peak)


def load_cx_or_ti(config: AnalysisConfig, flags: AnalysisFlags) -> SignalBundle:
    cx_data = load_pickle(config.path / str(config.shot) / "cx_data.pickle")
    ti_data = None
    if flags.use_ti:
        ti_data = load_pickle(config.path / str(config.shot) / "cx_ti_data.pickle")

    if flags.transp_mapping:
        cx_data = load_pickle(config.path / "51330_TRANSP_C04.pickle")

    use_cx_dataset = flags.use_cx or flags.transp_mapping
    data = cx_data if use_cx_dataset else ti_data
    if data is None:
        raise AnalysisError("No CX/Ti dataset is available for the selected flags.")

    title, ylabel, fact, namesave = (
        ("Toroidal velocity", "v [km/s]", 1.0, "vel")
        if use_cx_dataset
        else ("T ions", "Ti [keV]", 1e3, "ti")
    )

    time = np.asarray(data[config.key]["time"])
    time_slice = nearest_slice(time, config.time_window)
    time_masked = time[time_slice]

    if flags.eq_mapping:
        eq_data = load_pickle(config.path / str(config.shot) / "eq_data.pickle")[config.src]
        map_to_eq(cx_data[config.key], eq_data)
        mapped = cx_data[config.key]
        psi = np.asarray(mapped["psin"])
        psi_avg = np.mean(psi[:, time_slice], axis=1)
        valid = (psi_avg >= 0) & (psi_avg <= 1)
        values = np.asarray(mapped["data"])[valid, time_slice]
        detrended, mean = detrend_rows(values)
        return SignalBundle(
            title=title,
            ylabel=ylabel,
            namesave=namesave,
            fact=fact,
            base=np.sqrt(psi_avg[valid]),
            time=time_masked,
            values=values,
            detrended=detrended,
            mean=mean,
            x_label=r"$\rho_{pol}$",
        )

    radius = np.asarray(data[config.key]["radius"])
    radius_slice = nearest_slice(radius, config.r_window)
    values = np.asarray(data[config.key]["data"])[radius_slice, time_slice]
    errors = np.asarray(data[config.key]["error"])[radius_slice, time_slice]
    detrended, mean = detrend_rows(values)
    return SignalBundle(
        title=title,
        ylabel=ylabel,
        namesave=namesave,
        fact=fact,
        base=radius[radius_slice],
        time=time_masked,
        values=values,
        detrended=detrended,
        mean=mean,
        errors=errors,
    )


def load_te_or_ne(config: AnalysisConfig, flags: AnalysisFlags) -> SignalBundle:
    te_data = load_pickle(config.path / str(config.shot) / "ts_te_data.pickle")
    ne_data = load_pickle(config.path / str(config.shot) / "ts_ne_data.pickle")
    data = te_data if flags.use_te else ne_data

    title, ylabel, fact, namesave = (
        ("T electrons", "Te [keV]", 1e3, "te")
        if flags.use_te
        else ("Electron density", r"ne [#/m^3]", 1.0, "ne")
    )

    time = np.asarray(data[config.key]["time"])
    time_slice = nearest_slice(time, config.time_window)
    time_masked = time[time_slice]

    radius = np.asarray(data[config.key]["radius"])
    radius_axis = radius[0] if radius.ndim > 1 else radius
    radius_slice = nearest_slice(radius_axis, config.r_window)
    values = np.asarray(data[config.key]["data"])[radius_slice, time_slice].astype(float, copy=False)
    values[~np.isfinite(values)] = np.nan
    detrended, mean = detrend_rows(values)
    return SignalBundle(
        title=title,
        ylabel=ylabel,
        namesave=namesave,
        fact=fact,
        base=radius_axis[radius_slice],
        time=time_masked,
        values=values,
        detrended=detrended,
        mean=mean,
    )


def load_radius_signal(config: AnalysisConfig, flags: AnalysisFlags) -> SignalBundle:
    eq_data = load_pickle(config.path / str(config.shot) / "eq_data.pickle")[config.src]
    title, ylabel, fact, namesave = "Midplane radius", "mm", 1 / 1e3, "rmid"
    time = np.asarray(eq_data["time"])
    time_slice = nearest_slice(time, config.time_window)
    time_masked = time[time_slice]

    if flags.use_rsep:
        rsep = np.max(np.asarray(eq_data["rbbbs"]), axis=1)[time_slice]
        mean = np.array([np.nanmean(rsep)])
        return SignalBundle(
            title=title,
            ylabel=ylabel,
            namesave=namesave,
            fact=fact,
            base=np.array([]),
            time=time_masked,
            values=rsep,
            detrended=np.array([]),
            mean=mean,
            rtime_masked=time_masked,
            rsep_detrend=rsep - mean[0],
            x_label="Time [s]",
        )

    rr = np.column_stack([get_Rout(eq_data, current_time) for current_time in time_masked])
    detrended, mean = detrend_rows(rr)
    return SignalBundle(
        title=title,
        ylabel=ylabel,
        namesave=namesave,
        fact=fact,
        base=mean,
        time=time_masked,
        values=rr,
        detrended=detrended,
        mean=mean,
    )


def load_primary_signal(config: AnalysisConfig, flags: AnalysisFlags) -> SignalBundle:
    selected = [
        flags.use_cx or flags.use_ti,
        flags.use_te or flags.use_ne,
        flags.use_rsep or flags.use_radius,
    ]
    if sum(selected) != 1:
        raise AnalysisError("Select exactly one primary diagnostic group: CX/Ti, TE/NE, or radius/Rsep.")
    if flags.use_cx or flags.use_ti:
        return load_cx_or_ti(config, flags)
    if flags.use_te or flags.use_ne:
        return load_te_or_ne(config, flags)
    return load_radius_signal(config, flags)


def build_target_frequencies(shot: int, freqmax: float) -> tuple[np.ndarray, dict[float, int]]:
    if freqmax > 10:
        rounded = float(5 * round(freqmax / 5))
        return np.arange(20, 60, 5, dtype=float), {rounded: 0}
    fixed_target = 6.0 if shot == 50815 else 6.7
    return np.array([3, 4, 5, fixed_target, 8, 9, 11], dtype=float), {fixed_target: 0}


def compute_fit_summary(bundle: SignalBundle, frequency: float) -> FitSummary:
    fit_results = [fit_sinusoid_linear(bundle.time, row, frequency) for row in bundle.detrended]
    amplitudes = np.array([params[0] for params, _ in fit_results], dtype=float)
    errors = np.array([err[0] for _, err in fit_results], dtype=float)
    amp_plot = safe_divide(np.abs(amplitudes), bundle.mean)
    amp_err_plot = safe_divide(np.abs(errors), bundle.mean)
    snr = safe_divide(amp_plot, amp_err_plot)
    if np.all(np.isnan(snr)):
        peak_index = 0
    else:
        peak_index = int(np.nanargmax(snr))
    return FitSummary(frequency, amp_plot, amp_err_plot, snr, peak_index)


def create_line_plot(title: str) -> tuple[Figure, Any]:
    figure = Figure(figsize=(9, 6), dpi=100, constrained_layout=True)
    axis = figure.add_subplot(111)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    return figure, axis


def artifact_rmp_signal(rmp: RmpSignal) -> PlotArtifact:
    def draw(figure: Figure) -> None:
        axis = figure.add_subplot(111)
        axis.plot(rmp.time, rmp.signal, color="black")
        axis.set_title("RMP signal")
        axis.set_xlabel("Time [s]")
        axis.set_ylabel("Current [A]")
        axis.grid(True, alpha=0.25)

    return PlotArtifact("RMP signal", draw)


def artifact_rmp_fft(rmp: RmpSignal) -> PlotArtifact:
    def draw(figure: Figure) -> None:
        axis = figure.add_subplot(111)
        axis.stem(rmp.freqs, np.abs(rmp.spectrum), basefmt=" ")
        axis.axvline(rmp.dominant_freq, color="tab:red", linestyle="--", label=f"Peak {rmp.dominant_freq:.2f} Hz")
        axis.set_title("RMP FFT")
        axis.set_xlabel("Frequency [Hz]")
        axis.set_ylabel("|FFT|")
        axis.legend(loc="best")
        axis.grid(True, alpha=0.25)

    return PlotArtifact("RMP FFT", draw)


def artifact_signal_trace(bundle: SignalBundle) -> PlotArtifact:
    trace_index = min(5, len(bundle.values) - 1)

    def draw(figure: Figure) -> None:
        axis = figure.add_subplot(111)
        axis.plot(bundle.time, bundle.values[trace_index], color="black")
        axis.set_title(f"Sample trace: {bundle.title}")
        axis.set_xlabel("Time [s]")
        axis.set_ylabel(bundle.ylabel)
        axis.grid(True, alpha=0.25)

    return PlotArtifact("Sample trace", draw)


def artifact_snr_scan(
    config: AnalysisConfig,
    flags: AnalysisFlags,
    bundle: SignalBundle,
    rmp: RmpSignal,
    log_lines: list[str],
) -> PlotArtifact:
    target_freqs, subset = build_target_frequencies(config.shot, rmp.dominant_freq)
    summaries = [compute_fit_summary(bundle, frequency) for frequency in target_freqs]
    best_summary = None
    best_ratio = -np.inf
    second_peak_index = None

    for summary in summaries:
        if summary.frequency not in subset:
            continue
        ratio = summary.snr[summary.peak_index]
        if np.isnan(ratio) or ratio <= best_ratio:
            continue
        best_ratio = ratio
        best_summary = summary
        if summary.frequency < 10:
            masked_indices = np.flatnonzero(bundle.base < config.rlim)
            if masked_indices.size:
                local_snr = summary.snr[masked_indices]
                if not np.all(np.isnan(local_snr)):
                    second_peak_index = int(masked_indices[np.nanargmax(local_snr)])

    if best_summary is not None:
        log_lines.append(f"Dominant RMP frequency: {rmp.dominant_freq:.2f} Hz")
        log_lines.append(
            f"Best highlighted frequency: {best_summary.frequency:.2f} Hz | peak SNR={best_summary.snr[best_summary.peak_index]:.3f}"
        )
        if second_peak_index is not None:
            log_lines.append(
                f"Secondary inner peak at x={bundle.base[second_peak_index]:.4f} with SNR={best_summary.snr[second_peak_index]:.3f}"
            )

    def draw(figure: Figure) -> None:
        axis = figure.add_subplot(111)
        ylabel = "Amplitude" if flags.use_amp else "SNR"
        for summary in summaries:
            is_highlighted = summary.frequency in subset
            color = "tab:green" if is_highlighted else "0.7"
            alpha = 1.0 if is_highlighted else 0.35
            linestyle = LINESTYLES[subset[summary.frequency]] if is_highlighted else "-"
            label = f"{summary.frequency:g} Hz" if is_highlighted else None
            y_values = summary.amp_plot / bundle.fact if flags.use_amp else summary.snr
            axis.plot(bundle.base, y_values, "-o", color=color, alpha=alpha, linestyle=linestyle, label=label)
            if flags.plot_label:
                axis.text(bundle.base[summary.peak_index], y_values[summary.peak_index], f"{summary.frequency:g}", fontsize=8)

        if best_summary is not None:
            axis.scatter(
                bundle.base[best_summary.peak_index],
                (best_summary.amp_plot / bundle.fact if flags.use_amp else best_summary.snr)[best_summary.peak_index],
                color="tab:red",
                s=50,
                zorder=3,
                label="Best peak",
            )
        axis.set_title(bundle.title)
        axis.set_xlabel(bundle.x_label)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        if any(summary.frequency in subset for summary in summaries):
            axis.legend(loc="best")

    return PlotArtifact("SNR scan", draw)


def artifact_velocity_vs_rmp(bundle: SignalBundle, rmp: RmpSignal, peak_index: int) -> PlotArtifact:
    def draw(figure: Figure) -> None:
        axis = figure.add_subplot(111)
        axis.plot(bundle.time, bundle.values[peak_index] / 1e3, "-o", color="black")
        if bundle.errors is not None:
            axis.errorbar(bundle.time, bundle.values[peak_index] / 1e3, yerr=bundle.errors[peak_index] / 1e3, fmt="none", color="0.4")
        axis.set_xlabel("Time [s]")
        axis.set_ylabel(bundle.ylabel)
        axis.set_title(f"Signal vs RMP at x={bundle.base[peak_index]:.3f}")
        axis.grid(True, alpha=0.25)
        axis2 = axis.twinx()
        axis2.plot(rmp.time, rmp.signal / 1e3, linestyle="--", color="goldenrod")
        axis2.set_ylabel("I [kA]", color="goldenrod")
        axis2.tick_params(axis="y", colors="goldenrod")

    return PlotArtifact("Signal vs RMP", draw)


def artifact_rsep_frequency_scan(bundle: SignalBundle, target_freqs: np.ndarray, subset: dict[float, int], rmp: RmpSignal) -> PlotArtifact:
    snr_values = []
    for frequency in target_freqs:
        params, errors = fit_sinusoid_linear(bundle.time, bundle.rsep_detrend, frequency)
        snr_values.append(abs(params[0]) / errors[0] if errors[0] and np.isfinite(errors[0]) else np.nan)

    def draw(figure: Figure) -> None:
        axis = figure.add_subplot(211)
        markerline, stemlines, baseline = axis.stem(target_freqs, np.array(snr_values), basefmt=" ")
        markerline.set_color("tab:blue")
        plt.setp(stemlines, color="tab:blue")
        for frequency in subset:
            axis.axvline(frequency, color="tab:green", linestyle="--", label=f"{frequency:g} Hz")
        axis.set_title(r"$R_{sep}$ frequency scan")
        axis.set_xlabel("Frequency [Hz]")
        axis.set_ylabel("SNR")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

        axis2 = figure.add_subplot(212)
        axis2.plot(bundle.rtime_masked, bundle.rsep_detrend * 1e3, "-o", color="black")
        axis2.set_xlabel("Time [s]")
        axis2.set_ylabel(r"$r_{sep} - <r_{sep}>$ [mm]")
        axis2.grid(True, alpha=0.25)
        twin = axis2.twinx()
        twin.plot(rmp.time, rmp.signal / 1e3, linestyle="--", color="goldenrod")
        twin.set_ylabel("I [kA]", color="goldenrod")
        twin.tick_params(axis="y", colors="goldenrod")

    return PlotArtifact("Rsep frequency scan", draw)


def build_artifacts(config: AnalysisConfig, flags: AnalysisFlags, bundle: SignalBundle, rmp: RmpSignal) -> tuple[list[PlotArtifact], list[str]]:
    artifacts: list[PlotArtifact] = []
    log_lines: list[str] = [f"Shot {config.shot}", f"Primary diagnostic: {bundle.title}"]

    if flags.plot_rmp:
        artifacts.append(artifact_rmp_signal(rmp))
    if flags.plot_rmp_fft:
        artifacts.append(artifact_rmp_fft(rmp))
    if flags.plot_sig and np.ndim(bundle.values) > 1 and len(bundle.values) > 0:
        artifacts.append(artifact_signal_trace(bundle))
    if flags.plot_snr_singlef and np.ndim(bundle.detrended) > 1:
        snr_artifact = artifact_snr_scan(config, flags, bundle, rmp, log_lines)
        artifacts.append(snr_artifact)
    if flags.plot_vel_rmp and np.ndim(bundle.detrended) > 1:
        default_summary = compute_fit_summary(bundle, build_target_frequencies(config.shot, rmp.dominant_freq)[1].popitem()[0])
        artifacts.append(artifact_velocity_vs_rmp(bundle, rmp, default_summary.peak_index))
    if flags.use_rsep and bundle.rsep_detrend is not None:
        target_freqs, subset = build_target_frequencies(config.shot, rmp.dominant_freq)
        artifacts.append(artifact_rsep_frequency_scan(bundle, target_freqs, subset, rmp))

    if not artifacts:
        artifacts.append(artifact_rmp_signal(rmp))
    return artifacts, log_lines


def run_analysis(config: AnalysisConfig, flags: AnalysisFlags) -> AnalysisResult:
    bundle = load_primary_signal(config, flags)
    rmp = load_rmp_signal(config)
    artifacts, log_lines = build_artifacts(config, flags, bundle, rmp)
    return AnalysisResult(config=config, flags=flags, bundle=bundle, rmp=rmp, figures=artifacts, log_lines=log_lines)


class AnalysisApp(tk.Tk):
    """Desktop front-end for configuring and running the analysis."""

    def __init__(self) -> None:
        super().__init__()
        self.title("RMP/CX/TS Analysis GUI")
        self.geometry("1500x900")
        self.minsize(1200, 700)

        self.result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.figure_canvas: FigureCanvasTkAgg | None = None
        self.toolbar: NavigationToolbar2Tk | None = None
        self.current_result: AnalysisResult | None = None
        self.current_figure_index = 0

        self._build_state()
        self._build_layout()
        self.after(100, self._poll_queue)

    def _build_state(self) -> None:
        default_config = AnalysisConfig()
        default_flags = AnalysisFlags()

        self.shot_var = tk.StringVar(value=str(default_config.shot))
        self.src_var = tk.StringVar(value=default_config.src)
        self.key_var = tk.StringVar(value=default_config.key)
        self.time_start_var = tk.StringVar(value=str(default_config.time_window[0]))
        self.time_end_var = tk.StringVar(value=str(default_config.time_window[1]))
        self.r_start_var = tk.StringVar(value=str(default_config.r_window[0]))
        self.r_end_var = tk.StringVar(value=str(default_config.r_window[1]))
        self.rhop_var = tk.StringVar(value=str(default_config.rhop))
        self.rlim_var = tk.StringVar(value=str(default_config.rlim))
        self.path_var = tk.StringVar(value=str(default_config.path))
        self.save_path_var = tk.StringVar(value=str(default_config.pathsave))
        self.status_var = tk.StringVar(value="Ready")

        self.flag_vars = {
            field.name: tk.BooleanVar(value=getattr(default_flags, field.name))
            for field in fields(AnalysisFlags)
        }

    def _build_layout(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=12)
        controls.grid(row=0, column=0, sticky="nsw")
        controls.columnconfigure(1, weight=1)

        viewer = ttk.Frame(self, padding=12)
        viewer.grid(row=0, column=1, sticky="nsew")
        viewer.rowconfigure(1, weight=1)
        viewer.columnconfigure(0, weight=1)

        self._build_input_section(controls)
        self._build_flag_section(controls)
        self._build_action_section(controls)
        self._build_viewer_section(viewer)

    def _build_input_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Inputs", padding=10)
        frame.grid(row=0, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)

        rows = [
            ("Shot", self.shot_var),
            ("Source", self.src_var),
            ("Key", self.key_var),
            ("Time start", self.time_start_var),
            ("Time end", self.time_end_var),
            ("R start", self.r_start_var),
            ("R end", self.r_end_var),
            ("Rhop", self.rhop_var),
            ("R limit", self.rlim_var),
        ]
        for row_index, (label, variable) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=row_index, column=0, sticky="w", pady=2)
            ttk.Entry(frame, textvariable=variable, width=20).grid(row=row_index, column=1, sticky="ew", pady=2)

        ttk.Label(frame, text="Data path").grid(row=len(rows), column=0, sticky="w", pady=2)
        ttk.Entry(frame, textvariable=self.path_var).grid(row=len(rows), column=1, sticky="ew", pady=2)
        ttk.Button(frame, text="Browse", command=self._choose_data_path).grid(row=len(rows), column=2, padx=(6, 0))

        ttk.Label(frame, text="Save path").grid(row=len(rows) + 1, column=0, sticky="w", pady=2)
        ttk.Entry(frame, textvariable=self.save_path_var).grid(row=len(rows) + 1, column=1, sticky="ew", pady=2)
        ttk.Button(frame, text="Browse", command=self._choose_save_path).grid(row=len(rows) + 1, column=2, padx=(6, 0))

    def _build_flag_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Flags", padding=10)
        frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))

        for index, (name, variable) in enumerate(self.flag_vars.items()):
            checkbox = ttk.Checkbutton(frame, text=name, variable=variable)
            checkbox.grid(row=index // 2, column=index % 2, sticky="w", padx=4, pady=2)

    def _build_action_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        self.run_button = ttk.Button(frame, text="Run analysis", command=self._run_analysis_async)
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(frame, text="Save figure", command=self._save_current_figure).grid(row=0, column=1, sticky="ew")

        ttk.Label(frame, textvariable=self.status_var, foreground="navy").grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _build_viewer_section(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)

        ttk.Label(header, text="Plot").grid(row=0, column=0, sticky="w")
        self.figure_selector = ttk.Combobox(header, state="readonly")
        self.figure_selector.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.figure_selector.bind("<<ComboboxSelected>>", self._on_plot_selected)

        self.canvas_host = ttk.Frame(parent)
        self.canvas_host.grid(row=1, column=0, sticky="nsew")
        self.canvas_host.rowconfigure(0, weight=1)
        self.canvas_host.columnconfigure(0, weight=1)

        log_frame = ttk.LabelFrame(parent, text="Run log", padding=8)
        log_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=8, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="ew")
        self.log_text.configure(state="disabled")

    def _choose_data_path(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.path_var.get() or str(DEFAULT_DATA_PATH))
        if selected:
            self.path_var.set(selected)

    def _choose_save_path(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.save_path_var.get() or str(DEFAULT_SAVE_PATH))
        if selected:
            self.save_path_var.set(selected)

    def _build_config(self) -> tuple[AnalysisConfig, AnalysisFlags]:
        try:
            config = AnalysisConfig(
                shot=int(self.shot_var.get()),
                src=self.src_var.get().strip(),
                key=self.key_var.get().strip(),
                time_window=(float(self.time_start_var.get()), float(self.time_end_var.get())),
                r_window=(float(self.r_start_var.get()), float(self.r_end_var.get())),
                rhop=float(self.rhop_var.get()),
                rlim=float(self.rlim_var.get()),
                path=Path(self.path_var.get()).expanduser(),
                pathsave=Path(self.save_path_var.get()).expanduser(),
            )
        except ValueError as exc:
            raise AnalysisError(f"Invalid numeric input: {exc}") from exc

        flags = AnalysisFlags(**{name: variable.get() for name, variable in self.flag_vars.items()})
        return config, flags

    def _run_analysis_async(self) -> None:
        try:
            config, flags = self._build_config()
        except AnalysisError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self.run_button.configure(state="disabled")
        self.status_var.set("Running analysis…")
        worker = threading.Thread(target=self._worker_run_analysis, args=(config, flags), daemon=True)
        worker.start()

    def _worker_run_analysis(self, config: AnalysisConfig, flags: AnalysisFlags) -> None:
        try:
            result = run_analysis(config, flags)
        except Exception as exc:  # noqa: BLE001
            self.result_queue.put(("error", (exc, traceback.format_exc())))
            return
        self.result_queue.put(("success", result))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.result_queue.get_nowait()
                if kind == "success":
                    self._handle_success(payload)
                else:
                    self._handle_error(*payload)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)

    def _handle_success(self, result: AnalysisResult) -> None:
        self.current_result = result
        self.current_figure_index = 0
        self.figure_selector["values"] = [artifact.title for artifact in result.figures]
        self.figure_selector.current(0)
        self._render_figure(0)
        self._set_log_lines(result.log_lines)
        self.status_var.set(f"Completed shot {result.config.shot}")
        self.run_button.configure(state="normal")

    def _handle_error(self, exc: Exception, tb_text: str) -> None:
        self._set_log_lines([str(exc), "", tb_text])
        self.status_var.set("Run failed")
        self.run_button.configure(state="normal")
        messagebox.showerror("Analysis failed", str(exc))

    def _set_log_lines(self, lines: list[str]) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, "\n".join(lines))
        self.log_text.configure(state="disabled")

    def _on_plot_selected(self, event: tk.Event | None = None) -> None:
        if self.current_result is None:
            return
        self._render_figure(self.figure_selector.current())

    def _render_figure(self, index: int) -> None:
        if self.current_result is None:
            return
        self.current_figure_index = index

        for child in self.canvas_host.winfo_children():
            child.destroy()

        figure = Figure(figsize=(9, 6), dpi=100, constrained_layout=True)
        self.current_result.figures[index].draw(figure)

        canvas = FigureCanvasTkAgg(figure, master=self.canvas_host)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(self.canvas_host)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=0, column=0, sticky="w")

        self.figure_canvas = canvas
        self.toolbar = toolbar

    def _save_current_figure(self) -> None:
        if self.current_result is None:
            messagebox.showinfo("No figure", "Run the analysis before saving a figure.")
            return

        default_name = self.current_result.figures[self.current_figure_index].title.lower().replace(" ", "_") + ".png"
        destination = filedialog.asksaveasfilename(
            initialdir=str(self.current_result.config.pathsave),
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")],
        )
        if not destination:
            return
        self.figure_canvas.figure.savefig(destination, dpi=150)
        self.status_var.set(f"Saved {Path(destination).name}")


def main() -> None:
    app = AnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()
