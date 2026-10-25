"""
МД-симуляция 2D жидкости Леннарда-Джонса.

Запуск из CLI:
    python lab2.py --n 200 --rho 0.3 --T 1.0 --steps 20000
    python lab2.py --mode euler      # сравнение Верле vs Эйлер
    python lab2.py --mode relax      # демонстрация релаксации
    python lab2.py --mode all        # всё сразу

Тесты:
    pytest lab2.py -v
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # безопасно для headless; при интерактиве можно убрать
import matplotlib.pyplot as plt

# ============================================================
# Параметры по умолчанию (приведённые единицы: m = sigma = epsilon = 1)
# ============================================================
DEFAULTS = dict(
    N=200,
    rho=0.3,
    T_target=1.0,
    dt=0.005,
    N_steps=20000,
    N_equil=5000,
    N_sample=50,
    cutoff=2.5,
    seed=42,
)


# ============================================================
# Инициализация
# ============================================================
def init_positions(N, L, rng):
    """Начальные координаты на решётке + небольшой шум."""
    n_side = int(np.ceil(np.sqrt(N)))
    spacing = L / n_side
    pos = []
    for i in range(n_side):
        for j in range(n_side):
            if len(pos) < N:
                pos.append([(i + 0.5) * spacing, (j + 0.5) * spacing])
    pos = np.array(pos)
    pos += rng.uniform(-0.05, 0.05, pos.shape)
    return pos


def init_velocities_maxwell(N, T_target, rng):
    """Максвелловские скорости с убранным суммарным импульсом и точной T."""
    vel = rng.normal(0.0, np.sqrt(T_target), size=(N, 2))
    vel -= vel.mean(axis=0)
    # В 2D: K = 0.5 * sum(v^2); после удаления импульса степеней свободы 2N-2
    # Хотим K/(N-1) = T  =>  K = (N-1)*T
    KE = 0.5 * np.sum(vel**2)
    vel *= np.sqrt((N - 1) * T_target / KE)
    return vel


def init_velocities_equal_speed(N, T_target, rng):
    """Все частицы с одинаковым |v| (для демонстрации релаксации)."""
    angles = rng.uniform(0, 2 * np.pi, size=N)
    # |v| такой, чтобы K/(N-1) = T
    v_mag = np.sqrt(2.0 * (N - 1) * T_target / N)
    vel = v_mag * np.column_stack([np.cos(angles), np.sin(angles)])
    vel -= vel.mean(axis=0)  # осторожно: чуть меняет |v|, но сохраняет T
    KE = 0.5 * np.sum(vel**2)
    vel *= np.sqrt((N - 1) * T_target / KE)
    return vel


# ============================================================
# Силы Леннарда-Джонса (2D)
# ============================================================
def compute_forces(pos, L, cutoff=2.5):
    """
    Возвращает силы (N,2) и потенциальную энергию U.

    ВАЖНО: суммируем только по парам i < j (np.triu(..., k=1)),
    иначе каждая пара учитывается дважды.
    """
    N = len(pos)
    forces = np.zeros_like(pos)
    r2_cut = cutoff**2
    U_cut = 4.0 * (1.0 / r2_cut**6 - 1.0 / r2_cut**3)

    d = pos[:, None, :] - pos[None, :, :]     # (N, N, 2)
    d -= L * np.round(d / L)                  # минимальный образ
    r2 = np.sum(d * d, axis=-1)

    # Только пары i < j
    mask = np.triu(r2 < r2_cut, k=1)

    idx_i, idx_j = np.where(mask)
    d_pairs = d[idx_i, idx_j]                 # (M, 2)
    r2_pairs = r2[idx_i, idx_j]               # (M,)

    inv_r2 = 1.0 / r2_pairs
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2

    # U = sum по парам [4*(r^-12 - r^-6) - U_cut]
    U = np.sum(4.0 * (inv_r12 - inv_r6) - U_cut)

    # F_ij = 24*(2*inv_r12 - inv_r6) * inv_r2 * d_ij  (действует на i)
    coef = 24.0 * (2.0 * inv_r12 - inv_r6) * inv_r2
    f_contrib = coef[:, None] * d_pairs

    np.add.at(forces, idx_i, f_contrib)
    np.add.at(forces, idx_j, -f_contrib)

    return forces, U


def kinetic_energy(vel):
    return 0.5 * np.sum(vel**2)


def temperature(vel):
    """T = K/(N-1) в 2D после удаления суммарного импульса (2N-2 dof)."""
    return kinetic_energy(vel) / (len(vel) - 1)


# ============================================================
# Интеграторы
# ============================================================
def velocity_verlet_step(pos, vel, forces, dt, L, cutoff=2.5):
    vel += 0.5 * dt * forces
    pos += dt * vel
    pos %= L
    forces_new, U = compute_forces(pos, L, cutoff)
    vel += 0.5 * dt * forces_new
    return pos, vel, forces_new, U


def euler_step(pos, vel, forces, dt, L, cutoff=2.5):
    """Явный Эйлер (для сравнения дрейфа энергии)."""
    pos = (pos + dt * vel) % L
    vel = vel + dt * forces
    forces_new, U = compute_forces(pos, L, cutoff)
    return pos, vel, forces_new, U


# ============================================================
# Основной прогон МД
# ============================================================
def run_md(N=200, rho=0.3, T_target=1.0, dt=0.005,
           N_steps=20000, N_equil=5000, N_sample=50,
           cutoff=2.5, seed=42, integrator="verlet",
           init_mode="maxwell", return_traj=False):
    """
    Возвращает dict с результатами:
      T_hist, E_hist, V (сэмплы скоростей), pos, vel, U,
      T_measured, E_mean, E_std, E0
    """
    rng = np.random.default_rng(seed)
    L = np.sqrt(N / rho)

    pos = init_positions(N, L, rng)
    if init_mode == "maxwell":
        vel = init_velocities_maxwell(N, T_target, rng)
    elif init_mode == "equal_speed":
        vel = init_velocities_equal_speed(N, T_target, rng)
    else:
        raise ValueError(f"unknown init_mode: {init_mode}")

    step_fn = velocity_verlet_step if integrator == "verlet" else euler_step

    forces, U = compute_forces(pos, L, cutoff)
    E0 = kinetic_energy(vel) + U

    T_hist, E_hist, v_samples = [], [], []
    traj_pos = [] if return_traj else None

    for step in range(N_steps):
        pos, vel, forces, U = step_fn(pos, vel, forces, dt, L, cutoff)

        if step >= N_equil:
            T_hist.append(temperature(vel))
            E_hist.append(kinetic_energy(vel) + U)
            if (step - N_equil) % N_sample == 0:
                v_samples.append(vel.copy())

        if return_traj and step % max(1, N_steps // 50) == 0:
            traj_pos.append(pos.copy())

    T_hist = np.array(T_hist)
    E_hist = np.array(E_hist)
    V = np.vstack(v_samples) if v_samples else np.zeros((0, 2))

    out = dict(
        T_hist=T_hist, E_hist=E_hist, V=V,
        pos=pos, vel=vel, U=U,
        T_measured=float(np.mean(T_hist)) if len(T_hist) else float("nan"),
        E_mean=float(np.mean(E_hist)) if len(E_hist) else float("nan"),
        E_std=float(np.std(E_hist)) if len(E_hist) else float("nan"),
        E0=float(E0),
        L=L,
    )
    if return_traj:
        out["traj_pos"] = np.array(traj_pos)
    return out


# ============================================================
# Графики
# ============================================================
def plot_maxwell(result, outfile="md_lj_2d_maxwell.png"):
    V = result["V"]
    T = result["T_measured"]
    E_hist = result["E_hist"]
    if len(V) == 0:
        print("[plot_maxwell] нет сэмплов скоростей — пропуск")
        return

    vx, vy = V[:, 0], V[:, 1]
    speed = np.sqrt(vx**2 + vy**2)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    axs[0].hist(vx, bins=60, density=True, alpha=0.7, color="steelblue", label="MD")
    v_grid = np.linspace(-4, 4, 200)
    f_norm = 1 / np.sqrt(2 * np.pi * T) * np.exp(-v_grid**2 / (2 * T))
    axs[0].plot(v_grid, f_norm, "r-", lw=2, label=f"N(0, T={T:.3f})")
    axs[0].set_xlabel("v_x"); axs[0].set_ylabel("P(v_x)")
    axs[0].set_title("Компонента скорости"); axs[0].legend()

    axs[1].hist(speed, bins=60, density=True, alpha=0.7, color="seagreen", label="MD")
    v_grid = np.linspace(0, speed.max(), 300)
    f_speed = (v_grid / T) * np.exp(-v_grid**2 / (2 * T))
    axs[1].plot(v_grid, f_speed, "r-", lw=2, label=f"v/T·e^(-v²/2T), T={T:.3f}")
    axs[1].set_xlabel("|v|"); axs[1].set_ylabel("f(|v|)")
    axs[1].set_title("Распределение модуля скорости"); axs[1].legend()

    axs[2].plot(E_hist, lw=0.8)
    axs[2].set_xlabel("шаг"); axs[2].set_ylabel("E = K + U")
    axs[2].set_title("Полная энергия (после термализации)")
    axs[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close(fig)
    print(f"[plot_maxwell] сохранено: {outfile}")


def plot_energy_drift(dt_list, N=200, rho=0.3, T_target=1.0,
                      N_steps=2000, outfile="md_energy_drift.png"):
    """Относительный дрейф (E - E0)/E0 для Верле и Эйлера при разных dt."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))

    for integ, ax_i in zip(["verlet", "euler"], ax):
        for dt in dt_list:
            res = run_md(N=N, rho=rho, T_target=T_target, dt=dt,
                         N_steps=N_steps, N_equil=0, N_sample=10**9,
                         integrator=integ, seed=42)
            # Полная энергия по всем шагам нам не отдаётся — пересчитаем
            # траекторию коротко ещё раз с сохранением E на каждом шаге.
            rng = np.random.default_rng(42)
            L = np.sqrt(N / rho)
            pos = init_positions(N, L, rng)
            vel = init_velocities_maxwell(N, T_target, rng)
            forces, U = compute_forces(pos, L)
            E0 = kinetic_energy(vel) + U
            Es = [E0]
            step_fn = velocity_verlet_step if integ == "verlet" else euler_step
            for _ in range(N_steps):
                pos, vel, forces, U = step_fn(pos, vel, forces, dt, L)
                Es.append(kinetic_energy(vel) + U)
            Es = np.array(Es)
            drift = (Es - E0) / abs(E0)
            ax_i.plot(drift, lw=1.0, label=f"dt={dt:g}")
        ax_i.set_title(f"{integ.capitalize()}")
        ax_i.set_xlabel("шаг"); ax_i.set_ylabel("(E - E₀)/E₀")
        ax_i.grid(alpha=0.3); ax_i.legend()

    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close(fig)
    print(f"[plot_energy_drift] сохранено: {outfile}")


def plot_relaxation(N=200, rho=0.3, T_target=1.0, dt=0.005,
                    N_steps=3000, outfile="md_relaxation.png",
                    snapshots=(0, 50, 200, 1000)):
    """Старт со всех одинаковых |v| -> релаксация к распределению Рэлея."""
    rng = np.random.default_rng(42)
    L = np.sqrt(N / rho)
    pos = init_positions(N, L, rng)
    vel = init_velocities_equal_speed(N, T_target, rng)
    forces, U = compute_forces(pos, L)

    snaps = {}
    all_v = []
    for step in range(N_steps + 1):
        if step in snapshots:
            snaps[step] = vel.copy()
        all_v.append(vel.copy())
        if step < N_steps:
            pos, vel, forces, U = velocity_verlet_step(pos, vel, forces, dt, L)

    fig, axs = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4))
    if len(snapshots) == 1:
        axs = [axs]
    for ax, s in zip(axs, snapshots):
        V = snaps[s]
        sp = np.sqrt(np.sum(V**2, axis=1))
        ax.hist(sp, bins=30, density=True, alpha=0.7, color="steelblue", label=f"t={s*dt:.2f}")
        v_grid = np.linspace(0, max(sp.max(), 3.0), 300)
        # T оцениваем по текущему срезу
        T_now = 0.5 * np.mean(np.sum(V**2, axis=1)) * N / (N - 1)
        f = (v_grid / T_now) * np.exp(-v_grid**2 / (2 * T_now))
        ax.plot(v_grid, f, "r-", lw=2, label=f"Рэлей, T={T_now:.2f}")
        ax.set_xlabel("|v|"); ax.set_ylabel("f(|v|)")
        ax.set_title(f"шаг {s}"); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close(fig)
    print(f"[plot_relaxation] сохранено: {outfile}")


def report_checks(result):
    V = result["V"]
    T = result["T_measured"]
    print(f"Измеренная температура (2D): T = {T:.4f}")
    if len(V):
        vx, vy = V[:, 0], V[:, 1]
        sp = np.sqrt(vx**2 + vy**2)
        print(f"<v_x²> = {np.mean(vx**2):.4f}  (ожидание T = {T:.4f})")
        print(f"<v_y²> = {np.mean(vy**2):.4f}  (ожидание T = {T:.4f})")
        print(f"<v²>   = {np.mean(sp**2):.4f}  (ожидание 2T = {2*T:.4f})")
        from scipy.stats import kstest
        cdf = lambda v: 1 - np.exp(-v**2 / (2 * T))
        ks = kstest(sp, cdf)
        # ВНИМАНИЕ: p-value неинформативно — сэмплы коррелированы,
        # а T оценена по тем же данным. Смотрим на statistic.
        print(f"KS модуля скорости: statistic={ks.statistic:.4f}, p={ks.pvalue:.4g} "
              f"(p неинформативно: коррелированные сэмплы)")
    print(f"Средняя полная энергия: {result['E_mean']:.4f} ± {result['E_std']:.4f}")


# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser(description="2D LJ MD")
    p.add_argument("--mode", choices=["maxwell", "euler", "relax", "all"],
                   default="all")
    p.add_argument("--n", type=int, default=DEFAULTS["N"])
    p.add_argument("--rho", type=float, default=DEFAULTS["rho"])
    p.add_argument("--T", type=float, default=DEFAULTS["T_target"])
    p.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    p.add_argument("--steps", type=int, default=DEFAULTS["N_steps"])
    p.add_argument("--equil", type=int, default=DEFAULTS["N_equil"])
    p.add_argument("--sample", type=int, default=DEFAULTS["N_sample"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    args = p.parse_args()

    if args.mode in ("maxwell", "all"):
        res = run_md(N=args.n, rho=args.rho, T_target=args.T, dt=args.dt,
                     N_steps=args.steps, N_equil=args.equil,
                     N_sample=args.sample, seed=args.seed,
                     integrator="verlet", init_mode="maxwell")
        report_checks(res)
        plot_maxwell(res)

    if args.mode in ("euler", "all"):
        plot_energy_drift(dt_list=[0.001, 0.002, 0.005])

    if args.mode in ("relax", "all"):
        plot_relaxation(N=args.n, rho=args.rho, T_target=args.T, dt=args.dt)


# ============================================================
# pytest-тесты
# ============================================================
def test_force_two_particles_r1():
    """Две частицы на расстоянии r=1: |F| = 24 (по модулю)."""
    pos = np.array([[0.0, 0.0], [1.0, 0.0]])
    L = 100.0  # большой бокс, чтобы минимальный образ не влиял
    forces, U = compute_forces(pos, L, cutoff=2.5)
    # Частица 0 отталкивается от 1: F0 направлена от 1 к 0, т.е. -x
    assert np.isclose(forces[0, 0], -24.0), forces[0, 0]
    assert np.isclose(forces[1, 0], +24.0), forces[1, 0]
    assert np.isclose(forces[0, 1], 0.0)
    assert np.isclose(forces[1, 1], 0.0)


def test_force_zero_at_rmin():
    """При r = 2^(1/6) сила Леннарда-Джонса равна нулю."""
    r_min = 2.0 ** (1.0 / 6.0)
    pos = np.array([[0.0, 0.0], [r_min, 0.0]])
    L = 100.0
    forces, _ = compute_forces(pos, L, cutoff=2.5)
    assert np.allclose(forces, 0.0, atol=1e-10), forces


def test_energy_two_particles_r1():
    """U(r=1) = 4*(1 - 1) - U_cut = -U_cut (по паре, без двойного счёта)."""
    pos = np.array([[0.0, 0.0], [1.0, 0.0]])
    L = 100.0
    cutoff = 2.5
    r2_cut = cutoff**2
    U_cut = 4.0 * (1.0 / r2_cut**6 - 1.0 / r2_cut**3)
    _, U = compute_forces(pos, L, cutoff)
    # Внутри cutoff: 4*(1 - 1) = 0, минус U_cut
    assert np.isclose(U, -U_cut), U


def test_no_double_counting_N3():
    """Для трёх частиц U должна быть суммой по трём парам (не шести)."""
    r_min = 2.0 ** (1.0 / 6.0)
    pos = np.array([[0.0, 0.0], [r_min, 0.0], [0.0, r_min]])
    L = 100.0
    cutoff = 2.5
    r2_cut = cutoff**2
    U_cut = 4.0 * (1.0 / r2_cut**6 - 1.0 / r2_cut**3)
    _, U = compute_forces(pos, L, cutoff)
    # Две пары на r_min (U=0 каждая) + одна на r_min*sqrt(2) с U != 0
    r2 = 2 * r_min**2
    u_one = 4.0 * (r2**-6 - r2**-3) - U_cut
    assert np.isclose(U, u_one), (U, u_one)


def test_temperature_definition():
    """T = K/(N-1) в 2D после удаления импульса."""
    N = 100
    rng = np.random.default_rng(0)
    vel = init_velocities_maxwell(N, T_target=1.5, rng=rng)
    T = temperature(vel)
    assert np.isclose(T, 1.5, atol=1e-10), T


def test_momentum_removed():
    N = 50
    rng = np.random.default_rng(1)
    vel = init_velocities_maxwell(N, 1.0, rng)
    assert np.allclose(vel.sum(axis=0), 0.0, atol=1e-12)


# ============================================================
# Точка входа
# ============================================================
if __name__ == "__main__":
    main()
