import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# 1. Символьные вычисления в SymPy

print("Анализ потенциальной энергии (SymPy)")

# Определяем символьные переменные
theta, m, g, L = sp.symbols('theta m g L', real=True, positive=True)

U = m * g * L * (1 - sp.cos(theta))

# Находим производные
dU_dtheta = sp.diff(U, theta)
d2U_dtheta2 = sp.diff(U, theta, 2)

print(f"Потенциальная энергия U(θ) = {U}")
print(f"Первая производная dU/dθ = {dU_dtheta}")
print(f"Вторая производная d²U/dθ² = {d2U_dtheta2}")

# Ищем положение равновесия (где первая производная равна нулю)
equilibrium_points = sp.solve(dU_dtheta, theta)
print(f"Точки равновесия (решения solve(dU/dθ = 0)): {equilibrium_points}")

# Проверяем theta = 0 (устойчивое равновесие, т.к. вторая производная > 0)
is_stable_at_zero = d2U_dtheta2.subs(theta, 0) > 0
print(f"Является ли θ=0 положением устойчивого равновесия? {is_stable_at_zero}\n")


# 2. Численное интегрирование в SciPy

print("Численное решение ОДУ (SciPy)")

# Заданные параметры
g_over_L = 4.0
theta0 = 2.5
omega0 = 0.0
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 500) # точки для графика

# Функция для solve_ivp (y[0] = theta, y[1] = omega)
def pendulum_ode(t, y):
    theta_val, omega_val = y
    dtheta_dt = omega_val
    domega_dt = -g_over_L * np.sin(theta_val)
    return [dtheta_dt, domega_dt]

# Решение
sol = solve_ivp(pendulum_ode, t_span, [theta0, omega0], t_eval=t_eval, method='RK45')

t = sol.t
theta_vals = sol.y[0]
omega_vals = sol.y[1]

# 3. Построение графиков в Matplotlib

print(" Построение графиков ")
# Создаем окно с двумя подграфиками (1 строка, 2 столбца)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 1. График зависимости угла от времени
ax1.plot(t, theta_vals, color='blue', linewidth=2)
ax1.set_title('Зависимость угла θ(t) от времени')
ax1.set_xlabel('Время t, с')
ax1.set_ylabel('Угол θ, рад')
ax1.grid(True)
ax1.axhline(0, color='black', linewidth=0.8, linestyle='--') # Ось нуля

# 2. Фазовый портрет (θ' от θ)
ax2.plot(theta_vals, omega_vals, color='red', linewidth=2)
ax2.set_title('Фазовый портрет θ\'(θ)')
ax2.set_xlabel('Угол θ, рад')
ax2.set_ylabel('Угловая скорость θ\', рад/с')
ax2.grid(True)
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.axvline(0, color='black', linewidth=0.8, linestyle='--')

# Красиво размещаем элементы
plt.tight_layout()
plt.show()