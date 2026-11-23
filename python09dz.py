import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Параметры двойного маятника

m1 = 1.0
m2 = 1.0

L1 = 1.0
L2 = 1.0

g = 9.81

t0 = 0
t1 = 20

N = 1000

rtol = 1e-8
atol = 1e-8

# Начальные условия

y0 = np.array([
    np.pi / 2,
    0,
    np.pi / 2 + 0.01,
    0
])
# Начальные условия для проверки неустойчивости
delta = 1e-6

y0_A = y0.copy()

y0_B = np.array([
    np.pi / 2,
    0,
    np.pi / 2 + 0.01 + delta,
    0
])

# Вывод параметров

print("Двойной маятник")
print("Масса первого груза:", m1)
print("Масса второго груза:", m2)
print("Начальные условия:", y0)

# 4. СИСТЕМА ДИФФЕРЕНЦИАЛЬНЫХ УРАВНЕНИЙ


def equations(t, y):

    # Получаем переменные состояния
    theta1, omega1, theta2, omega2 = y

    # Разность углов
    delta = theta1 - theta2

    # Общий знаменатель
    D = 2*m1 + m2 - m2*np.cos(2*delta)

    # Угловое ускорение первого стержня
    domega1 = (
        -g*(2*m1 + m2)*np.sin(theta1)
        -m2*g*np.sin(theta1 - 2*theta2)
        -2*m2*np.sin(delta)*(
            omega2**2*L2
            +omega1**2*L1*np.cos(delta)
        )
    ) / (L1*D)

    # Угловое ускорение второго стержня
    domega2 = (
        2*np.sin(delta)*(
            omega1**2*L1*(m1 + m2)
            +g*(m1 + m2)*np.cos(theta1)
            +omega2**2*L2*m2*np.cos(delta)
        )
    ) / (L2*D)

    # Возвращаем четыре производные
    return np.array([
        omega1,
        domega1,
        omega2,
        domega2
    ])

# 5. ПРОВЕРКА ФУНКЦИИ

print("Проверка системы уравнений:")

print(equations(0, y0))

# 6. ЧИСЛЕННОЕ РЕШЕНИЕ СИСТЕМЫ УРАВНЕНИЙ


t = np.linspace(t0, t1, N)

solution = solve_ivp(
    equations,
    (t0, t1),
    y0,
    t_eval=t,
    rtol=rtol,
    atol=atol
)
# Решение A
solution_A = solve_ivp(
    equations,
    (t0, t1),
    y0_A,
    t_eval=t,
    rtol=rtol,
    atol=atol
)

# Решение B
solution_B = solve_ivp(
    equations,
    (t0, t1),
    y0_B,
    t_eval=t,
    rtol=rtol,
    atol=atol
)

print("Численное решение системы:")

if solution.success:
    print("Расчёт выполнен успешно!")
    print("Количество точек:", len(solution.t))
else:
    print("Ошибка расчёта:", solution.message)


    
# 7. СОЗДАНИЕ АНИМАЦИИ ДВОЙНОГО МАЯТНИКА

# 8. ПРОВЕРКА НЕУСТОЙЧИВОСТИ

# Координаты для решения A
x1_A = L1 * np.sin(solution_A.y[0])
y1_A = -L1 * np.cos(solution_A.y[0])

x2_A = x1_A + L2 * np.sin(solution_A.y[2])
y2_A = y1_A - L2 * np.cos(solution_A.y[2])

# Координаты для решения B
x1_B = L1 * np.sin(solution_B.y[0])
y1_B = -L1 * np.cos(solution_B.y[0])

x2_B = x1_B + L2 * np.sin(solution_B.y[2])
y2_B = y1_B - L2 * np.cos(solution_B.y[2])

# Построение двух графиков
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x2_A, y2_A)
ax[0].set_title("Траектория A")

ax[1].plot(x2_B, y2_B)
ax[1].set_title("Траектория B")

for a in ax:
    a.set_aspect('equal')
    a.grid()
    a.set_xlabel("X")
    a.set_ylabel("Y")

plt.tight_layout()
plt.show()

# 9. ПРОВЕРКА СОХРАНЕНИЯ ЭНЕРГИИ

def energy(y):
    theta1, omega1, theta2, omega2 = y

    delta = theta1 - theta2

    T = (
        (m1 + m2) / 2 * L1**2 * omega1**2
        + m2 / 2 * L2**2 * omega2**2
        + m2 * L1 * L2 * omega1 * omega2 * np.cos(delta)
    )

    V = (
        -(m1 + m2) * g * L1 * np.cos(theta1)
        - m2 * g * L2 * np.cos(theta2)
    )

    return T + V

E = np.array([energy(solution_A.y[:, i])
              for i in range(len(t))])

relative_error = np.abs(E - E[0]) / abs(E[0])

plt.figure(figsize=(8, 5))
plt.plot(t, relative_error)

plt.title("Относительное изменение энергии")
plt.xlabel("Время")
plt.ylabel("Относительная ошибка")
plt.grid()
plt.show()

print("Максимальная ошибка энергии:", np.max(relative_error))

# Координаты первого груза
x1 = L1 * np.sin(solution.y[0])
y1 = -L1 * np.cos(solution.y[0])

# Координаты второго груза
x2 = x1 + L2 * np.sin(solution.y[2])
y2 = y1 - L2 * np.cos(solution.y[2])

# Создание графика
fig, ax = plt.subplots(figsize=(8, 8))

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.grid()

ax.set_title("Анимация двойного маятника")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Стержни и грузы
line, = ax.plot([], [], 'o-', lw=3, markersize=10)
trail, = ax.plot([], [], '-', color='red', lw=1.5)
# Функция обновления кадров
def update(frame):
    line.set_data(
        [0, x1[frame], x2[frame]],
        [0, y1[frame], y2[frame]]
    )

    # Рисуем след за нижним грузом
    trail.set_data(
        x2[:frame + 1],
        y2[:frame + 1]
    )

    return line, trail

# Создание анимации
ani = FuncAnimation(
    fig,
    update,
    frames=N,
    interval=20,
    blit=False
)

plt.show()
# 8. Проверка устойчивости системы

difference = np.sqrt(
    (solution_A.y[0] - solution_B.y[0])**2 +
    (solution_A.y[2] - solution_B.y[2])**2
)

plt.figure(figsize=(8, 5))
plt.plot(solution_A.t, difference)

plt.title("Расхождение траекторий A и B")
plt.xlabel("Время")
plt.ylabel("Разница углов")
plt.grid()

plt.show()

# 9. Оценка показателя Ляпунова

log_difference = np.log(np.maximum(difference, 1e-12))

plt.figure(figsize=(8, 5))

plt.plot(solution_A.t, log_difference)

plt.title("Логарифм расхождения траекторий")
plt.xlabel("Время")
plt.ylabel("ln(разницы углов)")
plt.grid()

plt.show()

# 10. Расчёт показателя Ляпунова

mask = (solution_A.t >= 9) & (solution_A.t <= 13)

coefficients = np.polyfit(
    solution_A.t[mask],
    log_difference[mask],
    1
)

lyapunov = coefficients[0]

print("Показатель Ляпунова:", lyapunov)