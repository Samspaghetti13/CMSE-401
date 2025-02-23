import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def wave_sim(xmin=0, xmax=10,nx=512,tmin=0,tmax=10,nt=1000,gamma=1):

    dx = (xmax-xmin)/nx
    x = np.linspace(xmin, xmax, nx)

    dt = (tmax-tmin)/nt
    times = np.linspace(tmin,tmax,nt)

    y = np.exp(-((x-5)**2))
    y_dot = np.zeros_like(x)
    y_ddot = np.zeros_like(x)

    history = [y.copy()]

    for t in times:
        y_ddot[1:-1] = gamma * (y[2:] + y[:-2] - 2 * y[1:-1]) / dx**2

        y_ddot[0] = 0
        y_ddot[-1] = 0

        y += y_dot * dt
        y_dot += y_ddot * dt

        if len(history) < 500:
            history.append(y.copy())
    return x, history

def timing_study():
    nx = 512
    nt_list = [10,100,1000,10000,100000,1000000]
    repetitions = 5
    
    for nt in nt_list:
        durations=[]
        for _ in range(repetitions):
            start_time = time.time()
            wave_sim(nt=nt, nx=nx)
            end_time = time.time()
            durations.append(end_time - start_time)
            
        avg_duration = np.mean(durations)
        print(f"Average runtime for nt={nt}: {avg_duration:.4f} seconds over {repetitions} runs.")

def visualize_wave():
    x, history = wave_sim(nt = 1000)
    fig, ax = plt.subplots()
    line, = ax.plot(x, history[0], color='blue')
    ax.set_xlim(0,10)
    ax.set_ylim(-1,1)
    ax.set_title("1D Wave Equation")
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Amplitude (y)")

    def update(frame):
        line.set_ydata(history[frame])
        return line,

    ani = FuncAnimation(fig, update, frames = len(history), blit=True, interval=30)
    ani.save("wave_simulation.gif", writer="imagemagick")
    plt.show()

if __name__ == "__main__":
    visualize_wave()

