import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

images = [np.random.rand(10, 10) for _ in range(9)]  # replace this with your images
colors = ['red'] * 9

fig, axs = plt.subplots(3, 3)

submit_ax = plt.axes([0.45, 0.05, 0.1, 0.075])  # Position of the 'Submit' button
button = Button(submit_ax, 'Submit', color='lightgoldenrodyellow')
button.label.set_fontsize(14)

fig.suptitle('Select all good-looking instances', fontsize=14)  # Title above images
fig.text(0.95, 0.05, 'Info Label', ha='right', bbox=dict(facecolor='lightgray', alpha=1.0), fontsize=14)


def plot_images(_images, _colors, _axs):
    for img, color, ax in zip(_images, _colors, _axs.flatten()):
        ax.clear()  # Clear previous plot
        ax.imshow(img)
        for spine in ax.spines.values():  # Change border color
            spine.set_edgecolor(color)
            spine.set_linewidth(3.5)  # Increase border width
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
    fig.canvas.draw()  # Redraw the canvas


def on_click(event):
    global colors
    if submit_ax.in_axes(event):
        # If the click occurred within the "Submit" button, ignore it
        return
    for i, ax in enumerate(axs.flatten()):
        if ax.in_axes(event) and colors[i] == 'red':
            colors[i] = 'green'
            break
        elif ax.in_axes(event) and colors[i] == 'green':
            colors[i] = 'red'
            break
    plot_images(images, colors, axs)


def on_submit(_event):
    print('Final colors:', colors)  # Replace this with your function


def main():
    plot_images(images, colors, axs)
    fig.canvas.mpl_connect('button_press_event', on_click)
    button.on_clicked(on_submit)  # Connect the button press event to the on_submit function
    plt.subplots_adjust(bottom=0.2)  # Make room for the button
    plt.tight_layout()  # Adjust subplot positions
    plt.show()


if __name__ == '__main__':
    main()
