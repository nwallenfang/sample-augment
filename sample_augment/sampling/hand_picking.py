import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button

from sample_augment.models.generator import StyleGANGenerator, GC10_CLASSES
from sample_augment.utils.path_utils import shared_dir
"""
this script is horrible code and was just used for selecting a hand-picked subset of synthetic images one-off
"""
if __name__ == '__main__':
    images = [np.random.rand(10, 10) for _ in range(9)]  # replace this with your images
    colors = ['red'] * 9
    current_class_idx = 0
    class_counts = {}
    TARGET_CLASS_COUNTS = 50
    SEED = 1
    latents = []

    fig, axs = plt.subplots(3, 3)

    submit_ax = plt.axes([0.45, 0.05, 0.1, 0.075])  # Position of the 'Submit' button
    button = Button(submit_ax, 'Submit', color='lightgoldenrodyellow')
    button.label.set_fontsize(14)

    # save_ax = plt.axes([0.85, 0.05, 0.1, 0.075])  # Position of the 'Submit' button
    # save_button = Button(save_ax, 'Save', color='lightgoldenrodyellow')
    # button.label.set_fontsize(14)

    title = fig.suptitle(f'Select good {GC10_CLASSES[current_class_idx]} instances', fontsize=14)  # Title above images
    info_text = fig.text(0.95, 0.05, 'Info Label', ha='right', bbox=dict(facecolor='lightgray', alpha=1.0), fontsize=14)

    gen_name = 'wdataaug-028_012200'
    gen = StyleGANGenerator.load_from_name('wdataaug-028_012200')

    handpicked_dir = shared_dir / 'generated' / f'handpicked-{gen_name}'
    handpicked_dir.mkdir(exist_ok=True)

    for i, class_name in enumerate(GC10_CLASSES):
        class_dir = handpicked_dir / class_name
        if class_dir.is_dir():
            num_files = len(list(class_dir.glob('*')))
            assert num_files % 2 == 0
            class_counts[i] = int(num_files / 2)
        else:
            class_dir.mkdir(exist_ok=True)
            class_counts[i] = 0

    print("-- Class Counts --")
    print(class_counts)


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
        global images, current_class_idx, latents, colors
        selected_idx = [i for i in range(9) if colors[i] == 'green']
        selected_imgs = [images[i] for i in selected_idx]
        selected_latents = [latents[i] for i in selected_idx]
        _class_dir = handpicked_dir / GC10_CLASSES[current_class_idx]

        for img, latent in zip(selected_imgs, selected_latents):
            plt.imsave(_class_dir / f'hand_picked_{class_counts[current_class_idx]}.jpg', img)
            np.save(_class_dir / f'hand_picked_{class_counts[current_class_idx]}.npy', latent)
            class_counts[current_class_idx] += 1

        # go to next class if class count reached
        if class_counts[current_class_idx] >= TARGET_CLASS_COUNTS:
            title.set_text(f'Select good {GC10_CLASSES[current_class_idx]} instances')
            current_class_idx += 1

        c = torch.zeros((9, 10))
        c[:, current_class_idx] = 1.0

        global SEED
        w = gen.c_to_w(c, seed=SEED)
        SEED += 1
        # generate new images
        gen_images = gen.w_to_img(w).cpu().numpy()

        global images, latents, info_text
        latents = w.cpu().numpy()
        images = gen_images

        # reset colors
        colors = ['red'] * 9
        info_text.set_text(f'{class_counts[current_class_idx]} / {TARGET_CLASS_COUNTS}')
        plot_images(gen_images, colors, axs)


    def setup_handpicking():
        # determine initial class idx
        for i, num in class_counts.items():
            if num < TARGET_CLASS_COUNTS:
                current_class_idx = i
                break
        else:
            print('All classes picked :)')
        global info_text
        info_text.set_text(f'{class_counts[current_class_idx]} / {TARGET_CLASS_COUNTS}')
        title.set_text(f'Select good {GC10_CLASSES[current_class_idx]} instances')

        c = torch.zeros((9, 10))
        c[:, current_class_idx] = 1.0
        global SEED
        w = gen.c_to_w(c, seed=SEED)
        SEED += 1
        # generate new images
        gen_images = gen.w_to_img(w).cpu().numpy()

        global images, latents
        latents = w.cpu().numpy()
        images = gen_images

        plot_images(images, colors, axs)
        fig.canvas.mpl_connect('button_press_event', on_click)
        button.on_clicked(on_submit)  # Connect the button press event to the on_submit function
        plt.subplots_adjust(bottom=0.2)  # Make room for the button
        # plt.tight_layout()  # Adjust subplot positions
        plt.show()


    setup_handpicking()
